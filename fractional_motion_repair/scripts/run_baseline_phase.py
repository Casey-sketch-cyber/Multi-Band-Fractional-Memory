"""Run each baseline fold and seed in a pinned GPU worker."""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from traceback import print_exc

PAIRS = tuple(itertools.product(range(5), (20261201, 20261202, 20261203)))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def worker(args):
    import torch

    sys.path.insert(0, str(args.source))
    from baseline_train import train_saits, validate
    from baseline_test import evaluate

    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print(json.dumps({"device": torch.cuda.get_device_name(0), "pid": os.getpid()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        fold, seed = request["fold"], request["seed"]
        directory = args.runs / f"fold{fold}" / f"seed{seed}"
        try:
            with Path(request["log"]).open("x", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if args.phase == "evaluate":
                        evaluate(argparse.Namespace(freeze=args.freeze, fold=fold, seed=seed, device="cuda"))
                    else:
                        opts = dict(fold=fold, seed=seed, cache=args.cache, ignore_list=args.ignore_list, device="cuda")
                        train_saits(argparse.Namespace(**opts, out_dir=directory / "saits"))
                        if (fold, seed) == PAIRS[0]:
                            validate(argparse.Namespace(**opts, out_dir=args.state / "validation", runs=args.runs))
            print(json.dumps({"ok": True}), flush=True)
        except Exception as error:
            print_exc()
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
            break
        torch.cuda.empty_cache()


def dispatch(args):
    args.state.mkdir(parents=True, exist_ok=False)
    pending = queue.Queue()
    for pair in PAIRS:
        pending.put(pair)
    state = {"stage": args.phase, "jobs": {}, "failure": None, "started_at_utc": now()}
    guard, stop = Lock(), Event()
    path = args.state / "queue.json"

    def mark(key, stage, gpu, **fields):
        with guard:
            state["jobs"][key] = {"stage": stage, "gpu": gpu, "at_utc": now(), **fields}
            save(path, state)
        print(now(), key, stage, flush=True)

    def channel(gpu):
        proc = None
        key = f"gpu{gpu}"
        with (args.state / f"{key}.log").open("x", encoding="utf-8") as start:
            params = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), args.phase, "--worker"]
            for flag in ("source", "state", "runs", "cache", "ignore_list", "freeze"):
                value = getattr(args, flag)
                if value is not None:
                    params.extend(["--" + flag.replace("_", "-"), str(value)])
            try:
                proc = subprocess.Popen(params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=start, text=True,
                                        env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu)))
                ready = proc.stdout.readline()
                if not ready or json.loads(ready).get("device") != "NVIDIA RTX A6000":
                    raise RuntimeError(f"CUDA worker did not start on A6000 GPU {gpu}")
                while not stop.is_set():
                    try:
                        fold, seed = pending.get_nowait()
                    except queue.Empty:
                        break
                    key = f"fold{fold}-seed{seed}"
                    log = args.state / f"{key}.log"
                    mark(key, "running", gpu, worker_pid=proc.pid, log=str(log))
                    proc.stdin.write(json.dumps({"fold": fold, "seed": seed, "log": str(log)}) + "\n")
                    proc.stdin.flush()
                    line = proc.stdout.readline()
                    reply = json.loads(line) if line else {"ok": False, "error": "Worker exited"}
                    if not reply["ok"]:
                        raise RuntimeError(reply["error"])
                    mark(key, "complete", gpu, log=str(log))
            except Exception as error:
                stop.set()
                with guard:
                    state["failure"] = state["failure"] or {"job": key, "error": str(error)}
                    save(path, state)
                mark(key, "failed", gpu)
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        try:
                            proc.stdin.write('{"stop": true}\n')
                            proc.stdin.flush()
                        except BrokenPipeError:
                            pass
                    proc.wait()

    save(path, state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(channel, (0, 1)))
    if state["failure"] is not None or sum(job["stage"] == "complete" for job in state["jobs"].values()) != 15:
        raise RuntimeError(f"Baseline phase incomplete, see {path}")
    save(args.state / "phase_complete.json", {"completed_at_utc": now(), "phase": args.phase, "completed_pairs": 15})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train-saits", "evaluate"))
    parser.add_argument("--worker", action="store_true")
    for flag in ("source", "state", "runs"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    for flag in ("cache", "ignore-list", "freeze"):
        parser.add_argument(f"--{flag}", type=Path)
    args = parser.parse_args()
    if args.phase == "train-saits" and (args.cache is None or args.ignore_list is None):
        parser.error("train-saits requires --cache and --ignore-list")
    if args.phase == "evaluate" and args.freeze is None:
        parser.error("evaluate requires --freeze")
    (worker if args.worker else dispatch)(args)
