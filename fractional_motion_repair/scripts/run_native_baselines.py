"""Dispatch frozen-revision native baseline training without reading test motions."""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("bigru", "transformer", "patch_transformer")
PAIRS = tuple(itertools.product(range(5), (20261201, 20261202, 20261203)))


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def worker(args):
    import torch

    sys.path.insert(0, str(args.source.resolve(strict=True)))
    from kernel_test import digest, evaluation_runtime, file_record, source_hashes
    from raw_suite import main as train

    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    runtime = evaluation_runtime(torch.device("cuda"))
    sources = source_hashes()
    print(json.dumps({"ready": True, "pid": os.getpid(), "device": runtime["device_name"]}), flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            return
        fold, seed = request["fold"], request["seed"]
        directory = args.runs / f"fold{fold}" / f"seed{seed}" / "native"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            with (directory / "run.log").open("x", encoding="utf-8", buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    sys.argv = [
                        "raw_suite.py", "--raw-cache-dir", str(args.cache),
                        "--ignore-list", str(args.ignore_list), "--out-dir", str(directory),
                        "--fold", str(fold), "--seed", str(seed),
                        "--variants", ",".join(METHODS), "--epochs", "24",
                        "--batch-size", "16", "--train-windows", "2", "--val-windows", "3",
                    ]
                    train()
            manifest = json.loads((directory / "raw_suite_manifest.json").read_text())
            if manifest["test_set_accessed"] is not False or set(manifest["variants"]) != set(METHODS):
                raise ValueError("Baseline training manifest is invalid")
            results = {}
            for method in METHODS:
                history = json.loads((directory / f"{method}_history.json").read_text())
                if not history or any(
                    not math.isfinite(row[key]) for row in history for key in ("train_loss", "val_mpjpe")
                ):
                    raise ValueError(f"Nonfinite training history for {method}")
                best = history[0]
                for row in history[1:]:
                    if row["val_mpjpe"] < best["val_mpjpe"] - 1e-6:
                        best = row
                checkpoint_path = directory / f"{method}_best.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if any(not torch.isfinite(item).all() for item in checkpoint["model"].values()):
                    raise ValueError(f"Nonfinite selected checkpoint for {method}")
                results[method] = {"checkpoint": file_record(checkpoint_path),
                                   "selected_epoch": best["epoch"], "inner_val_mpjpe": best["val_mpjpe"],
                                   "trained_epochs": len(history), "history_sha256": digest(directory / f"{method}_history.json")}
            save(directory / "training_complete.json", {
                "fold": fold, "seed": seed, "test_set_accessed": False, "methods": results,
                "source_sha256": sources, "training_runtime": evaluation_runtime(torch.device("cuda")), "amp": True,
                "completed_at_utc": timestamp(),
            })
            print(json.dumps({"ok": True}), flush=True)
        except Exception as error:
            traceback.print_exc()
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
            return
        torch.cuda.empty_cache()


def dispatch(args):
    args.state.mkdir(parents=True, exist_ok=False)
    pending = queue.Queue()
    for pair in PAIRS:
        pending.put(pair)
    recorded = {"started_at_utc": timestamp(), "jobs": {}, "failure": None, "test_set_accessed": False}
    lock, stop = threading.Lock(), threading.Event()
    status = args.state / "queue.json"

    def note(key, stage, gpu, **kwargs):
        with lock:
            recorded["jobs"][key] = {"stage": stage, "gpu": gpu, "at_utc": timestamp(), **kwargs}
            save(status, recorded)
        print(timestamp(), key, stage, flush=True)

    def channel(gpu):
        child = None
        key = f"gpu{gpu}"
        try:
            with (args.state / f"gpu{gpu}.log").open("x", encoding="utf-8") as log:
                # Remap each worker's selected card to cuda:0.
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
                child = subprocess.Popen(
                    [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "worker",
                     "--source", str(args.source), "--cache", str(args.cache), "--ignore-list", str(args.ignore_list),
                     "--runs", str(args.runs), "--state", str(args.state)],
                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                )
                line = child.stdout.readline()
                if not line or not json.loads(line).get("ready"):
                    raise RuntimeError(f"CUDA worker exited before startup (GPU {gpu}); inspect gpu{gpu}.log")
                while not stop.is_set():
                    try:
                        fold, seed = pending.get_nowait()
                    except queue.Empty:
                        break
                    key = f"fold{fold}-seed{seed}"
                    note(key, "training", gpu, worker_pid=child.pid)
                    child.stdin.write(json.dumps({"fold": fold, "seed": seed}) + "\n")
                    child.stdin.flush()
                    line = child.stdout.readline()
                    result = json.loads(line) if line else {"ok": False, "error": "Training worker exited"}
                    if not result["ok"]:
                        raise RuntimeError(result["error"])
                    note(key, "complete", gpu)
        except Exception as error:
            stop.set()
            with lock:
                if recorded["failure"] is None:
                    recorded["failure"] = {"job": key, "error": str(error), "at_utc": timestamp()}
                save(status, recorded)
            note(key, "failed", gpu, error=str(error))
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        child.stdin.write('{"stop": true}\n')
                        child.stdin.flush()
                    except BrokenPipeError:
                        pass
                child.wait()

    save(status, recorded)
    workers = [threading.Thread(target=channel, args=(gpu,)) for gpu in (0, 1)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if recorded["failure"]:
        raise RuntimeError(f"Native baseline queue stopped, see {status}")
    if sum(item["stage"] == "complete" for item in recorded["jobs"].values()) != len(PAIRS):
        raise RuntimeError("Native baseline queue did not complete the full fold/seed grid")
    save(args.state / "training_complete.json", {"completed_at_utc": timestamp(),
         "completed_pairs": len(PAIRS), "completed_models": len(PAIRS) * len(METHODS), "test_set_accessed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "worker"))
    for option in ("source", "cache", "ignore-list", "runs", "state"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    (worker if args.mode == "worker" else dispatch)(args)
