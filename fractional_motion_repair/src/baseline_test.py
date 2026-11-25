"""Freeze and run the seven new T3.3 baselines on the original T3.2 test cells."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline_train import METHODS, NATIVE, TRAINED, load_method, select_epoch, split_data
from bridge_ablation import CONDITIONS, MASK_SEEDS, TRAINING_SEEDS, save_results
from kernel_test import PAIRS, digest, evaluation_runtime, file_record, now, read_json, source_hashes, verify_cell, verify_file, write_json
from pilot import PilotConfig, seed_everything
from raw_suite import RawWindowDataset, evaluate_detailed
from revision_metrics import load_aist_smpl_scales
from revision_splits import SPLIT_MANIFEST

STAGE = "revision_frozen_baseline_test"
REFERENCE_STAGE = "revision_frozen_kernel_test"


def environment() -> dict:
    return {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "pypots")}


def output_pair(root: Path, fold: int, seed: int) -> Path:
    return root / f"fold{fold}" / f"seed{seed}"


def completed_training(root: Path, run: dict, sources: dict) -> dict:
    fold, seed = run["fold"], run["seed"]
    pair = output_pair(root, fold, seed)
    methods = {name: {"checkpoint": {"sha256": None}, "selected_epoch": None} for name in METHODS}
    for category, names in (("native", TRAINED[:-1]), ("saits", ("saits",))):
        directory = pair / category
        split_path = directory / SPLIT_MANIFEST
        finished = directory / "training_complete.json"
        manifest_path = directory / ("raw_suite_manifest.json" if category == "native" else "saits_training_manifest.json")
        state, manifest = read_json(finished), read_json(manifest_path)
        if (
            read_json(split_path) != run["split"] or (state["fold"], state["seed"]) != (fold, seed)
            or state["test_set_accessed"] is not False or manifest["test_set_accessed"] is not False
            or set(state["methods"]) != set(names)
            or manifest["fold"] != fold
            or state["training_runtime"]["device_name"] != "NVIDIA RTX A6000"
            or any(sources.get(name) != checksum for name, checksum in state["source_sha256"].items())
        ):
            raise ValueError(f"Training provenance differs in {directory}")
        if category == "native" and (
            set(manifest["variants"]) != set(NATIVE)
            or (manifest["train_windows_per_sequence_per_epoch"], manifest["validation_windows_per_sequence"]) != (2, 3)
        ):
            raise ValueError("The native training schedule does not match the frozen baseline protocol")
        if category == "saits" and manifest["dependencies"] != environment():
            raise ValueError("SAITS training and evaluation dependency versions differ")
        expected = PilotConfig(seed=seed, batch_size=16, epochs=24)
        if any(manifest["config"][name] != getattr(expected, name) for name in (
            "seed", "frames", "epochs", "batch_size", "learning_rate", "weight_decay",
        )):
            raise ValueError(f"Training configuration differs in {directory}")
        for method in names:
            record = state["methods"][method]
            history = directory / ("saits_history_state.json" if method == "saits" else f"{method}_history.json")
            records = read_json(history)
            if method == "saits":
                records = records["history"]
            chosen = select_epoch(records)
            if (
                not 1 <= len(records) <= 24 or len(records) != record["trained_epochs"]
                or digest(history) != record["history_sha256"] or chosen["epoch"] != record["selected_epoch"]
                or chosen["val_mpjpe"] != record["inner_val_mpjpe"]
            ):
                raise ValueError(f"Selected training history differs for {method} at {pair}")
            checkpoint = verify_file(record["checkpoint"])
            if checkpoint.resolve() != (directory / f"{method}_best.pt").resolve(strict=True):
                raise ValueError(f"Checkpoint path differs for {method}")
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
            if any(not bool(torch.isfinite(value).all()) for value in state_dict.values()):
                raise FloatingPointError(f"Nonfinite selected {method} checkpoint")
            load_method(method, expected, checkpoint, torch.device("cpu"))
            methods[method] = {**record, "evidence": [file_record(path) for path in (
                split_path, finished, manifest_path, history,
            )]}
    return methods


def freeze(args: argparse.Namespace) -> None:
    root = args.out_dir.resolve()
    destination = root / "test_freeze.json"
    if destination.exists():
        raise FileExistsError("Baseline test freeze already exists")
    reference_record = file_record(args.kernel_freeze)
    manifest = read_json(args.kernel_summary / "kernel_test_manifest.json")
    reference_outputs = {
        "result_sha256": file_record(args.kernel_summary / "kernel_ablation.csv"),
        "sequence_result_sha256": file_record(args.kernel_summary / "kernel_sequence_results.csv"),
    }
    reference = read_json(args.kernel_freeze)
    if (
        manifest["stage"] != REFERENCE_STAGE or manifest["test_set_accessed"] is not True
        or manifest["freeze_sha256"] != reference_record["sha256"]
        or any(record["sha256"] != manifest[key] for key, record in reference_outputs.items())
        or reference["stage"] != REFERENCE_STAGE or reference["models_frozen"] is not True
        or reference["test_set_accessed"] is not False or reference["proposed_method"] != "B3_K5"
        or reference["mask_seeds"] != MASK_SEEDS or reference["training_seeds"] != list(TRAINING_SEEDS)
        or (reference["frames"], reference["stride"], reference["batch_size"], reference["num_workers"]) != (128, 128, 32, 0)
        or len(reference["runs"]) != len(PAIRS)
        or {(run["fold"], run["seed"]) for run in reference["runs"]} != set(PAIRS)
    ):
        raise ValueError("C results and original freeze must agree")
    decisions = args.decisions.read_text(encoding="utf-8")
    if "**G1：走 S1 线（作者同意）。**" not in decisions:
        raise ValueError("Author's S1 decision is missing")
    for key in ("metadata", "ignore_list", "protocol"):
        verify_file(reference[key])
    cache = Path(reference["raw_cache_dir"])
    verify_motions(reference)
    sources = source_hashes()
    runs = []
    with np.load(verify_file(reference["metadata"]), allow_pickle=False) as metadata:
        names, lengths = metadata["names"].astype(str), metadata["lengths"]
    for run in reference["runs"]:
        splits, split = split_data(cache, run["fold"], Path(reference["ignore_list"]["path"]))
        indices = [int(index) for index in splits["test"] if lengths[index] >= 128]
        if run["split"] != split or run["eligible_test_indices"] != indices or run["eligible_test_sequences"] != names[indices].tolist():
            raise ValueError("T3.3 test fold differs from T3.2")
        fields = {name: run[name] for name in ("fold", "seed", "eligible_test_indices", "eligible_test_sequences",
                                                "excluded_short_test_sequences", "split")}
        runs.append({**fields, "methods": completed_training(args.runs, run, sources)})
        print(f"freeze checked fold={run['fold']} seed={run['seed']}; test motion data unopened", flush=True)
    value = {
        **{name: reference[name] for name in ("metadata", "ignore_list", "protocol", "raw_cache_dir", "motion_file_stat",
                                              "aist_motion_zip", "frames", "stride", "batch_size", "num_workers",
                                              "mask_seeds", "training_seeds")},
        "stage": STAGE, "created_at_utc": now(), "models_frozen": True, "proposed_method": "B3_K5",
        "test_set_accessed": False, "test_output_dir": str(root),
        "decisions": file_record(args.decisions), "reference_kernel_freeze": reference_record,
        "reference_kernel_manifest": file_record(args.kernel_summary / "kernel_test_manifest.json"),
        "reference_kernel_results": reference_outputs,
        "source_sha256": sources, "dependencies": environment(), "runs": runs,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(destination, value, exclusive=True)
    print(json.dumps({"freeze_sha256": digest(destination), "trained_checkpoints": 60}), flush=True)


def verify_motions(frozen: dict) -> None:
    motion = (Path(frozen["raw_cache_dir"]) / "motions_f32.npy").stat()
    if frozen["motion_file_stat"] != {"bytes": motion.st_size, "mtime_ns": motion.st_mtime_ns}:
        raise ValueError("Motion cache metadata differs from the original freeze")
    archive = frozen["aist_motion_zip"]
    if archive and Path(archive["path"]).stat().st_size != archive["bytes"]:
        raise ValueError("AIST++ archive size changed")


def verified_freeze(path: Path) -> dict:
    frozen = read_json(path)
    if (
        frozen["stage"] != STAGE or frozen["models_frozen"] is not True or frozen["test_set_accessed"] is not False
        or frozen["source_sha256"] != source_hashes() or frozen["dependencies"] != environment()
        or path.resolve() != Path(frozen["test_output_dir"]) / "test_freeze.json"
        or frozen["mask_seeds"] != MASK_SEEDS or frozen["training_seeds"] != list(TRAINING_SEEDS)
        or (frozen["frames"], frozen["stride"], frozen["batch_size"], frozen["num_workers"]) != (128, 128, 32, 0)
        or len(frozen["runs"]) != len(PAIRS)
        or {(run["fold"], run["seed"]) for run in frozen["runs"]} != set(PAIRS)
    ):
        raise ValueError("Test freeze does not match this evaluator")
    for field in ("metadata", "ignore_list", "protocol", "decisions", "reference_kernel_freeze", "reference_kernel_manifest"):
        verify_file(frozen[field])
    for record in frozen["reference_kernel_results"].values():
        verify_file(record)
    verify_motions(frozen)
    for run in frozen["runs"]:
        if set(run["methods"]) != set(METHODS):
            raise ValueError("Incomplete baseline method grid")
        for name in METHODS:
            record = run["methods"][name]
            if name in TRAINED:
                verify_file(record["checkpoint"])
                for artifact in record["evidence"]:
                    verify_file(artifact)
            elif record != {"checkpoint": {"sha256": None}, "selected_epoch": None}:
                raise ValueError("An analytic baseline has unexpected checkpoint metadata")
    return frozen


def cell_paths(directory: Path):
    return {(method, condition): directory / "cells" / f"{method}-{condition}.json"
            for method, condition in product(METHODS, CONDITIONS)}


def evaluate(args: argparse.Namespace) -> None:
    frozen = verified_freeze(args.freeze)
    freeze_hash = digest(args.freeze)
    run = next(run for run in frozen["runs"] if (run["fold"], run["seed"]) == (args.fold, args.seed))
    directory = output_pair(Path(frozen["test_output_dir"]), args.fold, args.seed)
    manifest_path = directory / "test_manifest.json"
    paths = cell_paths(directory)
    if manifest_path.exists():
        read_pair(directory, run, freeze_hash)
        print("Baseline evaluation already complete", flush=True)
        return
    (directory / "cells").mkdir(parents=True, exist_ok=True)
    lock = directory / "evaluation.lock.json"
    write_json(lock, {"started_at_utc": now(), "freeze_sha256": freeze_hash}, exclusive=True)
    try:
        if any(path.with_suffix(".intent.json").exists() and not path.exists() for path in paths.values()):
            raise RuntimeError("An unfinished held-out test cell requires review")
        config = PilotConfig(seed=args.seed, batch_size=32)
        seed_everything(args.seed)
        device = torch.device(args.device)
        runtime = evaluation_runtime(device)
        reference_manifest = read_json(verify_file(frozen["reference_kernel_manifest"]))
        original = next(record for record in reference_manifest["runtime_by_run"]
                        if (record["fold"], record["seed"]) == (args.fold, args.seed))
        if runtime != {key: value for key, value in original.items() if key not in {"fold", "seed"}}:
            raise ValueError("Baseline evaluation runtime differs from the reference kernel evaluation")
        dataset = RawWindowDataset(Path(frozen["raw_cache_dir"]), np.asarray(run["eligible_test_indices"]),
                                   config.frames, mode="fixed", seed=args.seed, stride=128)
        if dataset.names[run["eligible_test_indices"]].tolist() != run["eligible_test_sequences"]:
            raise ValueError("Frozen test sequence order differs")
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        archive = frozen["aist_motion_zip"]
        scales = load_aist_smpl_scales(Path(archive["path"]), run["eligible_test_sequences"]) if archive else {}
        for method in METHODS:
            record = run["methods"][method]
            model = load_method(method, config, Path(record["checkpoint"]["path"]) if method in TRAINED else None, device)
            parameters = sum(parameter.numel() for parameter in model.parameters()) if isinstance(model, nn.Module) else 0
            for condition, mask_seed in MASK_SEEDS.items():
                path = paths[method, condition]
                if path.exists():
                    verify_cell(read_json(path), run, method, condition, freeze_hash, expected_runtime=runtime)
                    continue
                write_json(path.with_suffix(".intent.json"), {"started_at_utc": now(), "freeze_sha256": freeze_hash}, exclusive=True)
                metrics, details = evaluate_detailed(model, loader, dataset, condition, mask_seed, device, config,
                                                       extended_metrics=True, smpl_scales=scales)
                del metrics["milliseconds_per_window"]
                cell = {
                    "freeze_sha256": freeze_hash, "fold": args.fold, "seed": args.seed,
                    "method": method, "condition": condition, "mask_seed": mask_seed, "parameters": parameters,
                    "checkpoint_sha256": record["checkpoint"]["sha256"], "selected_epoch": record["selected_epoch"],
                    "runtime": runtime, "metrics": metrics, "sequences": details,
                }
                verify_cell(cell, run, method, condition, freeze_hash, expected_runtime=runtime)
                write_json(path, cell)
                print(f"test fold={args.fold} seed={args.seed} {method}/{condition} completed", flush=True)
            del model
        write_json(manifest_path, {
            "stage": STAGE, "freeze_sha256": freeze_hash, "test_set_accessed": True,
            "fold": args.fold, "seed": args.seed, "completed_at_utc": now(), "runtime": runtime,
            "cells": {path.name: digest(path) for path in paths.values()},
        }, exclusive=True)
    except BaseException as error:
        write_json(lock, {"failed_at_utc": now(), "freeze_sha256": freeze_hash, "error": str(error), "requires_review": True})
        raise
    lock.unlink()


def read_pair(directory: Path, run: dict, freeze_hash: str) -> tuple[list[dict], list[dict], dict]:
    manifest = read_json(directory / "test_manifest.json")
    paths = cell_paths(directory)
    if (
        manifest["stage"] != STAGE or manifest["freeze_sha256"] != freeze_hash or manifest["test_set_accessed"] is not True
        or (manifest["fold"], manifest["seed"]) != (run["fold"], run["seed"])
        or manifest["cells"] != {path.name: digest(path) for path in paths.values()}
    ):
        raise ValueError(f"Completed baseline evidence changed in {directory}")
    scores, sequences = [], []
    for (method, condition), path in paths.items():
        cell = read_json(path)
        verify_cell(cell, run, method, condition, freeze_hash, expected_runtime=manifest["runtime"])
        keys = {key: cell[key] for key in ("fold", "seed", "method", "condition", "mask_seed")}
        scores.append({**keys, **{key: cell[key] for key in ("parameters", "checkpoint_sha256", "selected_epoch")}, **cell["metrics"]})
        sequences.extend({**keys, **record} for record in cell["sequences"])
    return scores, sequences, manifest["runtime"]


def summarize(args: argparse.Namespace) -> None:
    frozen = verified_freeze(args.freeze)
    freeze_hash = digest(args.freeze)
    root = Path(frozen["test_output_dir"])
    rows, sequences, runtimes = [], [], []
    for run in frozen["runs"]:
        fold, seed = run["fold"], run["seed"]
        pair_rows, pair_sequences, runtime = read_pair(output_pair(root, fold, seed), run, freeze_hash)
        rows.extend(pair_rows)
        sequences.extend(pair_sequences)
        runtimes.append({"fold": fold, "seed": seed, **runtime})
    if any({k: v for k, v in runtime.items() if k not in {"fold", "seed"}} !=
           {k: v for k, v in runtimes[0].items() if k not in {"fold", "seed"}} for runtime in runtimes[1:]):
        raise ValueError("Baseline evaluation runtime differs across fold-seed pairs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = args.out_dir / "baseline_test.csv"
    details = args.out_dir / "baseline_sequence_results.csv"
    save_results(result, rows)
    save_results(details, sequences)
    write_json(args.out_dir / "baseline_test_manifest.json", {
        "stage": STAGE, "freeze_sha256": freeze_hash, "test_set_accessed": True,
        "reference_kernel_freeze_sha256": frozen["reference_kernel_freeze"]["sha256"],
        "rows": len(rows), "sequence_rows": len(sequences), "runtime_by_run": runtimes,
        "result_sha256": digest(result), "sequence_result_sha256": digest(details),
        "dependencies": frozen["dependencies"], "created_at_utc": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    for flag in ("runs", "kernel-freeze", "kernel-summary", "decisions", "out-dir"):
        freezing.add_argument(f"--{flag}", required=True, type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--freeze", required=True, type=Path)
    evaluation.add_argument("--fold", required=True, type=int, choices=range(5))
    evaluation.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    summary = commands.add_parser("summarize")
    summary.add_argument("--freeze", required=True, type=Path)
    summary.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
