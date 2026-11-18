from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bridge_ablation import CONDITIONS, MASK_SEEDS, TRAINING_SEEDS, check_training_run, read_json, save_results
from bridge_stability import nonfinite_fields, selected_epoch
from fractional_screen import build_long_expert, build_short_expert, load_checkpoint, nonfinite_tensor_names
from kernel_diagnostics import kernel_statistics
from pilot import DurationAwareExpertRouter, PilotConfig, seed_everything
from raw_suite import RawWindowDataset, evaluate_detailed
from revision_metrics import load_aist_smpl_scales
from revision_splits import SPLIT_MANIFEST, revision_split


METHODS = {"B0": "continued", **{f"B3_K{i}": f"B3_K{i}" for i in range(8)}}
PAIRS = tuple(product(range(5), TRAINING_SEEDS))
STAGE = "revision_frozen_kernel_test"
BRIDGE_DECISION = "**确认桥选择 B3**"
BRIDGE_APPROVAL_URL = "https://github.com/Casey-sketch-cyber/Multi-Band-Fractional-Memory/pull/15#pullrequestreview-5957389282"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    else:
        pending = path.with_name(path.name + ".tmp")
        pending.write_text(text, encoding="utf-8")
        pending.replace(path)


def file_record(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def verify_file(record: dict) -> Path:
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"Frozen file changed: {path}")
    return path


def source_hashes() -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def training_record(directory: Path, variant: str, fold: int, seed: int, split: dict) -> tuple[dict, dict]:
    manifest = check_training_run(directory, "screen_manifest.json", split, fold, seed)
    if manifest["stage"] != "revision_validation_only" or variant not in manifest["variants"]:
        raise ValueError(f"Invalid training stage or variant in {directory}")
    history = read_json(directory / f"{variant}_history.json")
    status = read_json(directory / f"{variant}_stability.json")
    chosen = selected_epoch(history)
    if (
        not 1 <= len(history) <= 8
        or any(nonfinite_fields(row) for row in history)
        or status["status"] != "completed"
        or status["first_nonfinite"] is not None
        or status["finite_optimizer_steps"] <= 0
        or status["best_epoch"] != chosen["epoch"]
    ):
        raise ValueError(f"Training is incomplete or nonfinite in {directory}/{variant}")
    timing = read_json(directory / "training_timing.json")
    checkpoint = file_record(directory / f"{variant}_best.pt")
    if checkpoint["sha256"] != timing["experts"][variant]["checkpoint_sha256"]:
        raise ValueError(f"Training checkpoint changed in {directory}/{variant}")
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)["model"]
    if nonfinite_tensor_names(state.items()):
        raise ValueError(f"Nonfinite selected state in {directory}/{variant}")
    expert, _ = build_long_expert(variant)
    if load_checkpoint(expert, Path(checkpoint["path"]))["missing"]:
        raise ValueError(f"Incomplete selected model in {directory}/{variant}")
    record = {
        "variant": variant, "checkpoint": checkpoint, "selected_epoch": chosen["epoch"],
        "epochs": len(history), "training_source_commit": timing["source_commit"],
        "training_source_sha256": timing["source_sha256"],
        "evidence": [file_record(directory / filename) for filename in (
            "screen_manifest.json", SPLIT_MANIFEST, "training_environment.json", "training_timing.json",
            f"{variant}_history.json", f"{variant}_stability.json",
        )],
    }
    return record, manifest


def freeze(args: argparse.Namespace) -> None:
    destination = args.out_dir.resolve() / "test_freeze.json"
    if destination.exists():
        raise FileExistsError("A test freeze already exists; it must not be replaced")
    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = metadata["names"].astype(str)
        lengths = metadata["lengths"]
    decisions = args.decisions.read_text(encoding="utf-8")
    if BRIDGE_DECISION not in decisions:
        raise ValueError("The approved B3 decision is absent from DECISIONS.md")
    records = []
    analysis_sources = source_hashes()
    for fold, seed in PAIRS:
        relative = Path(f"fold{fold}") / f"seed{seed}"
        base = args.base_runs_dir / relative / "base"
        stable = args.stable_runs_dir / relative / "bridges"
        splits, split = revision_split(names, fold, args.ignore_list)
        check_training_run(base, "raw_suite_manifest.json", split, fold, seed)
        reference = read_json(stable / "screen_manifest.json")
        reference_environment = read_json(stable / "training_environment.json")
        short = file_record(base / "matched_tcn_best.pt")
        base_initial = file_record(base / "gap_router_best.pt")
        short_state = torch.load(short["path"], map_location="cpu", weights_only=True)["model"]
        if nonfinite_tensor_names(short_state.items()):
            raise ValueError(f"Nonfinite short expert in {base}")
        build_short_expert(Path(short["path"]), torch.device("cpu"))
        methods = {}
        for method, variant in METHODS.items():
            directory = (
                stable if method in {"B0", "B3_K5"}
                else args.runs_dir / relative / ("k6" if method == "B3_K6" else "kernels")
            )
            record, manifest = training_record(directory, variant, fold, seed, split)
            if (
                manifest["config"] != reference["config"]
                or manifest["config"]["epochs"] != 8
                or manifest["config"]["batch_size"] != 16
                or manifest["memory_lr_multiplier"] != 10
            ):
                raise ValueError(f"Training settings differ in {directory}")
            if record["training_source_sha256"] != read_json(directory / "training_environment.json")["source_sha256"]:
                raise ValueError(f"Training timing and environment provenance differ in {directory}")
            environment = read_json(directory / "training_environment.json")
            if environment["base_checkpoint_sha256"] != {
                "matched_tcn": short["sha256"], "gap_router": base_initial["sha256"],
            }:
                raise ValueError(f"Base initialization differs in {directory}")
            for name, checksum in reference_environment["source_sha256"].items():
                # Reporting and synthetic checks do not participate in model numerics.
                if name not in {"bridge_stability.py", "smoke_fractional.py"} and (
                    environment["source_sha256"][name] != checksum
                    or analysis_sources.get(name) != checksum
                ):
                    raise ValueError(f"Numerical training/evaluation source differs: {name}")
            methods[method] = record
        eligible = [int(index) for index in splits["test"] if lengths[index] >= 128]
        if not eligible:
            raise ValueError("Empty test fold")
        records.append({
            "fold": fold, "seed": seed, "split": split, "short_checkpoint": short,
            "eligible_test_sequences": names[eligible].tolist(),
            "eligible_test_indices": eligible,
            "excluded_short_test_sequences": [str(names[index]) for index in splits["test"] if lengths[index] < 128],
            "methods": methods,
        })
        print(f"freeze checked fold={fold} seed={seed}; test motion data unopened", flush=True)
    value = {
        "stage": STAGE, "created_at_utc": now(), "models_frozen": True,
        "test_set_accessed": False, "bridge": "B3", "bridge_decision": BRIDGE_DECISION,
        "bridge_approval_url": BRIDGE_APPROVAL_URL,
        "protocol": file_record(args.protocol), "decisions": file_record(args.decisions),
        "ignore_list": file_record(args.ignore_list),
        "metadata": file_record(args.raw_cache_dir / "metadata.npz"),
        "raw_cache_dir": str(args.raw_cache_dir.resolve()),
        "motion_file_stat": {
            "bytes": (args.raw_cache_dir / "motions_f32.npy").stat().st_size,
            "mtime_ns": (args.raw_cache_dir / "motions_f32.npy").stat().st_mtime_ns,
        },
        "test_output_dir": str(args.out_dir.resolve()),
        "frames": 128, "stride": 128, "batch_size": 32, "num_workers": 0,
        "mask_seeds": MASK_SEEDS, "training_seeds": TRAINING_SEEDS,
        "proposed_method": "B3_K5", "checkpoint_selection": "inner_validation_only",
        "evaluation_source_sha256": analysis_sources, "runs": records,
        # Stat only: hashing motion payloads would read held-out data before the freeze.
        "aist_motion_zip": {
            "path": str(args.aist_motion_zip.resolve(strict=True)),
            "bytes": args.aist_motion_zip.stat().st_size,
        } if args.aist_motion_zip else None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, value, exclusive=True)
    print(json.dumps({"freeze": str(destination), "sha256": digest(destination), "long_checkpoints": 135}), flush=True)


def verify_freeze(path: Path) -> dict:
    value = read_json(path)
    if (
        value["stage"] != STAGE or value["models_frozen"] is not True
        or value["test_set_accessed"] is not False or value["bridge"] != "B3"
        or value["proposed_method"] != "B3_K5" or value["mask_seeds"] != MASK_SEEDS
        or value["bridge_decision"] != BRIDGE_DECISION or value["bridge_approval_url"] != BRIDGE_APPROVAL_URL
        or value["training_seeds"] != list(TRAINING_SEEDS)
        or (value["frames"], value["stride"], value["batch_size"], value["num_workers"]) != (128, 128, 32, 0)
        or value["evaluation_source_sha256"] != source_hashes()
        or len(value["runs"]) != 15
        or {(row["fold"], row["seed"]) for row in value["runs"]} != set(PAIRS)
        or path.resolve() != Path(value["test_output_dir"]) / "test_freeze.json"
    ):
        raise ValueError("Invalid or incompatible complete test freeze")
    for key in ("protocol", "decisions", "ignore_list", "metadata"):
        verify_file(value[key])
    motion_stat = (Path(value["raw_cache_dir"]) / "motions_f32.npy").stat()
    if value["motion_file_stat"] != {"bytes": motion_stat.st_size, "mtime_ns": motion_stat.st_mtime_ns}:
        raise ValueError("Frozen motion cache size or modification time changed")
    # Check the entire selected grid before any test dataset can be constructed.
    for run in value["runs"]:
        if set(run["methods"]) != set(METHODS):
            raise ValueError("Incomplete frozen method grid")
        verify_file(run["short_checkpoint"])
        for method, record in run["methods"].items():
            if record["variant"] != METHODS[method]:
                raise ValueError("Frozen method mapping differs")
            verify_file(record["checkpoint"])
            for evidence in record["evidence"]:
                verify_file(evidence)
    return value


def band_columns(bands: list[dict]) -> dict:
    fields = {}
    for index in range(3):
        band = bands[index] if index < len(bands) else {}
        for key in ("mean_lag_frames", "zero_lag_mass", "beta", "lambda"):
            fields[f"kernel_band{index + 1}_{key}"] = band.get(key)
    return fields


def verify_cell(cell: dict, run: dict, method: str, condition: str, freeze_sha: str) -> None:
    if (
        cell["freeze_sha256"] != freeze_sha
        or (cell["fold"], cell["seed"]) != (run["fold"], run["seed"])
        or cell["method"] != method or cell["condition"] != condition
        or cell["mask_seed"] != MASK_SEEDS[condition]
        or cell["checkpoint_sha256"] != run["methods"][method]["checkpoint"]["sha256"]
    ):
        raise ValueError("Test result provenance differs from the freeze")
    details = cell["sequences"]
    names = [row["sequence"] for row in details]
    if len(names) != len(set(names)) or set(names) != set(run["eligible_test_sequences"]):
        raise ValueError("Missing, duplicate, or unexpected test sequences")
    values = [row["mpjpe_window_mean"] for row in details]
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Nonfinite sequence test score")
    if not math.isclose(math.fsum(values) / len(values), cell["metrics"]["mpjpe_window_mean"], rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Test score is not the sequence mean of window means")


def pair_outputs(directory: Path, run: dict, freeze_sha: str) -> tuple[list[dict], list[dict], dict]:
    rows, details, files = [], [], {}
    for method, condition in product(METHODS, CONDITIONS):
        path = directory / "cells" / f"{method}-{condition}.json"
        cell = read_json(path)
        verify_cell(cell, run, method, condition, freeze_sha)
        files[path.name] = digest(path)
        fields = {key: cell[key] for key in ("fold", "seed", "method", "condition", "mask_seed", "parameters", "checkpoint_sha256", "selected_epoch")}
        fields.update(band_columns(cell["kernel_bands"]))
        rows.append({**fields, **cell["metrics"]})
        details.extend({**{key: fields[key] for key in ("fold", "seed", "method", "condition", "mask_seed")}, **row} for row in cell["sequences"])
    return rows, details, files


def evaluate(args: argparse.Namespace) -> None:
    frozen = verify_freeze(args.freeze)
    freeze_sha = digest(args.freeze)
    run = next(row for row in frozen["runs"] if (row["fold"], row["seed"]) == (args.fold, args.seed))
    directory = Path(frozen["test_output_dir"]) / f"fold{args.fold}" / f"seed{args.seed}"
    if (directory / "test_manifest.json").exists():
        pair_outputs(directory, run, freeze_sha)
        print("Completed frozen evaluation already exists; no test data read", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cells").mkdir(exist_ok=True)
    lock = directory / "evaluation.lock.json"
    write_json(lock, {"started_at_utc": now(), "freeze_sha256": freeze_sha}, exclusive=True)
    try:
        # An interrupted cell requires review; it cannot be silently evaluated twice.
        for method, condition in product(METHODS, CONDITIONS):
            stem = directory / "cells" / f"{method}-{condition}"
            if stem.with_suffix(".intent.json").exists() and not stem.with_suffix(".json").exists():
                raise RuntimeError(f"Unfinished test transaction requires review: {stem}")
        config = PilotConfig(seed=args.seed, frames=128, batch_size=32)
        seed_everything(args.seed)
        with np.load(verify_file(frozen["metadata"])) as metadata:
            names = metadata["names"].astype(str)
        indices = np.asarray(run["eligible_test_indices"], dtype=np.int64)
        if names[indices].tolist() != run["eligible_test_sequences"]:
            raise ValueError("Frozen test sequence order differs")
        dataset = RawWindowDataset(Path(frozen["raw_cache_dir"]), indices, 128, mode="fixed", seed=args.seed, stride=128)
        device = torch.device(args.device)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        if frozen["aist_motion_zip"] and Path(frozen["aist_motion_zip"]["path"]).stat().st_size != frozen["aist_motion_zip"]["bytes"]:
            raise ValueError("Frozen AIST++ archive size changed")
        smpl_scales = load_aist_smpl_scales(
            Path(frozen["aist_motion_zip"]["path"]), run["eligible_test_sequences"],
        ) if frozen["aist_motion_zip"] else {}
        short = build_short_expert(Path(run["short_checkpoint"]["path"]), device)
        for method, record in run["methods"].items():
            expert, _ = build_long_expert(record["variant"])
            if load_checkpoint(expert, Path(record["checkpoint"]["path"]))["missing"]:
                raise ValueError(f"Incomplete selected checkpoint for {method}")
            bands = [] if method == "B0" else kernel_statistics(expert)
            model = DurationAwareExpertRouter(short, expert).to(device).eval()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            for condition, mask_seed in MASK_SEEDS.items():
                path = directory / "cells" / f"{method}-{condition}.json"
                if path.exists():
                    verify_cell(read_json(path), run, method, condition, freeze_sha)
                    continue
                intent = path.with_suffix(".intent.json")
                write_json(intent, {"started_at_utc": now(), "freeze_sha256": freeze_sha}, exclusive=True)
                metrics, details = evaluate_detailed(
                    model, loader, dataset, condition, mask_seed, device, config,
                    extended_metrics=True, smpl_scales=smpl_scales,
                )
                # Dedicated, synchronized efficiency measurements belong to T3.8.
                del metrics["milliseconds_per_window"]
                cell = {
                    "freeze_sha256": freeze_sha, "fold": args.fold, "seed": args.seed,
                    "method": method, "condition": condition, "mask_seed": mask_seed,
                    "checkpoint_sha256": record["checkpoint"]["sha256"],
                    "selected_epoch": record["selected_epoch"], "parameters": parameters,
                    "kernel_bands": bands, "metrics": metrics, "sequences": details,
                }
                verify_cell(cell, run, method, condition, freeze_sha)
                write_json(path, cell)
                print(f"test fold={args.fold} seed={args.seed} {method}/{condition} completed", flush=True)
            del model, expert
        rows, details, files = pair_outputs(directory, run, freeze_sha)
        save_results(directory / "kernel_ablation.csv", rows)
        save_results(directory / "kernel_sequence_results.csv", details)
        write_json(directory / "test_manifest.json", {
            "stage": STAGE, "test_set_accessed": True, "freeze_sha256": freeze_sha,
            "fold": args.fold, "seed": args.seed, "completed_at_utc": now(),
            "cell_sha256": files, "rows": len(rows), "sequence_rows": len(details),
            "result_sha256": digest(directory / "kernel_ablation.csv"),
            "sequence_result_sha256": digest(directory / "kernel_sequence_results.csv"),
        })
    except BaseException as error:
        write_json(lock, {"failed_at_utc": now(), "freeze_sha256": freeze_sha, "error": str(error), "requires_review": True})
        raise
    lock.unlink()


def summarize(args: argparse.Namespace) -> None:
    frozen = verify_freeze(args.freeze)
    freeze_sha = digest(args.freeze)
    rows, details = [], []
    for run in frozen["runs"]:
        directory = Path(frozen["test_output_dir"]) / f"fold{run['fold']}" / f"seed{run['seed']}"
        manifest = read_json(directory / "test_manifest.json")
        pair_rows, pair_details, files = pair_outputs(directory, run, freeze_sha)
        if (
            manifest["stage"] != STAGE or manifest["freeze_sha256"] != freeze_sha
            or manifest["cell_sha256"] != files
            or manifest["result_sha256"] != digest(directory / "kernel_ablation.csv")
            or manifest["sequence_result_sha256"] != digest(directory / "kernel_sequence_results.csv")
        ):
            raise ValueError(f"Completed test evidence changed in {directory}")
        rows.extend(pair_rows)
        details.extend(pair_details)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_results(args.out_dir / "kernel_ablation.csv", rows)
    # This file stays on the experiment server and must not be committed to Git.
    save_results(args.out_dir / "kernel_sequence_results.csv", details)
    write_json(args.out_dir / "kernel_test_manifest.json", {
        "stage": STAGE, "freeze_sha256": freeze_sha, "test_set_accessed": True,
        "rows": len(rows), "sequence_rows": len(details), "created_at_utc": now(),
        "result_sha256": digest(args.out_dir / "kernel_ablation.csv"),
        "sequence_result_sha256": digest(args.out_dir / "kernel_sequence_results.csv"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and evaluate the complete milestone C grid")
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    for name in ("raw-cache-dir", "ignore-list", "base-runs-dir", "stable-runs-dir", "runs-dir", "out-dir", "protocol", "decisions"):
        freezing.add_argument(f"--{name}", type=Path, required=True)
    freezing.add_argument("--aist-motion-zip", type=Path)
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
