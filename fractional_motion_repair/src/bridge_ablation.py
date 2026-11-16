from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fractional_screen import build_long_expert, build_short_expert, load_checkpoint
from pilot import (
    REVISION_FOLDS,
    DurationAwareExpertRouter,
    PilotConfig,
    seed_everything,
)
from raw_suite import RawWindowDataset, evaluate_detailed, write_csv
from revision_metrics import load_aist_smpl_scales
from revision_splits import SPLIT_MANIFEST, revision_split


CONDITIONS = (
    "limb24", "limb36", "limb48", "limb60", "limb72", "limb84", "limb96",
    "temporal32", "temporal48", "mixed",
)
MASK_SEEDS = {condition: 20271101 + offset for offset, condition in enumerate(CONDITIONS)}
TRAINING_SEEDS = (20261201, 20261202, 20261203)
VARIANTS = {"B0": "continued", "B2_K5": "K5", "B3_K5": "B3_K5"}
RESULTS_FILE = "bridge_ablation.csv"
MANIFEST_FILE = "bridge_validation_manifest.json"
VALIDATION_STAGE = "revision_inner_validation"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_training_run(
    directory: Path,
    manifest_name: str,
    split: dict[str, object],
    fold: int,
    seed: int,
) -> dict:
    if read_json(directory / SPLIT_MANIFEST) != split:
        raise ValueError(f"Sequence lists differ from the requested primary fold in {directory}")
    manifest = read_json(directory / manifest_name)
    if (
        manifest.get("fold") != fold
        or manifest["config"]["seed"] != seed
        or manifest["config"]["frames"] != 128
        or manifest.get("test_set_accessed") is not False
    ):
        raise ValueError(f"Cannot use run {directory} for this fold and training seed")
    return manifest


def save_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".csv.tmp")
    write_csv(pending, rows)
    pending.replace(path)


def evaluate(args: argparse.Namespace) -> None:
    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = metadata["names"].astype(str)
    splits, split = revision_split(names, args.fold, args.ignore_list)
    check_training_run(args.base_run, "raw_suite_manifest.json", split, args.fold, args.seed)
    bridge_manifest = check_training_run(
        args.bridge_run, "screen_manifest.json", split, args.fold, args.seed,
    )
    if (
        bridge_manifest.get("stage") != "revision_validation_only"
        or set(bridge_manifest["variants"]) != set(VARIANTS.values())
    ):
        raise ValueError("Expected a training run for the three bridge variants")

    config = PilotConfig(seed=args.seed, frames=128, batch_size=32)
    seed_everything(config.seed)
    dataset = RawWindowDataset(
        args.raw_cache_dir, splits["val"], config.frames, mode="fixed", seed=args.seed, stride=128,
    )
    if not dataset:
        raise ValueError("The inner validation set has no complete windows")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    smpl_scales = (
        load_aist_smpl_scales(
            args.aist_motion_zip, (dataset.names[index] for index, _ in dataset.records),
        )
        if args.aist_motion_zip is not None else {}
    )
    short_checkpoint = args.base_run / "matched_tcn_best.pt"
    short_expert = build_short_expert(short_checkpoint, device)
    checkpoints = {"matched_tcn": short_checkpoint}
    rows: list[dict[str, object]] = []
    for method, variant in VARIANTS.items():
        checkpoint = args.bridge_run / f"{variant}_best.pt"
        checkpoints[method] = checkpoint
        long_expert, _ = build_long_expert(variant)
        if load_checkpoint(long_expert, checkpoint)["missing"]:
            raise ValueError(f"Missing trained parameters in {checkpoint}")
        model = DurationAwareExpertRouter(short_expert, long_expert).to(device).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        for condition, mask_seed in MASK_SEEDS.items():
            metrics, _ = evaluate_detailed(
                model, loader, dataset, condition, mask_seed, device, config,
                extended_metrics=True, smpl_scales=smpl_scales,
            )
            validation_windows = int(metrics.pop("test_windows"))
            validation_sequences = int(metrics.pop("test_sequences"))
            del metrics["milliseconds_per_window"], metrics["mpjpe"]
            score = metrics.pop("mpjpe_window_mean")
            if score is None or not math.isfinite(score) or score < 0:
                raise ValueError(f"Invalid validation score for {method}/{condition}")
            rows.append({
                "fold": args.fold, "seed": args.seed, "method": method, "condition": condition,
                "mask_seed": mask_seed, "parameters": parameters,
                "validation_sequences": validation_sequences, "validation_windows": validation_windows,
                "mpjpe_window_mean": score,
                **metrics,
            })
            print(
                f"fold={args.fold} seed={args.seed} {method} {condition} mpjpe_window_mean={score:.9f}",
                flush=True,
            )

    manifest = {
        "stage": VALIDATION_STAGE, "test_set_accessed": False, "fold": args.fold, "seed": args.seed,
        "split_manifest_sha256": hashlib.sha256((args.base_run / SPLIT_MANIFEST).read_bytes()).hexdigest(),
        "frames": dataset.frames, "stride": 128, "batch_size": config.batch_size, "num_workers": 0,
        "mask_seeds": MASK_SEEDS, "aist_motion_sequences": len(smpl_scales),
        "checkpoint_sha256": {
            name: hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for name, checkpoint in checkpoints.items()
        },
    }
    save_results(args.out_dir / RESULTS_FILE, rows)
    (args.out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def summarize(runs_dir: Path, out_dir: Path) -> dict[str, float]:
    expected_pairs = set(product(VARIANTS, CONDITIONS))
    rows: list[dict[str, object]] = []
    for fold, seed in product(range(REVISION_FOLDS), TRAINING_SEEDS):
        directory = runs_dir / f"fold{fold}" / f"seed{seed}" / "validation"
        manifest = read_json(directory / MANIFEST_FILE)
        if (
            manifest.get("stage") != VALIDATION_STAGE or manifest.get("test_set_accessed") is not False
            or manifest["fold"] != fold or manifest["seed"] != seed or manifest["mask_seeds"] != MASK_SEEDS
            or (manifest["frames"], manifest["stride"], manifest["batch_size"], manifest["num_workers"])
            != (128, 128, 32, 0)
        ):
            raise ValueError(f"Invalid inner validation manifest in {directory}")
        with (directory / RESULTS_FILE).open(newline="", encoding="utf-8") as handle:
            run_rows = list(csv.DictReader(handle))
        cells = {(row["method"], row["condition"]) for row in run_rows}
        if len(run_rows) != len(expected_pairs) or cells != expected_pairs:
            raise ValueError(f"Missing, duplicate, or unexpected validation cells in {directory}")
        for row in run_rows:
            score = float(row["mpjpe_window_mean"])
            if (
                int(row["fold"]) != fold or int(row["seed"]) != seed
                or int(row["mask_seed"]) != MASK_SEEDS[row["condition"]]
                or int(row["validation_windows"]) <= 0 or int(row["validation_sequences"]) <= 0
                or not math.isfinite(score) or score < 0
            ):
                raise ValueError(f"Invalid inner validation row in {directory}")
        rows.extend(run_rows)

    save_results(out_dir / RESULTS_FILE, rows)
    # The complete grid gives equal weight to each fold, training seed, and condition.
    count = len(rows) // len(VARIANTS)
    return {
        method: math.fsum(float(row["mpjpe_window_mean"]) for row in rows if row["method"] == method) / count
        for method in VARIANTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone B bridge ablation on the inner validation folds")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate one trained fold and seed")
    evaluation.add_argument("--fold", required=True, type=int, choices=range(REVISION_FOLDS))
    evaluation.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    evaluation.add_argument("--raw-cache-dir", required=True, type=Path)
    evaluation.add_argument("--ignore-list", required=True, type=Path)
    evaluation.add_argument("--base-run", required=True, type=Path)
    evaluation.add_argument("--bridge-run", required=True, type=Path)
    evaluation.add_argument("--out-dir", required=True, type=Path)
    evaluation.add_argument("--aist-motion-zip", type=Path, help="Trusted official AIST++ motion archive")
    evaluation.add_argument("--device", choices=("cpu", "cuda"))
    summary = commands.add_parser("summarize", help="Combine the complete five-fold, three-seed grid")
    summary.add_argument("--runs-dir", required=True, type=Path)
    summary.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        print(json.dumps(summarize(args.runs_dir, args.out_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
