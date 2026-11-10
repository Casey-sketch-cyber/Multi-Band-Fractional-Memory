from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fractional_screen import build_long_expert, build_short_expert, load_checkpoint
from pilot import DurationAwareExpertRouter, PilotConfig, grouped_split, parameter_count
from raw_suite import RawWindowDataset, evaluate_detailed


DEFAULT_GAPS = (24, 36, 48, 60, 72, 84, 96)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("data/raw60"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260829)
    parser.add_argument("--mask-seed", type=int, default=20270829)
    parser.add_argument("--mask-repeats", type=int, default=3)
    parser.add_argument("--test-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gaps", default=",".join(map(str, DEFAULT_GAPS)))
    parser.add_argument("--seeds", default="20260829,20260830,20260831")
    parser.add_argument("--base-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--fractional-runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    gaps = [int(value) for value in args.gaps.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(args.base_runs) or len(seeds) != len(args.fractional_runs):
        raise ValueError("seeds, base runs, and fractional runs must align")
    if any(gap < 1 or gap >= 128 for gap in gaps):
        raise ValueError("all gaps must be between 1 and 127 frames")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)
    splits = grouped_split(names, args.split_seed)
    dataset = RawWindowDataset(
        args.raw_cache_dir,
        splits["test"],
        frames=128,
        mode="fixed",
        seed=args.split_seed,
        stride=args.test_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = {
        "stage": "post_confirmatory_exploratory_gap_sweep",
        "protocol": "reports/fractional_exploratory_protocol.md",
        "models_frozen": True,
        "split_seed": args.split_seed,
        "test_sequences": int(len(splits["test"])),
        "test_windows": len(dataset),
        "test_stride": args.test_stride,
        "training_seeds": seeds,
        "mask_seeds": [args.mask_seed + index for index in range(args.mask_repeats)],
        "limb_gap_frames": gaps,
    }
    (args.out_dir / "gap_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    result_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for seed, base_run, fractional_run in zip(
        seeds,
        args.base_runs,
        args.fractional_runs,
        strict=True,
    ):
        config = PilotConfig(seed=seed, frames=128, batch_size=args.batch_size)
        short_expert = build_short_expert(base_run / "matched_tcn_best.pt", device)
        methods = {}
        for method_name, variant, checkpoint in (
            (
                "finite_bridge",
                "integer_memory",
                fractional_run / "integer_memory_best.pt",
            ),
            (
                "fractional_bridge",
                "fractional_memory_loss",
                fractional_run / "fractional_memory_loss_best.pt",
            ),
        ):
            long_expert, _ = build_long_expert(variant)
            load_checkpoint(long_expert, checkpoint)
            long_expert.to(device).eval()
            methods[method_name] = DurationAwareExpertRouter(
                short_expert,
                long_expert,
            ).to(device)

        for gap in gaps:
            condition = f"limb{gap}"
            for repeat in range(args.mask_repeats):
                mask_seed = args.mask_seed + repeat
                for method_name, model in methods.items():
                    metrics, details = evaluate_detailed(
                        model,
                        loader,
                        dataset,
                        condition,
                        mask_seed,
                        device,
                        config,
                    )
                    result = {
                        "seed": seed,
                        "mask_repeat": repeat,
                        "mask_seed": mask_seed,
                        "method": method_name,
                        "gap_frames": gap,
                        "parameters": parameter_count(model),
                        **metrics,
                    }
                    result_rows.append(result)
                    sequence_rows.extend(
                        {
                            "seed": seed,
                            "mask_repeat": repeat,
                            "mask_seed": mask_seed,
                            "method": method_name,
                            "gap_frames": gap,
                            **detail,
                        }
                        for detail in details
                    )
                    print(json.dumps(result), flush=True)

    write_csv(args.out_dir / "gap_sweep_results.csv", result_rows)
    write_csv(args.out_dir / "gap_sweep_sequence_results.csv", sequence_rows)


if __name__ == "__main__":
    main()
