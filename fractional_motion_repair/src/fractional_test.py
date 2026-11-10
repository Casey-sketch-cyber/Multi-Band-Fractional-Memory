from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fractional_screen import (
    build_long_expert,
    build_short_expert,
    load_checkpoint,
)
from pilot import DurationAwareExpertRouter, PilotConfig, grouped_split, parameter_count
from raw_suite import RawWindowDataset, evaluate_detailed


CONDITIONS = ("limb72", "limb48", "mixed", "temporal32", "temporal48")


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
    parser.add_argument("--test-mask-seed", type=int, default=20270829)
    parser.add_argument("--test-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage", default="focused_confirmatory_test")
    parser.add_argument(
        "--protocol",
        default="protocols/fractional_confirmatory_protocol.md",
    )
    parser.add_argument("--seeds", default="20260829,20260830,20260831")
    parser.add_argument("--base-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--finite-runs", nargs="+", type=Path)
    parser.add_argument("--fractional-runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--fractional-variant",
        default="fractional_memory_loss",
        choices=("fractional_memory", "fractional_memory_loss"),
    )
    parser.add_argument(
        "--fractional-checkpoint",
        default="fractional_memory_loss_best.pt",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    finite_runs = args.finite_runs or args.fractional_runs
    if (
        len(seeds) != len(args.base_runs)
        or len(seeds) != len(finite_runs)
        or len(seeds) != len(args.fractional_runs)
    ):
        raise ValueError(
            "seeds, base runs, finite runs, and fractional runs must align"
        )
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
    result_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []

    manifest = {
        "stage": args.stage,
        "protocol": args.protocol,
        "split_seed": args.split_seed,
        "test_mask_seed": args.test_mask_seed,
        "test_stride": args.test_stride,
        "test_sequences": int(len(splits["test"])),
        "test_windows": len(dataset),
        "seeds": seeds,
        "conditions": CONDITIONS,
        "primary_condition": "limb72",
        "models_frozen": True,
        "fractional_variant": args.fractional_variant,
        "fractional_checkpoint": args.fractional_checkpoint,
    }
    (args.out_dir / "test_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    for seed, base_run, finite_run, fractional_run in zip(
        seeds,
        args.base_runs,
        finite_runs,
        args.fractional_runs,
        strict=True,
    ):
        config = PilotConfig(seed=seed, frames=128, batch_size=args.batch_size)
        short_expert = build_short_expert(
            base_run / "matched_tcn_best.pt",
            device,
        )
        methods = {}
        for method_name, variant, checkpoint in (
            ("dancerouter", "continued", base_run / "gap_router_best.pt"),
            (
                "finite_bridge",
                "integer_memory",
                finite_run / "integer_memory_best.pt",
            ),
            (
                "fractional_bridge",
                args.fractional_variant,
                fractional_run / args.fractional_checkpoint,
            ),
        ):
            long_expert, _ = build_long_expert(variant)
            load_checkpoint(long_expert, checkpoint)
            long_expert.to(device).eval()
            methods[method_name] = DurationAwareExpertRouter(
                short_expert,
                long_expert,
            ).to(device)

        for condition_index, condition in enumerate(CONDITIONS):
            mask_seed = args.test_mask_seed + condition_index
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
                    "method": method_name,
                    "condition": condition,
                    "mask_seed": mask_seed,
                    "parameters": parameter_count(model),
                    **metrics,
                }
                result_rows.append(result)
                sequence_rows.extend(
                    {
                        "seed": seed,
                        "method": method_name,
                        "condition": condition,
                        **detail,
                    }
                    for detail in details
                )
                print(json.dumps(result), flush=True)

    write_csv(args.out_dir / "fractional_test_results.csv", result_rows)
    write_csv(
        args.out_dir / "fractional_test_sequence_results.csv",
        sequence_rows,
    )
    (args.out_dir / "test_config.json").write_text(
        json.dumps(
            {
                "pilot_config": asdict(PilotConfig()),
                **manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
