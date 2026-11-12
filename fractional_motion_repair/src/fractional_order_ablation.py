from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fractional_screen import build_long_expert, build_short_expert, load_checkpoint
from pilot import DurationAwareExpertRouter, PilotConfig, grouped_split
from raw_suite import RawWindowDataset, evaluate_detailed
from revision_splits import require_legacy_runs


MODES = ("learned", "band_centers", "joint_mean", "joint_permuted")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alter_orders(model: torch.nn.Module, mode: str, seed: int) -> None:
    raw_orders = model.backbone.memory.raw_orders
    with torch.no_grad():
        if mode == "learned":
            return
        if mode == "band_centers":
            raw_orders.zero_()
        elif mode == "joint_mean":
            raw_orders.copy_(raw_orders.mean(dim=0, keepdim=True).expand_as(raw_orders))
        elif mode == "joint_permuted":
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(raw_orders), generator=generator)
            raw_orders.copy_(raw_orders[permutation])
        else:
            raise ValueError(f"unknown mode: {mode}")


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default="20260829,20260830,20260831")
    parser.add_argument("--base-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--fractional-runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    require_legacy_runs([*args.base_runs, *args.fractional_runs])
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(args.base_runs) or len(seeds) != len(args.fractional_runs):
        raise ValueError("seeds, base runs, and fractional runs must align")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)
    splits = grouped_split(names, args.split_seed)
    dataset = RawWindowDataset(
        args.raw_cache_dir,
        splits["val"],
        frames=128,
        mode="fixed",
        seed=args.split_seed,
        max_windows=3,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for seed, base_run, fractional_run in zip(
        seeds,
        args.base_runs,
        args.fractional_runs,
        strict=True,
    ):
        config = PilotConfig(seed=seed, frames=128, batch_size=args.batch_size)
        short_expert = build_short_expert(base_run / "matched_tcn_best.pt", device)
        for mode in MODES:
            long_expert, _ = build_long_expert("fractional_memory_loss")
            load_checkpoint(
                long_expert,
                fractional_run / "fractional_memory_loss_best.pt",
            )
            alter_orders(long_expert, mode, seed=20260829)
            actual_orders = (
                long_expert.backbone.memory.fractional_orders()
                .detach()
                .mean(dim=0)
                .tolist()
            )
            model = DurationAwareExpertRouter(
                short_expert,
                long_expert.to(device),
            ).to(device)
            metrics, _ = evaluate_detailed(
                model,
                loader,
                dataset,
                "limb72",
                args.mask_seed,
                device,
                config,
            )
            row = {
                "seed": seed,
                "mode": mode,
                "condition": "limb72",
                "order_band1_mean": actual_orders[0],
                "order_band2_mean": actual_orders[1],
                "order_band3_mean": actual_orders[2],
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, long_expert
    write_csv(args.out_dir / "fractional_order_ablation.csv", rows)

    by_mode = {
        mode: [float(row["mpjpe"]) for row in rows if row["mode"] == mode]
        for mode in MODES
    }
    learned = float(np.mean(by_mode["learned"]))
    summary = {
        "stage": "post_confirmatory_validation_order_ablation",
        "split": "development_validation",
        "condition": "limb72",
        "mean_mpjpe": {
            mode: float(np.mean(values)) for mode, values in by_mode.items()
        },
        "relative_change_vs_learned_percent": {
            mode: 100.0 * (float(np.mean(values)) - learned) / learned
            for mode, values in by_mode.items()
        },
    }
    (args.out_dir / "fractional_order_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
