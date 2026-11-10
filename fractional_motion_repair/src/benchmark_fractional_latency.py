from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from fractional_screen import build_long_expert, build_short_expert, load_checkpoint
from pilot import DurationAwareExpertRouter, make_mask


def count_parameters(model: torch.nn.Module, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


@torch.no_grad()
def benchmark(
    model: torch.nn.Module,
    batch_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + batch_size)
    motion = torch.randn(batch_size, 128, 17, 3, generator=generator).to(device)
    mask = make_mask(batch_size, 128, 17, "limb72", generator).to(device)
    model.eval()
    for _ in range(warmup):
        model(motion, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(motion, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(1000.0 * (time.perf_counter() - started))

    timing = np.asarray(timings, dtype=np.float64)
    peak_memory = (
        float(torch.cuda.max_memory_allocated()) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "median_ms_per_batch": float(np.median(timing)),
        "p95_ms_per_batch": float(np.percentile(timing, 95)),
        "median_ms_per_window": float(np.median(timing) / batch_size),
        "p95_ms_per_window": float(np.percentile(timing, 95) / batch_size),
        "peak_allocated_mib": peak_memory,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--fractional-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for method_name, variant, checkpoint in (
        ("dancerouter", "continued", args.base_run / "gap_router_best.pt"),
        (
            "finite_bridge",
            "integer_memory",
            args.fractional_run / "integer_memory_best.pt",
        ),
        (
            "fractional_bridge",
            "fractional_memory_loss",
            args.fractional_run / "fractional_memory_loss_best.pt",
        ),
    ):
        short_expert = build_short_expert(
            args.base_run / "matched_tcn_best.pt",
            device,
        )
        long_expert, _ = build_long_expert(variant)
        load_checkpoint(long_expert, checkpoint)
        model = DurationAwareExpertRouter(short_expert, long_expert.to(device)).to(device)
        for batch_size in (1, 32):
            row = {
                "method": method_name,
                "batch_size": batch_size,
                "total_parameters": count_parameters(model, trainable_only=False),
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                **benchmark(
                    model,
                    batch_size,
                    args.warmup,
                    args.repetitions,
                    device,
                    args.seed,
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model, long_expert, short_expert
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.out_dir / "fractional_latency.csv", rows)
    (args.out_dir / "fractional_latency.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
