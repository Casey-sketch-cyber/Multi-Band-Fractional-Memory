from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fractional_models import TemporalMemoryAdapter, endpoint_constrained_bridge
from fractional_screen import (
    SCREEN_CONDITIONS,
    build_long_expert,
    load_checkpoint,
    write_csv,
)
from pilot import PilotConfig, grouped_split, linear_interpolate, make_mask
from raw_suite import RawWindowDataset


ATTRIBUTION_METHODS = (
    "continued",
    "fractional_loss",
    "integer_memory",
    "fractional_memory",
    "fractional_memory_loss",
)
BRIDGE_METHODS = ("integer_memory", "fractional_memory")


def read_validation_run(run: Path) -> tuple[dict, dict[tuple[str, str], float]]:
    manifest = json.loads((run / "screen_manifest.json").read_text(encoding="utf-8"))
    if manifest["stage"] != "development_validation_only" or manifest["test_set_accessed"]:
        raise ValueError(f"Expected a validation-only screen run in {run}")
    expected = {
        (method, condition)
        for method in ("dancerouter", *ATTRIBUTION_METHODS)
        for condition in SCREEN_CONDITIONS
    }
    results = {}
    with (run / "screen_results.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = row["method"], row["condition"]
            if key not in expected:
                continue
            mpjpe = float(row["mpjpe"])
            if (
                key in results
                or int(row["seed"]) != manifest["config"]["seed"]
                or not math.isfinite(mpjpe)
                or mpjpe <= 0
            ):
                raise ValueError(f"Invalid screen row {key} in {run}")
            results[key] = mpjpe
    if results.keys() != expected:
        raise ValueError(f"Missing validation comparisons in {run}: {expected - results.keys()}")
    return manifest, results


def measure_bands(
    adapter: TemporalMemoryAdapter,
    signal: torch.Tensor,
    visibility: torch.Tensor,
    increment_visibility: torch.Tensor,
    check_decomposition: bool,
) -> torch.Tensor:
    """Return energy before the backbone, split by projection path and band."""
    increments = torch.zeros_like(signal)
    increments[..., 1:] = signal[..., 1:] - signal[..., :-1]
    bands = adapter._band_features(increments, increment_visibility)
    mixing = adapter.mix_logits.softmax(dim=-1)
    gate = adapter.gate_logits.sigmoid().view(1, -1, 1)
    weights = adapter.projection.weight.squeeze(-1)
    diagonal = weights.diagonal()
    off_diagonal = (weights - diagonal.diag()).unsqueeze(-1)
    energies = []
    corrections = []
    for index, feature in enumerate(bands.unbind(dim=2)):
        mixed = feature * mixing[:, index].view(1, -1, 1)
        same = endpoint_constrained_bridge(
            mixed * gate * diagonal.view(1, -1, 1), visibility
        )
        cross = endpoint_constrained_bridge(
            F.conv1d(mixed, off_diagonal) * gate, visibility
        )
        combined = same + cross
        unit = endpoint_constrained_bridge(feature, visibility)
        energies.append(torch.stack([
            item.square().sum(dtype=torch.float64) for item in (same, cross, combined, unit)
        ]))
        if check_decomposition:
            corrections.append(combined)
    if check_decomposition:
        torch.testing.assert_close(
            torch.stack(corrections).sum(dim=0),
            adapter(signal, visibility) - signal,
            rtol=2e-5,
            atol=5e-6,
        )
    return torch.stack(energies)


@torch.inference_mode()
def diagnose_condition(
    adapters: dict[str, TemporalMemoryAdapter],
    loader: DataLoader,
    condition: str,
    mask_seed: int,
    device: torch.device,
) -> tuple[int, list, list[int]]:
    generator = torch.Generator().manual_seed(mask_seed)
    windows = adapters["integer_memory"].finite_windows
    energy = torch.zeros((len(adapters), len(windows), 4), dtype=torch.float64, device=device)
    blind = torch.zeros(len(windows), dtype=torch.int64, device=device)
    bounded_count = torch.zeros((), dtype=torch.int64, device=device)
    for batch_index, target in enumerate(loader):
        mask = make_mask(
            len(target), target.shape[1], target.shape[2], condition, generator
        ).to(device)
        initial = linear_interpolate(target.to(device, non_blocking=True), mask)
        signal = initial.flatten(2).transpose(1, 2)
        visibility = mask.expand_as(initial).flatten(2).transpose(1, 2)
        observed = visibility.bool()
        has_left_endpoint = observed.cumsum(dim=-1).bool()
        has_right_endpoint = observed.flip(-1).cumsum(dim=-1).flip(-1).bool()
        bounded = ~observed & has_left_endpoint & has_right_endpoint
        bounded_count += bounded.sum()
        increment_visibility = torch.zeros_like(visibility)
        increment_visibility[..., 1:] = visibility[..., 1:] * visibility[..., :-1]
        for band_index, width in enumerate(windows):
            reachable = F.max_pool1d(
                increment_visibility, 2 * width - 1, stride=1, padding=width - 1
            ).bool()
            blind[band_index] += (bounded & ~reachable).sum()
        for method_index, adapter in enumerate(adapters.values()):
            energy[method_index] += measure_bands(
                adapter, signal, visibility, increment_visibility, batch_index == 0
            )
    count = int(bounded_count)
    if count == 0 or not torch.isfinite(energy).all():
        raise ValueError(f"No finite bridge energies for {condition}")
    return count, (energy / count).cpu().tolist(), blind.cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the T0.1b validation screens and bridge diagnostics.")
    parser.add_argument("--raw-cache-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation diagnostics on {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)

    attribution_rows = []
    diagnostic_rows = []
    seen_seeds = set()
    split_seed = None
    for run in args.runs:
        manifest, results = read_validation_run(run)
        config = PilotConfig(**manifest["config"])
        if config.seed in seen_seeds or split_seed not in (None, manifest["split_seed"]):
            raise ValueError("Runs must have unique training seeds on the same validation split")
        seen_seeds.add(config.seed)
        split_seed = manifest["split_seed"]
        splits = grouped_split(names, split_seed)
        if len(splits["val"]) != manifest["split_counts"]["val"]:
            raise ValueError(f"Validation split size differs from the saved run in {run}")
        validation = RawWindowDataset(
            args.raw_cache_dir, splits["val"], config.frames,
            mode="fixed", seed=config.seed, max_windows=3,
        )
        # Preserve the screen batch order without requiring worker IPC on mounted drives.
        loader = DataLoader(validation, batch_size=config.batch_size)
        adapters = {}
        for method in BRIDGE_METHODS:
            model, _ = build_long_expert(method)
            keys = load_checkpoint(model, run / f"{method}_best.pt")
            if keys["missing"]:
                raise ValueError(f"Missing trained adapter weights for {method} in {run}")
            adapters[method] = model.backbone.memory.to(device).eval()

        for index, condition in enumerate(SCREEN_CONDITIONS):
            mask_seed = config.seed + 50_000 + index
            common = {
                "split_seed": split_seed,
                "seed": config.seed,
                "condition": condition,
                "mask_seed": mask_seed,
                "validation_sequences": len(splits["val"]),
                "validation_windows": len(loader.dataset),
            }
            continued = results["continued", condition]
            for method in ATTRIBUTION_METHODS:
                mpjpe = results[method, condition]
                attribution_rows.append({
                    **common,
                    "method": method,
                    "mpjpe": mpjpe,
                    "original_router_mpjpe": results["dancerouter", condition],
                    "continued_mpjpe": continued,
                    "gain_vs_continued_percent": 100 * (continued - mpjpe) / continued,
                })

            count, energies, blind = diagnose_condition(adapters, loader, condition, mask_seed, device)
            for method, method_energies in zip(adapters, energies, strict=True):
                adapter = adapters[method]
                for band, (same, cross, combined, unit) in enumerate(method_energies):
                    finite = adapter.mode == "integer"
                    diagnostic_rows.append({
                        **common,
                        "method": method,
                        "band": band + 1,
                        "kernel_taps": adapter.finite_windows[band] if finite else adapter.kernel_size,
                        "bounded_missing_coordinate_frames": count,
                        "same_channel_mean_square": same,
                        "cross_channel_mean_square": cross,
                        "combined_mean_square": combined,
                        "unit_projection_mean_square": unit,
                        "finite_blind_coordinate_frames": blind[band] if finite else "",
                        "finite_blind_fraction": blind[band] / count if finite else "",
                    })
            print(f"Seed {config.seed}, {condition}: {len(loader.dataset)} validation windows", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "attribution_validation.csv", attribution_rows)
    write_csv(args.out_dir / "band_diagnostics.csv", diagnostic_rows)


if __name__ == "__main__":
    main()
