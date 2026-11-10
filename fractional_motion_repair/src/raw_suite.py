from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pilot import (
    AdaptiveKinematicCompletion,
    BODY_EDGES,
    DurationAwareExpertRouter,
    FixedKinematicCompletion,
    PilotConfig,
    bone_lengths,
    grouped_split,
    hankel_low_rank,
    linear_interpolate,
    make_mask,
    parameter_count,
    seed_everything,
    train_model,
)
from run_suite import DEFAULT_CONDITIONS, build_model


class RawWindowDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        cache_dir: Path,
        sequence_indices: np.ndarray,
        frames: int,
        mode: str,
        seed: int,
        windows_per_sequence: int = 1,
        stride: int = 64,
        max_windows: int | None = None,
    ) -> None:
        if mode not in {"random", "fixed"}:
            raise ValueError(f"Unknown window mode: {mode}")
        self.cache_dir = Path(cache_dir)
        self.motion_path = self.cache_dir / "motions_f32.npy"
        with np.load(self.cache_dir / "metadata.npz") as metadata:
            self.names = np.asarray(metadata["names"]).astype(str)
            self.lengths = np.asarray(metadata["lengths"], dtype=np.int64)
            self.offsets = np.asarray(metadata["offsets"], dtype=np.int64)
        self.frames = frames
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self._motions: np.ndarray | None = None
        self.records: list[tuple[int, int]] = []

        for sequence_index in np.asarray(sequence_indices, dtype=np.int64):
            length = int(self.lengths[sequence_index])
            if length < frames:
                continue
            if mode == "random":
                self.records.extend(
                    (int(sequence_index), repeat)
                    for repeat in range(windows_per_sequence)
                )
            else:
                last = length - frames
                starts = list(range(0, last + 1, max(1, stride)))
                if not starts or starts[-1] != last:
                    starts.append(last)
                if max_windows is not None and len(starts) > max_windows:
                    selected = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
                    starts = [starts[index] for index in sorted(set(selected.tolist()))]
                self.records.extend((int(sequence_index), int(start)) for start in starts)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _motion_memmap(self) -> np.ndarray:
        if self._motions is None:
            self._motions = np.load(self.motion_path, mmap_mode="r")
        return self._motions

    def window_record(self, index: int) -> tuple[int, int]:
        sequence_index, value = self.records[index]
        if self.mode == "fixed":
            return sequence_index, value
        last = int(self.lengths[sequence_index]) - self.frames
        digest = hashlib.sha1(
            f"{self.seed}:{self.epoch}:{sequence_index}:{value}".encode()
        ).hexdigest()
        start = int(digest[:12], 16) % (last + 1)
        return sequence_index, start

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence_index, local_start = self.window_record(index)
        start = int(self.offsets[sequence_index]) + local_start
        stop = start + self.frames
        motion = np.array(self._motion_memmap()[start:stop], dtype=np.float32, copy=True)
        return torch.from_numpy(motion)


def make_loaders(
    cache_dir: Path,
    splits: dict[str, np.ndarray],
    config: PilotConfig,
    train_windows: int,
    val_windows: int,
    test_stride: int,
    quick: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, RawWindowDataset]:
    train_dataset = RawWindowDataset(
        cache_dir,
        splits["train"],
        config.frames,
        mode="random",
        seed=config.seed + 101,
        windows_per_sequence=train_windows,
    )
    val_dataset = RawWindowDataset(
        cache_dir,
        splits["val"],
        config.frames,
        mode="fixed",
        seed=config.seed,
        max_windows=2 if quick else val_windows,
    )
    test_dataset = RawWindowDataset(
        cache_dir,
        splits["test"],
        config.frames,
        mode="fixed",
        seed=config.seed,
        stride=test_stride,
        max_windows=2 if quick else None,
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 31),
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_dataset


def predict(
    method: nn.Module | str,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: PilotConfig,
) -> torch.Tensor:
    initial = linear_interpolate(target, mask)
    if method == "linear":
        return initial
    if method == "hankel":
        return hankel_low_rank(
            initial,
            target,
            mask,
            config.hankel_window,
            config.hankel_rank,
        )
    if isinstance(method, nn.Module):
        return method(target, mask)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def evaluate_detailed(
    method: nn.Module | str,
    loader: DataLoader,
    dataset: RawWindowDataset,
    condition: str,
    mask_seed: int,
    device: torch.device,
    config: PilotConfig,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if isinstance(method, nn.Module):
        method.eval()
    generator = torch.Generator().manual_seed(mask_seed)
    totals = defaultdict(float)
    per_sequence: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cursor = 0
    elapsed = 0.0
    timed_windows = 0

    left = torch.as_tensor([edge[0] for edge in BODY_EDGES], device=device)
    right = torch.as_tensor([edge[1] for edge in BODY_EDGES], device=device)
    for batch_index, target in enumerate(loader):
        target = target.to(device, non_blocking=True)
        batch_size = len(target)
        sequence_indices = [dataset.records[cursor + offset][0] for offset in range(batch_size)]
        cursor += batch_size
        mask = make_mask(
            batch_size,
            target.shape[1],
            target.shape[2],
            condition,
            generator,
        ).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = predict(method, target, mask, config)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if batch_index > 0:
            elapsed += time.perf_counter() - started
            timed_windows += batch_size

        missing = (1.0 - mask[..., 0]).bool()
        distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        pred_velocity = prediction[:, 1:] - prediction[:, :-1]
        true_velocity = target[:, 1:] - target[:, :-1]
        velocity_distance = torch.linalg.vector_norm(pred_velocity - true_velocity, dim=-1)
        transition = missing[:, 1:] | missing[:, :-1]
        pred_bone = bone_lengths(prediction)
        true_bone = bone_lengths(target)
        bone_missing = missing[:, :, left] | missing[:, :, right]
        bone_relative = (pred_bone - true_bone).abs() / true_bone.clamp_min(1e-3)

        for item, sequence_index in enumerate(sequence_indices):
            values = per_sequence[sequence_index]
            values["distance"] += float(distance[item][missing[item]].sum())
            values["points"] += float(missing[item].sum())
            values["velocity"] += float(velocity_distance[item][transition[item]].sum())
            values["transitions"] += float(transition[item].sum())
            values["bone"] += float(bone_relative[item][bone_missing[item]].sum())
            values["bones"] += float(bone_missing[item].sum())
            values["windows"] += 1.0

    for values in per_sequence.values():
        for key in ("distance", "points", "velocity", "transitions", "bone", "bones"):
            totals[key] += values[key]
    aggregate = {
        "mpjpe": totals["distance"] / max(totals["points"], 1.0),
        "velocity_error": totals["velocity"] / max(totals["transitions"], 1.0),
        "bone_relative_error": totals["bone"] / max(totals["bones"], 1.0),
        "milliseconds_per_window": 1000.0 * elapsed / max(timed_windows, 1),
        "test_windows": float(len(dataset)),
        "test_sequences": float(len(per_sequence)),
    }
    sequence_rows: list[dict[str, object]] = []
    for sequence_index, values in sorted(per_sequence.items()):
        sequence_rows.append(
            {
                "sequence_index": sequence_index,
                "sequence": dataset.names[sequence_index],
                "windows": int(values["windows"]),
                "mpjpe": values["distance"] / max(values["points"], 1.0),
                "velocity_error": values["velocity"] / max(values["transitions"], 1.0),
                "bone_relative_error": values["bone"] / max(values["bones"], 1.0),
                "missing_points": int(values["points"]),
            }
        )
    return aggregate, sequence_rows


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
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--mask-seed", type=int, default=20270815)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-windows", type=int, default=2)
    parser.add_argument("--val-windows", type=int, default=3)
    parser.add_argument("--test-stride", type=int, default=64)
    parser.add_argument("--variants", default="patch_transformer,tcn,matched_tcn")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-rejected-projections", action="store_true")
    args = parser.parse_args()

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    config = PilotConfig(
        seed=args.seed,
        epochs=3 if args.quick else args.epochs,
        batch_size=min(args.batch_size, 8) if args.quick else args.batch_size,
    )
    seed_everything(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)
    splits = grouped_split(names, args.split_seed)
    if args.quick:
        splits = {
            "train": splits["train"][:128],
            "val": splits["val"][:48],
            "test": splits["test"][:64],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "config": asdict(config),
        "fps": 60,
        "window_seconds": config.frames / 60.0,
        "split_seed": args.split_seed,
        "mask_seed": args.mask_seed,
        "split_counts": {key: int(len(value)) for key, value in splits.items()},
        "variants": variants,
        "conditions": conditions,
        "duration_router": {
            "linear_through": 1,
            "short_full_from": 8,
            "long_transition_start": 16,
            "long_full_from": 32,
        },
        "train_windows_per_sequence_per_epoch": args.train_windows,
        "validation_windows_per_sequence": args.val_windows,
        "test_stride": args.test_stride,
    }
    (args.out_dir / "raw_suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    trained: dict[str, nn.Module] = {}
    parameters: dict[str, int] = {"linear": 0, "hankel": 0}
    for variant in variants:
        seed_everything(config.seed)
        train_loader, val_loader, _, _ = make_loaders(
            args.raw_cache_dir,
            splits,
            config,
            args.train_windows,
            args.val_windows,
            args.test_stride,
            args.quick,
        )
        model, model_config = build_model(variant, config)
        model = model.to(device)
        parameters[variant] = parameter_count(model)
        print(f"training={variant} parameters={parameters[variant]}", flush=True)
        trained[variant] = train_model(
            variant,
            model,
            train_loader,
            val_loader,
            model_config,
            device,
            args.out_dir,
        )

    methods: list[tuple[str, nn.Module | str]] = [("linear", "linear"), ("hankel", "hankel")]
    methods.extend((name, trained[name]) for name in variants)
    if "matched_tcn" in trained and args.include_rejected_projections:
        backbone = trained["matched_tcn"]
        methods.extend(
            (
                ("matched_tcn_fixed_projection", FixedKinematicCompletion(backbone)),
                ("dancefill_adaptive", AdaptiveKinematicCompletion(backbone)),
            )
        )
        parameters["matched_tcn_fixed_projection"] = parameters["matched_tcn"]
        parameters["dancefill_adaptive"] = parameters["matched_tcn"]
    if "matched_tcn" in trained and "gap_router" in trained:
        methods.append(
            (
                "duration_router",
                DurationAwareExpertRouter(
                    trained["matched_tcn"],
                    trained["gap_router"],
                ),
            )
        )
        parameters["duration_router"] = (
            parameters["matched_tcn"] + parameters["gap_router"]
        )
    result_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for method_name, method in methods:
        _, _, test_loader, test_dataset = make_loaders(
            args.raw_cache_dir,
            splits,
            config,
            args.train_windows,
            args.val_windows,
            args.test_stride,
            args.quick,
        )
        for condition_index, condition in enumerate(conditions):
            metrics, detailed = evaluate_detailed(
                method,
                test_loader,
                test_dataset,
                condition,
                args.mask_seed + condition_index,
                device,
                config,
            )
            row: dict[str, object] = {
                "seed": config.seed,
                "method": method_name,
                "condition": condition,
                "parameters": parameters[method_name],
                **metrics,
            }
            result_rows.append(row)
            for detail in detailed:
                sequence_rows.append(
                    {
                        "seed": config.seed,
                        "method": method_name,
                        "condition": condition,
                        **detail,
                    }
                )
            print(json.dumps(row), flush=True)

    write_csv(args.out_dir / "raw_results.csv", result_rows)
    write_csv(args.out_dir / "raw_sequence_results.csv", sequence_rows)
    (args.out_dir / "raw_results.json").write_text(
        json.dumps(result_rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
