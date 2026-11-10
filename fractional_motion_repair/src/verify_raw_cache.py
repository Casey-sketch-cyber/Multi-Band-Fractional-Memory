from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot import choreography_group, grouped_split


def resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    old_grid = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    flat = sequence.reshape(len(sequence), -1)
    output = np.empty((frames, flat.shape[1]), dtype=np.float32)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(new_grid, old_grid, flat[:, column])
    return output.reshape(frames, 17, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("data/raw60"),
    )
    parser.add_argument(
        "--legacy-cache",
        type=Path,
        default=Path("data/sequence_cache_128.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw_cache_verification.json"),
    )
    args = parser.parse_args()

    motions = np.load(args.raw_cache_dir / "motions_f32.npy", mmap_mode="r")
    confidence = np.load(args.raw_cache_dir / "confidence_f16.npy", mmap_mode="r")
    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)
        lengths = np.asarray(metadata["lengths"], dtype=np.int64)
        offsets = np.asarray(metadata["offsets"], dtype=np.int64)
    with np.load(args.legacy_cache, allow_pickle=True) as legacy:
        legacy_names = np.asarray(legacy["names"]).astype(str)
        legacy_motion = np.asarray(legacy["sequences"][:, :3], dtype=np.float32).transpose(0, 2, 3, 1)

    if len(set(names.tolist())) != len(names):
        raise AssertionError("Raw cache sequence names are not unique")
    if int(lengths.sum()) != len(motions) or int(offsets[-1]) != len(motions):
        raise AssertionError("Metadata offsets do not match the motion array")
    if confidence.shape != (len(motions), 9, 17):
        raise AssertionError(f"Unexpected confidence shape: {confidence.shape}")

    sample_indices = np.linspace(0, len(motions) - 1, 200_000).round().astype(np.int64)
    motion_sample = np.asarray(motions[sample_indices])
    confidence_sample = np.asarray(confidence[sample_indices], dtype=np.float32)
    if not np.isfinite(motion_sample).all() or not np.isfinite(confidence_sample).all():
        raise AssertionError("Cache contains non-finite sampled values")
    if float(confidence_sample.min()) < 0.0 or float(confidence_sample.max()) > 1.0:
        raise AssertionError("Confidence values fall outside [0,1]")

    legacy_for_name = {name: index for index, name in enumerate(legacy_names)}
    generator = np.random.default_rng(20260815)
    selected = sorted(generator.choice(len(names), size=25, replace=False).tolist())
    comparison_errors: list[float] = []
    for sequence_index in selected:
        name = names[sequence_index]
        legacy_index = legacy_for_name[name]
        start, stop = int(offsets[sequence_index]), int(offsets[sequence_index + 1])
        reconstructed = resample(np.asarray(motions[start:stop]), 128)
        comparison_errors.append(float(np.max(np.abs(reconstructed - legacy_motion[legacy_index]))))

    splits = grouped_split(names, 20260815)
    split_groups = {
        split: {choreography_group(names[index])[1] for index in indices}
        for split, indices in splits.items()
    }
    overlap = {
        "train_val": sorted(split_groups["train"] & split_groups["val"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "val_test": sorted(split_groups["val"] & split_groups["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"Choreography leakage detected: {overlap}")

    report = {
        "passed": True,
        "sequences": int(len(names)),
        "total_frames": int(len(motions)),
        "motion_shape": list(motions.shape),
        "confidence_shape": list(confidence.shape),
        "motion_sample_min": float(motion_sample.min()),
        "motion_sample_max": float(motion_sample.max()),
        "motion_sample_clipped_fraction": float(np.mean(np.abs(motion_sample) >= 19.999)),
        "confidence_sample_min": float(confidence_sample.min()),
        "confidence_sample_max": float(confidence_sample.max()),
        "legacy_comparison_sequences": len(selected),
        "legacy_comparison_max_abs_error": float(max(comparison_errors)),
        "split_counts": {key: int(len(value)) for key, value in splits.items()},
        "split_group_overlap": overlap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
