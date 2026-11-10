from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np


def load_keypoints3d(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        values = payload.get("keypoints3d_optim", payload.get("keypoints3d"))
    else:
        values = payload
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(f"Expected (T,17,3), got {array.shape} for {path}")
    return array


def normalize_pose(keypoints: np.ndarray) -> tuple[np.ndarray, float]:
    pelvis = (keypoints[:, 11] + keypoints[:, 12]) / 2.0
    shoulders = (keypoints[:, 5] + keypoints[:, 6]) / 2.0
    torso = np.linalg.norm(shoulders - pelvis, axis=1)
    valid = torso[np.isfinite(torso) & (torso > 1e-6)]
    scale = float(np.median(valid)) if len(valid) else 1.0
    normalized = (keypoints - pelvis[:, None, :]) / max(scale, 1e-6)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.clip(normalized, -20.0, 20.0).astype(np.float32), scale


def load_confidence(
    archive: zipfile.ZipFile,
    member: str,
    target_frames: int,
) -> tuple[np.ndarray, bool]:
    with archive.open(member) as handle:
        payload = pickle.load(handle)
    values = payload.get("keypoints2d") if isinstance(payload, dict) else payload
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[0] != 9 or array.shape[2:] != (17, 3):
        raise ValueError(f"Expected (9,T,17,3), got {array.shape} for {member}")
    confidence = np.nan_to_num(
        array[..., 2], nan=0.0, posinf=1.0, neginf=0.0
    )
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    resampled = confidence.shape[1] != target_frames
    if resampled:
        indices = np.linspace(0, confidence.shape[1] - 1, target_frames).round().astype(np.int64)
        confidence = confidence[:, indices]
    return confidence.transpose(1, 0, 2), resampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-3d-dir",
        type=Path,
        default=Path("data/aistpp/annotations/keypoints3d"),
    )
    parser.add_argument(
        "--keypoints2d-zip",
        type=Path,
        default=Path("data/keypoints2d.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw60"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.raw_3d_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No 3D keypoint files found in {args.raw_3d_dir}")
    names = np.asarray([path.stem for path in paths], dtype="U")

    lengths = np.empty(len(paths), dtype=np.int32)
    for index, path in enumerate(paths):
        lengths[index] = len(load_keypoints3d(path))
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total_frames = int(offsets[-1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = args.output_dir / "motions_f32.npy"
    confidence_path = args.output_dir / "confidence_f16.npy"
    metadata_path = args.output_dir / "metadata.npz"
    summary_path = args.output_dir / "cache_summary.json"
    outputs = (motion_path, confidence_path, metadata_path, summary_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing cache files: {existing}")

    motion_tmp = motion_path.with_suffix(".partial.npy")
    confidence_tmp = confidence_path.with_suffix(".partial.npy")
    motions = np.lib.format.open_memmap(
        motion_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, 17, 3),
    )
    confidences = np.lib.format.open_memmap(
        confidence_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(total_frames, 9, 17),
    )
    scales = np.empty(len(paths), dtype=np.float32)
    confidence_resampled: list[str] = []

    with zipfile.ZipFile(args.keypoints2d_zip) as archive:
        member_map = {
            Path(member).stem: member
            for member in archive.namelist()
            if member.startswith("keypoints2d/") and member.endswith(".pkl")
        }
        missing = sorted(set(names.tolist()) - set(member_map))
        if missing:
            raise FileNotFoundError(f"Missing 2D confidence files for {len(missing)} sequences")

        for index, (name, path) in enumerate(zip(names, paths, strict=True)):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            normalized, scale = normalize_pose(load_keypoints3d(path))
            confidence, was_resampled = load_confidence(
                archive, member_map[str(name)], stop - start
            )
            motions[start:stop] = normalized
            confidences[start:stop] = confidence.astype(np.float16)
            scales[index] = scale
            if was_resampled:
                confidence_resampled.append(str(name))
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(
                    f"processed={index + 1}/{len(paths)} frames={stop}/{total_frames}",
                    flush=True,
                )

    motions.flush()
    confidences.flush()
    del motions, confidences
    os.replace(motion_tmp, motion_path)
    os.replace(confidence_tmp, confidence_path)
    np.savez_compressed(
        metadata_path,
        names=names,
        lengths=lengths,
        offsets=offsets,
        scales=scales,
    )

    confidence_view = np.load(confidence_path, mmap_mode="r")
    sample_stride = max(1, total_frames // 200_000)
    finite_sample = np.asarray(confidence_view[::sample_stride], dtype=np.float32).reshape(-1)
    finite_sample = finite_sample[np.isfinite(finite_sample)]
    summary = {
        "sequences": int(len(paths)),
        "total_frames": total_frames,
        "fps": 60,
        "motion_shape": list(np.load(motion_path, mmap_mode="r").shape),
        "confidence_shape": list(confidence_view.shape),
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "confidence_quantiles_sampled": {
            str(q): float(np.quantile(finite_sample, q))
            for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        },
        "confidence_length_resampled_sequences": confidence_resampled,
        "normalization": "per-frame pelvis centering and per-sequence median torso scaling",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(
        json.dumps(
            {
                "motions_bytes": motion_path.stat().st_size,
                "confidence_bytes": confidence_path.stat().st_size,
                "metadata_bytes": metadata_path.stat().st_size,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
