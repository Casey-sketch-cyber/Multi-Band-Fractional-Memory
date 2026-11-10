from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_gl_integral(beta: float, kernel_size: int) -> np.ndarray:
    coefficients = [1.0]
    for index in range(1, kernel_size):
        coefficients.append(
            coefficients[-1] * (beta + float(index - 1)) / float(index)
        )
    output = np.asarray(coefficients, dtype=np.float64)
    return output / output.sum()


def summarize_checkpoint(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    prefix = "backbone.memory."
    lower = state[f"{prefix}order_lower"]
    upper = state[f"{prefix}order_upper"]
    raw_orders = state[f"{prefix}raw_orders"]
    orders = lower[None, :] + (upper - lower)[None, :] * torch.sigmoid(raw_orders)
    mixing = torch.softmax(state[f"{prefix}mix_logits"], dim=1).reshape(17, 3, 3)
    mixing = mixing.mean(dim=1)
    gates = torch.sigmoid(state[f"{prefix}gate_logits"]).reshape(17, 3).mean(dim=1)
    return {
        "checkpoint": str(path),
        "orders": orders.numpy(),
        "mixing": mixing.numpy(),
        "gates": gates.numpy(),
        "projection_norm": float(state[f"{prefix}projection.weight"].norm()),
        "kernel_size": 97,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [summarize_checkpoint(path) for path in args.checkpoints]
    orders = np.stack([item["orders"] for item in checkpoints])
    mixing = np.stack([item["mixing"] for item in checkpoints])
    gates = np.stack([item["gates"] for item in checkpoints])
    joint_rows: list[dict[str, object]] = []
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        row: dict[str, object] = {
            "joint_index": joint_index,
            "joint": joint_name,
            "gate_mean": float(gates[:, joint_index].mean()),
            "gate_std": float(gates[:, joint_index].std(ddof=1)),
        }
        for band in range(3):
            band_orders = orders[:, joint_index, band]
            band_mixing = mixing[:, joint_index, band]
            expected_lags = np.asarray(
                [
                    np.dot(
                        np.arange(int(item["kernel_size"])),
                        normalized_gl_integral(
                            float(item["orders"][joint_index, band]),
                            int(item["kernel_size"]),
                        ),
                    )
                    for item in checkpoints
                ]
            )
            row[f"order_band{band + 1}_mean"] = float(band_orders.mean())
            row[f"order_band{band + 1}_std"] = float(band_orders.std(ddof=1))
            row[f"mix_band{band + 1}_mean"] = float(band_mixing.mean())
            row[f"mix_band{band + 1}_std"] = float(band_mixing.std(ddof=1))
            row[f"expected_lag_band{band + 1}_frames"] = float(
                expected_lags.mean()
            )
        joint_rows.append(row)
    write_csv(args.out_dir / "learned_fractional_memory_by_joint.csv", joint_rows)

    global_summary = {
        "training_seeds": len(checkpoints),
        "kernel_size": 97,
        "initial_order_centers": [0.225, 0.5, 0.775],
        "learned_order_mean": orders.mean(axis=(0, 1)).tolist(),
        "learned_order_std_across_seed_joint": orders.std(axis=(0, 1)).tolist(),
        "mixture_mean": mixing.mean(axis=(0, 1)).tolist(),
        "gate_mean": float(gates.mean()),
        "gate_std_across_seed_joint": float(gates.std()),
        "projection_norm_mean": float(
            np.mean([item["projection_norm"] for item in checkpoints])
        ),
        "projection_norm_std": float(
            np.std([item["projection_norm"] for item in checkpoints], ddof=1)
        ),
        "checkpoint_summaries": [
            {
                "checkpoint": item["checkpoint"],
                "projection_norm": item["projection_norm"],
            }
            for item in checkpoints
        ],
    }
    (args.out_dir / "learned_fractional_memory_summary.json").write_text(
        json.dumps(global_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()
