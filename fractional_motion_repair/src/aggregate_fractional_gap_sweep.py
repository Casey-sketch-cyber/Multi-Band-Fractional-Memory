from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_raw import paired_bootstrap, read_rows, write_rows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = ranks[indices].mean()
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270829)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.run_dir / "gap_sweep_sequence_results.csv")
    metrics = ("mpjpe", "velocity_error", "bone_relative_error")
    values: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            key = (
                metric,
                int(row["gap_frames"]),
                row["method"],
                row["sequence"],
            )
            values[key].append(float(row[metric]))

    gaps = sorted({key[1] for key in values})
    sequences = sorted({key[3] for key in values})
    summary_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    gains = []
    for gap_index, gap in enumerate(gaps):
        finite = np.asarray(
            [
                np.mean(values[("mpjpe", gap, "finite_bridge", sequence)])
                for sequence in sequences
            ]
        )
        fractional = np.asarray(
            [
                np.mean(values[("mpjpe", gap, "fractional_bridge", sequence)])
                for sequence in sequences
            ]
        )
        statistics = paired_bootstrap(
            finite,
            fractional,
            args.bootstrap,
            args.seed + gap_index * 101,
        )
        gains.append(float(statistics["mean_gain_percent"]))
        summary_rows.append(
            {
                "gap_frames": gap,
                "gap_seconds_at_60hz": gap / 60.0,
                "sequences": len(sequences),
                "replicates_per_sequence_method": len(
                    values[("mpjpe", gap, "finite_bridge", sequences[0])]
                ),
                "finite_mpjpe": float(finite.mean()),
                "fractional_mpjpe": float(fractional.mean()),
                **statistics,
                "ci_excludes_zero": statistics["difference_ci_low"] > 0.0,
            }
        )
        for metric_index, metric in enumerate(metrics):
            baseline_values = np.asarray(
                [
                    np.mean(values[(metric, gap, "finite_bridge", sequence)])
                    for sequence in sequences
                ]
            )
            proposed_values = np.asarray(
                [
                    np.mean(values[(metric, gap, "fractional_bridge", sequence)])
                    for sequence in sequences
                ]
            )
            metric_statistics = paired_bootstrap(
                baseline_values,
                proposed_values,
                args.bootstrap,
                args.seed + gap_index * 101 + metric_index * 10_007,
            )
            metric_rows.append(
                {
                    "gap_frames": gap,
                    "metric": metric,
                    "finite_mean": float(baseline_values.mean()),
                    "fractional_mean": float(proposed_values.mean()),
                    "mean_difference": metric_statistics[
                        "mean_mpjpe_difference"
                    ],
                    "difference_ci_low": metric_statistics[
                        "difference_ci_low"
                    ],
                    "difference_ci_high": metric_statistics[
                        "difference_ci_high"
                    ],
                    "mean_gain_percent": metric_statistics[
                        "mean_gain_percent"
                    ],
                    "gain_ci_low": metric_statistics["gain_ci_low"],
                    "gain_ci_high": metric_statistics["gain_ci_high"],
                    "sequence_win_fraction": metric_statistics[
                        "sequence_win_fraction"
                    ],
                    "cohen_dz": metric_statistics["cohen_dz"],
                    "ci_excludes_zero": metric_statistics[
                        "difference_ci_low"
                    ]
                    > 0.0,
                }
            )
    write_rows(args.out_dir / "gap_sweep_bootstrap.csv", summary_rows)
    write_rows(args.out_dir / "gap_sweep_metric_bootstrap.csv", metric_rows)

    gap_array = np.asarray(gaps, dtype=np.float64)
    gain_array = np.asarray(gains, dtype=np.float64)
    correlation = float(
        np.corrcoef(rankdata(gap_array), rankdata(gain_array))[0, 1]
    )
    slope, intercept = np.polyfit(gap_array, gain_array, deg=1)
    report = {
        "stage": "post_confirmatory_exploratory_gap_sweep",
        "gaps": gaps,
        "gain_percent": gains,
        "spearman_duration_gain": correlation,
        "linear_gain_slope_percent_per_frame": float(slope),
        "linear_gain_intercept_percent": float(intercept),
        "monotone_non_decreasing": all(
            right >= left for left, right in zip(gains, gains[1:])
        ),
        "confirmatory_result_unchanged": True,
    }
    (args.out_dir / "gap_sweep_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
