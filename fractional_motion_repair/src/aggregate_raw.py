from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def paired_bootstrap(
    baseline: np.ndarray,
    proposed: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if baseline.shape != proposed.shape or baseline.ndim != 1:
        raise ValueError("Paired arrays must be one-dimensional with equal shape")
    generator = np.random.default_rng(seed)
    differences = baseline - proposed
    sample_indices = generator.integers(0, len(differences), size=(repetitions, len(differences)))
    sampled_difference = differences[sample_indices].mean(axis=1)
    sampled_baseline = baseline[sample_indices].mean(axis=1)
    sampled_percent = 100.0 * sampled_difference / np.maximum(sampled_baseline, 1e-12)
    mean_difference = float(differences.mean())
    return {
        "mean_mpjpe_difference": mean_difference,
        "difference_ci_low": float(np.quantile(sampled_difference, 0.025)),
        "difference_ci_high": float(np.quantile(sampled_difference, 0.975)),
        "mean_gain_percent": 100.0 * mean_difference / max(float(baseline.mean()), 1e-12),
        "gain_ci_low": float(np.quantile(sampled_percent, 0.025)),
        "gain_ci_high": float(np.quantile(sampled_percent, 0.975)),
        "sequence_win_fraction": float(np.mean(differences > 0.0)),
        "cohen_dz": float(mean_difference / max(float(differences.std(ddof=1)), 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--proposed", default="duration_router")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, str]] = []
    sequence_rows: list[dict[str, str]] = []
    for run_dir in args.run_dirs:
        aggregate_rows.extend(read_rows(run_dir / "raw_results.csv"))
        sequence_rows.extend(read_rows(run_dir / "raw_sequence_results.csv"))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[(row["method"], row["condition"])].append(row)
    summary_rows: list[dict[str, object]] = []
    metrics = ("mpjpe", "velocity_error", "bone_relative_error", "milliseconds_per_window")
    for (method, condition), rows in sorted(grouped.items()):
        output: dict[str, object] = {
            "method": method,
            "condition": condition,
            "training_seeds": len(rows),
            "parameters": int(float(rows[0]["parameters"])),
        }
        for metric in metrics:
            mean, std = mean_std([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
        summary_rows.append(output)
    write_rows(args.output_dir / "raw_mean_std.csv", summary_rows)

    sequence_index: dict[tuple[int, str, str, str], float] = {}
    for row in sequence_rows:
        sequence_index[
            (int(row["seed"]), row["method"], row["condition"], row["sequence"])
        ] = float(row["mpjpe"])
    seeds = sorted({key[0] for key in sequence_index})
    methods = sorted({key[1] for key in sequence_index})
    conditions = sorted({key[2] for key in sequence_index})
    sequences = sorted({key[3] for key in sequence_index})

    comparisons: list[dict[str, object]] = []
    baselines = [
        method
        for method in (
            "linear",
            "hankel",
            "bigru",
            "transformer",
            "patch_transformer",
            "tcn",
            "matched_tcn",
        )
        if method in methods
    ]
    ablations = [
        method
        for method in (
            "gap_router",
            "no_graph",
            "no_hankel",
            "no_velocity",
            "no_bone",
            "short_context",
        )
        if method in methods
    ]
    for condition_index, condition in enumerate(conditions):
        for baseline_name in baselines + ablations:
            comparison_seeds = sorted(
                seed
                for seed in seeds
                if any(
                    (seed, baseline_name, condition, sequence) in sequence_index
                    and (seed, args.proposed, condition, sequence) in sequence_index
                    for sequence in sequences
                )
            )
            common_sequences: list[str] = []
            baseline_values: list[float] = []
            proposed_values: list[float] = []
            for sequence in sequences:
                paired_seed_values = [
                    (
                        sequence_index[(seed, baseline_name, condition, sequence)],
                        sequence_index[(seed, args.proposed, condition, sequence)],
                    )
                    for seed in comparison_seeds
                    if (seed, baseline_name, condition, sequence) in sequence_index
                    and (seed, args.proposed, condition, sequence) in sequence_index
                ]
                if len(paired_seed_values) != len(comparison_seeds):
                    continue
                common_sequences.append(sequence)
                baseline_values.append(float(np.mean([item[0] for item in paired_seed_values])))
                proposed_values.append(float(np.mean([item[1] for item in paired_seed_values])))
            if not common_sequences:
                continue
            statistics = paired_bootstrap(
                np.asarray(baseline_values),
                np.asarray(proposed_values),
                args.bootstrap,
                args.seed + condition_index * 101 + len(comparisons),
            )
            comparisons.append(
                {
                    "condition": condition,
                    "comparison": f"{baseline_name}_minus_{args.proposed}",
                    "comparison_type": "ablation" if baseline_name in ablations else "baseline",
                    "sequences": len(common_sequences),
                    "training_seeds": len(comparison_seeds),
                    **statistics,
                    "ci_excludes_zero": statistics["difference_ci_low"] > 0.0,
                }
            )
    write_rows(args.output_dir / "raw_paired_bootstrap.csv", comparisons)

    primary_conditions = {"limb48", "limb72", "mixed"}
    primary_rows = [
        row
        for row in comparisons
        if row["condition"] in primary_conditions
        and int(row["training_seeds"]) >= 3
        and row["comparison"] == f"matched_tcn_minus_{args.proposed}"
    ]
    boundary_conditions = {"temporal16", "temporal32", "temporal48"}
    boundary_rows = [
        row
        for row in comparisons
        if row["condition"] in boundary_conditions
        and int(row["training_seeds"]) >= 3
        and row["comparison"] == f"matched_tcn_minus_{args.proposed}"
    ]
    gate = {
        "training_seeds": seeds,
        "proposed_method": args.proposed,
        "primary_claim": "partial-body and heterogeneous dance-skeleton occlusion",
        "required_primary_conditions": sorted(primary_conditions),
        "primary_comparison_rows": primary_rows,
        "primary_gate_passed": len(primary_rows) == len(primary_conditions)
        and all(bool(row["ci_excludes_zero"]) for row in primary_rows),
        "boundary_claim": "full-body temporal gaps are reported as secondary scope tests",
        "boundary_comparison_rows": boundary_rows,
        "all_boundary_cis_exclude_zero": len(boundary_rows)
        == len(boundary_conditions)
        and all(bool(row["ci_excludes_zero"]) for row in boundary_rows),
    }
    (args.output_dir / "raw_evidence_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
