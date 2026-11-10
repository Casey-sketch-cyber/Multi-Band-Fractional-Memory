from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_raw import mean_std, paired_bootstrap, read_rows, write_rows


METHODS = ("dancerouter", "finite_bridge", "fractional_bridge")
CONDITIONS = ("limb72", "limb48", "mixed", "temporal32", "temporal48")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270829)
    parser.add_argument("--stage", default="focused_confirmatory_test")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = read_rows(args.run_dir / "fractional_test_results.csv")
    sequence_rows = read_rows(
        args.run_dir / "fractional_test_sequence_results.csv"
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["method"], row["condition"])].append(row)

    mean_rows: list[dict[str, object]] = []
    for method in METHODS:
        for condition in CONDITIONS:
            rows = grouped[(method, condition)]
            output: dict[str, object] = {
                "method": method,
                "condition": condition,
                "training_seeds": len(rows),
            }
            for metric in (
                "mpjpe",
                "velocity_error",
                "bone_relative_error",
                "milliseconds_per_window",
            ):
                metric_mean, metric_std = mean_std(
                    [float(row[metric]) for row in rows]
                )
                output[f"{metric}_mean"] = metric_mean
                output[f"{metric}_std"] = metric_std
            mean_rows.append(output)
    write_rows(args.out_dir / "fractional_test_mean_std.csv", mean_rows)

    result_index = {
        (int(row["seed"]), row["method"], row["condition"]): float(row["mpjpe"])
        for row in result_rows
    }
    seeds = sorted({key[0] for key in result_index})
    seed_gain_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in seeds:
            finite = result_index[(seed, "finite_bridge", condition)]
            fractional = result_index[(seed, "fractional_bridge", condition)]
            seed_gain_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "finite_mpjpe": finite,
                    "fractional_mpjpe": fractional,
                    "gain_percent": 100.0 * (finite - fractional) / finite,
                }
            )
    write_rows(args.out_dir / "fractional_test_seed_gains.csv", seed_gain_rows)

    sequence_index = {
        (
            int(row["seed"]),
            row["method"],
            row["condition"],
            row["sequence"],
        ): float(row["mpjpe"])
        for row in sequence_rows
    }
    sequences = sorted({key[3] for key in sequence_index})
    bootstrap_rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for comparison_index, (baseline, proposed) in enumerate(
            (
                ("dancerouter", "finite_bridge"),
                ("dancerouter", "fractional_bridge"),
                ("finite_bridge", "fractional_bridge"),
            )
        ):
            baseline_values = []
            proposed_values = []
            for sequence in sequences:
                pairs = [
                    (
                        sequence_index[(seed, baseline, condition, sequence)],
                        sequence_index[(seed, proposed, condition, sequence)],
                    )
                    for seed in seeds
                ]
                baseline_values.append(float(np.mean([pair[0] for pair in pairs])))
                proposed_values.append(float(np.mean([pair[1] for pair in pairs])))
            statistics = paired_bootstrap(
                np.asarray(baseline_values),
                np.asarray(proposed_values),
                args.bootstrap,
                args.seed + condition_index * 101 + comparison_index,
            )
            bootstrap_rows.append(
                {
                    "condition": condition,
                    "comparison": f"{baseline}_minus_{proposed}",
                    "sequences": len(sequences),
                    "training_seeds": len(seeds),
                    **statistics,
                    "ci_excludes_zero": statistics["difference_ci_low"] > 0.0,
                }
            )
    write_rows(
        args.out_dir / "fractional_test_paired_bootstrap.csv",
        bootstrap_rows,
    )

    limb72_seed_gains = [
        float(row["gain_percent"])
        for row in seed_gain_rows
        if row["condition"] == "limb72"
    ]
    limb72_bootstrap = next(
        row
        for row in bootstrap_rows
        if row["condition"] == "limb72"
        and row["comparison"] == "finite_bridge_minus_fractional_bridge"
    )
    secondary_gains = {
        condition: float(
            next(
                row["mean_gain_percent"]
                for row in bootstrap_rows
                if row["condition"] == condition
                and row["comparison"]
                == "finite_bridge_minus_fractional_bridge"
            )
        )
        for condition in ("limb48", "mixed")
    }
    decision = {
        "stage": args.stage,
        "primary_condition": "limb72",
        "per_seed_gain_percent": limb72_seed_gains,
        "all_seeds_positive": all(value > 0.0 for value in limb72_seed_gains),
        "all_seeds_at_least_0_2": all(
            value >= 0.2 for value in limb72_seed_gains
        ),
        "bootstrap_mean_gain_percent": limb72_bootstrap["mean_gain_percent"],
        "bootstrap_gain_ci": [
            limb72_bootstrap["gain_ci_low"],
            limb72_bootstrap["gain_ci_high"],
        ],
        "bootstrap_ci_excludes_zero": limb72_bootstrap["ci_excludes_zero"],
        "secondary_gain_percent": secondary_gains,
        "secondary_no_regression_over_1_5": all(
            value > -1.5 for value in secondary_gains.values()
        ),
    }
    rule_met = bool(
        decision["all_seeds_positive"]
        and decision["all_seeds_at_least_0_2"]
        and decision["bootstrap_mean_gain_percent"] >= 0.2
        and decision["bootstrap_ci_excludes_zero"]
        and decision["secondary_no_regression_over_1_5"]
    )
    if args.stage == "focused_confirmatory_test":
        decision["focused_hypothesis_supported"] = rule_met
    else:
        decision["posthoc_descriptive_rule_met"] = rule_met
    (args.out_dir / "fractional_test_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
