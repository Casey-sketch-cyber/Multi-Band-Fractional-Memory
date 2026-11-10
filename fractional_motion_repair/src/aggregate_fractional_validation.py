from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CONDITIONS = ("temporal32", "temporal48", "limb48", "limb72", "mixed")
PRIMARY = ("limb48", "limb72", "mixed")
FINITE = "integer_memory"
FRACTIONAL = "fractional_memory_loss"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_run(path: Path) -> dict[str, object]:
    rows = json.loads((path / "screen_results.json").read_text(encoding="utf-8"))
    seed = int(rows[0]["seed"])
    by_method = {
        method: {
            row["condition"]: float(row["mpjpe"])
            for row in rows
            if row["method"] == method
        }
        for method in (FINITE, FRACTIONAL, "dancerouter")
    }
    relative_primary = {
        method: mean(
            [
                by_method[method][condition]
                / by_method["dancerouter"][condition]
                for condition in PRIMARY
            ]
        )
        for method in (FINITE, FRACTIONAL)
    }
    condition_advantage = {
        condition: 100.0
        * (
            by_method[FINITE][condition] - by_method[FRACTIONAL][condition]
        )
        / by_method[FINITE][condition]
        for condition in CONDITIONS
    }
    primary_advantage = 100.0 * (
        relative_primary[FINITE] - relative_primary[FRACTIONAL]
    ) / relative_primary[FINITE]
    return {
        "seed": seed,
        "relative_primary": relative_primary,
        "primary_advantage_percent": primary_advantage,
        "condition_advantage_percent": condition_advantage,
        "mpjpe": by_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(path) for path in args.runs]
    primary_advantages = [
        float(run["primary_advantage_percent"]) for run in runs
    ]
    condition_advantages = {
        condition: [
            float(run["condition_advantage_percent"][condition])
            for run in runs
        ]
        for condition in CONDITIONS
    }
    summary = {
        "stage": "three_seed_development_validation",
        "test_set_accessed": False,
        "seeds": [int(run["seed"]) for run in runs],
        "runs": runs,
        "primary_advantage_percent": {
            "mean": mean(primary_advantages),
            "sample_std": statistics.stdev(primary_advantages),
            "values": primary_advantages,
            "all_positive": all(value > 0.0 for value in primary_advantages),
            "all_at_least_0_2": all(value >= 0.2 for value in primary_advantages),
        },
        "condition_advantage_percent": {
            condition: {
                "mean": mean(values),
                "sample_std": statistics.stdev(values),
                "values": values,
                "all_positive": all(value > 0.0 for value in values),
                "all_at_least_0_2": all(value >= 0.2 for value in values),
            }
            for condition, values in condition_advantages.items()
        },
    }
    summary["decision"] = {
        "original_aggregate_gate_passed": (
            summary["primary_advantage_percent"]["mean"] >= 0.2
            and summary["primary_advantage_percent"]["all_positive"]
        ),
        "longest_limb_gap_replication_passed": summary[
            "condition_advantage_percent"
        ]["limb72"]["all_at_least_0_2"],
        "advance_to_test": False,
        "reason": (
            "The predeclared aggregate 0.2% advantage gate was not met across "
            "three development seeds; the test set remains sealed."
        ),
    }
    json_path = args.out_dir / "three_seed_validation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Three-seed fractional bridge validation",
        "",
        "The test set was not accessed.",
        "",
        "| Seed | Finite relative score | Fractional relative score | Fractional advantage (%) |",
        "|---:|---:|---:|---:|",
    ]
    for run in runs:
        relative = run["relative_primary"]
        lines.append(
            f"| {run['seed']} | {relative[FINITE]:.6f} | "
            f"{relative[FRACTIONAL]:.6f} | "
            f"{run['primary_advantage_percent']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Aggregate advantage: {mean(primary_advantages):.4f}% "
            f"+/- {statistics.stdev(primary_advantages):.4f}% (sample SD).",
            "",
            "| Condition | Mean advantage (%) | SD | All positive | All >= 0.2% |",
            "|---|---:|---:|:---:|:---:|",
        ]
    )
    for condition in CONDITIONS:
        item = summary["condition_advantage_percent"][condition]
        lines.append(
            f"| {condition} | {item['mean']:.4f} | {item['sample_std']:.4f} | "
            f"{item['all_positive']} | {item['all_at_least_0_2']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"]["reason"],
        ]
    )
    (args.out_dir / "three_seed_validation_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
