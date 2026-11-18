from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from bridge_ablation import CONDITIONS, MASK_SEEDS, TRAINING_SEEDS, save_results
from kernel_test import METHODS, STAGE, digest, read_json, verify_freeze
from pilot import choreography_group


BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20261114
ENDPOINTS = {"L": ("limb60", "limb72", "limb84", "limb96"), "W": ("temporal32", "temporal48")}
EQUIVALENCE_MARGIN = 0.2


def holm(pvalues: list[float]) -> tuple[np.ndarray, np.ndarray]:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or not len(pvalues) or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("Holm correction requires valid p values")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(len(pvalues))
    ranks = np.empty(len(pvalues), dtype=np.int64)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    ranks[order] = np.arange(1, len(pvalues) + 1)
    return adjusted, ranks


def cluster_draw_counts(groups: list[tuple[str, str]], repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED) -> tuple[list[tuple[str, str]], np.ndarray]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("Cluster bootstrap requires at least two choreography groups")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    counts = np.zeros((repetitions, len(unique)), dtype=np.int64)
    np.add.at(counts, (np.arange(repetitions)[:, None], indices), 1)
    return unique, counts


def cluster_gain(baseline: np.ndarray, proposed: np.ndarray, groups: list[tuple[str, str]], unique: list[tuple[str, str]], counts: np.ndarray) -> tuple[dict, np.ndarray]:
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if (
        baseline.ndim != 1 or baseline.shape != proposed.shape or len(baseline) != len(groups)
        or not np.isfinite(baseline).all() or not np.isfinite(proposed).all()
        or (baseline < 0).any() or (proposed < 0).any() or baseline.sum() <= 0
    ):
        raise ValueError("Invalid paired sequence errors")
    group_indices = {group: index for index, group in enumerate(unique)}
    sums = np.zeros((len(unique), 2), dtype=np.float64)
    for b, p, group in zip(baseline, proposed, groups, strict=True):
        sums[group_indices[group]] += (b, p)
    draws = counts @ sums
    if (draws[:, 0] <= 0).any():
        raise ValueError("A cluster bootstrap draw has zero baseline error")
    percent = 100 * (draws[:, 0] - draws[:, 1]) / draws[:, 0]
    point = 100 * (baseline.sum() - proposed.sum()) / baseline.sum()
    ci95 = np.quantile(percent, (0.025, 0.975))
    ci90 = np.quantile(percent, (0.05, 0.95))
    # Invert percentile bootstrap intervals. The +1 correction avoids zero p values.
    lower_tail = (1 + np.count_nonzero(percent <= 0)) / (len(percent) + 1)
    upper_tail = (1 + np.count_nonzero(percent >= 0)) / (len(percent) + 1)
    p_difference = min(1.0, 2 * min(lower_tail, upper_tail))
    p_equivalence = max(
        (1 + np.count_nonzero(percent <= -EQUIVALENCE_MARGIN)) / (len(percent) + 1),
        (1 + np.count_nonzero(percent >= EQUIVALENCE_MARGIN)) / (len(percent) + 1),
    )
    return {
        "sequences": len(baseline), "choreography_groups": len(unique),
        "baseline_mpjpe_window_mean": float(baseline.mean()),
        "proposed_mpjpe_window_mean": float(proposed.mean()),
        "gain_percent": float(point), "gain_ci95_low": float(ci95[0]), "gain_ci95_high": float(ci95[1]),
        "gain_ci90_low": float(ci90[0]), "gain_ci90_high": float(ci90[1]),
        "bootstrap_tail_p_difference": float(p_difference),
        "bootstrap_tail_p_equivalence": float(p_equivalence),
        "equivalence_margin_percent": EQUIVALENCE_MARGIN,
        "bootstrap_repetitions": len(percent), "bootstrap_seed": BOOTSTRAP_SEED,
    }, percent


def correct_family(records: list[dict], draws: list[np.ndarray], *, equivalence: bool) -> None:
    adjusted, ranks = holm([record["bootstrap_tail_p_difference"] for record in records])
    for record, samples, pvalue, rank in zip(records, draws, adjusted, ranks, strict=True):
        alpha = 0.05 / (len(records) - int(rank) + 1)
        low, high = np.quantile(samples, (alpha / 2, 1 - alpha / 2))
        record.update(
            difference_holm_p=float(pvalue), difference_holm_rank=int(rank),
            difference_holm_ci_level=1 - alpha,
            difference_holm_ci_low=float(low), difference_holm_ci_high=float(high),
            difference_rejected=bool(pvalue <= 0.05 and (low > 0 or high < 0)),
        )
    if equivalence:
        adjusted_eq, ranks_eq = holm([record["bootstrap_tail_p_equivalence"] for record in records])
        for record, samples, pvalue, rank in zip(records, draws, adjusted_eq, ranks_eq, strict=True):
            alpha = 0.05 / (len(records) - int(rank) + 1)
            low, high = np.quantile(samples, (alpha, 1 - alpha))
            record.update(
                equivalence_holm_p=float(pvalue), equivalence_holm_rank=int(rank),
                equivalence_holm_ci_level=1 - 2 * alpha,
                equivalence_holm_ci_low=float(low), equivalence_holm_ci_high=float(high),
                equivalent=bool(pvalue <= 0.05 and low > -EQUIVALENCE_MARGIN and high < EQUIVALENCE_MARGIN),
            )


def sequence_grid(path: Path, frozen: dict) -> tuple[list[str], dict[tuple, float], dict[str, int]]:
    expected_fold = {}
    for run in frozen["runs"]:
        for name in run["eligible_test_sequences"]:
            previous = expected_fold.setdefault(name, run["fold"])
            if previous != run["fold"]:
                raise ValueError("A choreography test sequence occurs in multiple folds")
    group_folds: dict[tuple, set[int]] = defaultdict(set)
    for name, fold in expected_fold.items():
        group_folds[choreography_group(name)].add(fold)
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A choreography group crosses outer test folds")
    index = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name, method, condition, seed = row["sequence"], row["method"], row["condition"], int(row["seed"])
            key = (name, method, condition, seed)
            if (
                key in index or name not in expected_fold or method not in METHODS
                or condition not in CONDITIONS or seed not in TRAINING_SEEDS
                or int(row["fold"]) != expected_fold[name] or int(row["mask_seed"]) != MASK_SEEDS[condition]
            ):
                raise ValueError("Missing provenance, duplicate, or unexpected sequence result")
            value = float(row["mpjpe_window_mean"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid sequence window-mean MPJPE")
            index[key] = value
    expected_size = len(expected_fold) * len(METHODS) * len(CONDITIONS) * len(TRAINING_SEEDS)
    if len(index) != expected_size:
        raise ValueError(f"Incomplete sequence grid: {len(index)} cells, expected {expected_size}")
    return sorted(expected_fold), index, expected_fold


def endpoint_values(names: list[str], index: dict[tuple, float], method: str, conditions: tuple[str, ...]) -> np.ndarray:
    # Windows are averaged in evaluate_detailed; average seeds before endpoint conditions.
    return np.asarray([
        np.mean([np.mean([index[(name, method, condition, seed)] for seed in TRAINING_SEEDS]) for condition in conditions])
        for name in names
    ], dtype=np.float64)


def infer(args: argparse.Namespace) -> None:
    frozen = verify_freeze(args.freeze)
    manifest = read_json(args.results_dir / "kernel_test_manifest.json")
    sequence_path = args.results_dir / "kernel_sequence_results.csv"
    if (
        frozen["stage"] != STAGE or frozen["models_frozen"] is not True
        or manifest["stage"] != STAGE or manifest["freeze_sha256"] != digest(args.freeze)
        or manifest["test_set_accessed"] is not True
        or manifest["result_sha256"] != digest(args.results_dir / "kernel_ablation.csv")
        or manifest["sequence_result_sha256"] != digest(sequence_path)
    ):
        raise ValueError("Test results do not match the complete frozen evaluation")
    names, index, folds = sequence_grid(sequence_path, frozen)
    groups = [choreography_group(name) for name in names]
    unique, counts = cluster_draw_counts(groups)
    all_records, all_draws = [], []
    specifications = [
        ("H1", "H1", "B0", endpoint) for endpoint in ENDPOINTS
    ] + [
        (hypothesis, "H2", method, endpoint)
        for hypothesis, method in (("H2a", "B3_K1"), ("H2b", "B3_K4"))
        for endpoint in ENDPOINTS
    ] + [("H3", "H3", "B3_K1", "W")] + [
        ("G1_K0_equivalence", "G1_exploratory", "B3_K0", endpoint) for endpoint in ENDPOINTS
    ]
    for hypothesis, family, baseline, endpoint in specifications:
        statistics, draws = cluster_gain(
            endpoint_values(names, index, baseline, ENDPOINTS[endpoint]),
            endpoint_values(names, index, "B3_K5", ENDPOINTS[endpoint]), groups, unique, counts,
        )
        all_records.append({"hypothesis": hypothesis, "family": family, "baseline": baseline, "proposed": "B3_K5", "endpoint": endpoint, **statistics})
        all_draws.append(draws)
    for family in ("H1", "H2"):
        positions = [position for position, record in enumerate(all_records) if record["family"] == family]
        correct_family([all_records[position] for position in positions], [all_draws[position] for position in positions], equivalence=family == "H2")
    for record in all_records:
        if record["family"] in {"H1", "H2"}:
            record["better"] = record["difference_rejected"] and record["gain_percent"] > 0
            record["worse"] = record["difference_rejected"] and record["gain_percent"] < 0
        if record["family"] == "H3":
            record["noninferiority_margin_percent"] = -0.5
            record["noninferior"] = record["gain_ci95_low"] > -0.5
        if record["family"] == "G1_exploratory":
            record["equivalent"] = record["gain_ci90_low"] > -EQUIVALENCE_MARGIN and record["gain_ci90_high"] < EQUIVALENCE_MARGIN
    fields = list(dict.fromkeys(key for record in all_records for key in record))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_results(args.out_dir / "kernel_hypotheses.csv", [{key: record.get(key) for key in fields} for record in all_records])
    summary = []
    for method, condition in product(METHODS, CONDITIONS):
        seed_means = [float(np.mean([index[(name, method, condition, seed)] for name in names])) for seed in TRAINING_SEEDS]
        summary.append({
            "method": method, "condition": condition, "test_sequences": len(names),
            "choreography_groups": len(unique), "training_seeds": len(TRAINING_SEEDS),
            "mpjpe_window_mean": float(endpoint_values(names, index, method, (condition,)).mean()),
            "training_seed_std": float(np.std(seed_means, ddof=1)),
        })
    save_results(args.out_dir / "kernel_ablation_summary.csv", summary)
    h1 = all(record["better"] for record in all_records if record["family"] == "H1")
    h2a = all(record["better"] or record["equivalent"] for record in all_records if record["hypothesis"] == "H2a")
    h3 = next(record["noninferior"] for record in all_records if record["family"] == "H3")
    k0_equivalent = all(record["equivalent"] for record in all_records if record["family"] == "G1_exploratory")
    report = {
        "stage": "milestone_C_confirmatory_inference", "author_G1_decision_required": True,
        "freeze_sha256": digest(args.freeze), "sequence_result_sha256": digest(sequence_path),
        "sequences": len(names), "choreography_groups": len(unique),
        "outer_fold_sequences": {str(fold): sum(value == fold for value in folds.values()) for fold in range(5)},
        "bootstrap_unit": "pooled genre+choreography groups, sampled uniformly with replacement; all member sequences retained",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": BOOTSTRAP_SEED,
        "shared_bootstrap_draws_for_all_comparisons": True,
        "p_value_definition": "percentile-bootstrap tail inversion with +1 finite-Monte-Carlo correction",
        "multiplicity": "Holm H1 family=2; H2 family=4, applied separately to two-sided difference and intersection-union equivalence p values",
        "stepdown_interval_interpretation": "Rank-specific critical percentile intervals accompany Holm decisions; they are not simultaneous confidence regions. Unadjusted 95% and 90% intervals are also reported.",
        "equivalence": "TOST-style percentile tails; both raw 90% and Holm-adjusted equivalence intervals reported",
        "H1_positive_both_endpoints": h1, "H2a_better_or_equivalent_both_endpoints": h2a,
        "H3_noninferior_W": h3, "K0_equivalent_both_endpoints_exploratory": k0_equivalent,
        "G1_evidence": {"S1_rule_satisfied": bool(h1 and h2a and h3), "S3_kernel_equivalence_rule_satisfied": bool(h1 and k0_equivalent), "H1_failure_stop_required": not h1},
        "source_sha256": {path.name: digest(path) for path in (Path(__file__), Path(__file__).with_name("kernel_test.py"))},
    }
    (args.out_dir / "kernel_inference_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen milestone C sequence/cluster inference")
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    infer(parser.parse_args())


if __name__ == "__main__":
    main()
