"""Assemble T3.3 from existing paired, frozen C and D test results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from bridge_ablation import CONDITIONS, MASK_SEEDS, TRAINING_SEEDS, save_results
from kernel_inference import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, ENDPOINTS, cluster_draw_counts, cluster_gain,
)
from kernel_test import METHODS as KERNEL_METHODS
from kernel_test import STAGE as KERNEL_STAGE
from kernel_test import digest, read_json
from pilot import choreography_group


BASELINE_METHODS = (
    "linear", "pchip", "hankel", "bigru", "transformer", "patch_transformer", "saits",
)
REUSED_METHODS = ("B0", "B3_K5")
PROPOSED = "B3_K5"
METHODS = BASELINE_METHODS + REUSED_METHODS
FOLDS = range(5)
TEST_STAGE = "revision_frozen_baseline_test"
SUMMARY_STAGE = "revision_baseline_main_table"
PRIMARY = "mpjpe_window_mean"
METRICS = (
    PRIMARY, "mpjpe", "mpjpe_raw_units", "mpjpe_mm", "velocity_error", "bone_relative_error",
    "endpoint_velocity_jump_error", "acceleration_error",
    *(f"mpjpe_joint_{joint:02d}" for joint in range(17)),
)
SUPPORT = ("sequence_index", "windows", "missing_points", "endpoint_velocity_jump_points", "acceleration_points")
PLAN_G2_CANDIDATES = ("saits", "transformer")
EXTENDED_G2_CANDIDATES = (*PLAN_G2_CANDIDATES, "patch_transformer")


def frozen_pairs(frozen: dict, stage: str) -> dict[tuple[int, int], tuple[str, ...]]:
    if (
        frozen["stage"] != stage or frozen.get("models_frozen", True) is not True
        or frozen.get("test_set_accessed", False) is not False
        or frozen["mask_seeds"] != MASK_SEEDS
        or frozen.get("training_seeds", list(TRAINING_SEEDS)) != list(TRAINING_SEEDS)
    ):
        raise ValueError(f"Invalid freeze state, stage, or seeds for {stage}")
    pairs = {}
    for run in frozen["runs"]:
        key = run["fold"], run["seed"]
        names = tuple(run["eligible_test_sequences"])
        if key in pairs or not names or not all(names) or len(names) != len(set(names)):
            raise ValueError(f"Duplicate fold/seed or invalid sequence list in {stage}")
        pairs[key] = names
    if set(pairs) != set(product(FOLDS, TRAINING_SEEDS)):
        raise ValueError(f"Incomplete frozen fold and seed grid for {stage}")
    for fold in FOLDS:
        if any(pairs[fold, seed] != pairs[fold, TRAINING_SEEDS[0]] for seed in TRAINING_SEEDS):
            raise ValueError(f"Frozen test sequence order differs by seed for fold {fold}")
    return pairs


def verify_inputs(
    manifest_path: Path, aggregate_path: Path, sequence_path: Path, stage: str, freeze_sha: str,
) -> dict[str, str]:
    manifest = read_json(manifest_path)
    hashes = {p.name: digest(p) for p in (manifest_path, aggregate_path, sequence_path)}
    if (
        manifest["stage"] != stage or manifest["freeze_sha256"] != freeze_sha
        or manifest.get("test_set_accessed", True) is not True
        or manifest["result_sha256"] != hashes[aggregate_path.name]
        or manifest["sequence_result_sha256"] != hashes[sequence_path.name]
    ):
        raise ValueError(f"Evaluation files differ from {manifest_path.name}")
    return hashes


def csv_rows(path: Path, required: set[str]):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if missing := required.difference(reader.fieldnames or ()):
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[key] is None for key in required):
                raise ValueError(f"Malformed row {line} in {path.name}")
            yield line, row


def count(value: str) -> int:
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError("Invalid count in test results")
    return int(number)


def metric(value: str) -> float:
    if value == "":
        return math.nan
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid numeric metric in test results")
    return number


def read_aggregates(path: Path, methods: tuple[str, ...]) -> tuple[dict, dict[str, int]]:
    rows, parameters = {}, {}
    required = {
        "fold", "seed", "method", "condition", "mask_seed", "parameters", "test_windows",
        "test_sequences", "mpjpe_mm_sequences", *METRICS,
    }
    expected = set(product(FOLDS, TRAINING_SEEDS, methods, CONDITIONS))
    for line, row in csv_rows(path, required):
        fold, seed = int(row["fold"]), int(row["seed"])
        method, condition = row["method"], row["condition"]
        key = fold, seed, method, condition
        if key not in expected or key in rows or int(row["mask_seed"]) != MASK_SEEDS[condition]:
            raise ValueError(f"Unexpected or duplicate aggregate cell in {path.name} at line {line}")
        param_count = count(row["parameters"])
        if parameters.setdefault(method, param_count) != param_count:
            raise ValueError(f"Parameter count differs within {method}")
        for name in METRICS:
            metric(row[name])
        rows[key] = row
    if set(rows) != expected:
        raise ValueError(f"Incomplete aggregate grid in {path.name}: {len(rows)} of {len(expected)}")
    return rows, parameters


def read_sequences(
    path: Path, methods: tuple[str, ...], names: list[str], folds: dict[str, int],
    support: dict[tuple[str, int], tuple[int, ...]],
) -> np.ndarray:
    # Axes: method, condition, training seed, held-out sequence, metric.
    seen = np.zeros((len(methods), len(CONDITIONS), len(TRAINING_SEEDS), len(names)), dtype=bool)
    values = np.empty((*seen.shape, len(METRICS)), dtype=np.float64)
    locations = [{name: offset for offset, name in enumerate(axis)}
                 for axis in (methods, CONDITIONS, TRAINING_SEEDS, names)]
    required = {"sequence", "fold", "seed", "method", "condition", "mask_seed", *METRICS, *SUPPORT}
    for line, row in csv_rows(path, required):
        name, method = row["sequence"], row["method"]
        condition, seed = row["condition"], int(row["seed"])
        try:
            position = tuple(axis[key] for axis, key in zip(locations, (method, condition, seed, name), strict=True))
        except KeyError:
            raise ValueError(f"Unexpected test sequence cell in {path.name} at line {line}") from None
        if seen[position] or int(row["fold"]) != folds[name] or int(row["mask_seed"]) != MASK_SEEDS[condition]:
            raise ValueError(f"Duplicate or mismatched sequence cell in {path.name} at line {line}")
        counts = tuple(count(row[column]) for column in SUPPORT)
        if not counts[1] or not counts[2] or support.setdefault((condition, position[3]), counts) != counts:
            raise ValueError(f"Paired window or mask support differs in {path.name} at line {line}")
        data = [metric(row[column]) for column in METRICS]
        if not math.isfinite(data[0]):
            raise ValueError(f"Missing primary metric in {path.name} at line {line}")
        seen[position] = True
        values[position] = data
    if not seen.all():
        raise ValueError(f"Incomplete sequence grid in {path.name}: {seen.sum()} of {seen.size}")
    return values


def verify_fold_aggregates(
    rows: dict, values: np.ndarray, methods: tuple[str, ...], names: list[str],
    folds: dict[str, int], support: dict[tuple[str, int], tuple[int, ...]],
) -> None:
    fold_indices = {fold: [i for i, name in enumerate(names) if folds[name] == fold] for fold in FOLDS}
    for fold, (m, method), (s, seed), (c, condition) in product(
        FOLDS, enumerate(methods), enumerate(TRAINING_SEEDS), enumerate(CONDITIONS),
    ):
        key = fold, seed, method, condition
        row, indices = rows[key], fold_indices[fold]
        per_sequence = values[m, c, s, indices]
        windows = sum(support[condition, i][1] for i in indices)
        if count(row["test_sequences"]) != len(indices) or count(row["test_windows"]) != windows:
            raise ValueError(f"Sequence or window counts differ from fold aggregate for {key}")
        for name in (PRIMARY, "mpjpe_raw_units", "mpjpe_mm"):
            column = per_sequence[:, METRICS.index(name)]
            covered = column[np.isfinite(column)]
            expected = float(covered.mean()) if covered.size else math.nan
            actual = metric(row[name])
            if not ((math.isnan(expected) and math.isnan(actual))
                    or math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError(f"{name} differs from fold aggregate for {key}")
            if name == "mpjpe_mm" and count(row["mpjpe_mm_sequences"]) != covered.size:
                raise ValueError(f"Millimeter coverage differs from fold aggregate for {key}")


def build_table(
    values: dict[str, np.ndarray], parameters: dict[str, int], groups: list[tuple[str, str]],
    support: dict[tuple[str, int], tuple[int, ...]],
) -> list[dict]:
    clusters, draws = cluster_draw_counts(groups)
    conditions = {name: (name,) for name in CONDITIONS} | ENDPOINTS
    rows = []
    for condition, members in conditions.items():
        indices = [CONDITIONS.index(name) for name in members]
        proposed = values[PROPOSED][indices, :, :, 0].mean(axis=(0, 1))
        for method in METHODS:
            selected = values[method][indices]
            sequence_means = selected.mean(axis=(0, 1))
            primary = sequence_means[:, 0]
            score = float(primary.mean())
            gain = low = high = None
            if score > 0:
                gain = 100 * (score - float(proposed.mean())) / score
                if condition in ENDPOINTS and method != PROPOSED:
                    result, _ = cluster_gain(primary, proposed, groups, clusters, draws)
                    low, high = result["gain_ci95_low"], result["gain_ci95_high"]
            row = {
                "method": method, "condition": condition,
                "row_kind": "endpoint" if condition in ENDPOINTS else "mask",
                "mask_seed": MASK_SEEDS.get(condition), "parameters": parameters[method],
                "test_sequences": len(groups), "choreography_groups": len(clusters),
                "training_seeds": len(TRAINING_SEEDS),
                "test_windows_per_condition": sum(support[members[0], i][1] for i in range(len(groups))),
                "mpjpe_training_seed_std": float(selected[..., 0].mean(axis=(0, 2)).std(ddof=1)),
                "proposed_gain_over_method_percent": gain,
                "proposed_gain_descriptive_ci95_low": low, "proposed_gain_descriptive_ci95_high": high,
            }
            # A sequence contributes only if its metric is present for every seed and condition.
            for name, estimates in zip(METRICS, sequence_means.T, strict=True):
                covered = estimates[np.isfinite(estimates)]
                row[name] = float(covered.mean()) if covered.size else None
                row[f"{name}_sequences"] = int(covered.size)
            rows.append(row)
    return rows


def g2_evidence(rows: list[dict]) -> dict:
    limb_rows = {row["method"]: row for row in rows if row["condition"] == "L"}
    proposed_score = limb_rows[PROPOSED][PRIMARY]
    fields = (PRIMARY, "proposed_gain_over_method_percent",
              "proposed_gain_descriptive_ci95_low", "proposed_gain_descriptive_ci95_high")
    plan_best = min(PLAN_G2_CANDIDATES, key=lambda method: limb_rows[method][PRIMARY])
    extended_best = min(EXTENDED_G2_CANDIDATES, key=lambda method: limb_rows[method][PRIMARY])
    return {
        "endpoint": "L", "conditions": list(ENDPOINTS["L"]), "metric": PRIMARY,
        "proposed_method": PROPOSED, "proposed_mpjpe_window_mean": proposed_score,
        "plan_candidate_methods_evaluated": list(PLAN_G2_CANDIDATES),
        "plan_candidate_brits_evaluated": False,
        "plan_strongest_method": plan_best,
        "plan_stop_condition_observed": proposed_score > limb_rows[plan_best][PRIMARY],
        "extended_candidate_methods_evaluated": list(EXTENDED_G2_CANDIDATES),
        "extended_strongest_method": extended_best,
        "proposed_underperforms_extended_strongest": proposed_score > limb_rows[extended_best][PRIMARY],
        "candidate_results": {method: {field: limb_rows[method][field] for field in fields}
                              for method in EXTENDED_G2_CANDIDATES},
        "author_G2_decision_required": True,
    }


def aggregate(baseline_summary: Path, kernel_summary: Path, freeze: Path, out_dir: Path) -> None:
    baseline_frozen = read_json(freeze)
    baseline_sha = digest(freeze)
    reference = baseline_frozen["reference_kernel_freeze"]
    kernel_freeze = Path(reference["path"])
    if not kernel_freeze.is_absolute():
        kernel_freeze = freeze.parent / kernel_freeze
    kernel_sha = digest(kernel_freeze)
    if kernel_sha != reference["sha256"]:
        raise ValueError("Referenced milestone C freeze differs from the recorded hash")
    kernel_frozen = read_json(kernel_freeze)
    pairs = frozen_pairs(baseline_frozen, TEST_STAGE)
    if pairs != frozen_pairs(kernel_frozen, KERNEL_STAGE):
        raise ValueError("Frozen C and D test sequences or ordering differ")
    for field in ("frames", "stride", "batch_size", "num_workers"):
        if field in baseline_frozen and baseline_frozen[field] != kernel_frozen[field]:
            raise ValueError(f"Frozen evaluation settings differ for {field}")
    folds = {name: fold for fold in FOLDS for name in pairs[fold, TRAINING_SEEDS[0]]}
    if len(folds) != sum(len(pairs[fold, TRAINING_SEEDS[0]]) for fold in FOLDS):
        raise ValueError("A frozen sequence occurs in multiple outer folds")
    names = sorted(folds)
    groups = [choreography_group(name) for name in names]
    group_folds = {}
    for name, group in zip(names, groups, strict=True):
        if group_folds.setdefault(group, folds[name]) != folds[name]:
            raise ValueError("A choreography group crosses outer folds")

    values, parameters, hashes, support = {}, {}, {}, {}
    inputs = (
        (kernel_summary, "kernel", "kernel_ablation.csv", tuple(KERNEL_METHODS), REUSED_METHODS, KERNEL_STAGE, kernel_sha),
        (baseline_summary, "baseline", "baseline_test.csv", BASELINE_METHODS, BASELINE_METHODS, TEST_STAGE, baseline_sha),
    )
    for directory, stem, aggregate_filename, methods, retained, stage, freeze_sha in inputs:
        aggregate_path = directory / aggregate_filename
        sequence_path = directory / f"{stem}_sequence_results.csv"
        manifest_path = directory / f"{stem}_test_manifest.json"
        hashes.update(verify_inputs(manifest_path, aggregate_path, sequence_path, stage, freeze_sha))
        aggregates, counts = read_aggregates(aggregate_path, methods)
        sequences = read_sequences(sequence_path, methods, names, folds, support)
        verify_fold_aggregates(aggregates, sequences, methods, names, folds, support)
        for method in retained:
            values[method] = sequences[methods.index(method)].copy()
            parameters[method] = counts[method]
        del sequences
    for i in range(len(names)):
        if len({support[condition, i][:2] for condition in CONDITIONS}) != 1:
            raise ValueError("Test window counts or sequence indices differ across conditions")

    rows = build_table(values, parameters, groups, support)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "main_table.csv"
    save_results(output, rows)
    manifest = {
        "stage": SUMMARY_STAGE, "analysis_role": "descriptive_baseline_comparison",
        "from_existing_frozen_test_results": True,
        "baseline_freeze_sha256": baseline_sha, "kernel_freeze_sha256": kernel_sha,
        "input_sha256": hashes, "source_sha256": digest(Path(__file__)),
        "main_table_sha256": digest(output),
        "rows": len(rows), "methods": list(METHODS), "conditions": list(CONDITIONS),
        "reused_milestone_C_methods": list(REUSED_METHODS),
        "training_seeds": list(TRAINING_SEEDS), "mask_seeds": MASK_SEEDS,
        "test_sequences": len(names), "choreography_groups": len(set(groups)),
        "pairing": "Same outer fold, sequence, windows, training seed, and mask seed across methods",
        "endpoints": ENDPOINTS,
        "estimand": "Equal sequence weighting after averaging seeds and constituent conditions within each sequence; primary sequence errors average windows",
        "metric_coverage": "Each <metric>_sequences field counts sequences with a finite value for every constituent condition and seed; means use these complete sequences",
        "gain_definition": "100 * (comparator MPJPE - proposed MPJPE) / comparator MPJPE",
        "descriptive_gain_ci": {
            "interval": "Unadjusted two-sided 95% percentile interval for paired L and W gains",
            "bootstrap_unit": "Whole genre/choreography groups sampled with replacement",
            "repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED,
        },
        "g2_evidence": g2_evidence(rows),
    }
    (out_dir / "main_table_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--kernel-summary", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True, help="Milestone D test_freeze.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.baseline_summary, args.kernel_summary, args.freeze, args.out_dir)


if __name__ == "__main__":
    main()
