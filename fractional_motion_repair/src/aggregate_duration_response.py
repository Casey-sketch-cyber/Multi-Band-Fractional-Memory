"""T3.4 prespecified exploratory limb-duration gains from the frozen T3.2 summary.

The frozen protocol (section 9) lists this analysis as exploratory before any
test data existed, so it is labelled exploratory_prespecified, not post hoc.

Each input MPJPE is the mean over sequences after averaging each sequence's
windows and training seeds. Report the ratio of these means, as in the
frozen protocol, for each duration separately. B0 is the CT baseline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


LIMB_LENGTHS = (24, 36, 48, 60, 72, 84, 96)
CONDITIONS = tuple(f"limb{length}" for length in LIMB_LENGTHS)
KERNEL_METHODS = tuple(f"B3_K{index}" for index in range(8))
BASELINE = "B0"
SAMPLING_RATE_HZ = 60
TRAINING_SEEDS = 3
SUMMARY_NAME = "kernel_ablation_summary.csv"
OUTPUT_NAME = "duration_response.csv"


def duration_response(summary: Path) -> list[dict[str, object]]:
    expected = {
        (method, condition)
        for method in (BASELINE, *KERNEL_METHODS)
        for condition in CONDITIONS
    }
    scores: dict[tuple[str, str], float] = {}
    cohorts: set[tuple[int, int, int]] = set()
    with summary.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {
            "method", "condition", "mpjpe_window_mean", "test_sequences",
            "choreography_groups", "training_seeds",
        }
        if not fields.issubset(reader.fieldnames or ()):
            raise ValueError("The T3.2 summary is missing required columns")
        for row in reader:
            key = (row["method"], row["condition"])
            if key not in expected:
                continue
            if key in scores:
                raise ValueError(f"Duplicate T3.2 summary row for {key}")
            score = float(row["mpjpe_window_mean"])
            if not math.isfinite(score) or score < 0 or (key[0] == BASELINE and score == 0):
                raise ValueError(f"Invalid MPJPE for {key}")
            samples = int(row["test_sequences"])
            groups = int(row["choreography_groups"])
            seeds = int(row["training_seeds"])
            if not 1 <= groups <= samples or seeds != TRAINING_SEEDS:
                raise ValueError(f"Invalid sample counts for {key}")
            scores[key] = score
            cohorts.add((samples, groups, seeds))

    missing = sorted(expected.difference(scores))
    if missing:
        raise ValueError(f"Missing T3.2 summary rows: {missing}")
    if len(cohorts) != 1:
        raise ValueError("The duration comparisons require matching sample counts")
    sequences, clusters, seeds = cohorts.pop()

    result = []
    for length, condition in zip(LIMB_LENGTHS, CONDITIONS, strict=True):
        baseline_score = scores[(BASELINE, condition)]
        for method in KERNEL_METHODS:
            method_score = scores[(method, condition)]
            result.append({
                "method": method,
                "baseline": BASELINE,
                "condition": condition,
                "gap_frames": length,
                "gap_seconds_at_60hz": length / SAMPLING_RATE_HZ,
                "test_sequences": sequences,
                "choreography_groups": clusters,
                "training_seeds": seeds,
                "baseline_mpjpe_window_mean": baseline_score,
                "method_mpjpe_window_mean": method_score,
                "gain_percent": 100 * (baseline_score - method_score) / baseline_score,
                "analysis_stage": "exploratory_prespecified",
            })
    return result


def main() -> None:
    revision_results = Path(__file__).resolve().parents[1] / "results" / "revision"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=revision_results / SUMMARY_NAME)
    parser.add_argument("--out", type=Path, default=revision_results / OUTPUT_NAME)
    args = parser.parse_args()
    if args.summary.resolve() == args.out.resolve():
        parser.error("--out must differ from --summary")
    rows = duration_response(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
