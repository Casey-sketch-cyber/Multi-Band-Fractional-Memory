from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import kernel_inference as inference
import kernel_test as evaluation


def expect_error(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def check_clusters() -> None:
    groups = [("gA", "ch1")] * 3 + [("gB", "ch2")]
    unique = sorted(set(groups))
    counts = np.asarray([[2, 0], [0, 2], [1, 1]])
    baseline = np.asarray([1, 2, 3, 8], dtype=float)
    proposed = np.asarray([0.9, 1.8, 2.7, 4], dtype=float)
    statistics, draws = inference.cluster_gain(baseline, proposed, groups, unique, counts)
    np.testing.assert_allclose(draws, [10, 50, 100 * 4.6 / 14], rtol=1e-12)
    np.testing.assert_allclose(statistics["gain_percent"], 100 * 4.6 / 14, rtol=1e-12)
    # A group mean of gains would give 30%, violating sequence-equal point weighting.
    assert abs(statistics["gain_percent"] - 30) > 2
    unique_a, a = inference.cluster_draw_counts(groups)
    unique_b, b = inference.cluster_draw_counts(groups)
    assert unique_a == unique_b == unique
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a.sum(axis=1), np.full(10_000, 2))
    assert {tuple(row) for row in a} == {(2, 0), (1, 1), (0, 2)}
    expect_error(ValueError, inference.cluster_gain, baseline, proposed * np.nan, groups, unique, counts)
    expect_error(ValueError, inference.cluster_draw_counts, [("gA", "ch1")] * 4)


def check_decisions() -> None:
    adjusted, ranks = inference.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    np.testing.assert_array_equal(ranks, [1, 3, 2])
    groups = [("g", f"ch{i}") for i in range(5)]
    unique, counts = inference.cluster_draw_counts(groups)
    records, draws = [], []
    for gain in (0.0, 0.1, 1.0, -1.0):
        record, samples = inference.cluster_gain(np.ones(5), np.ones(5) * (1 - gain / 100), groups, unique, counts)
        records.append(record)
        draws.append(samples)
    inference.correct_family(records, draws, equivalence=True)
    assert [record["difference_rejected"] for record in records] == [False, True, True, True]
    assert [record["equivalent"] for record in records] == [True, True, False, False]
    assert records[2]["gain_ci95_low"] > -0.5
    assert records[3]["gain_ci95_low"] < -0.5
    assert all(0 < record["bootstrap_tail_p_difference"] <= 1 for record in records)


def check_grid(directory: Path) -> None:
    names = ["gBR_sFM_d01_mBR0_ch01", "gHO_sFM_d02_mHO0_ch02"]
    frozen = {"runs": [{"fold": fold, "seed": seed, "eligible_test_sequences": [names[fold]]}
                       for fold in range(2) for seed in evaluation.TRAINING_SEEDS]}
    rows = []
    for fold, name in enumerate(names):
        for method in evaluation.METHODS:
            for condition in evaluation.CONDITIONS:
                for seed in evaluation.TRAINING_SEEDS:
                    rows.append({"sequence": name, "fold": fold, "seed": seed, "method": method,
                                 "condition": condition, "mask_seed": evaluation.MASK_SEEDS[condition],
                                 "mpjpe_window_mean": (fold + 1) + (seed % 100) / 1000})
    path = directory / "sequences.csv"

    def write(values):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(values)

    write(rows)
    actual_names, index, folds = inference.sequence_grid(path, frozen)
    assert actual_names == names and folds == dict(zip(names, range(2)))
    values = inference.endpoint_values(actual_names, index, "B3_K5", inference.ENDPOINTS["L"])
    np.testing.assert_allclose(values, [1.002, 2.002], atol=1e-15)
    write(rows[:-1])
    expect_error(ValueError, inference.sequence_grid, path, frozen)
    write([*rows, rows[0]])
    expect_error(ValueError, inference.sequence_grid, path, frozen)
    changed = {**rows[0], "mask_seed": 0}
    write([changed, *rows[1:]])
    expect_error(ValueError, inference.sequence_grid, path, frozen)


def check_access_guards(directory: Path) -> None:
    with patch.object(evaluation, "RawWindowDataset") as dataset:
        args = SimpleNamespace(freeze=directory / "missing.json", fold=0, seed=20261201, device="cpu")
        expect_error(FileNotFoundError, evaluation.evaluate, args)
        dataset.assert_not_called()
    freeze_path = directory / "test_freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    run = {"fold": 0, "seed": 20261201}
    root = directory / "test"
    cell_dir = root / "fold0" / "seed20261201" / "cells"
    cell_dir.mkdir(parents=True)
    (cell_dir / "B0-limb24.intent.json").write_text("{}", encoding="utf-8")
    frozen = {"runs": [run], "test_output_dir": str(root)}
    with patch.object(evaluation, "verify_freeze", return_value=frozen), patch.object(evaluation, "RawWindowDataset") as dataset:
        args = SimpleNamespace(freeze=freeze_path, fold=0, seed=20261201, device="cpu")
        expect_error(RuntimeError, evaluation.evaluate, args)
        dataset.assert_not_called()
        lock = root / "fold0" / "seed20261201" / "evaluation.lock.json"
        assert json.loads(lock.read_text())["requires_review"] is True
        expect_error(FileExistsError, evaluation.evaluate, args)
        dataset.assert_not_called()


def check_freeze_contract(directory: Path) -> None:
    root = directory / "freeze-contract"
    root.mkdir()
    cache = root / "cache"
    cache.mkdir()
    payload = root / "evidence.txt"
    payload.write_text("synthetic evidence", encoding="utf-8")
    checkpoint = root / "selected.bin"
    checkpoint.write_bytes(b"synthetic checkpoint")
    motion = cache / "motions_f32.npy"
    motion.write_bytes(b"not a real motion file; must never be opened")
    evidence = evaluation.file_record(payload)
    selected = evaluation.file_record(checkpoint)
    value = {
        "stage": evaluation.STAGE, "models_frozen": True, "test_set_accessed": False,
        "bridge": "B3", "proposed_method": "B3_K5",
        "bridge_decision": evaluation.BRIDGE_DECISION, "bridge_approval_url": evaluation.BRIDGE_APPROVAL_URL,
        "mask_seeds": evaluation.MASK_SEEDS, "training_seeds": list(evaluation.TRAINING_SEEDS),
        "frames": 128, "stride": 128, "batch_size": 32, "num_workers": 0,
        "evaluation_source_sha256": evaluation.source_hashes(),
        "test_output_dir": str(root.resolve()), "raw_cache_dir": str(cache.resolve()),
        "motion_file_stat": {"bytes": motion.stat().st_size, "mtime_ns": motion.stat().st_mtime_ns},
        **{key: evidence for key in ("protocol", "decisions", "ignore_list", "metadata")},
        "runs": [
            {"fold": fold, "seed": seed, "short_checkpoint": selected,
             "methods": {method: {"variant": variant, "checkpoint": selected, "evidence": [evidence]}
                         for method, variant in evaluation.METHODS.items()}}
            for fold, seed in evaluation.PAIRS
        ],
    }
    path = root / "test_freeze.json"
    evaluation.write_json(path, value)
    evaluation.verify_freeze(path)
    with patch.object(evaluation, "RawWindowDataset") as dataset:
        args = SimpleNamespace(freeze=path, fold=0, seed=20261201, device="cpu")
        checkpoint.write_bytes(b"changed checkpoint")
        expect_error(ValueError, evaluation.evaluate, args)
        dataset.assert_not_called()
    checkpoint.write_bytes(b"synthetic checkpoint")
    changed = {**value, "runs": value["runs"][:-1]}
    evaluation.write_json(path, changed)
    expect_error(ValueError, evaluation.verify_freeze, path)
    evaluation.write_json(path, {**value, "mask_seeds": {}})
    expect_error(ValueError, evaluation.verify_freeze, path)
    evaluation.write_json(path, {**value, "evaluation_source_sha256": {}})
    expect_error(ValueError, evaluation.verify_freeze, path)
    evaluation.write_json(path, value)
    payload.write_text("changed evidence", encoding="utf-8")
    expect_error(ValueError, evaluation.verify_freeze, path)


def check_cell_integrity() -> None:
    run = {"fold": 0, "seed": 20261201, "eligible_test_sequences": ["a", "b"],
           "methods": {"B0": {"checkpoint": {"sha256": "checkpoint"}}}}
    cell = {"fold": 0, "seed": 20261201, "method": "B0", "condition": "limb24",
            "mask_seed": evaluation.MASK_SEEDS["limb24"], "freeze_sha256": "freeze",
            "checkpoint_sha256": "checkpoint", "metrics": {"mpjpe_window_mean": 2.0},
            "sequences": [{"sequence": "a", "mpjpe_window_mean": 1.0}, {"sequence": "b", "mpjpe_window_mean": 3.0}]}
    evaluation.verify_cell(cell, run, "B0", "limb24", "freeze")
    bad = {**cell, "metrics": {"mpjpe_window_mean": 1.9}}
    expect_error(ValueError, evaluation.verify_cell, bad, run, "B0", "limb24", "freeze")
    bad = {**cell, "sequences": [cell["sequences"][0]] * 2}
    expect_error(ValueError, evaluation.verify_cell, bad, run, "B0", "limb24", "freeze")
    expect_error(ValueError, evaluation.verify_cell, cell, run, "B0", "limb24", "changed_freeze")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic regression checks; no AIST++ motion access")
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    check_clusters()
    check_decisions()
    check_cell_integrity()
    with tempfile.TemporaryDirectory(prefix="kernel-inference-", dir=args.work_dir) as temporary:
        directory = Path(temporary)
        check_grid(directory)
        check_access_guards(directory)
        check_freeze_contract(directory)
    print(json.dumps({"cluster_weighting": True, "holm_and_equivalence": True, "complete_grid": True,
                      "freeze_access_guard": True, "interrupted_test_guard": True, "sequence_mean_integrity": True,
                      "full_freeze_hash_and_grid_guards": True, "test_set_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
