from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot import REVISION_FOLDS, grouped_split


REVISION_SPLIT_SEED = 20261112
SPLIT_MANIFEST = "revision_split_manifest.json"


def add_fold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fold", type=int, choices=range(REVISION_FOLDS))
    parser.add_argument("--ignore-list", type=Path, help="AIST++ ignore_list.txt for the primary folds")
    parser.add_argument("--include-ignored", action="store_true", help="Use all sequences for sensitivity analysis")


def revision_split(
    names: np.ndarray,
    fold: int,
    ignore_list: Path | None,
    include_ignored: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    names = names.astype(str)
    if len(set(names.tolist())) != len(names):
        raise ValueError("Cache sequence names must be unique")
    ignored: set[str] = set()
    if not include_ignored:
        if ignore_list is None:
            raise ValueError("Supply --ignore-list for the primary folds")
        ignored = {
            line.strip()
            for line in ignore_list.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
    splits = grouped_split(names, REVISION_SPLIT_SEED, fold=fold, ignored_names=ignored)
    sequences = {split: names[indices].tolist() for split, indices in splits.items()}
    manifest = {
        "split_seed": REVISION_SPLIT_SEED,
        "fold": fold,
        "fold_count": REVISION_FOLDS,
        "include_ignored": include_ignored,
        "ignored_sequences": sorted(ignored.intersection(names)),
        "sequences": sequences,
    }
    return splits, manifest


def _save_manifest_once(path: Path, manifest: dict[str, object]) -> None:
    contents = json.dumps(manifest, indent=2)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(contents):
            raise ValueError(f"Manifest differs from the existing file {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(contents + "\n")


def save_split_manifest(out_dir: Path, manifest: dict[str, object]) -> None:
    path = out_dir / SPLIT_MANIFEST
    if not path.exists() and (
        any(out_dir.glob("*_best.pt")) or any(out_dir.glob("*_manifest.json"))
    ):
        raise ValueError(f"Existing run files in {out_dir} lack a revision split manifest")
    _save_manifest_once(path, manifest)


def save_training_manifests(
    path: Path,
    manifest: dict[str, object],
    split_manifest: dict[str, object] | None,
) -> None:
    if split_manifest is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return
    save_split_manifest(path.parent, split_manifest)
    _save_manifest_once(path, manifest)


def training_split(
    names: np.ndarray,
    args: argparse.Namespace,
    legacy_seed: int,
) -> tuple[dict[str, np.ndarray], int, dict[str, object] | None]:
    if args.fold is None:
        if args.ignore_list is not None or args.include_ignored:
            raise ValueError("Ignore-list options require --fold")
        if (args.out_dir / SPLIT_MANIFEST).exists():
            raise ValueError("A fold run directory requires --fold")
        seed = legacy_seed if args.split_seed is None else args.split_seed
        return grouped_split(names, seed), seed, None

    if args.quick or args.split_seed not in (None, REVISION_SPLIT_SEED):
        raise ValueError(f"Fold training requires the full split and seed {REVISION_SPLIT_SEED}")
    splits, manifest = revision_split(names, args.fold, args.ignore_list, args.include_ignored)
    if not all(len(indices) for indices in splits.values()):
        raise ValueError("The selected fold has an empty training, validation, or test set")
    return splits, REVISION_SPLIT_SEED, manifest


def require_legacy_runs(runs: list[Path]) -> None:
    for run in runs:
        if (run / SPLIT_MANIFEST).exists():
            raise ValueError(f"Fold run {run} must use the revision protocol evaluator")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIST++ sequence lists for the frozen grouped folds")
    parser.add_argument("--raw-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    add_fold_arguments(parser)
    args = parser.parse_args()

    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = metadata["names"]
    folds = range(REVISION_FOLDS) if args.fold is None else (args.fold,)
    for fold in folds:
        splits, manifest = revision_split(names, fold, args.ignore_list, args.include_ignored)
        out_dir = args.out_dir / f"fold{fold}" if args.fold is None else args.out_dir
        save_split_manifest(out_dir, manifest)
        print(json.dumps({"fold": fold, **{name: len(indices) for name, indices in splits.items()}}))


if __name__ == "__main__":
    main()
