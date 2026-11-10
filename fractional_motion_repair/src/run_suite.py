from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pilot import (
    BiGRUCompletion,
    DanceFillGatedNet,
    DanceFillReliabilityNet,
    DanceFillNet,
    GapAdaptiveResidualCompletion,
    MotionDataset,
    PatchTransformerCompletion,
    PilotConfig,
    TemporalConvNet,
    TransformerCompletion,
    evaluate_method,
    grouped_split,
    load_cache,
    parameter_count,
    seed_everything,
    train_model,
)


DEFAULT_VARIANTS = (
    "bigru",
    "transformer",
    "patch_transformer",
    "tcn",
    "dancefill",
    "no_graph",
    "no_hankel",
    "no_velocity",
    "no_bone",
    "short_context",
)

DEFAULT_CONDITIONS = (
    "random10",
    "random20",
    "random40",
    "temporal16",
    "temporal32",
    "temporal48",
    "limb24",
    "limb48",
    "limb72",
    "mixed",
)


def build_model(name: str, config: PilotConfig) -> tuple[nn.Module, PilotConfig]:
    if name == "bigru":
        return BiGRUCompletion(config.frames), config
    if name == "transformer":
        return TransformerCompletion(config.frames), config
    if name == "patch_transformer":
        return PatchTransformerCompletion(config.frames), config
    if name == "tcn":
        return TemporalConvNet(config.frames), config
    if name == "wide_tcn":
        return TemporalConvNet(config.frames, width=320), config
    if name == "matched_tcn":
        return TemporalConvNet(
            config.frames,
            width=248,
            dilations=(1, 2, 4, 8, 16, 32),
        ), config
    if name == "gap_router":
        return GapAdaptiveResidualCompletion(
            TemporalConvNet(
                config.frames,
                width=248,
                dilations=(1, 2, 4, 8, 16, 32),
            ),
            zero_through=1,
            full_from=8,
        ), config
    if name == "dancefill_gate":
        return DanceFillGatedNet(config.frames), config
    if name == "gate_fixed":
        return DanceFillGatedNet(
            config.frames,
            learn_gate=False,
            use_severity_prior=False,
        ), config
    if name == "gate_prior_only":
        return DanceFillGatedNet(config.frames, learn_gate=False), config
    if name == "gate_no_graph":
        return DanceFillGatedNet(config.frames, use_graph=False), config
    if name == "gate_no_velocity":
        return DanceFillGatedNet(config.frames, use_velocity=False), config
    if name == "gate_short_only":
        return DanceFillGatedNet(config.frames, branch_mode="short"), config
    if name == "gate_long_only":
        return DanceFillGatedNet(config.frames, branch_mode="long"), config
    if name == "dancefill_reliability":
        return DanceFillReliabilityNet(config.frames), config
    if name == "reliability_unweighted":
        return DanceFillReliabilityNet(config.frames, reliability_aware=False), config
    if name == "reliability_no_graph":
        return DanceFillReliabilityNet(config.frames, use_graph=False), config
    if name == "reliability_no_velocity":
        return DanceFillReliabilityNet(config.frames, use_velocity=False), config

    kwargs: dict[str, object] = {}
    model_config = config
    if name == "no_graph":
        kwargs["use_graph"] = False
    elif name == "no_hankel":
        kwargs["use_hankel"] = False
    elif name == "no_velocity":
        kwargs["use_velocity"] = False
    elif name == "no_bone":
        model_config = replace(config, bone_weight=0.0)
    elif name == "short_context":
        kwargs["dilations"] = (1, 2, 4, 8, 16)
    elif name != "dancefill":
        raise ValueError(f"Unknown variant: {name}")
    return (
        DanceFillNet(
            config.frames,
            config.hankel_window,
            config.hankel_rank,
            **kwargs,
        ),
        model_config,
    )


def make_loaders(
    motions,
    splits,
    config: PilotConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": config.batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train = DataLoader(
        MotionDataset(motions, splits["train"]),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed + 31),
        **common,
    )
    val = DataLoader(
        MotionDataset(motions, splits["val"]), shuffle=False, drop_last=False, **common
    )
    test = DataLoader(
        MotionDataset(motions, splits["test"]), shuffle=False, drop_last=False, **common
    )
    return train, val, test


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/sequence_cache_128.npz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    config = PilotConfig(
        seed=args.seed,
        epochs=3 if args.quick else args.epochs,
        batch_size=16 if args.quick else args.batch_size,
    )
    seed_everything(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    motions, names = load_cache(args.cache, config.frames)
    splits = grouped_split(names, args.split_seed)
    if args.quick:
        splits = {
            "train": splits["train"][:256],
            "val": splits["val"][:96],
            "test": splits["test"][:128],
        }
    manifest = {
        "config": asdict(config),
        "split_seed": args.split_seed,
        "variants": variants,
        "conditions": conditions,
        "split_counts": {key: int(len(value)) for key, value in splits.items()},
    }
    (args.out_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained: dict[str, nn.Module] = {}
    parameters: dict[str, int] = {"linear": 0, "hankel": 0}
    for variant in variants:
        seed_everything(config.seed)
        train_loader, val_loader, _ = make_loaders(motions, splits, config)
        model, model_config = build_model(variant, config)
        model = model.to(device)
        parameters[variant] = parameter_count(model)
        print(f"training={variant} parameters={parameters[variant]}", flush=True)
        trained[variant] = train_model(
            variant,
            model,
            train_loader,
            val_loader,
            model_config,
            device,
            args.out_dir,
        )

    methods: list[tuple[str, nn.Module | str]] = [("linear", "linear"), ("hankel", "hankel")]
    methods.extend((name, trained[name]) for name in variants)
    rows: list[dict[str, object]] = []
    for method_name, method in methods:
        _, _, test_loader = make_loaders(motions, splits, config)
        for condition_index, condition in enumerate(conditions):
            metrics = evaluate_method(
                method,
                test_loader,
                condition,
                config.seed + 10_000 + condition_index,
                device,
                hankel_window=config.hankel_window,
                hankel_rank=config.hankel_rank,
            )
            row: dict[str, object] = {
                "seed": config.seed,
                "method": method_name,
                "condition": condition,
                "parameters": parameters[method_name],
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_rows(args.out_dir / "suite_results.csv", rows)
    (args.out_dir / "suite_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if "dancefill" in variants:
        indexed = {(str(row["method"]), str(row["condition"])): row for row in rows}
        gains: dict[str, dict[str, float]] = {}
        for condition in conditions:
            proposed = float(indexed[("dancefill", condition)]["mpjpe"])
            gains[condition] = {}
            for baseline in ("linear", "tcn"):
                if (baseline, condition) not in indexed:
                    continue
                reference = float(indexed[(baseline, condition)]["mpjpe"])
                gains[condition][f"vs_{baseline}_percent"] = 100.0 * (reference - proposed) / reference
        (args.out_dir / "suite_summary.json").write_text(
            json.dumps({"mpjpe_gain_percent": gains}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
