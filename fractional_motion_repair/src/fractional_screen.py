from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fractional_models import MemoryTemporalConvNet, fractional_trajectory_loss
from pilot import (
    DurationAwareExpertRouter,
    GapAdaptiveResidualCompletion,
    PilotConfig,
    TemporalConvNet,
    completion_loss,
    evaluate_method,
    grouped_split,
    make_mask,
    parameter_count,
    seed_everything,
)
from raw_suite import RawWindowDataset


SCREEN_CONDITIONS = (
    "temporal32",
    "temporal48",
    "limb48",
    "limb72",
    "mixed",
)
PRIMARY_CONDITIONS = ("limb48", "limb72", "mixed")
LONG_GAP_TRAIN_CONDITIONS = (
    "temporal32",
    "temporal48",
    "limb48",
    "limb72",
    "mixed",
)
VARIANTS = (
    "continued",
    "fractional_loss",
    "integer_memory",
    "fractional_memory",
    "fractional_memory_loss",
    "fractional_fixed",
    "fractional_fixed_loss",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_long_gap_train_mask(
    batch: int,
    frames: int,
    joints: int,
    generator: torch.Generator,
) -> torch.Tensor:
    masks = []
    for _ in range(batch):
        condition_index = int(
            torch.randint(
                len(LONG_GAP_TRAIN_CONDITIONS),
                (1,),
                generator=generator,
            )
        )
        masks.append(
            make_mask(
                1,
                frames,
                joints,
                LONG_GAP_TRAIN_CONDITIONS[condition_index],
                generator,
            )[0]
        )
    return torch.stack(masks)


def make_train_val_loaders(
    cache_dir: Path,
    splits: dict[str, np.ndarray],
    config: PilotConfig,
    train_windows: int,
    val_windows: int,
    quick: bool,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = RawWindowDataset(
        cache_dir,
        splits["train"],
        config.frames,
        mode="random",
        seed=config.seed + 101,
        windows_per_sequence=train_windows,
    )
    val_dataset = RawWindowDataset(
        cache_dir,
        splits["val"],
        config.frames,
        mode="fixed",
        seed=config.seed,
        max_windows=2 if quick else val_windows,
    )
    common = {
        "batch_size": config.batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 31),
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def load_checkpoint(module: nn.Module, path: Path) -> dict[str, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = module.load_state_dict(payload["model"], strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    allowed_prefixes = ("backbone.memory.",)
    disallowed = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if disallowed:
        raise RuntimeError(f"Checkpoint mismatch outside memory adapter: {disallowed}")
    return {"missing": missing, "unexpected": unexpected}


def build_short_expert(checkpoint: Path, device: torch.device) -> nn.Module:
    model = TemporalConvNet(
        frames=128,
        width=248,
        dilations=(1, 2, 4, 8, 16, 32),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_long_expert(variant: str) -> tuple[nn.Module, float]:
    if variant in {"continued", "fractional_loss"}:
        backbone: nn.Module = TemporalConvNet(
            frames=128,
            width=248,
            dilations=(1, 2, 4, 8, 16, 32),
        )
    elif variant == "integer_memory":
        backbone = MemoryTemporalConvNet(frames=128, mode="integer")
    elif variant in {"fractional_memory", "fractional_memory_loss"}:
        backbone = MemoryTemporalConvNet(frames=128, mode="fractional")
    elif variant in {"fractional_fixed", "fractional_fixed_loss"}:
        backbone = MemoryTemporalConvNet(
            frames=128,
            mode="fractional",
            learn_orders=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model = GapAdaptiveResidualCompletion(
        backbone,
        zero_through=1,
        full_from=8,
    )
    fractional_weight = 0.1 if "loss" in variant else 0.0
    return model, fractional_weight


def fractional_orders(model: nn.Module) -> list[float] | None:
    backbone = getattr(model, "backbone", None)
    memory = getattr(backbone, "memory", None)
    if memory is None or memory.mode != "fractional":
        return None
    values = memory.fractional_orders().detach().cpu().mean(dim=0)
    return [float(value) for value in values]


@torch.no_grad()
def evaluate_router(
    router: nn.Module,
    val_loader: DataLoader,
    config: PilotConfig,
    baseline: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, float]], float]:
    metrics: dict[str, dict[str, float]] = {}
    for condition_index, condition in enumerate(SCREEN_CONDITIONS):
        metrics[condition] = evaluate_method(
            router,
            val_loader,
            condition,
            config.seed + 50_000 + condition_index,
            next(router.parameters()).device,
            warmup=False,
        )
    if baseline is None:
        score = float(np.mean([metrics[item]["mpjpe"] for item in PRIMARY_CONDITIONS]))
    else:
        score = float(
            np.mean(
                [
                    metrics[item]["mpjpe"] / baseline[item]
                    for item in PRIMARY_CONDITIONS
                ]
            )
        )
    return metrics, score


def train_extension(
    name: str,
    long_expert: nn.Module,
    short_expert: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: PilotConfig,
    device: torch.device,
    out_dir: Path,
    baseline: dict[str, float],
    fractional_weight: float,
    memory_lr_multiplier: float,
) -> tuple[nn.Module, list[dict[str, object]]]:
    memory_parameters = []
    backbone_parameters = []
    for parameter_name, parameter in long_expert.named_parameters():
        destination = (
            memory_parameters
            if ".memory." in parameter_name
            else backbone_parameters
        )
        destination.append(parameter)
    parameter_groups = [
        {
            "params": backbone_parameters,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]
    if memory_parameters:
        parameter_groups.append(
            {
                "params": memory_parameters,
                "lr": config.learning_rate * memory_lr_multiplier,
                "weight_decay": config.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed + 17)
    router = DurationAwareExpertRouter(short_expert, long_expert)
    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        long_expert.train()
        short_expert.eval()
        total_loss = 0.0
        total_fractional = 0.0
        samples = 0
        started = time.perf_counter()
        for target in train_loader:
            target = target.to(device, non_blocking=True)
            mask = make_long_gap_train_mask(
                len(target),
                target.shape[1],
                target.shape[2],
                generator,
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = long_expert(target, mask)
                base_loss, _ = completion_loss(prediction, target, mask, config)
                if fractional_weight > 0.0:
                    fractional = fractional_trajectory_loss(
                        prediction,
                        target,
                        mask,
                        alpha=0.5,
                        kernel_size=5,
                    )
                else:
                    fractional = prediction.new_zeros(())
                loss = base_loss + fractional_weight * fractional
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(long_expert.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            total_fractional += float(fractional.detach()) * len(target)
            samples += len(target)

        metrics, score = evaluate_router(router, val_loader, config, baseline)
        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "fractional_loss": total_fractional / max(samples, 1),
            "val_relative_primary": score,
            "seconds": time.perf_counter() - started,
            "fractional_orders": fractional_orders(long_expert),
            "memory_lr_multiplier": memory_lr_multiplier,
        }
        for condition in SCREEN_CONDITIONS:
            record[f"val_{condition}_mpjpe"] = metrics[condition]["mpjpe"]
        history.append(record)
        print(json.dumps({"variant": name, **record}), flush=True)

        if score < best_score - 1e-5:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in long_expert.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4:
            print(f"[{name}] early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {name}")
    long_expert.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "config": asdict(config),
            "variant": name,
            "fractional_weight": fractional_weight,
        },
        out_dir / f"{name}_best.pt",
    )
    (out_dir / f"{name}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return long_expert, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("data/raw60"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--split-seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-windows", type=int, default=2)
    parser.add_argument("--val-windows", type=int, default=3)
    parser.add_argument("--memory-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    config = PilotConfig(
        seed=args.seed,
        epochs=min(args.epochs, 3) if args.quick else args.epochs,
        batch_size=min(args.batch_size, 8) if args.quick else args.batch_size,
        learning_rate=1e-4,
    )
    seed_everything(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.raw_cache_dir / "metadata.npz") as metadata:
        names = np.asarray(metadata["names"]).astype(str)
    splits = grouped_split(names, args.split_seed)
    if args.quick:
        splits = {
            "train": splits["train"][:128],
            "val": splits["val"][:48],
            "test": np.asarray([], dtype=np.int64),
        }
    manifest = {
        "stage": "development_validation_only",
        "test_set_accessed": False,
        "config": asdict(config),
        "split_seed": args.split_seed,
        "split_counts": {key: int(len(value)) for key, value in splits.items()},
        "screen_conditions": SCREEN_CONDITIONS,
        "primary_conditions": PRIMARY_CONDITIONS,
        "training_mask_curriculum": LONG_GAP_TRAIN_CONDITIONS,
        "variants": variants,
        "memory_lr_multiplier": args.memory_lr_multiplier,
        "initialization": str(args.base_run / "gap_router_best.pt"),
        "fractional_loss_reference": {
            "alpha": 0.5,
            "kernel_size": 5,
            "weight": 0.1,
        },
        "memory_adapter": {
            "signal": "endpoint_constrained_visible_increment_bridge",
            "order_sharing": "joint_specific_xyz_shared",
            "kernel_size": 97,
            "fractional_order_ranges": [
                [0.10, 0.35],
                [0.35, 0.65],
                [0.65, 0.90],
            ],
            "finite_window_control": [9, 33, 97],
        },
    }
    (args.out_dir / "screen_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short_expert = build_short_expert(
        args.base_run / "matched_tcn_best.pt", device
    )
    original_long, _ = build_long_expert("continued")
    load_checkpoint(original_long, args.base_run / "gap_router_best.pt")
    original_long.to(device).eval()
    baseline_router = DurationAwareExpertRouter(short_expert, original_long)
    _, baseline_val_loader = make_train_val_loaders(
        args.raw_cache_dir,
        splits,
        config,
        args.train_windows,
        args.val_windows,
        args.quick,
    )
    baseline_metrics, _ = evaluate_router(
        baseline_router,
        baseline_val_loader,
        config,
    )
    baseline_mpjpe = {
        condition: baseline_metrics[condition]["mpjpe"]
        for condition in SCREEN_CONDITIONS
    }

    trained: dict[str, nn.Module] = {}
    initialization: dict[str, dict[str, list[str]]] = {}
    for variant in variants:
        seed_everything(config.seed)
        train_loader, val_loader = make_train_val_loaders(
            args.raw_cache_dir,
            splits,
            config,
            args.train_windows,
            args.val_windows,
            args.quick,
        )
        long_expert, fractional_weight = build_long_expert(variant)
        initialization[variant] = load_checkpoint(
            long_expert, args.base_run / "gap_router_best.pt"
        )
        long_expert.to(device)
        print(
            f"training={variant} parameters={parameter_count(long_expert)} "
            f"fractional_weight={fractional_weight}",
            flush=True,
        )
        trained[variant], _ = train_extension(
            variant,
            long_expert,
            short_expert,
            train_loader,
            val_loader,
            config,
            device,
            args.out_dir,
            baseline_mpjpe,
            fractional_weight,
            args.memory_lr_multiplier,
        )

    rows: list[dict[str, object]] = []
    for method_name, long_expert in (
        ("dancerouter", original_long),
        *trained.items(),
    ):
        _, val_loader = make_train_val_loaders(
            args.raw_cache_dir,
            splits,
            config,
            args.train_windows,
            args.val_windows,
            args.quick,
        )
        router = DurationAwareExpertRouter(short_expert, long_expert)
        metrics, score = evaluate_router(
            router,
            val_loader,
            config,
            baseline_mpjpe if method_name != "dancerouter" else None,
        )
        for condition in SCREEN_CONDITIONS:
            rows.append(
                {
                    "seed": config.seed,
                    "method": method_name,
                    "condition": condition,
                    "mpjpe": metrics[condition]["mpjpe"],
                    "velocity_error": metrics[condition]["velocity_error"],
                    "bone_relative_error": metrics[condition]["bone_relative_error"],
                    "relative_to_dancerouter": (
                        metrics[condition]["mpjpe"] / baseline_mpjpe[condition]
                    ),
                    "relative_primary_score": score,
                    "parameters_long_expert": parameter_count(long_expert),
                    "fractional_orders": fractional_orders(long_expert),
                }
            )

    score_by_method = {
        method: float(
            np.mean(
                [
                    row["relative_to_dancerouter"]
                    for row in rows
                    if row["method"] == method
                    and row["condition"] in PRIMARY_CONDITIONS
                ]
            )
        )
        for method in ["dancerouter", *variants]
    }
    fractional_candidates = [
        name
        for name in (
            "fractional_memory",
            "fractional_memory_loss",
            "fractional_fixed",
            "fractional_fixed_loss",
        )
        if name in score_by_method
    ]
    best_fractional = min(
        fractional_candidates,
        key=score_by_method.__getitem__,
        default=None,
    )
    controls = [
        name
        for name in ("continued", "fractional_loss", "integer_memory")
        if name in score_by_method
    ]
    best_control_score = min(
        (score_by_method[name] for name in controls),
        default=math.inf,
    )
    regressions = []
    if best_fractional is not None:
        regressions = [
            row["condition"]
            for row in rows
            if row["method"] == best_fractional
            and row["relative_to_dancerouter"] > 1.015
        ]
    gate = {
        "best_fractional_variant": best_fractional,
        "relative_primary_scores": score_by_method,
        "best_fractional_beats_original_by_0_5_percent": (
            best_fractional is not None
            and score_by_method[best_fractional] <= 0.995
        ),
        "best_fractional_beats_controls_by_0_2_percent": (
            best_fractional is not None
            and score_by_method[best_fractional] <= best_control_score - 0.002
        ),
        "conditions_with_more_than_1_5_percent_regression": regressions,
        "advance_to_test": (
            best_fractional is not None
            and score_by_method[best_fractional] <= 0.995
            and score_by_method[best_fractional] <= best_control_score - 0.002
            and not regressions
        ),
    }
    write_csv(args.out_dir / "screen_results.csv", rows)
    (args.out_dir / "screen_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (args.out_dir / "screen_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (args.out_dir / "initialization_report.json").write_text(
        json.dumps(initialization, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
