"""Train SAITS or validate the milestone D baselines on an inner fold."""
from __future__ import annotations

import argparse
import importlib.metadata
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bridge_ablation import TRAINING_SEEDS
from external_baselines import SAITS_CONFIG, SaitsCompletion, pchip_interpolate
from kernel_test import digest, evaluation_runtime, file_record, now, read_json, source_hashes, write_json
from pilot import PilotConfig, evaluate_method, make_mask, seed_everything
from raw_suite import RawWindowDataset, evaluate_detailed, make_train_val_loaders, write_csv
from revision_splits import SPLIT_MANIFEST, revision_split
from run_suite import build_model

NATIVE = ("bigru", "transformer", "patch_transformer")
NONTRAINED = ("linear", "pchip", "hankel")
METHODS = (*NONTRAINED, *NATIVE, "saits")
TRAINED = (*NATIVE, "saits")


class PchipCompletion(nn.Module):
    def forward(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return pchip_interpolate(target, mask)


def split_data(cache: Path, fold: int, ignore: Path):
    with np.load(cache / "metadata.npz", allow_pickle=False) as metadata:
        names = metadata["names"].astype(str)
    return revision_split(names, fold, ignore)


def load_method(method: str, config: PilotConfig, checkpoint: Path | None, device: torch.device):
    if method in ("linear", "hankel"):
        return method
    if method == "pchip":
        return PchipCompletion()
    if method not in TRAINED or checkpoint is None:
        raise ValueError(f"{method} requires a trained checkpoint")
    model = SaitsCompletion(config.frames) if method == "saits" else build_model(method, config)[0]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def select_epoch(history: list[dict]) -> dict:
    if not history or any(not math.isfinite(row[key]) for row in history for key in ("train_loss", "val_mpjpe")):
        raise ValueError("Missing or nonfinite training history")
    best = history[0]
    for row in history[1:]:
        if row["val_mpjpe"] < best["val_mpjpe"] - 1e-6:
            best = row
    return best


def train_saits(args: argparse.Namespace) -> None:
    config = PilotConfig(seed=args.seed, batch_size=16, epochs=24)
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "pypots")}
    if dependencies["pypots"] != SAITS_CONFIG["pypots_version"]:
        raise ValueError("The SAITS training environment does not contain the pinned PyPOTS version")
    seed_everything(config.seed)
    splits, manifest = split_data(args.cache, args.fold, args.ignore_list)
    train, val = make_train_val_loaders(args.cache, splits, config, 2, 3, False, num_workers=0)
    device = torch.device(args.device)
    model = SaitsCompletion(config.frames).to(device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / SPLIT_MANIFEST, manifest, exclusive=True)
    runtime = evaluation_runtime(device)
    write_json(args.out_dir / "saits_training_manifest.json", {
        "fold": args.fold, "seed": args.seed, "stage": "revision_baseline_inner_validation",
        "config": asdict(config), "saits": SAITS_CONFIG, "training_runtime": runtime,
        "amp": device.type == "cuda", "patience": 5, "validation_condition": "mixed",
        "validation_mask_seed": config.seed + 1000, "test_set_accessed": False,
        "source_sha256": source_hashes(), "dependencies": dependencies,
    }, exclusive=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed + 17)
    best_score = math.inf
    stale = 0
    history = []
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train.dataset.set_epoch(epoch)
        model.train()
        loss_sum, windows = 0.0, 0
        for target in train:
            target = target.to(device, non_blocking=True)
            mask = make_mask(len(target), target.shape[1], target.shape[2], "train", generator).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model.training_loss(target, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite SAITS training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(target)
            windows += len(target)
        score = float(evaluate_method(model, val, "mixed", config.seed + 1000, device, warmup=False)["mpjpe"])
        if not math.isfinite(score):
            raise FloatingPointError(f"Nonfinite inner-validation MPJPE at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": loss_sum / windows, "val_mpjpe": score})
        write_json(args.out_dir / "saits_history_state.json", {"history": history})
        print(f"[saits] epoch={epoch:02d} val_mpjpe={score:.6f}", flush=True)
        if score < best_score - 1e-6:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 5:
            break

    if best_state is None or any(not bool(torch.isfinite(value).all()) for value in best_state.values()):
        raise FloatingPointError("Missing or nonfinite SAITS checkpoint")
    path = args.out_dir / "saits_best.pt"
    torch.save({"model": best_state, "config": asdict(config), "saits": SAITS_CONFIG}, path)
    write_json(args.out_dir / "training_complete.json", {
        "fold": args.fold, "seed": args.seed, "test_set_accessed": False,
        "methods": {"saits": {
            "checkpoint": file_record(path), "trained_epochs": len(history),
            "selected_epoch": select_epoch(history)["epoch"], "inner_val_mpjpe": best_score,
            "history_sha256": digest(args.out_dir / "saits_history_state.json"),
        }},
        "source_sha256": source_hashes(), "training_runtime": runtime, "amp": device.type == "cuda",
        "completed_at_utc": now(),
    }, exclusive=True)


def validate(args: argparse.Namespace) -> None:
    config = PilotConfig(seed=args.seed, batch_size=32)
    seed_everything(config.seed)
    splits, manifest = split_data(args.cache, args.fold, args.ignore_list)
    pair = args.runs / f"fold{args.fold}" / f"seed{args.seed}"
    for subdir in ("native", "saits"):
        record = pair / subdir / SPLIT_MANIFEST
        if read_json(record) != manifest:
            raise ValueError(f"Training and inner-validation splits differ: {record}")
    dataset = RawWindowDataset(args.cache, splits["val"], config.frames, mode="fixed",
                               seed=config.seed, max_windows=3)
    device = torch.device(args.device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in METHODS:
        where = "saits" if method == "saits" else "native"
        checkpoint = pair / where / f"{method}_best.pt" if method in TRAINED else None
        model = load_method(method, config, checkpoint, device)
        metrics, _ = evaluate_detailed(model, loader, dataset, "mixed", config.seed + 1000, device, config,
                                       extended_metrics=True)
        if not math.isfinite(float(metrics["mpjpe_window_mean"])):
            raise ValueError(f"Nonfinite inner-validation metric for {method}")
        rows.append({
            "stage": "inner_validation", "fold": args.fold, "seed": args.seed, "condition": "mixed",
            "method": method, "mask_seed": config.seed + 1000,
            "parameters": sum(parameter.numel() for parameter in model.parameters()) if isinstance(model, nn.Module) else 0,
            "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
            "validation_sequences": int(metrics["test_sequences"]), "validation_windows": int(metrics["test_windows"]),
            "mpjpe_window_mean": metrics["mpjpe_window_mean"],
        })
        print(f"validated {method}", flush=True)
    write_csv(args.out_dir / "baseline_validation_check.csv", rows)
    write_json(args.out_dir / "validation_manifest.json", {
        "stage": "inner_validation", "fold": args.fold, "seed": args.seed, "test_set_accessed": False,
        "split_sha256": digest(pair / "native" / SPLIT_MANIFEST),
        "rows": len(rows), "result_sha256": digest(args.out_dir / "baseline_validation_check.csv"),
        "runtime": evaluation_runtime(device), "source_sha256": source_hashes(),
    }, exclusive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-saits", "validate"))
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for option in ("cache", "ignore-list", "out-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    args = parser.parse_args()
    if args.command == "validate" and args.runs is None:
        parser.error("--runs is required for validate")
    (train_saits if args.command == "train-saits" else validate)(args)
