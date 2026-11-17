from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from bridge_ablation import TRAINING_SEEDS, VARIANTS, save_results
from fractional_screen import SCREEN_CONDITIONS


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return math.isfinite(value)


def nonfinite_fields(record: dict) -> list[str]:
    fields = ["train_loss", "fractional_loss", "val_relative_primary", "fractional_orders"]
    fields.extend(f"val_{condition}_mpjpe" for condition in SCREEN_CONDITIONS)
    return [field for field in fields if not finite(record[field])]


def selected_epoch(history: list[dict]) -> dict:
    selected = None
    best = math.inf
    for record in history:
        score = record["val_relative_primary"]
        if math.isfinite(score) and score < best - 1e-5:
            best, selected = score, record
    if selected is None or nonfinite_fields(selected):
        raise ValueError("No finite selected checkpoint epoch")
    return selected


def checkpoint_bands(path: Path) -> dict[str, list[float] | None]:
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError(f"Non-finite selected checkpoint in {path}")
    raw_beta = state.get("backbone.memory.raw_beta")
    if raw_beta is None:
        return {"beta": None, "lambda": None}
    beta = torch.exp(-raw_beta.square()).clamp_min(torch.finfo(raw_beta.dtype).tiny)
    return {"beta": beta.tolist(), "lambda": state["backbone.memory.raw_decay"].square().tolist()}


def add_bands(row: dict, prefix: str, bands: dict) -> None:
    for parameter in ("beta", "lambda"):
        values = bands[parameter]
        if values is not None and len(values) != 3:
            raise ValueError("The bridge audit expects the three-band K5 model")
        for band in range(3):
            row[f"{prefix}_{parameter}_{band + 1}"] = "" if values is None else values[band]


def audit_run(original: Path, retrained: Path, fold: int, seed: int, method: str, variant: str) -> dict:
    for directory in (original, retrained):
        manifest = read_json(directory / "screen_manifest.json")
        if (
            manifest.get("fold") != fold or manifest["config"]["seed"] != seed
            or manifest.get("stage") != "revision_validation_only"
            or manifest.get("test_set_accessed") is not False
            or set(manifest["variants"]) != set(VARIANTS.values())
        ):
            raise ValueError(f"Incorrect training provenance in {directory}")
    if read_json(original / "revision_split_manifest.json") != read_json(retrained / "revision_split_manifest.json"):
        raise ValueError("Original and retrained fold lists differ")
    old_manifest = read_json(original / "screen_manifest.json")
    new_manifest = read_json(retrained / "screen_manifest.json")
    if (
        old_manifest["config"] != new_manifest["config"]
        or old_manifest["memory_lr_multiplier"] != new_manifest["memory_lr_multiplier"]
    ):
        raise ValueError("Original and retrained configurations differ")

    history_path = original / f"{variant}_history.json"
    checkpoint = original / f"{variant}_best.pt"
    history = read_json(history_path)
    selected = selected_epoch(history)
    finite_epochs = [record for record in history if not nonfinite_fields(record)]
    abnormal = next((record for record in history if nonfinite_fields(record)), None)
    last_finite = finite_epochs[-1]
    bands = checkpoint_bands(checkpoint)
    final_decay = None
    if last_finite["epoch"] == selected["epoch"]:
        final_decay = bands["lambda"]
    row = {
        "method": method, "fold": fold, "seed": seed,
        "original_epochs": len(history), "original_finite_epochs": len(finite_epochs),
        "original_first_logged_nonfinite_epoch": "" if abnormal is None else abnormal["epoch"],
        "original_first_logged_nonfinite_fields": "" if abnormal is None else "|".join(nonfinite_fields(abnormal)),
        "original_batch_gradient_source": "not_recorded",
        "original_selected_epoch": selected["epoch"],
        "original_selected_train_loss": selected["train_loss"],
        "original_selected_val_relative_primary": selected["val_relative_primary"],
        "original_last_finite_epoch": last_finite["epoch"],
        "original_last_finite_lambda_source": (
            "not_applicable" if method == "B0" else
            "selected_checkpoint" if final_decay is not None else "not_recorded"
        ),
        "original_history_sha256": digest(history_path),
        "original_checkpoint_sha256": digest(checkpoint),
    }
    add_bands(row, "original_selected", bands)
    add_bands(row, "original_last_finite", {"beta": last_finite["fractional_orders"], "lambda": final_decay})

    new_history_path = retrained / f"{variant}_history.json"
    new_checkpoint = retrained / f"{variant}_best.pt"
    new_history = read_json(new_history_path)
    stability = read_json(retrained / f"{variant}_stability.json")
    new_selected = selected_epoch(new_history)
    if (
        stability["status"] != "completed" or stability["first_nonfinite"] is not None
        or stability["finite_optimizer_steps"] <= 0
        or stability["best_epoch"] != new_selected["epoch"]
        or any(nonfinite_fields(record) or not finite(record["tempering_decays"]) for record in new_history)
    ):
        raise ValueError(f"Retraining did not complete with finite states in {retrained}/{variant}")
    new_bands = checkpoint_bands(new_checkpoint)
    if new_bands["beta"] is not None:
        # CPU and CUDA exp may differ by one float32 ULP for the same raw beta.
        torch.testing.assert_close(
            torch.tensor(new_bands["beta"]), torch.tensor(new_selected["fractional_orders"]), rtol=1e-6, atol=0,
        )
        torch.testing.assert_close(
            torch.tensor(new_bands["lambda"]), torch.tensor(new_selected["tempering_decays"]), rtol=0, atol=0,
        )
    row.update({
        "retrained_epochs": len(new_history), "retrained_finite_epochs": len(new_history),
        "retrained_status": stability["status"], "retrained_nonfinite_events": 0,
        "retrained_finite_optimizer_steps": stability["finite_optimizer_steps"],
        "retrained_minimum_amp_scale": stability["minimum_amp_scale"],
        "retrained_selected_epoch": new_selected["epoch"],
        "retrained_selected_train_loss": new_selected["train_loss"],
        "retrained_selected_val_relative_primary": new_selected["val_relative_primary"],
        "retrained_last_finite_epoch": new_history[-1]["epoch"],
        "retrained_history_sha256": digest(new_history_path),
        "retrained_checkpoint_sha256": digest(new_checkpoint),
    })
    add_bands(row, "retrained_selected", new_bands)
    add_bands(row, "retrained_last_finite", {
        "beta": new_history[-1]["fractional_orders"], "lambda": new_history[-1]["tempering_decays"],
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the original and corrected milestone B bridge training")
    parser.add_argument("--original-runs-dir", required=True, type=Path)
    parser.add_argument("--retrained-runs-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for fold in range(5):
        for seed in TRAINING_SEEDS:
            relative = Path(f"fold{fold}") / f"seed{seed}" / "bridges"
            for method, variant in VARIANTS.items():
                rows.append(audit_run(
                    args.original_runs_dir / relative, args.retrained_runs_dir / relative,
                    fold, seed, method, variant,
                ))
    if len(rows) != 45:
        raise ValueError("Expected 45 paired original/retrained runs")
    save_results(args.out_dir / "bridge_training_stability.csv", rows)
    print(json.dumps({"runs": len(rows), "retrained_nonfinite_events": 0}), flush=True)


if __name__ == "__main__":
    main()
