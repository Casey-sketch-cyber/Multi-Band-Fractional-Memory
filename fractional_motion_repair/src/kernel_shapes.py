from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from bridge_ablation import TRAINING_SEEDS, read_json, save_results
from bridge_stability import selected_epoch
from fractional_screen import build_long_expert, load_checkpoint, nonfinite_tensor_names
from kernel_diagnostics import kernel_profiles, kernel_statistics


FAMILIES = {
    "power_law": ((), (1,)),
    "exponential": ((), (2,)),
    "tempered_power_law": ((), (1,), (2,), (1, 2)),
}
PARAMETERS = {"power_law": 3, "exponential": 3, "tempered_power_law": 4}


def fit_kernel_shapes(weights: np.ndarray) -> list[dict[str, object]]:
    """Fit nonnegative decay slopes to log tap values with a free intercept."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) <= max(PARAMETERS.values()):
        raise ValueError("Shape fitting requires a one-dimensional kernel with at least five taps")
    valid = np.isfinite(weights).all() and (weights > 0).all()
    lags = np.arange(len(weights), dtype=np.float64)
    design = np.column_stack((np.ones(len(weights)), -np.log1p(lags), -lags))
    target = np.log(weights) if valid else None
    results = []
    for family, active_sets in FAMILIES.items():
        record = {
            "family": family,
            "fit_status": "nonpositive_or_nonfinite_taps",
            "fit_points": len(weights),
            "aic_parameters": PARAMETERS[family],
            "log_amplitude": None,
            "power_exponent": None,
            "exponential_decay": None,
            "rss_log_weights": None,
            "aic": None,
            "delta_aic": None,
        }
        if valid:
            candidates = []
            for active in active_sets:
                columns = (0, *active)
                values, _, rank, _ = np.linalg.lstsq(design[:, columns], target, rcond=None)
                if rank != len(columns) or (values[1:] < 0).any():
                    continue
                coefficients = np.zeros(3, dtype=np.float64)
                coefficients[list(columns)] = values
                residual = target - design @ coefficients
                candidates.append((float(residual @ residual), coefficients))
            rss, coefficients = min(candidates, key=lambda item: item[0])
            aic = len(weights) * math.log(rss / len(weights)) + 2 * PARAMETERS[family] if rss > 0 else None
            record.update(
                fit_status="ok" if aic is not None else "zero_residual_variance",
                log_amplitude=float(coefficients[0]),
                power_exponent=float(coefficients[1]),
                exponential_decay=float(coefficients[2]),
                rss_log_weights=rss,
                aic=aic,
            )
        results.append(record)
    aics = [record["aic"] for record in results if record["aic"] is not None]
    for record in results:
        if record["aic"] is not None:
            record["delta_aic"] = record["aic"] - min(aics)
    return results


def fitted_profile(record: dict[str, object], lags: np.ndarray) -> np.ndarray:
    return np.exp(
        record["log_amplitude"]
        - record["power_exponent"] * np.log1p(lags)
        - record["exponential_decay"] * lags
    )


def selected_profile(directory: Path, variant: str, fold: int, seed: int) -> dict:
    manifest = read_json(directory / "screen_manifest.json")
    if (
        manifest["fold"] != fold
        or manifest["config"]["seed"] != seed
        or manifest["test_set_accessed"] is not False
        or manifest["stage"] != "revision_validation_only"
        or variant not in manifest["variants"]
    ):
        raise ValueError(f"Incorrect selected-kernel provenance in {directory}")
    status = read_json(directory / f"{variant}_stability.json")
    chosen = selected_epoch(read_json(directory / f"{variant}_history.json"))
    if (
        status["status"] != "completed"
        or status["first_nonfinite"] is not None
        or status["best_epoch"] != chosen["epoch"]
    ):
        raise ValueError(f"Kernel training did not complete finitely in {directory}")
    checkpoint = directory / f"{variant}_best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    timing = read_json(directory / "training_timing.json")
    if digest != timing["experts"][variant]["checkpoint_sha256"]:
        raise ValueError(f"Selected checkpoint hash differs in {directory}")
    expert, _ = build_long_expert(variant)
    if load_checkpoint(expert, checkpoint)["missing"]:
        raise ValueError(f"Selected checkpoint is incomplete in {directory}")
    if nonfinite_tensor_names(expert.state_dict().items()):
        raise ValueError(f"Selected checkpoint contains nonfinite tensors in {directory}")
    expert.eval()
    return {
        "metadata": {
            "fold": fold,
            "seed": seed,
            "method": variant,
            "selected_epoch": chosen["epoch"],
            "checkpoint_sha256": digest,
            "training_source_commit": timing["source_commit"],
        },
        "profiles": kernel_profiles(expert).numpy(),
        "statistics": kernel_statistics(expert),
    }


def plot_shapes(records: dict[str, list[dict]], out_dir: Path, suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference, _ = build_long_expert("B3_K3")
    reference_profiles = kernel_profiles(reference).numpy()
    lags = np.arange(reference_profiles.shape[1])
    colors = {"power_law": "#D17C24", "exponential": "#28836B", "tempered_power_law": "#AB4264"}
    styles = {"power_law": "-", "exponential": "-.", "tempered_power_law": ":"}
    labels = {"power_law": "Power law fit", "exponential": "Exponential fit", "tempered_power_law": "Tempered fit"}
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharex=True)
    for row, method in enumerate(("B3_K5", "B3_K6")):
        samples = records[method]
        for band, axis in enumerate(axes[row]):
            profiles = np.stack([item["profiles"][band] for item in samples])
            mean = profiles.mean(axis=0)
            low, high = np.quantile(profiles, (0.1, 0.9), axis=0)
            axis.semilogy(lags, np.where(mean > 0, mean, np.nan), color="#245B9A", linewidth=1.9, label="Checkpoint mean")
            axis.fill_between(lags, low, high, where=low > 0, color="#245B9A", alpha=0.17, label="10–90% range")
            axis.semilogy(lags, reference_profiles[band], color="#797979", linestyle="--", linewidth=1.2, label="Fixed GL reference")
            if method == "B3_K6":
                for family in FAMILIES:
                    curves = [
                        fitted_profile(fit, lags)
                        for item in samples
                        for fit in item["fits"][band]
                        if fit["family"] == family and fit["fit_status"] == "ok"
                    ]
                    if curves:
                        axis.semilogy(
                            lags, np.mean(curves, axis=0), color=colors[family],
                            linestyle=styles[family], linewidth=1.5, label=labels[family],
                        )
            mean_lag = np.mean([item["statistics"][band]["mean_lag_frames"] for item in samples])
            zero_mass = np.mean([item["statistics"][band]["zero_lag_mass"] for item in samples])
            axis.set_title(f"{method}, band {band + 1}\nMean lag {mean_lag:.3f} frames, zero-lag mass {zero_mass:.3f}", fontsize=10)
            axis.set_xlim(0, int(lags[-1]))
            axis.grid(axis="y", which="major", alpha=0.18)
            axis.tick_params(labelsize=9)
            if row == 1:
                axis.set_xlabel("Absolute lag (frames)", fontsize=10)
            if band == 0:
                axis.set_ylabel("Kernel tap weight (log scale)", fontsize=10)
    title_prefix = "Preview · " if suffix else ""
    figure.suptitle(
        f"{title_prefix}Selected temporal kernels · K5 n={len(records['B3_K5'])}, K6 n={len(records['B3_K6'])} fold–seed pairs",
        fontsize=12,
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.16, hspace=0.4, wspace=0.27)
    figure.savefig(out_dir / f"kernel_shapes{suffix}.png", dpi=240)
    figure.savefig(out_dir / f"kernel_shapes{suffix}.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selected K5/K6 kernel shapes and exploratory log-profile fits")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--stable-runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Inspect the currently completed K6 runs")
    args = parser.parse_args()
    records = {"B3_K5": [], "B3_K6": []}
    fit_rows = []
    for fold, seed in product(range(5), TRAINING_SEEDS):
        relative = Path(f"fold{fold}") / f"seed{seed}"
        for method, directory in (
            ("B3_K5", args.stable_runs_dir / relative / "bridges"),
            ("B3_K6", args.runs_dir / relative / "k6"),
        ):
            if not directory.is_dir() and args.preview and method == "B3_K6":
                continue
            item = selected_profile(directory, method, fold, seed)
            if method == "B3_K6":
                item["fits"] = [fit_kernel_shapes(weights) for weights in item["profiles"]]
                for statistics, fits in zip(item["statistics"], item["fits"], strict=True):
                    fit_rows.extend({**item["metadata"], **statistics, **fit} for fit in fits)
            records[method].append(item)
    if not records["B3_K6"]:
        raise ValueError("No completed K6 checkpoints are available")
    if not args.preview and any(len(items) != 15 for items in records.values()):
        raise ValueError("The final kernel-shape report requires all five folds and three seeds")
    suffix = "_preview" if args.preview else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_results(args.out_dir / f"kernel_shape_fit{suffix}.csv", fit_rows)
    plot_shapes(records, args.out_dir, suffix)
    manifest = {
        "stage": "partial_exploratory_kernel_shapes" if args.preview else "exploratory_kernel_shapes",
        "checkpoint_selection": "inner_validation",
        "test_set_accessed": False,
        "fit_domain": "log_normalized_tap_weights",
        "fit_equations": ["a-p*log(k+1)", "a-lambda*k", "a-p*log(k+1)-lambda*k"],
        "fit_constraints": "p >= 0, lambda >= 0",
        "aic_parameters_including_variance": PARAMETERS,
        "plot_summary": "checkpoint mean and 10th–90th percentile range",
        "checkpoints": {method: [{**item["metadata"], "bands": item["statistics"]} for item in items] for method, items in records.items()},
        "analysis_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("kernel_diagnostics.py"))
        },
    }
    (args.out_dir / f"kernel_shape_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoints": {method: len(items) for method, items in records.items()}, "fit_rows": len(fit_rows), "preview": args.preview}))


if __name__ == "__main__":
    main()
