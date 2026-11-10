from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#172033"
MUTED = "#5E6B7E"
GRID = "#DDE3EA"
BLUE = "#276FBF"
TEAL = "#159D8C"
ORANGE = "#D97706"
RED = "#C94C4C"
LIGHT_BLUE = "#EAF2FB"
LIGHT_TEAL = "#E8F6F3"
LIGHT_ORANGE = "#FFF3DF"
LIGHT_GRAY = "#F4F6F8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(
            out_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "white",
    edgecolor: str = GRID,
    title_size: float = 11.5,
    subtitle_size: float = 9.0,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * (0.60 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        color=INK,
        fontsize=title_size,
        fontweight="semibold",
        linespacing=1.15,
        zorder=3,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.29,
            subtitle,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=subtitle_size,
            linespacing=1.15,
            zorder=3,
        )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    linestyle: str = "-",
    linewidth: float = 1.4,
    mutation_scale: float = 11,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)


def add_elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    via_y: float,
    color: str = MUTED,
    linestyle: str = "-",
) -> None:
    ax.plot(
        [start[0], start[0], end[0] - 0.12],
        [start[1], via_y, via_y],
        color=color,
        linewidth=1.35,
        linestyle=linestyle,
        zorder=1,
    )
    add_arrow(
        ax,
        (end[0] - 0.12, via_y),
        end,
        color=color,
        linestyle=linestyle,
        linewidth=1.35,
    )


def method_figure(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.8, 5.0))
    ax.set_xlim(0, 20.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    add_box(
        ax,
        0.20,
        2.00,
        2.10,
        2.60,
        "Incomplete\nmotion",
        "$\\mathbf{Y},\\,\\mathbf{M}$",
        facecolor=LIGHT_GRAY,
        edgecolor="#B9C3CF",
    )

    add_box(
        ax,
        2.80,
        2.35,
        1.95,
        1.90,
        "Linear\ninterpolation",
        "$\\mathbf{X}_{\\mathrm{lin}}$",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_size=11.0,
    )
    add_arrow(ax, (2.30, 3.30), (2.80, 3.30), color=BLUE)

    outer = FancyBboxPatch(
        (5.25, 0.72),
        7.40,
        5.10,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#FBFDFC",
        edgecolor=TEAL,
        linewidth=1.6,
        zorder=1,
    )
    ax.add_patch(outer)
    ax.text(
        8.95,
        5.43,
        "Multi-band fractional memory bridge",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        8.95,
        5.06,
        "smooth power-law history with exact endpoint constraints",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )

    internal = (
        (5.65, 1.70, "Visible\nincrements"),
        (7.65, 2.10, "GL memory\n(three bands)"),
        (10.05, 2.10, "Endpoint\nbridge"),
    )
    for x0, width, title in internal:
        add_box(
            ax,
            x0,
            2.08,
            width,
            2.25,
            title,
            facecolor="white",
            edgecolor="#9ACFC6",
            title_size=12.3,
        )
    for (left, width, _), (right, _, _) in zip(internal, internal[1:]):
        add_arrow(
            ax,
            (left + width, 3.205),
            (right, 3.205),
            color=TEAL,
            linewidth=1.25,
            mutation_scale=9,
        )
    ax.text(
        7.15,
        1.42,
        "$h_k^{(\\beta)} \\propto \\Gamma(k+\\beta)/(\\Gamma(k+1)\\Gamma(\\beta))$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    ax.text(
        10.95,
        1.42,
        "$C_t=(S_t-S_a)-\\frac{t-a}{b-a}(S_b-S_a)$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    add_arrow(ax, (4.75, 3.30), (5.25, 3.30), color=TEAL)

    add_box(
        ax,
        13.30,
        3.55,
        2.20,
        1.50,
        "Long-gap\nresidual TCN",
        "fractional context\n+ mask",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
        title_size=11.5,
        subtitle_size=8.5,
    )
    add_box(
        ax,
        13.30,
        1.25,
        2.20,
        1.50,
        "Frozen short\nexpert",
        "interpolation\n+ mask",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_size=11.5,
        subtitle_size=8.5,
    )
    add_arrow(ax, (12.65, 4.30), (13.30, 4.30), color=TEAL)
    add_elbow_arrow(
        ax,
        (4.75, 3.00),
        (13.30, 2.00),
        via_y=0.40,
        color=BLUE,
    )

    add_box(
        ax,
        16.15,
        2.35,
        1.65,
        1.80,
        "Duration\nrouter",
        "joint-wise\nrun length",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        title_size=11.5,
        subtitle_size=8.5,
    )
    add_arrow(ax, (15.50, 4.30), (16.15, 3.72), color=ORANGE)
    add_arrow(ax, (15.50, 2.00), (16.15, 2.78), color=ORANGE)

    add_box(
        ax,
        18.55,
        2.00,
        1.30,
        2.60,
        "Repair",
        "$\\widehat{\\mathbf{X}}$",
        facecolor=LIGHT_GRAY,
        edgecolor="#B9C3CF",
    )
    add_arrow(ax, (17.80, 3.25), (18.55, 3.25), color=INK)
    ax.text(
        19.20,
        1.62,
        "$\\mathbf{M}\\odot\\widehat{\\mathbf{X}}=\\mathbf{M}\\odot\\mathbf{Y}$",
        ha="center",
        va="center",
        fontsize=9.0,
        color=INK,
    )
    add_box(
        ax,
        16.00,
        5.05,
        2.40,
        0.72,
        "Fractional trajectory loss\n(training only)",
        facecolor="white",
        edgecolor="#9AA4B2",
        title_size=9.5,
        linewidth=1.0,
    )
    add_arrow(
        ax,
        (15.50, 4.62),
        (16.00, 5.14),
        color="#7A8492",
        linestyle="--",
        linewidth=1.0,
        mutation_scale=8,
    )
    save_figure(fig, out_dir, "figure1_fractional_method")


def protocol_figure(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 4.99))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    add_box(
        ax,
        0.20,
        2.20,
        2.00,
        1.55,
        "AIST++",
        "1,408 sequences\nnative 60 Hz",
        facecolor=LIGHT_GRAY,
        edgecolor="#AEB8C5",
    )
    add_box(
        ax,
        2.75,
        2.20,
        2.30,
        1.55,
        "Group-held-out\nsplit",
        "genre + choreography\nno overlap",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_size=10.8,
        subtitle_size=8.8,
    )
    add_arrow(ax, (2.20, 2.975), (2.75, 2.975), color=BLUE)
    add_box(
        ax,
        5.70,
        4.25,
        2.20,
        1.35,
        "Training",
        "1,002 sequences\nthree seeds",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
    )
    add_box(
        ax,
        5.70,
        2.25,
        2.20,
        1.35,
        "Development",
        "185 sequences\nbroad gate failed",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_box(
        ax,
        5.70,
        0.25,
        2.20,
        1.35,
        "Held-out test",
        "221 sequences\nsealed until freeze",
        facecolor="#FDECEC",
        edgecolor=RED,
    )
    add_arrow(ax, (5.05, 2.975), (5.70, 4.925), color=TEAL)
    add_arrow(ax, (5.05, 2.975), (5.70, 2.925), color=ORANGE)
    add_arrow(ax, (5.05, 2.975), (5.70, 0.925), color=RED)

    add_box(
        ax,
        8.65,
        2.20,
        2.40,
        1.65,
        "Frozen model +\nhypothesis",
        "72-frame limb gap\nfinite vs. fractional",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
        title_size=10.8,
        subtitle_size=8.5,
    )
    add_arrow(ax, (7.90, 4.925), (8.65, 3.52), color=TEAL)
    add_arrow(ax, (7.90, 2.925), (8.65, 3.00), color=ORANGE)
    add_box(
        ax,
        11.75,
        2.20,
        2.15,
        1.65,
        "Confirmatory\ntest",
        "1,419 windows\n10,000 bootstraps",
        facecolor="#FDECEC",
        edgecolor=RED,
        title_size=10.8,
        subtitle_size=8.5,
    )
    add_arrow(ax, (11.05, 3.025), (11.75, 3.025), color=RED)
    ax.plot(
        [7.90, 8.25, 8.25, 11.52],
        [0.925, 0.925, 1.82, 1.82],
        color=RED,
        linewidth=1.35,
        zorder=1,
    )
    add_arrow(ax, (11.52, 1.82), (11.75, 2.45), color=RED)
    add_box(
        ax,
        14.55,
        2.20,
        1.70,
        1.65,
        "Decision",
        "four rules\npassed",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
    )
    add_arrow(ax, (13.90, 3.025), (14.55, 3.025), color=TEAL)

    add_box(
        ax,
        10.15,
        0.05,
        3.60,
        1.50,
        "Post-confirmatory\nduration response",
        "24-96-frame limb gaps\nthree mask repeats",
        facecolor=LIGHT_GRAY,
        edgecolor="#9AA4B2",
        title_size=10.5,
        subtitle_size=8.5,
    )
    add_arrow(
        ax,
        (12.825, 2.20),
        (12.825, 1.55),
        color="#7A8492",
        linestyle="--",
        linewidth=1.1,
    )
    save_figure(fig, out_dir, "figure2_evaluation_protocol")


def confirmatory_figure(
    test_dir: Path,
    test_report_dir: Path,
    out_dir: Path,
) -> None:
    result_rows = read_csv(test_dir / "fractional_test_results.csv")
    bootstrap_rows = read_csv(
        test_report_dir / "fractional_test_paired_bootstrap.csv"
    )
    seeds = sorted({int(row["seed"]) for row in result_rows})
    methods = ("dancerouter", "finite_bridge", "fractional_bridge")
    labels = ("DanceRouter", "Finite bridge", "Fractional bridge")
    colors = (MUTED, BLUE, TEAL)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    for seed_index, seed in enumerate(seeds):
        values = [
            float(
                next(
                    row["mpjpe"]
                    for row in result_rows
                    if int(row["seed"]) == seed
                    and row["method"] == method
                    and row["condition"] == "limb72"
                )
            )
            for method in methods
        ]
        ax.plot(
            np.arange(3),
            values,
            color="#B6BEC8",
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            np.arange(3),
            values,
            color=colors,
            s=34,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
    means = [
        np.mean(
            [
                float(row["mpjpe"])
                for row in result_rows
                if row["method"] == method and row["condition"] == "limb72"
            ]
        )
        for method in methods
    ]
    ax.scatter(np.arange(3), means, marker="D", s=52, color=colors, edgecolor=INK, linewidth=0.7, zorder=3)
    ax.set_xticks(np.arange(3), labels, rotation=18, ha="right")
    ax.set_ylabel("Missing-entry MPJPE")
    ax.set_title("a  Frozen 72-frame limb test", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    def forest(axis: plt.Axes, conditions: list[str], title: str) -> None:
        display = {
            "limb72": "Limb 72",
            "limb48": "Limb 48",
            "mixed": "Mixed",
            "temporal32": "Full-body 32",
            "temporal48": "Full-body 48",
        }
        rows = [
            next(
                row
                for row in bootstrap_rows
                if row["condition"] == condition
                and row["comparison"] == "finite_bridge_minus_fractional_bridge"
            )
            for condition in conditions
        ]
        y = np.arange(len(rows))[::-1]
        gains = np.asarray([float(row["mean_gain_percent"]) for row in rows])
        low = np.asarray([float(row["gain_ci_low"]) for row in rows])
        high = np.asarray([float(row["gain_ci_high"]) for row in rows])
        point_colors = [TEAL if condition == "limb72" else MUTED for condition in conditions]
        for index, point_color in enumerate(point_colors):
            axis.errorbar(
                gains[index],
                y[index],
                xerr=np.asarray(
                    [[gains[index] - low[index]], [high[index] - gains[index]]]
                ),
                fmt="none",
                ecolor=point_color,
                elinewidth=1.8,
                capsize=3,
            )
        axis.scatter(gains, y, c=point_colors, s=46, zorder=3)
        axis.axvline(0, color=INK, linewidth=0.9)
        axis.set_yticks(y, [display[item] for item in conditions])
        axis.set_xlabel("Relative MPJPE gain (%)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color=GRID, linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    forest(axes[1], ["limb72", "limb48", "mixed"], "b  Partial-body conditions")
    forest(axes[2], ["temporal32", "temporal48"], "c  Full-body stress tests")
    axes[1].text(
        0.02,
        -0.25,
        "Positive values favor the fractional bridge",
        transform=axes[1].transAxes,
        fontsize=11.5,
        color=MUTED,
    )
    fig.tight_layout(w_pad=2.5)
    save_figure(fig, out_dir, "figure3_confirmatory_results")


def duration_figure(gap_dir: Path, out_dir: Path) -> None:
    rows = read_csv(gap_dir / "gap_sweep_metric_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    settings = (
        ("mpjpe", "Position error", TEAL),
        ("bone_relative_error", "Relative bone-length error", ORANGE),
    )
    for axis, (metric, label, color) in zip(axes, settings):
        selected = sorted(
            (row for row in rows if row["metric"] == metric),
            key=lambda row: int(row["gap_frames"]),
        )
        frames = np.asarray([int(row["gap_frames"]) for row in selected])
        gains = np.asarray([float(row["mean_gain_percent"]) for row in selected])
        low = np.asarray([float(row["gain_ci_low"]) for row in selected])
        high = np.asarray([float(row["gain_ci_high"]) for row in selected])
        axis.fill_between(frames, low, high, color=color, alpha=0.16, linewidth=0)
        axis.plot(frames, gains, color=color, linewidth=2.0, marker="o", markersize=5)
        axis.axhline(0, color=INK, linewidth=0.9)
        axis.axvline(60, color="#9AA4B2", linewidth=0.9, linestyle="--")
        axis.text(61.5, axis.get_ylim()[1] * 0.86, "60 frames", fontsize=11.5, color=MUTED)
        axis.set_xticks(frames, [f"{value}\n{value / 60:.1f}s" for value in frames])
        axis.set_xlabel("Limb-gap duration at 60 Hz")
        axis.set_ylabel("Fractional over finite gain (%)")
        axis.set_title(label, loc="left", fontweight="bold")
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Post-confirmatory duration response (221 sequences, 9 paired replicates per method)",
        fontsize=15.0,
        color=INK,
        y=1.02,
    )
    fig.tight_layout(w_pad=2.8)
    save_figure(fig, out_dir, "figure4_duration_response")


def gl_integral(beta: float, size: int = 97) -> np.ndarray:
    coefficients = [1.0]
    for index in range(1, size):
        coefficients.append(coefficients[-1] * (beta + index - 1) / index)
    values = np.asarray(coefficients)
    return values / values.sum()


def memory_figure(
    memory_dir: Path,
    order_dir: Path,
    out_dir: Path,
) -> None:
    summary = json.loads(
        (memory_dir / "learned_fractional_memory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    joints = read_csv(memory_dir / "learned_fractional_memory_by_joint.csv")
    order_summary = json.loads(
        (order_dir / "fractional_order_ablation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    axis = axes[0]
    lags = np.arange(97)
    for beta, color, label in zip(
        summary["learned_order_mean"],
        (BLUE, TEAL, ORANGE),
        ("Short band", "Middle band", "Long band"),
    ):
        axis.plot(lags, gl_integral(float(beta)), color=color, linewidth=2.0, label=f"{label}: beta={beta:.3f}")
    axis.set_yscale("log")
    axis.set_xlabel("History lag (frames)")
    axis.set_ylabel("Normalized GL weight (log scale)")
    axis.set_title("a  Smooth power-law memory", loc="left", fontweight="bold")
    axis.grid(color=GRID, linewidth=0.6, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=11.0)

    regions = {
        "Head": list(range(0, 5)),
        "Arms": list(range(5, 11)),
        "Hips": [11, 12],
        "Legs": list(range(13, 17)),
    }
    axis = axes[1]
    x = np.arange(len(regions))
    width = 0.23
    for band in range(3):
        values = [
            np.mean([float(joints[index][f"mix_band{band + 1}_mean"]) for index in indices])
            for indices in regions.values()
        ]
        axis.bar(
            x + (band - 1) * width,
            values,
            width=width,
            color=(BLUE, TEAL, ORANGE)[band],
            label=("Short", "Middle", "Long")[band],
        )
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("Mean mixture weight")
    axis.set_xticks(x, list(regions), rotation=15, ha="right")
    axis.set_title("b  Band allocation", loc="left", fontweight="bold")
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=11.0, ncol=3, loc="upper center")

    axis = axes[2]
    changes = order_summary["relative_change_vs_learned_percent"]
    modes = ("learned", "band_centers", "joint_mean", "joint_permuted")
    labels = ("Learned", "Fixed centers", "Joint mean", "Permuted joints")
    values = np.asarray([float(changes[mode]) for mode in modes])
    colors = [TEAL if mode == "learned" else MUTED for mode in modes]
    axis.barh(np.arange(len(modes))[::-1], values, color=colors, height=0.56)
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.set_yticks(np.arange(len(modes))[::-1], labels)
    axis.set_xlabel("MPJPE change vs. learned orders (%)")
    axis.set_title("c  Order ablation (development)", loc="left", fontweight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, out_dir, "figure5_memory_analysis")


def efficiency_figure(latency_dir: Path, out_dir: Path) -> None:
    rows = json.loads((latency_dir / "fractional_latency.json").read_text(encoding="utf-8"))
    methods = ("dancerouter", "finite_bridge", "fractional_bridge")
    labels = ("DanceRouter", "Finite bridge", "Fractional bridge")
    colors = (MUTED, BLUE, TEAL)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))
    for axis, batch_size, title in (
        (axes[0], 1, "a  Batch 1"),
        (axes[1], 32, "b  Batch 32 (per window)"),
    ):
        selected = [
            next(
                row
                for row in rows
                if row["method"] == method and int(row["batch_size"]) == batch_size
            )
            for method in methods
        ]
        medians = np.asarray([float(row["median_ms_per_window"]) for row in selected])
        p95 = np.asarray([float(row["p95_ms_per_window"]) for row in selected])
        x = np.arange(3)
        axis.bar(x, medians, color=colors, width=0.62)
        axis.errorbar(
            x,
            medians,
            yerr=np.vstack((np.zeros(3), p95 - medians)),
            fmt="none",
            ecolor=INK,
            capsize=3,
            linewidth=1.0,
        )
        for index, value in enumerate(medians):
            axis.text(index, p95[index] + max(medians) * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=11.0)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylabel("Milliseconds per 128-frame window")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTX 3090 inference after 100 warm-up and 500 measured passes", fontsize=14.5, y=1.02)
    fig.tight_layout(w_pad=2.8)
    save_figure(fig, out_dir, "figure6_efficiency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--test-report-dir", type=Path, required=True)
    parser.add_argument("--gap-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--order-dir", type=Path, required=True)
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    method_figure(args.out_dir)
    protocol_figure(args.out_dir)
    confirmatory_figure(args.test_dir, args.test_report_dir, args.out_dir)
    duration_figure(args.gap_dir, args.out_dir)
    memory_figure(args.memory_dir, args.order_dir, args.out_dir)
    efficiency_figure(args.latency_dir, args.out_dir)
    print(json.dumps({"figures": sorted(path.name for path in args.out_dir.glob("*.pdf"))}, indent=2))


if __name__ == "__main__":
    main()
