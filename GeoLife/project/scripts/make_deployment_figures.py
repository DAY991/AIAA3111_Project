"""Export member-A research figures from measured CSV files only.

From the final repository root:
    python project/scripts/make_deployment_figures.py

The coverage figure can be made before model evaluation. Grid figures and
interaction contrasts require full-LOUO results on the common support, seed 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "现实关联" / "outputs"
SAMPLE_INTERVALS = [0, 5, 10]
NOISE_LEVELS = [0, 5, 10, 20]
CLASSES = ["walk", "bike", "bus", "car", "subway"]
MODEL_COLOURS = {"rf": "#377eb8", "gru": "#e07a25"}


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_retention(input_dir: Path) -> None:
    pilot = pd.read_csv(input_dir / "retention_pilot.csv")
    classes = pd.read_csv(input_dir / "retention_by_class.csv")
    selected = pilot.loc[pilot["noise_sigma_m"].eq(0)].set_index("downsample_sec")
    if selected.index.duplicated().any():
        raise ValueError("Duplicate coverage rows for a sampling interval")
    selected = selected.loc[[0, 5, 10, 30]]
    n_reference = int(selected.loc[0, "n_reference"])
    n_common = int(selected.loc[0, "n_common"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    positions = np.arange(4)
    values = selected["retention"].to_numpy() * 100
    bars = axes[0].bar(positions, values, color=["#777777", "#66a5cc", "#377eb8", "#cccccc"])
    for bar, (_, row) in zip(bars, selected.iterrows()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{row['retention']:.1%}\n(n={int(row['n_usable']):,})",
                     ha="center", va="bottom", fontsize=9)
    if int(selected.loc[30, "n_usable"]) == 0:
        axes[0].text(3, 19, "No usable\nwindows", ha="center", fontsize=9, color="#555555")
    axes[0].set_xticks(positions, ["Native", "5 s", "10 s", "30 s (stress)"])
    axes[0].set_title("Overall coverage")
    axes[0].set_xlabel("Sampling interval")

    positions = np.arange(len(CLASSES))
    width = 0.35
    for offset, step, colour in [(-0.5, 5, "#66a5cc"), (0.5, 10, "#377eb8")]:
        rows = classes.loc[classes["condition"].eq(f"ds{step}_n0")].set_index("label")
        if rows.index.duplicated().any():
            raise ValueError(f"Duplicate class coverage rows for ds{step}_n0")
        percentages = rows.loc[CLASSES, "retention"].to_numpy() * 100
        bars = axes[1].bar(positions + offset * width, percentages, width,
                           label=f"{step} s", color=colour)
        axes[1].bar_label(bars, labels=[f"{v:.1f}" for v in percentages], padding=3, fontsize=8)
    axes[1].set_xticks(positions, CLASSES)
    axes[1].set_title("Coverage by transportation mode")
    axes[1].set_xlabel("Reference label")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)
    for ax in axes:
        ax.set_ylim(0, 116)
        ax.set_ylabel("Usable reference windows (%)")
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"120 s windows; at least 10 points; maximum gap 30 s\n"
                 f"Reference: {n_reference:,} windows; common main-grid support: {n_common:,}", fontsize=12)
    fig.tight_layout()
    save_figure(fig, input_dir / "retention_pilot.png")


def read_primary_grid(path: Path) -> pd.DataFrame:
    grid = pd.read_csv(path)
    selected = grid.loc[
        grid["scope"].eq("full_louo")
        & grid["cohort"].eq("common")
        & grid["noise_seed"].eq(0)
        & grid["downsample_sec"].isin(SAMPLE_INTERVALS)
        & grid["noise_sigma_m"].isin(NOISE_LEVELS)
    ].copy()
    if selected.empty:
        return selected
    if selected.duplicated(["downsample_sec", "noise_sigma_m"]).any():
        raise ValueError("Duplicate grid cells: select one full-LOUO run before plotting")
    if selected["run_sha256"].nunique(dropna=False) != 1:
        raise ValueError("Grid results mix different experiment runs")
    if selected["n_scored"].nunique(dropna=False) != 1 or selected["n_scored"].iloc[0] <= 0:
        raise ValueError("Common-support rows must score the same positive window count")
    for model in MODEL_COLOURS:
        scores = selected[f"{model}_macro_f1"]
        if not scores.between(0, 1).all():
            raise ValueError(f"Invalid or missing {model} Macro-F1 values")
    return selected.set_index(["downsample_sec", "noise_sigma_m"])


def plot_grid(grid: pd.DataFrame, input_dir: Path) -> None:
    full_index = pd.MultiIndex.from_product([SAMPLE_INTERVALS, NOISE_LEVELS],
                                           names=["downsample_sec", "noise_sigma_m"])
    aligned = grid.reindex(full_index)
    differences = (aligned["gru_macro_f1"] - aligned["rf_macro_f1"]).to_numpy().reshape(3, 4) * 100
    limit = max(float(np.nanmax(np.abs(differences))), 1.0)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(7.7, 4.1))
    im = ax.imshow(differences, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for row in range(3):
        for col in range(4):
            value = differences[row, col]
            ax.text(col, row, f"{value:+.2f}" if np.isfinite(value) else "Not evaluated",
                    ha="center", va="center", fontsize=10,
                    color="white" if np.isfinite(value) and abs(value) > 0.65 * limit else "black")
    ax.set_xticks(range(4), NOISE_LEVELS)
    ax.set_yticks(range(3), ["Native", "5 s", "10 s"])
    ax.set_xlabel("Added noise: per-axis Gaussian sigma (m)")
    ax.set_ylabel("Sampling interval")
    ax.set_title("GRU minus RF Macro-F1 (percentage points)\nFull LOUO; common support; noise seed 0")
    fig.colorbar(im, ax=ax, label="GRU - RF (percentage points)")
    fig.tight_layout()
    save_figure(fig, input_dir / "deployment_difference_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, step in zip(axes, SAMPLE_INTERVALS):
        for model, colour in MODEL_COLOURS.items():
            scores = aligned.loc[step, f"{model}_macro_f1"].to_numpy()
            ax.plot(NOISE_LEVELS, scores, marker="o", color=colour, label=model.upper())
        ax.set_title("Native sampling" if step == 0 else f"{step} s sampling")
        ax.set_xticks(NOISE_LEVELS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Per-axis noise sigma (m)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1")
    axes[-1].legend(frameon=False)
    n_scored = int(grid["n_scored"].iloc[0])
    fig.suptitle(f"Full LOUO; identical {n_scored:,} windows in each cell; noise seed 0", fontsize=11)
    fig.tight_layout()
    save_figure(fig, input_dir / "deployment_f1_curves.png")

    contrasts = []
    for step in [5, 10]:
        for sigma in [5, 10, 20]:
            for model in MODEL_COLOURS:
                column = f"{model}_macro_f1"
                f_dn, f_d0, f_0n, f_00 = [aligned.loc[key, column]
                                          for key in [(step, sigma), (step, 0), (0, sigma), (0, 0)]]
                contrast = f_dn - f_d0 - f_0n + f_00
                contrasts.append({
                    "condition": f"ds{step}_n{sigma}", "model": model.upper(),
                    "downsample_sec": step, "noise_sigma_m": sigma,
                    "noise_seed": 0, "scope": "full_louo", "cohort": "common",
                    "n_scored": n_scored, "run_sha256": grid["run_sha256"].iloc[0],
                    "f_joint": f_dn, "f_downsample_only": f_d0,
                    "f_noise_only": f_0n, "f_native": f_00,
                    "interaction_f1": contrast, "interaction_percentage_points": 100 * contrast,
                    "status": "evaluated" if np.isfinite(contrast) else "missing_inputs",
                })
    # Positive means less F1 loss than the additive-loss prediction; this is a
    # descriptive contrast, not proof of a causal mechanism or significance.
    target = input_dir / "interaction_contrasts.csv"
    pd.DataFrame(contrasts).to_csv(target, index=False)
    print(f"Saved {target.name}")
    if len(grid) < 12:
        print(f"Only {len(grid)}/12 primary cells are available; missing cells remain blank.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Directory containing the real aggregate CSV files; outputs are saved here")
    args = parser.parse_args()
    plot_retention(args.input_dir)
    path = args.input_dir / "grid_results.csv"
    if not path.exists():
        print("grid_results.csv is absent; skipped model figures and interaction contrasts.")
        return
    grid = read_primary_grid(path)
    if grid.empty:
        print("No full_louo/common/noise_seed=0 results; skipped model figures and interaction contrasts.")
        return
    plot_grid(grid, args.input_dir)


if __name__ == "__main__":
    main()
