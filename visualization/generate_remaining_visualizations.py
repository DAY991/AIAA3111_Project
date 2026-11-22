from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GEOLIFE = ROOT / "GeoLife"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "RF": "#2A6F97",
    "GRU": "#C47A52",
    "walk": "#D97A4A",
    "bike": "#6F8F72",
    "bus": "#D2A84A",
    "car": "#5A7D9A",
    "subway": "#9A708F",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#9DA7B1",
    "axes.linewidth": 0.8,
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"created: {OUT / name}")


def add_panel_label(ax, text):
    ax.text(
        -0.11,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def create_method_flowchart():
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    steps = [
        (0.45, 2.55, 1.25, "GeoLife", "GPS trajectories"),
        (2.10, 2.55, 1.35, "Label alignment", "5 modes"),
        (3.85, 2.55, 1.50, "Same-label runs", "split points"),
        (5.75, 2.55, 1.38, "120 s windows", ">=10 points"),
        (7.53, 3.25, 1.65, "RF pipeline", "statistical features"),
        (7.53, 1.85, 1.65, "GRU pipeline", "sequences"),
        (9.58, 2.55, 1.45, "LOUO", "48 held-out folds"),
        (11.43, 2.55, 1.35, "Sampling x noise", "12-grid evaluation"),
    ]

    for x, y, width, title, subtitle in steps:
        if title in ("RF pipeline", "GRU pipeline"):
            face = "#EEF4F7" if title == "RF pipeline" else "#F7EFEA"
        else:
            face = "#F3F5F6"
        ax.add_patch(plt.Rectangle((x, y), width, 1.15, facecolor=face,
                                   edgecolor="#697786", linewidth=1.2,
                                   transform=ax.transData))
        ax.text(x + width / 2, y + 0.72, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#243443")
        ax.text(x + width / 2, y + 0.34, subtitle, ha="center", va="center",
                fontsize=8.5, color="#66717C")

    arrow_kw = dict(arrowstyle="-|>", color="#7D8994", linewidth=1.4,
                    mutation_scale=13)
    arrows = [
        ((1.70, 3.12), (2.10, 3.12)),
        ((3.45, 3.12), (3.85, 3.12)),
        ((5.35, 3.12), (5.75, 3.12)),
        ((7.13, 3.12), (7.53, 3.82)),
        ((7.13, 3.12), (7.53, 2.42)),
        ((9.18, 3.82), (9.58, 3.12)),
        ((9.18, 2.42), (9.58, 3.12)),
        ((11.03, 3.12), (11.43, 3.12)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_kw)

    ax.text(6.55, 4.92, "Processing and Evaluation Pipeline", ha="center",
            fontsize=20, fontweight="bold", color="#243443")
    ax.text(6.55, 0.58,
            "Label-aligned windows feed two fixed feature representations before leave-one-user-out evaluation.",
            ha="center", fontsize=10.5, color="#66717C")
    save(fig, "project_method_flowchart.png")


def create_class_distribution():
    csv = GEOLIFE / "现实关联" / "outputs" / "per_class_robustness.csv"
    df = pd.read_csv(csv)
    clean = df[df["condition"] == "ds0_n0"].drop_duplicates("label")
    counts = clean.set_index("label")["reference_class_count"]
    order = ["walk", "bike", "bus", "car", "subway"]
    counts = counts.reindex(order)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.barh(order[::-1], counts.reindex(order[::-1]),
                   color=[COLORS[x] for x in order[::-1]], height=0.62)
    ax.set_title("Clean reference window distribution")
    ax.set_xlabel("Number of windows")
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for bar, value in zip(bars, counts.reindex(order[::-1])):
        ax.text(value + 900, bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#374151")

    ax.set_xlim(0, 33500)
    ax.text(0.0, -0.22,
            "Five-class Macro-F1 gives subway the same per-class weight despite being the smallest class.",
            transform=ax.transAxes, fontsize=9.5, color="#66717C")
    save(fig, "class_distribution.png")


def create_speed_importance():
    csv = GEOLIFE / "member_b" / "outputs" / "group_permutation_summary.csv"
    df = pd.read_csv(csv)
    conditions = ["ds0_n0", "ds10_n0", "ds0_n20", "ds10_n20"]
    labels = ["Clean", "10 s + 0 m", "Native + 20 m", "10 s + 20 m"]
    speed = df[(df["feature_group"] == "speed") & df["condition"].isin(conditions)]
    speed = speed.set_index("condition")
    values = speed.loc[conditions, "mean_importance"]
    errors = speed.loc[conditions, "sd_importance"]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars = ax.bar(labels, values, yerr=errors, capsize=5,
                  color=["#4C7A65", "#6E8F99", "#C17857", "#B99A54"],
                  edgecolor="none", width=0.64,
                  error_kw={"elinewidth": 1.1, "ecolor": "#66717C"})
    ax.axhline(0, color="#697786", linewidth=0.8)
    ax.set_ylim(-0.02, 0.42)
    ax.set_ylabel("Macro-F1 importance")
    ax.set_title("RF dependence on speed features")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                f"{value:.4f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#374151")

    ax.text(0.0, -0.22,
            "Fixed-RF grouped permutation importance; error bars show repeat variation only.",
            transform=ax.transAxes, fontsize=9.5, color="#66717C")
    save(fig, "rf_speed_group_importance.png")


def create_feature_drift():
    csv = GEOLIFE / "member_b" / "outputs" / "feature_drift_by_feature.csv"
    df = pd.read_csv(csv)
    drift = df[(df["feature_group"] == "speed") &
               (df["feature"] == "mean_speed") &
               df["condition"].isin(["ds0_n20", "ds10_n20"])]
    drift = drift.set_index("condition")

    labels = ["Native + 20 m", "10 s + 20 m"]
    values = drift.loc[["ds0_n20", "ds10_n20"], "median_absolute_paired_difference"]

    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    bars = ax.bar(labels, values, width=0.58,
                  color=["#C17857", "#6E8F99"], edgecolor="none")
    ax.set_ylim(0, 14)
    ax.set_ylabel("Median absolute change (m/s)")
    ax.set_title("20 m noise changes mean-speed estimates")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45,
                f"{value:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#374151")

    ax.text(0.0, -0.22,
            "Paired median absolute change in mean_speed; thinning reduces the dense-sampling noise response.",
            transform=ax.transAxes, fontsize=9.5, color="#66717C")
    save(fig, "mean_speed_feature_drift.png")


def create_per_class_clean():
    csv = GEOLIFE / "现实关联" / "outputs" / "per_class_robustness.csv"
    df = pd.read_csv(csv)
    clean = df[df["condition"] == "ds0_n0"].copy()
    clean["label"] = pd.Categorical(clean["label"],
                                    ["walk", "bike", "bus", "car", "subway"],
                                    ordered=True)
    clean = clean.sort_values("label")

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    y = range(len(clean))
    height = 0.34
    ax.barh([v - height / 2 for v in y], clean["rf_f1"], height=height,
            color=COLORS["RF"], label="RF")
    ax.barh([v + height / 2 for v in y], clean["gru_f1"], height=height,
            color=COLORS["GRU"], label="GRU")

    ax.set_yticks(list(y), clean["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class F1")
    ax.set_title("Clean-condition per-class performance")
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    for i, (rf, gru) in enumerate(zip(clean["rf_f1"], clean["gru_f1"])):
        ax.text(rf + 0.012, i - height / 2, f"{rf:.3f}", va="center",
                fontsize=8.5, color="#374151")
        ax.text(gru + 0.012, i + height / 2, f"{gru:.3f}", va="center",
                fontsize=8.5, color="#374151")

    ax.text(0.0, -0.18,
            "GRU leads car by 17.49 percentage points; RF leads subway by 7.73 percentage points.",
            transform=ax.transAxes, fontsize=9.5, color="#66717C")
    save(fig, "clean_per_class_performance.png")


def create_noise_uncertainty():
    csv = GEOLIFE / "现实关联" / "outputs" / "grid_key_seed_summary.csv"
    df = pd.read_csv(csv)
    summary = df[(df["condition"] == "ds10_n10") &
                 (df["cohort"] == "common") &
                 (df["scope"] == "full_louo")]
    summary = summary.set_index("metric")

    means = [summary.loc["rf_macro_f1", "mean"],
             summary.loc["gru_macro_f1", "mean"]]
    stds = [summary.loc["rf_macro_f1", "std_ddof1"],
             summary.loc["gru_macro_f1", "std_ddof1"]]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.bar(["RF", "GRU"], means, yerr=stds, capsize=7,
                  color=[COLORS["RF"], COLORS["GRU"]], width=0.58,
                  error_kw={"elinewidth": 1.4, "ecolor": "#66717C"})
    ax.set_ylim(0.38, 0.46)
    ax.set_ylabel("Common-support Macro-F1")
    ax.set_title("10 s + 10 m across three noise seeds")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.002,
                f"{mean:.4f} +/- {std:.4f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#374151")

    ax.text(0.0, -0.22,
            "Three noise realizations with fixed models and users; not a training or cross-user confidence interval.",
            transform=ax.transAxes, fontsize=9.3, color="#66717C")
    save(fig, "noise_seed_uncertainty.png")


def main():
    create_method_flowchart()
    create_class_distribution()
    create_speed_importance()
    create_feature_drift()
    create_per_class_clean()
    create_noise_uncertainty()


if __name__ == "__main__":
    main()
