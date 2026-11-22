from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GEOLIFE = ROOT / "GeoLife"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = GEOLIFE / "现实关联" / "outputs"
MEMBER_B = GEOLIFE / "member_b" / "outputs"

COLORS = {
    "RF": "#2A6F97",
    "GRU": "#C47A52",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
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


def create_coverage_f1_tradeoff():
    grid = pd.read_csv(RESULTS / "grid_primary_condition.csv")
    retention = pd.read_csv(RESULTS / "retention_pilot.csv")

    grid = grid.drop(columns=["retention"])
    tradeoff = grid.merge(
        retention[["condition", "retention"]],
        on="condition",
        how="left",
        validate="one_to_one",
    )

    fig, ax = plt.subplots(figsize=(10.2, 6.2))

    sampling_markers = {0: "o", 5: "s", 10: "D"}
    sampling_names = {0: "Native", 5: "5 s", 10: "10 s"}

    for downsample, marker in sampling_markers.items():
        subset = tradeoff[tradeoff["downsample_sec"] == downsample].copy()
        subset = subset.sort_values("retention", ascending=False)
        ax.plot(
            subset["retention"],
            subset["rf_macro_f1"],
            color=COLORS["RF"],
            linewidth=1.2,
            alpha=0.35,
            zorder=1,
        )
        ax.plot(
            subset["retention"],
            subset["gru_macro_f1"],
            color=COLORS["GRU"],
            linewidth=1.2,
            alpha=0.35,
            zorder=1,
        )

        ax.scatter(
            subset["retention"],
            subset["rf_macro_f1"],
            s=76,
            marker=marker,
            facecolor="white",
            edgecolor=COLORS["RF"],
            linewidth=1.8,
            label=f"RF - {sampling_names[downsample]}" if downsample == 0 else None,
            zorder=3,
        )
        ax.scatter(
            subset["retention"],
            subset["gru_macro_f1"],
            s=76,
            marker=marker,
            facecolor=COLORS["GRU"],
            edgecolor="white",
            linewidth=0.8,
            label=f"GRU - {sampling_names[downsample]}" if downsample == 0 else None,
            zorder=3,
        )

        for _, row in subset.iterrows():
            ax.annotate(
                f"{int(row['noise_sigma_m'])}m",
                (row["retention"], row["rf_macro_f1"]),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=7.5,
                color=COLORS["RF"],
                alpha=0.95,
            )
            ax.annotate(
                f"{int(row['noise_sigma_m'])}m",
                (row["retention"], row["gru_macro_f1"]),
                xytext=(5, -11),
                textcoords="offset points",
                fontsize=7.5,
                color=COLORS["GRU"],
                alpha=0.95,
            )

    ax.axvline(0.7982800710164225, color="#9DA7B1", linestyle="--", linewidth=1.0)
    ax.axvline(0.934254327563249, color="#9DA7B1", linestyle="--", linewidth=1.0)

    ax.text(
        0.802,
        0.03,
        "10 s: 79.83% retained",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="#66717C",
    )

    ds30 = retention[retention["downsample_sec"] == 30]
    if not ds30.empty:
        ax.text(
            0.008,
            0.655,
            "30 s: 0% retained\nnot evaluated",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            color="#7A4E4E",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#F7EEEE",
                "edgecolor": "#C7A1A1",
                "linewidth": 0.8,
            },
        )

    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.05, 0.72)
    ax.set_xlabel("Window retention rate")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Coverage–Macro-F1 trade-off")
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    rf_handle = plt.Line2D([], [], color=COLORS["RF"], marker="o", linestyle="-", markersize=7,
                           markerfacecolor="white", markeredgewidth=1.5)
    gru_handle = plt.Line2D([], [], color=COLORS["GRU"], marker="o", linestyle="-", markersize=7,
                           markerfacecolor=COLORS["GRU"], markeredgewidth=0.8)
    ax.legend([rf_handle, gru_handle], ["RF", "GRU"], frameon=False,
              loc="lower right", ncol=2)

    ax.text(
        0.0,
        -0.18,
        "Marker shape indicates sampling interval; labels indicate added noise in metres. 30 s has no usable windows and is shown only as an annotation.",
        transform=ax.transAxes,
        fontsize=9,
        color="#66717C",
    )
    save(fig, "coverage_f1_tradeoff.png")


def create_interaction_heatmap():
    df = pd.read_csv(RESULTS / "interaction_contrasts.csv")
    df = df[df["status"] == "evaluated"].copy()

    noise_levels = [5, 10, 20]
    downsample_levels = [5, 10]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), constrained_layout=False)

    for ax, model in zip(axes, ["RF", "GRU"]):
        subset = df[df["model"] == model]
        matrix = subset.pivot(
            index="noise_sigma_m",
            columns="downsample_sec",
            values="interaction_percentage_points",
        ).reindex(index=noise_levels, columns=downsample_levels)

        abs_max = float(max(abs(matrix.min().min()), abs(matrix.max().max())))
        vmax = max(abs_max, 1.0)

        im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")

        ax.set_xticks(range(len(downsample_levels)), [f"{value} s" for value in downsample_levels])
        ax.set_yticks(range(len(noise_levels)), [f"{value} m" for value in noise_levels])
        ax.set_xlabel("Downsampling")
        if model == "RF":
            ax.set_ylabel("Added noise")
        ax.set_title(model)

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                text_color = "white" if abs(value) > vmax * 0.55 else "#243443"
                ax.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=text_color,
                )

        ax.spines[:].set_visible(False)

    fig.suptitle("joint degradation interaction on bounded Macro-F1",
                 fontsize=17, fontweight="bold", x=0.5, y=1.02)

    cbar = fig.colorbar(axes[1].images[0], ax=axes, fraction=0.035,
                        pad=0.05, aspect=25)
    cbar.set_label("Interaction (percentage points)")

    fig.text(
        0.5,
        -0.11,
        "Descriptive model of joint vs. additive degradation on a bounded F1 metric; not a causal effect.",
        ha="center",
        fontsize=9.5,
        color="#66717C",
    )
    save(fig, "joint_degradation_interaction.png")


def create_keyword_result_strip():
    items = [
        ("90,120", "clean windows"),
        ("71,941", "common windows"),
        ("79.83%", "10 s retention"),
        ("+14.28 pp", "maximum GRU advantage"),
        ("0%", "30 s retention"),
        ("0.3572", "clean speed importance"),
    ]

    fig, ax = plt.subplots(figsize=(15.5, 2.8))
    ax.axis("off")

    ax.text(0.5, 1.04, "Key study results", ha="center", va="bottom",
            fontsize=17, fontweight="bold", color="#243443", transform=ax.transAxes)

    start = 0.02
    width = 0.156
    gap = 0.006

    for index, (value, label) in enumerate(items):
        left = start + index * (width + gap)
        face = "#F3F6F7" if index % 2 == 0 else "#F7F3EF"
        ax.add_patch(plt.Rectangle(
            (left, 0.16),
            width,
            0.66,
            transform=ax.transAxes,
            facecolor=face,
            edgecolor="#CDD5DB",
            linewidth=0.9,
        ))
        ax.text(
            left + width / 2,
            0.58,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#243443",
        )
        ax.text(
            left + width / 2,
            0.34,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#66717C",
        )

    save(fig, "keyword_result_strip.png")


def main():
    create_coverage_f1_tradeoff()
    create_interaction_heatmap()
    create_keyword_result_strip()


if __name__ == "__main__":
    main()
