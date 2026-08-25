"""
Regenerate every figure in figures/.

Sources are kept separate and always labelled:
  - thesis results  -> results/*.csv exported from the MATLAB pipeline
  - replication     -> results/replication_*.csv produced by the modules in src/

Run the src/ modules first (they write the replication CSVs), then:
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

DPI = 150

SCENARIO_ORDER = [
    "Net Zero 2050",
    "Delayed transition",
    "Below 2°C",
    "Nationally Determined Contributions (NDCs)",
    "Fragmented World",
]
SCENARIO_SHORT = {
    "Net Zero 2050": "Net Zero 2050",
    "Delayed transition": "Delayed transition",
    "Below 2°C": "Below 2°C",
    "Nationally Determined Contributions (NDCs)": "NDCs",
    "Fragmented World": "Fragmented World",
}
SCENARIO_COLOURS = {
    "Net Zero 2050": "#1b7837",
    "Delayed transition": "#7fbc41",
    "Below 2°C": "#2166ac",
    "Nationally Determined Contributions (NDCs)": "#d6604d",
    "Fragmented World": "#8c510a",
}


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the src/ modules first.")
    return path


def _save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- model comparison

def fig_model_comparison():
    """
    Out-of-sample R2 of the four thesis models, plus the Python replication of
    the winning one. The y axis starts just below the lowest bar on purpose:
    the point of the figure is how close the four models are.
    """
    thesis = pd.read_csv(_need(RESULTS / "oos_model_comparison.csv"))
    replication = pd.read_csv(_need(RESULTS / "replication_vs_thesis_metrics.csv"))
    replicated_r2 = float(
        replication.loc[
            replication["Metric"] == "R2 out-of-sample", "Replication (Python)"
        ].iloc[0]
    )

    display = {
        "ElasticNet": "Elastic Net",
        "RandomForest": "Random Forest",
        "Panel": "Panel (linear)",
        "GradientBoosting": "Gradient Boosting",
    }
    order = ["ElasticNet", "RandomForest", "Panel", "GradientBoosting"]
    thesis = thesis.set_index("Models").loc[order]

    labels = [display[m] for m in order] + ["Gradient Boosting\n(Python replication)"]
    values = list(thesis["R2"]) + [replicated_r2]
    colours = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#2166ac", "#7fbc41"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.62)

    low, high = min(values), max(values)
    pad = (high - low) * 0.6 if high > low else 0.01
    ax.set_ylim(low - pad, high + pad)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + pad * 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_title(
        "Out-of-sample $R^2$ on the sealed 2021-2024 test block\n"
        "grey and blue: thesis (MATLAB), green: Python replication",
        fontsize=11,
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, "fig_model_comparison.png")


# ------------------------------------------------------------------- trajectories

def _logratio_long():
    """Thesis log-ratio, wide monthly columns reshaped to long."""
    wide = pd.read_csv(_need(RESULTS / "logratio_green_brown.csv"))
    long = wide.melt(
        id_vars=["Scenario", "Component", "Sector"],
        var_name="Month",
        value_name="LogRatio",
    )
    long["Date"] = pd.to_datetime(
        long["Month"].str.slice(1).str.replace("_", "-"), format="%Y-%m"
    )
    return long


def fig_trajectories(sector, sector_name, filename):
    """
    Two panels, physical | transition, one line per scenario.

    The physical panel is flat and stays in: that flatness is the finding.
    The combined component is deliberately not plotted.
    """
    long = _logratio_long()
    subset = long[long["Sector"] == sector]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, component in zip(axes, ["physical", "transition"]):
        block = subset[subset["Component"] == component]
        for scenario in SCENARIO_ORDER:
            line = block[block["Scenario"] == scenario].sort_values("Date")
            if line.empty:
                continue
            ax.plot(
                line["Date"],
                line["LogRatio"],
                label=SCENARIO_SHORT[scenario],
                color=SCENARIO_COLOURS[scenario],
                linewidth=1.8,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{component} component", fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative log-ratio Green / Brown")
    axes[1].legend(fontsize=8.5, loc="best", frameon=True)
    fig.suptitle(
        f"{sector_name}: within-sector Green/Brown log-ratio, 2025-2050 (thesis results)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, filename)


# --------------------------------------------------------------------- PDP / ICE

def fig_pdp():
    """Two rows by three columns, one panel per defence pair. Replication only."""
    pdp = pd.read_csv(_need(RESULTS / "replication_pdp_curves.csv"))
    pairs = list(dict.fromkeys(pdp["Pair"]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pair in zip(axes.ravel(), pairs):
        block = pdp[pdp["Pair"] == pair].sort_values("GridValue")
        ax.plot(block["GridValue"], block["PDP"], color="#2166ac", linewidth=2)
        ax.set_title(pair, fontsize=10)
        ax.set_xlabel("feature value")
        ax.set_ylabel("average prediction")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Partial dependence, recomputed on the Python replication of the model",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig_pdp_2x3.png")


def fig_ice():
    """Same six pairs: thin grey individual curves with the average on top."""
    ice = pd.read_csv(_need(RESULTS / "replication_ice_curves.csv"))
    pdp = pd.read_csv(_need(RESULTS / "replication_pdp_curves.csv"))
    pairs = list(dict.fromkeys(pdp["Pair"]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pair in zip(axes.ravel(), pairs):
        block = ice[ice["Pair"] == pair]
        grid = np.sort(block["GridValue"].unique())
        matrix = block.pivot_table(
            index="CurveId", columns="GridValue", values="Prediction"
        )
        matrix = matrix[grid]

        for _, curve in matrix.iterrows():
            ax.plot(grid, curve.to_numpy(), color="#999999", linewidth=0.4, alpha=0.35)

        mean_curve = pdp[pdp["Pair"] == pair].sort_values("GridValue")
        ax.plot(mean_curve["GridValue"], mean_curve["PDP"], color="black", linewidth=2.2)

        ax.set_title(pair, fontsize=10)
        ax.set_xlabel("feature value")
        ax.set_ylabel("prediction")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Individual conditional expectation curves (grey) and their average (black),\n"
        "recomputed on the Python replication of the model",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig_ice_2x3.png")


# ------------------------------------------------------------------- importances

def fig_feature_importance(top_n=15):
    """Top features of the thesis model, next to the same features in the replication."""
    thesis = pd.read_csv(_need(RESULTS / "gb_feature_importance.csv"))
    replication = pd.read_csv(_need(RESULTS / "replication_gb_feature_importance.csv"))

    top = thesis.head(top_n).copy()
    lookup = replication.set_index("Feature")["SharePct"]
    top["ReplicationShare"] = top["Feature"].map(lookup).fillna(0.0)

    positions = np.arange(len(top))[::-1]
    height = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.barh(
        positions + height / 2,
        top["SharePct"],
        height=height,
        color="#2166ac",
        label="Thesis (MATLAB)",
    )
    ax.barh(
        positions - height / 2,
        top["ReplicationShare"],
        height=height,
        color="#7fbc41",
        label="Python replication",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(top["Feature"], fontsize=9)
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title(
        f"Top {top_n} features of the thesis Gradient Boosting model,\n"
        "with the share the same features take in the replication",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_feature_importance.png")


def main():
    fig_model_comparison()
    fig_trajectories("ENRG", "Energy", "fig_energy_trajectories.png")
    fig_trajectories("MATS", "Materials", "fig_materials_trajectories.png")
    fig_pdp()
    fig_ice()
    fig_feature_importance()


if __name__ == "__main__":
    main()
