"""
Regenerate every figure in figures/.

Inputs come from results/: the aggregated scenario results, and the model
outputs written by the modules in src/.

The trajectory panels are drawn as single lines, with no band around them: they
are point projections and nothing in this repository estimates a distribution
around them.

Run the src/ modules first (they write the CSVs used here), then:
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
SCENARIO_LABEL = {
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

# Readable names for the drivers, used in the importance labels.
DRIVER_NAMES = {
    "UNRATE_DIFF": "UNRATE",
    "CPI_DIFF": "CPI",
    "R_WTI": "WTI",
    "R_NatGas": "NatGas",
    "Tas_ANOM": "Temp",
    "Precip_ANOM": "Precip",
    "Wind_ANOM": "Wind",
    "ExMkt": "ExMkt",
}
LAG_NAMES = {"L0": "(t)", "L1": "(t-1)", "L2": "(t-2)"}


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


def _entity_names():
    """Entity code -> readable portfolio name, e.g. 6 -> ENRG-Brown."""
    from build_features import load_entity_labels

    readable = {}
    for code, label in load_entity_labels().items():
        sector, leg = label.rsplit("_", 1)
        readable[code] = f"{sector}-{'Green' if leg == 'G' else 'Brown'}"
    return readable


def pretty_feature(name, entity_names):
    """
    Turn a raw feature name into a readable label.

        R_WTI_L1_x_Entity_6 -> WTI(t-1) x ENRG-Brown
        ExMkt_L1            -> ExMkt(t-1)
    """
    base, _, entity = name.partition("_x_Entity_")
    driver, _, lag = base.rpartition("_")
    label = DRIVER_NAMES.get(driver, driver) + LAG_NAMES.get(lag, "")
    if entity:
        label += f" x {entity_names[int(entity)]}"
    return label


# ---------------------------------------------------------------- model comparison

def fig_model_comparison():
    """
    Out-of-sample R2 of the four candidate models. The y axis starts just below
    the lowest bar on purpose: the point of the figure is how close they are.
    """
    comparison = pd.read_csv(_need(RESULTS / "oos_model_comparison.csv"))
    metrics = pd.read_csv(_need(RESULTS / "model_metrics.csv"))
    gb_r2 = float(metrics.loc[metrics["Sample"].str.startswith("out-of-sample"), "R2"].iloc[0])

    display = {
        "ElasticNet": "Elastic Net",
        "RandomForest": "Random Forest",
        "Panel": "Panel (linear)",
    }
    order = ["ElasticNet", "RandomForest", "Panel"]
    comparison = comparison.set_index("Models").loc[order]

    labels = [display[m] for m in order] + ["Gradient Boosting"]
    values = list(comparison["R2"]) + [gb_r2]
    colours = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#2166ac"]

    print("\n[fig_model_comparison] values plotted:")
    for label, value in zip(labels, values):
        print(f"    {label:<20} {value:.4f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.62)

    low, high = min(values), max(values)
    pad = (high - low) * 0.6
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
        "Gradient Boosting refit here; the other three as reported in the thesis",
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

    end = subset["Date"].max()
    print(f"\n[{filename}] log-ratio at {end:%Y-%m}, one line per scenario:")
    for component in ["physical", "transition"]:
        block = subset[(subset["Component"] == component) & (subset["Date"] == end)]
        block = block.set_index("Scenario")
        print(f"  {component}:")
        for scenario in SCENARIO_ORDER:
            if scenario in block.index:
                print(f"    {SCENARIO_LABEL[scenario]:<22} {block.loc[scenario, 'LogRatio']:+.4f}")
        spread = block["LogRatio"].max() - block["LogRatio"].min()
        print(f"    {'spread across scenarios':<22} {spread:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    for ax, component in zip(axes, ["physical", "transition"]):
        block = subset[subset["Component"] == component]
        for scenario in SCENARIO_ORDER:
            line = block[block["Scenario"] == scenario].sort_values("Date")
            if line.empty:
                continue
            ax.plot(
                line["Date"],
                line["LogRatio"],
                label=SCENARIO_LABEL[scenario],
                color=SCENARIO_COLOURS[scenario],
                linewidth=1.8,
            )
            ax.plot(
                line["Date"].iloc[-1],
                line["LogRatio"].iloc[-1],
                marker="o",
                markersize=4.5,
                color=SCENARIO_COLOURS[scenario],
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
    fig.text(
        0.5,
        -0.02,
        "Single lines, no uncertainty band: these are point projections, "
        "with the 2050 endpoints marked.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, filename)


# --------------------------------------------------------------------- PDP / ICE

def fig_pdp():
    """Two rows by three columns, one panel per defence pair."""
    pdp = pd.read_csv(_need(RESULTS / "model_pdp_curves.csv"))
    pairs = list(dict.fromkeys(pdp["Pair"]))

    print("\n[fig_pdp_2x3] vertical range of each average curve (max - min):")
    for pair in pairs:
        block = pdp[pdp["Pair"] == pair]
        print(f"    {pair:<26} {block['PDP'].max() - block['PDP'].min():.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pair in zip(axes.ravel(), pairs):
        block = pdp[pdp["Pair"] == pair].sort_values("GridValue")
        ax.plot(block["GridValue"], block["PDP"], color="#2166ac", linewidth=2)
        ax.set_title(pair, fontsize=10)
        ax.set_xlabel("feature value")
        ax.set_ylabel("average prediction")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Partial dependence over the six defence pairs", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_pdp_2x3.png")


def fig_ice():
    """Same six pairs: thin grey individual curves with the average on top."""
    ice = pd.read_csv(_need(RESULTS / "model_ice_curves.csv"))
    pdp = pd.read_csv(_need(RESULTS / "model_pdp_curves.csv"))
    pairs = list(dict.fromkeys(pdp["Pair"]))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pair in zip(axes.ravel(), pairs):
        block = ice[ice["Pair"] == pair]
        grid = np.sort(block["GridValue"].unique())
        matrix = block.pivot_table(index="CurveId", columns="GridValue", values="Prediction")
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
        "Individual conditional expectation curves (grey) and their average (black)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig_ice_2x3.png")


# ------------------------------------------------------------------- importances

def fig_feature_importance(top_n=15):
    """
    Top features by importance, market factor excluded.

    The contemporaneous market factor takes about two thirds of the budget on its
    own; leaving it in flattens every other bar to nothing and shows what is
    already known. It is dropped and the remaining shares are renormalised, so
    the figure answers "what matters once the market is taken out".
    """
    importance = pd.read_csv(_need(RESULTS / "model_feature_importance.csv"))

    market_share = float(importance.loc[importance["Feature"] == "ExMkt_L0", "SharePct"].iloc[0])
    without_market = importance[importance["Feature"] != "ExMkt_L0"].copy()
    without_market["SharePct"] = (
        100 * without_market["Importance"] / without_market["Importance"].sum()
    )

    entity_names = _entity_names()
    top = without_market.head(top_n).copy()
    top["Label"] = top["Feature"].map(lambda n: pretty_feature(n, entity_names))

    print(f"\n[fig_feature_importance] market factor excluded ({market_share:.1f}% of the "
          f"full budget); shares below are renormalised over the rest:")
    for label, share in zip(top["Label"], top["SharePct"]):
        print(f"    {label:<28} {share:5.2f}%")

    positions = np.arange(len(top))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(positions, top["SharePct"], height=0.62, color="#2166ac")

    ax.set_yticks(positions)
    ax.set_yticklabels(top["Label"], fontsize=9.5)
    ax.set_xlabel("Share of importance, market factor excluded (%)")
    ax.set_title(f"Top {top_n} features once the market factor is taken out", fontsize=11)
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
