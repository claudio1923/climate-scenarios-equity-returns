"""
Regenerate every figure in figures/.

Sources are kept separate and always labelled:
  - thesis results  -> results/*.csv exported from the MATLAB pipeline
  - replication     -> results/replication_*.csv, produced by the modules in src/
                       from the two fits (189 months for metrics, 237 for the
                       scenario projection)

One deliberate omission. The trajectory panels are drawn as single lines, with no
uncertainty band around them, even though the projection is known to be poorly
conditioned. The dispersion that was measured is the dispersion of the 2050
endpoint, not of the monthly paths, so a band would be an interpolation of a
quantity nobody computed. fig_conditioning.png carries that information instead,
where it is an actual measurement.

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
    Out-of-sample R2 of the four models. The y axis starts just below the lowest
    bar on purpose: the point of the figure is how close the four models are.

    The Gradient Boosting bar is the value this repository produces on FIT A;
    the other three are the thesis figures from oos_model_comparison.csv, which
    were not re-estimated here.
    """
    comparison = pd.read_csv(_need(RESULTS / "oos_model_comparison.csv"))
    metrics = pd.read_csv(_need(RESULTS / "replication_vs_thesis_metrics.csv"))
    gb_r2 = float(
        metrics.loc[metrics["Metric"] == "R2 out-of-sample", "Replication (Python)"].iloc[0]
    )

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
        "Single lines, no uncertainty band: these are point projections. "
        "The 2050 endpoints are marked, and their stability is measured separately "
        "in fig_conditioning.png.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    _save(fig, filename)


# -------------------------------------------------------------------- conditioning

def fig_conditioning():
    """
    What the 2050 endpoint does when the estimation matrix is nudged by one unit
    in the last representable bit: 150 refits, one swarm of points per scenario.

    This is the figure that carries the uncertainty. It shows directly that Net
    Zero and Delayed transition stay above zero almost always, while the other
    three straddle it.
    """
    from scenarios import SCENARIO_CODES

    table = pd.read_csv(_need(RESULTS / "conditioning" / "perturbation_dense.csv"))
    finest = table["level"].min()
    block = table[table["level"] == finest]

    comparison = pd.read_csv(_need(RESULTS / "replication_logratio_2050_comparison.csv"))
    unperturbed = comparison[
        (comparison["Sector"] == "ENRG") & (comparison["Component"] == "transition")
    ].set_index("Scenario")["LogRatioReplication"]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 5.6))

    print(f"\n[fig_conditioning] {len(block)} draws at relative noise {finest:.0e}")
    for position, scenario in enumerate(SCENARIO_ORDER):
        values = block[f"s_{SCENARIO_CODES[scenario]}"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=values.size)
        ax.scatter(
            position + jitter,
            values,
            s=11,
            alpha=0.35,
            color=SCENARIO_COLOURS[scenario],
            edgecolors="none",
        )
        reference = float(unperturbed[scenario])
        ax.plot([position - 0.3, position + 0.3], [reference, reference],
                color="black", linewidth=2, zorder=5)
        share = 100 * (values > 0).mean()
        ax.text(position, ax.get_ylim()[1], "", ha="center")
        print(f"    {SCENARIO_LABEL[scenario]:<22} unperturbed {reference:+.4f}   "
              f"positive in {share:.1f}% of draws   "
              f"range [{values.min():+.3f}, {values.max():+.3f}]")

    ax.axhline(0, color="#b2182b", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=9)
    ax.set_ylabel("Energy Green/Brown log-ratio, December 2050")
    ax.set_title(
        "Each point is one refit after nudging the estimation matrix\n"
        f"by one unit in the last representable bit ({len(block)} draws); "
        "black bars are the unperturbed values",
        fontsize=11,
    )

    for position, scenario in enumerate(SCENARIO_ORDER):
        values = block[f"s_{SCENARIO_CODES[scenario]}"].to_numpy()
        ax.annotate(
            f"{100 * (values > 0).mean():.0f}% > 0",
            xy=(position, ax.get_ylim()[0]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#333333",
        )

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_conditioning.png")


# --------------------------------------------------------------------- PDP / ICE

def fig_pdp():
    """Two rows by three columns, one panel per defence pair."""
    pdp = pd.read_csv(_need(RESULTS / "replication_pdp_curves.csv"))
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

    fig.suptitle("Partial dependence, recomputed on the replicated model", fontsize=12)
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
        "Individual conditional expectation curves (grey) and their average (black),\n"
        "recomputed on the replicated model",
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
    importance = pd.read_csv(_need(RESULTS / "replication_gb_feature_importance.csv"))

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
    fig_conditioning()
    fig_pdp()
    fig_ice()
    fig_feature_importance()


if __name__ == "__main__":
    main()
