"""
Scenario projection 2025-2050 with the replicated model.

scenario_design_K62.csv already contains the 73 features built in MATLAB for
every Scenario x Component x Entity x month, so this step only applies the
Python model to that design and aggregates the predictions the same way the
thesis does:

    cumulative index   = prod(1 + r_t / 100)
    log-ratio (sector) = log(cum of the Green portfolio) - log(cum of the Brown one)

One difference from the thesis, stated explicitly because it matters: the design
file carries no scenario risk-free path, so the compounding here uses the excess
return (yhat) whereas the thesis compounds yhat + RF. The risk-free component is
common to the Green and the Brown leg of a sector, so it nearly cancels in the
log-ratio; levels can shift slightly, the shape of the trajectories and the
ranking across scenarios do not.

"Current Policies" appears in the design file with the physical component only
and is not part of the five-scenario set reported in the thesis, so it is
projected but left out of the comparison table.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import DATA, RESULTS, _require
from train_gb import train

KEY_COLS = ["Entity", "EntityLabel", "Green", "Scenario", "Component", "Date", "SSP"]

THESIS_SCENARIOS = [
    "Net Zero 2050",
    "Delayed transition",
    "Below 2°C",
    "Nationally Determined Contributions (NDCs)",
    "Fragmented World",
]


def load_design():
    """Scenario design: keys plus the 73 winning features, already built."""
    design = pd.read_csv(_require(DATA / "scenario_design_K62.csv"))
    design["Date"] = pd.to_datetime(design["Date"], format="%d-%b-%Y")
    design["Sector"] = design["EntityLabel"].str.rsplit("_", n=1).str[0]
    return design


def predict_scenarios(model, feature_names, design=None):
    """Monthly predicted excess return for every scenario cell."""
    design = load_design() if design is None else design
    design = design.copy()
    design["yhat"] = model.predict(design[list(feature_names)])
    return design


def log_ratio(predictions):
    """
    Compound each portfolio, then take the within-sector Green/Brown log-ratio.
    Returns a long frame: Scenario, Component, Sector, Date, LogRatio.
    """
    frame = predictions.sort_values(["Scenario", "Component", "EntityLabel", "Date"]).copy()
    frame["Cum"] = frame.groupby(["Scenario", "Component", "EntityLabel"])["yhat"].transform(
        lambda s: (1 + s / 100).cumprod()
    )

    keys = ["Scenario", "Component", "Sector", "Date"]
    green = frame[frame["Green"] == 1].set_index(keys)["Cum"]
    brown = frame[frame["Green"] == 0].set_index(keys)["Cum"]

    ratio = (np.log(green) - np.log(brown)).rename("LogRatio").reset_index()
    return ratio.sort_values(keys).reset_index(drop=True)


def thesis_log_ratio():
    """The thesis log-ratio, reshaped long, for the qualitative comparison."""
    wide = pd.read_csv(_require(RESULTS / "logratio_green_brown.csv"))
    long = wide.melt(
        id_vars=["Scenario", "Component", "Sector"],
        var_name="Month",
        value_name="LogRatioThesis",
    )
    # Column names look like d2050_12.
    long["Date"] = pd.to_datetime(long["Month"].str.slice(1).str.replace("_", "-"), format="%Y-%m")
    long["Date"] = long["Date"] + pd.offsets.MonthEnd(0)
    return long.drop(columns="Month")


def comparison_2050(replication):
    """End-of-horizon comparison, replication against thesis, per sector."""
    thesis = thesis_log_ratio()
    end = thesis["Date"].max()

    left = replication[replication["Date"] == end]
    right = thesis[thesis["Date"] == end]
    merged = left.merge(right, on=["Scenario", "Component", "Sector", "Date"], how="inner")
    merged = merged[merged["Scenario"].isin(THESIS_SCENARIOS)]
    return merged.rename(columns={"LogRatio": "LogRatioReplication"})


def energy_sign_check(merged):
    """
    Does the Energy sign reversal survive the replication?

    In the thesis the transition component of Energy is positive under Net Zero
    and Delayed transition and negative under the other three scenarios.
    """
    energy = merged[(merged["Sector"] == "ENRG") & (merged["Component"] == "transition")]
    energy = energy.set_index("Scenario")
    rows = []
    for scenario in THESIS_SCENARIOS:
        if scenario not in energy.index:
            continue
        thesis_value = energy.loc[scenario, "LogRatioThesis"]
        replication_value = energy.loc[scenario, "LogRatioReplication"]
        rows.append(
            {
                "Scenario": scenario,
                "Thesis": round(float(thesis_value), 4),
                "Replication": round(float(replication_value), 4),
                "SameSign": bool(np.sign(thesis_value) == np.sign(replication_value)),
            }
        )
    return pd.DataFrame(rows)


def main():
    data = train()
    predictions = predict_scenarios(data["model"], data["x_fit"].columns)
    replication = log_ratio(predictions)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "replication_logratio_green_brown.csv"
    replication.to_csv(out, index=False)

    merged = comparison_2050(replication)
    comparison_out = RESULTS / "replication_logratio_2050_comparison.csv"
    merged.to_csv(comparison_out, index=False)

    root = Path(__file__).resolve().parents[1]
    print(f"Wrote {out.relative_to(root)} ({len(replication)} rows)")
    print(f"Wrote {comparison_out.relative_to(root)} ({len(merged)} rows)")

    check = energy_sign_check(merged)
    print("\nEnergy, transition component, December 2050:")
    print(check.to_string(index=False))
    if check["SameSign"].all():
        print("\nSign reversal reproduced: same sign as the thesis in all five scenarios.")
    else:
        failed = check.loc[~check["SameSign"], "Scenario"].tolist()
        print(f"\nSign reversal NOT fully reproduced. Scenarios that differ in sign: {failed}")


if __name__ == "__main__":
    main()
