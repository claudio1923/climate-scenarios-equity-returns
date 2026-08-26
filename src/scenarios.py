"""
Scenario projection 2025-2050 with the replicated model.

scenario_design_K62.csv already contains the 73 features built in MATLAB for
every Scenario x Component x Entity x month, so this step only applies the
Python model to that design and aggregates the predictions the same way the
thesis does:

    cumulative index   = prod(1 + r_t / 100)
    log-ratio (sector) = log(cum of the Green portfolio) - log(cum of the Brown one)

Compounding uses the total return, exactly as s3_K62_leaf10.m does:

    total = yhat + RF

scenario_design_K62.csv carries no risk-free column, so the path is joined from
results/scenario_monthly_predictions.csv, where RF is constant across entities
within a Scenario x Component x month. The risk-free leg is not optional and not
a refinement: leaving it out changes the log-ratio.

"Current Policies" appears in the design file with the physical component only.
It carries no risk-free path and is not part of the five-scenario set reported in
the thesis, so it is dropped before compounding, as s3_K62_leaf10.m does.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import DATA, RESULTS, _require
from train_gb import VARIANTS, train_for_projection

KEY_COLS = ["Entity", "EntityLabel", "Green", "Scenario", "Component", "Date", "SSP"]

# Scenarios projected but not reported: they have no risk-free path.
REFERENCE_SCENARIOS = ("current policies", "baseline")


THESIS_SCENARIOS = [
    "Net Zero 2050",
    "Delayed transition",
    "Below 2°C",
    "Nationally Determined Contributions (NDCs)",
    "Fragmented World",
]


def load_design():
    """Scenario design: keys plus the 73 winning features, already built."""
    # float_precision="round_trip" is required, not cosmetic: the default parser
    # is fast rather than correctly rounded and shifts values by about 1e-14,
    # which is enough to move a split threshold and with it the projection.
    design = pd.read_csv(_require(DATA / "scenario_design_K62.csv"), float_precision="round_trip")
    design["Date"] = pd.to_datetime(design["Date"], format="%d-%b-%Y")
    design["Sector"] = design["EntityLabel"].str.rsplit("_", n=1).str[0]
    return design


def predict_scenarios(model, feature_names, design=None):
    """Monthly predicted excess return for every scenario cell."""
    design = load_design() if design is None else design
    design = design.copy()
    design["yhat"] = model.predict(design[list(feature_names)])
    return design


def load_risk_free():
    """
    The scenario risk-free path, keyed by Scenario x Component x month.

    RF does not vary across entities inside one of those cells, so the join key
    carries no Entity.
    """
    predictions = pd.read_csv(
        _require(RESULTS / "scenario_monthly_predictions.csv"), float_precision="round_trip"
    )
    predictions["Date"] = pd.to_datetime(predictions["Date"], format="%d-%b-%Y")
    path = predictions[["Scenario", "Component", "Date", "RF"]].drop_duplicates()
    duplicated = path.duplicated(["Scenario", "Component", "Date"]).sum()
    assert duplicated == 0, f"RF is not unique in {duplicated} Scenario/Component/month cells"
    return path


def log_ratio(predictions, risk_free=None):
    """
    Compound each portfolio on the total return, then take the within-sector
    Green/Brown log-ratio. Returns Scenario, Component, Sector, Date, LogRatio.

    Follows s3_K62_leaf10.m: total = yhat + RF, compounded, then the log-ratio.
    """
    risk_free = load_risk_free() if risk_free is None else risk_free

    # Reference scenarios carry no risk-free path and are not part of the five
    # reported in the thesis; s3_K62_leaf10.m drops them before compounding.
    lowered = predictions["Scenario"].str.lower()
    is_reference = np.zeros(len(predictions), dtype=bool)
    for token in REFERENCE_SCENARIOS:
        is_reference |= lowered.str.contains(token, regex=False).to_numpy()
    predictions = predictions.loc[~is_reference]

    before = len(predictions)
    frame = predictions.merge(risk_free, on=["Scenario", "Component", "Date"], how="left")
    assert len(frame) == before, "the risk-free join duplicated rows"
    missing = int(frame["RF"].isna().sum())
    if missing:
        raise ValueError(
            f"no risk-free path for {missing} scenario rows; "
            "the projection cannot be compounded without it"
        )

    frame["total"] = frame["yhat"] + frame["RF"]
    frame = frame.sort_values(["Scenario", "Component", "EntityLabel", "Date"])
    frame["Cum"] = frame.groupby(["Scenario", "Component", "EntityLabel"])["total"].transform(
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
    # The projection uses FIT B, the 237-month refit, as s3_K62_leaf10.m does.
    # FIT A stops in 2020 and is only there to certify the out-of-sample metrics.
    data = train_for_projection()
    print(f"FIT B: {data['x_fit'].shape[0]} rows = 22 x "
          f"{data['x_fit'].shape[0] // 22} months, {VARIANTS[data['variant']]}")

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

    energy = merged[(merged["Sector"] == "ENRG") & (merged["Component"] == "transition")]
    spread = float(energy["LogRatioReplication"].max() - energy["LogRatioReplication"].min())
    thesis_spread = float(energy["LogRatioThesis"].max() - energy["LogRatioThesis"].min())
    print(f"\nspread across scenarios: {spread:.6f}   thesis {thesis_spread:.6f}")

    deviation = (merged["LogRatioReplication"] - merged["LogRatioThesis"]).abs()
    print(f"largest deviation at 2050 over {len(merged)} values: {deviation.max():.3e}")

    if not check["SameSign"].all():
        failed = check.loc[~check["SameSign"], "Scenario"].tolist()
        print(f"Scenarios whose sign differs from the thesis: {failed}")


if __name__ == "__main__":
    main()
