"""
Scenario projection, 2025-2050.

scenario_design_K62.csv holds the 73 features for every Scenario x Component x
Entity x month, so this step applies the model to that design and aggregates:

    cumulative index   = prod(1 + r_t / 100)
    log-ratio (sector) = log(cum of the Green portfolio) - log(cum of the Brown one)

Compounding uses the total return, total = yhat + RF. The design file carries no
risk-free column, so the path is joined from
results/scenario_monthly_predictions.csv, where RF is constant across entities
within a Scenario x Component x month. The risk-free leg is not optional and not
a refinement: leaving it out changes the log-ratio.

"Current Policies" appears in the design file with the physical component only.
It carries no risk-free path and is not one of the five scenarios reported, so it
is dropped before compounding.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import DATA, RESULTS, _require
from train_gb import VARIANTS, train_for_projection

KEY_COLS = ["Entity", "EntityLabel", "Green", "Scenario", "Component", "Date", "SSP"]

# Scenarios projected but not reported: they have no risk-free path.
REFERENCE_SCENARIOS = ("current policies", "baseline")


SCENARIOS = [
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

    Compounds total = yhat + RF, then takes the log-ratio.
    """
    risk_free = load_risk_free() if risk_free is None else risk_free

    # Reference scenarios carry no risk-free path and are not among the five
    # reported, so they are dropped before compounding.
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


def main():
    # The projection uses the 237-month window. The 189-month fit exists only to
    # score the sealed block and is not used here.
    data = train_for_projection()
    print(f"{data['x_fit'].shape[0]} rows = 22 x {data['x_fit'].shape[0] // 22} months, "
          f"{VARIANTS[data['variant']]}")

    predictions = predict_scenarios(data["model"], data["x_fit"].columns)
    ratio = log_ratio(predictions)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "scenario_logratio.csv"
    ratio.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(Path(__file__).resolve().parents[1])} ({len(ratio)} rows)")

    end = ratio["Date"].max()
    energy = ratio[
        (ratio["Sector"] == "ENRG")
        & (ratio["Component"] == "transition")
        & (ratio["Date"] == end)
    ].set_index("Scenario")["LogRatio"]

    print(f"\nEnergy, transition component, {end:%B %Y}:")
    for scenario in SCENARIOS:
        if scenario in energy.index:
            print(f"  {scenario:<44}{energy[scenario]:+.6f}")
    print(f"  {'range across scenarios':<44}{energy.max() - energy.min():.6f}")


if __name__ == "__main__":
    main()
