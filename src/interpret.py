"""
Partial dependence (PDP) and individual conditional expectation (ICE) curves,
computed on the fitted model.

This follows the logic of the calcPDP routine of the thesis. For an interaction
feature driver_x_Entity_e the sweep is local by construction: the column is only
non-zero on the rows of entity e, so the grid is applied to those rows and the
predictions are averaged over them. The main-effect column of the same driver is
left untouched, so the curve isolates the entity-specific channel.

The six pairs below are the ones discussed in the defence.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import RESULTS, load_entity_labels
from train_gb import train

# (feature in the winning set, entity code, label used in the figures)
PAIRS = [
    ("ExMkt_L0_x_Entity_2", 2, "ExMkt x CDSC-Brown"),
    ("ExMkt_L0_x_Entity_17", 17, "ExMkt x REIT-Green"),
    ("R_WTI_L1_x_Entity_6", 6, "WTI(t-1) x ENRG-Brown"),
    ("R_WTI_L1_x_Entity_15", 15, "WTI(t-1) x MATS-Green"),
    ("Tas_ANOM_L2_x_Entity_16", 16, "Temp(t-2) x MATS-Brown"),
    ("UNRATE_DIFF_L1_x_Entity_18", 18, "UNRATE(t-1) x REIT-Brown"),
]

GRID_POINTS = 25
PERCENTILE_RANGE = (1, 99)  # trim the tails so the grid is not driven by outliers


def _grid(values, n_points=GRID_POINTS):
    """Grid over the percentiles of the feature, tails trimmed."""
    quantiles = np.linspace(PERCENTILE_RANGE[0], PERCENTILE_RANGE[1], n_points)
    return np.unique(np.percentile(values, quantiles))


def curves_for_pair(model, x_all, entity_series, feature, entity):
    """
    Sweep `feature` over its grid on the rows of `entity`.

    Returns the grid, the ICE matrix (rows = observations of that entity,
    columns = grid points) and the PDP (column mean of the ICE matrix).
    """
    mask = (entity_series == entity).to_numpy()
    block = x_all.loc[mask].copy()
    if block.empty:
        raise ValueError(f"No observations for entity {entity}.")

    grid = _grid(block[feature].to_numpy())

    ice = np.empty((block.shape[0], grid.size))
    for j, value in enumerate(grid):
        perturbed = block.copy()
        perturbed[feature] = value
        ice[:, j] = model.predict(perturbed)

    return grid, ice, ice.mean(axis=0)


def compute_all(data=None):
    """Compute PDP and ICE for the six defence pairs on the full sample."""
    data = data or train()
    model = data["model"]
    panel = data["panel"]

    # Use the whole panel so each entity contributes all of its months.
    from build_features import build_winning_matrix

    x_all = build_winning_matrix(panel)
    entity_series = panel["Entity"]
    labels = load_entity_labels()

    pdp_rows, ice_rows = [], []
    for feature, entity, title in PAIRS:
        grid, ice, pdp = curves_for_pair(model, x_all, entity_series, feature, entity)

        for value, mean_pred in zip(grid, pdp):
            pdp_rows.append(
                {
                    "Pair": title,
                    "Feature": feature,
                    "Entity": entity,
                    "EntityLabel": labels[entity],
                    "GridValue": value,
                    "PDP": mean_pred,
                }
            )

        for curve_id in range(ice.shape[0]):
            for value, pred in zip(grid, ice[curve_id]):
                ice_rows.append(
                    {
                        "Pair": title,
                        "Feature": feature,
                        "CurveId": curve_id,
                        "GridValue": value,
                        "Prediction": pred,
                    }
                )

    return pd.DataFrame(pdp_rows), pd.DataFrame(ice_rows)


def main():
    pdp, ice = compute_all()

    RESULTS.mkdir(exist_ok=True)
    pdp_out = RESULTS / "model_pdp_curves.csv"
    ice_out = RESULTS / "model_ice_curves.csv"
    pdp.to_csv(pdp_out, index=False)
    ice.to_csv(ice_out, index=False)

    root = Path(__file__).resolve().parents[1]
    print(f"Wrote {pdp_out.relative_to(root)} ({len(pdp)} rows)")
    print(f"Wrote {ice_out.relative_to(root)} ({len(ice)} rows)")
    print("\nPDP range per pair (max - min of the average curve):")
    spread = pdp.groupby("Pair")["PDP"].agg(lambda s: s.max() - s.min()).round(4)
    print(spread.to_string())


if __name__ == "__main__":
    main()
