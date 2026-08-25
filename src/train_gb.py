"""
Gradient Boosting: the winning configuration of the thesis, refit in Python.

No tuning happens here, by design. The search that produced this model was run
in MATLAB as part of the thesis:
    - screening of the 552 candidates down to the 73-feature winning set,
    - validation over 30 seeds to pick depth, leaf size, learning rate, number
      of trees and the subsampling rate.
This module only refits the selected configuration, so that the Python results
are a replication and not a second, independent search.

Winning configuration:
    depth 4, min leaf 10, learning rate 0.03, 300 trees, no subsampling.
Subsampling is switched off, which makes the fit deterministic; random_state is
fixed anyway so the run is reproducible end to end.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from build_features import (
    RESULTS,
    build_winning_matrix,
    load_panel,
    split_frames,
)

# The winning hyper-parameters. Do not tune these here.
GB_PARAMS = dict(
    max_depth=4,
    min_samples_leaf=10,
    learning_rate=0.03,
    n_estimators=300,
    subsample=1.0,
    random_state=42,
)


def prepare_data():
    """Return the fit sample (tr70 + va10) and the sealed test block."""
    panel = load_panel()
    fit, test = split_frames(panel)

    x_fit = build_winning_matrix(fit)
    x_test = build_winning_matrix(test)
    return {
        "x_fit": x_fit,
        "y_fit": fit["Y"].to_numpy(),
        "fit_frame": fit,
        "x_test": x_test,
        "y_test": test["Y"].to_numpy(),
        "test_frame": test,
        "panel": panel,
    }


def fit_model(x_fit, y_fit):
    """Fit the Gradient Boosting model on the 80% fit sample."""
    model = GradientBoostingRegressor(**GB_PARAMS)
    model.fit(x_fit, y_fit)
    return model


def train():
    """Convenience wrapper used by the other modules: data + fitted model."""
    data = prepare_data()
    data["model"] = fit_model(data["x_fit"], data["y_fit"])
    return data


def feature_importance(model, feature_names):
    """Impurity-based importance, sorted, with the share of the total."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    table = importance.rename("Importance").reset_index()
    table.columns = ["Feature", "Importance"]
    table["SharePct"] = 100 * table["Importance"] / table["Importance"].sum()
    return table


def main():
    data = train()
    table = feature_importance(data["model"], data["x_fit"].columns)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "replication_gb_feature_importance.csv"
    table.to_csv(out, index=False)

    print(f"Fitted on {data['x_fit'].shape[0]} rows x {data['x_fit'].shape[1]} features "
          f"(tr70 + va10); sealed test block has {data['x_test'].shape[0]} rows.")
    print(f"Wrote {out.relative_to(Path(__file__).resolve().parents[1])}")
    print("\nTop 10 features (Python replication):")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
