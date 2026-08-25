"""
The two fits of the thesis, and the three tree growth policies.

Two fits, never one
-------------------
The MATLAB pipeline estimates the same configuration twice, and the two are used
for different things:

    FIT A   189 months, tr70 + va10, up to December 2020   (refit_finale_K62.m)
            certifies the out-of-sample metrics on the sealed 2021-2024 block
    FIT B   237 months, everything up to December 2024      (s3_K62_leaf10.m)
            produces the 2025-2050 scenario projection

Appendix A.3 of the thesis states the design: the model is re-estimated with the
hyper-parameters held at the chapter 3 values, and the scenario anchors are
recomputed on the same window as the estimation, so each configuration is
internally consistent. Using FIT A to project, or FIT B to report out-of-sample
metrics, mixes the two and is wrong in both directions: FIT B has seen the test
block, and FIT A projects from a model that stops in 2020.

Three growth policies
---------------------
MaxNumSplits is a budget on the number of splits, spent breadth-first, and
scikit-learn has no equivalent. The default here is the numpy builder that
follows the documented MATLAB procedure; the two scikit-learn policies are kept
as labelled diagnostics, because together the three bracket the thesis on
in-sample fit, out-of-sample fit and projection at once:

    max_depth=4        level-wise, truncated at depth 4     projection 0.5213
    matlab_policy      level-wise under a 15-split budget   projection 1.2199
    max_leaf_nodes=16  best-first, 15 splits                projection 1.4251

No tuning happens in this module. The screening of the 552 candidates down to
the 73-feature winning set, and the two-stage validation that fixed depth, leaf
size, learning rate, number of trees and the subsampling rate, were done in
MATLAB as part of the thesis. Only the selected configuration is refitted, so
the Python results are a replication and not a second, independent search.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from build_features import (
    RESULTS,
    build_winning_matrix,
    load_panel,
    split_frames,
)
from matlab_policy_gb import MatlabPolicyGB

FIT_A_SPLITS = ("tr70", "va10")
FIT_B_SPLITS = ("tr70", "va10", "test")

# The winning hyper-parameters. Do not tune these here.
LEARNING_RATE = 0.03
N_ESTIMATORS = 300
MIN_LEAF = 10
MIN_PARENT = 20  # MATLAB: max(MinParentSize, 2 * MinLeafSize)
MAX_SPLITS = 15  # MaxNumSplits

# scikit-learn settings shared by the two diagnostic variants.
GB_PARAMS = dict(
    max_depth=4,
    min_samples_leaf=MIN_LEAF,
    min_samples_split=MIN_PARENT,
    learning_rate=LEARNING_RATE,
    n_estimators=N_ESTIMATORS,
    subsample=1.0,
    max_features=None,
    random_state=42,
)

DEFAULT_VARIANT = "matlab_policy"

VARIANTS = {
    "matlab_policy": "numpy builder, level-wise under a 15-split budget",
    "sklearn_depth4": "scikit-learn max_depth=4, level-wise truncated at depth 4",
    "sklearn_bestfirst": "scikit-learn max_leaf_nodes=16, best-first",
}


def build_model(variant=DEFAULT_VARIANT):
    """An unfitted model for one of the three growth policies."""
    if variant == "matlab_policy":
        return MatlabPolicyGB(
            max_splits=MAX_SPLITS,
            min_leaf=MIN_LEAF,
            min_parent=MIN_PARENT,
            learning_rate=LEARNING_RATE,
            n_estimators=N_ESTIMATORS,
        )
    if variant == "sklearn_depth4":
        return GradientBoostingRegressor(**GB_PARAMS)
    if variant == "sklearn_bestfirst":
        params = dict(GB_PARAMS, max_depth=None, max_leaf_nodes=16)
        return GradientBoostingRegressor(**params)
    raise ValueError(f"unknown variant {variant!r}; choose one of {sorted(VARIANTS)}")


def fit_model(x, y, variant=DEFAULT_VARIANT):
    """Fit one variant. Inputs go in as arrays so every variant sees the same thing."""
    return build_model(variant).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def _stack(panel, splits):
    """
    Stack the requested splits in block order.

    tr70 first, then va10, then test: this is the row order of the MATLAB design
    matrices, and build_features validates against it.
    """
    return pd.concat([panel[panel["Split"] == s] for s in splits], ignore_index=True)


def prepare_fit_a():
    """FIT A: 189 months for fitting, plus the sealed test block for scoring."""
    panel = load_panel()
    fit, test = split_frames(panel)
    return {
        "x_fit": build_winning_matrix(fit),
        "y_fit": fit["Y"].to_numpy(dtype=float),
        "fit_frame": fit,
        "x_test": build_winning_matrix(test),
        "y_test": test["Y"].to_numpy(dtype=float),
        "test_frame": test,
        "panel": panel,
    }


def prepare_fit_b():
    """FIT B: all 237 months. There is no held-out block left, and none is used."""
    panel = load_panel()
    frame = _stack(panel, FIT_B_SPLITS)
    return {
        "x_fit": build_winning_matrix(frame),
        "y_fit": frame["Y"].to_numpy(dtype=float),
        "fit_frame": frame,
        "panel": panel,
    }


# Kept under the old name: FIT A is what the metric modules ask for.
prepare_data = prepare_fit_a


def train_for_metrics(variant=DEFAULT_VARIANT):
    """FIT A plus a fitted model. Use for R2, RMSE and interpretation."""
    data = prepare_fit_a()
    data["model"] = fit_model(data["x_fit"], data["y_fit"], variant)
    data["variant"] = variant
    return data


def train_for_projection(variant=DEFAULT_VARIANT):
    """FIT B plus a fitted model. Use for the 2025-2050 scenario projection."""
    data = prepare_fit_b()
    data["model"] = fit_model(data["x_fit"], data["y_fit"], variant)
    data["variant"] = variant
    return data


train = train_for_metrics


def feature_importance(model, feature_names, x=None, y=None):
    """
    Share of the total SSE reduction attributed to each column.

    The numpy builder does not store the gains, so it replays the splits and
    needs the sample back; scikit-learn keeps them on the fitted estimator.
    """
    if isinstance(model, MatlabPolicyGB):
        if x is None or y is None:
            raise ValueError("the numpy builder needs the fitting sample to replay the splits")
        values = model.feature_importance(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    else:
        values = model.feature_importances_

    importance = pd.Series(values, index=list(feature_names)).sort_values(ascending=False)
    table = importance.rename("Importance").reset_index()
    table.columns = ["Feature", "Importance"]
    table["SharePct"] = 100 * table["Importance"] / table["Importance"].sum()
    return table


def main():
    data = train_for_metrics()
    table = feature_importance(
        data["model"], data["x_fit"].columns, data["x_fit"], data["y_fit"]
    )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "replication_gb_feature_importance.csv"
    table.to_csv(out, index=False)

    shape = data["model"].tree_shape()
    print(f"FIT A, {VARIANTS[data['variant']]}")
    print(f"  fitted on {data['x_fit'].shape[0]} rows x {data['x_fit'].shape[1]} features; "
          f"sealed test block has {data['x_test'].shape[0]} rows")
    print(f"  splits per tree {shape['splits'].mean():.2f}, "
          f"leaves {shape['leaves'].mean():.2f}, max depth {shape['depth'].max()}")
    print(f"  wrote {out.relative_to(Path(__file__).resolve().parents[1])}")
    print("\nTop 10 features:")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
