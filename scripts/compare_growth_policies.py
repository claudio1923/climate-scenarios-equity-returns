"""
Three-way comparison of tree growth policies on the K62 design.

    A  scikit-learn max_depth=4        level-wise, stops at depth 4
    B  scikit-learn max_leaf_nodes=16  best-first, 15 splits
    C  numpy builder max_splits=15     MATLAB policy: level-wise under a budget

Everything else is held fixed: same 4158 x 73 design, same residual loss, same
learning rate and number of trees, same leaf and parent constraints, no
subsampling. The only thing that varies is how a tree spends its splits.

Thesis reference values, quoted as such and never recomputed here:
    R2 out-of-sample 0.406, RMSE 5.108, Energy transition spread 2050 = 1.22.

Run from the repository root:  python scripts/compare_growth_policies.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from build_features import load_feature_names, load_winning_set  # noqa: E402
from matlab_policy_gb import MatlabPolicyGB, check_equivalence_with_sklearn  # noqa: E402
from scenarios import THESIS_SCENARIOS, load_design, log_ratio  # noqa: E402
from train_gb import prepare_data  # noqa: E402

TARGET_FEATURE = "R_WTI_L1_x_Entity_6"  # WTI(t-1) x ENRG-Brown
WINDOW = (0.04, 1.845)                  # range the scenario paths occupy
END = pd.Timestamp("2050-12-31")

# Thesis figures, for reference only.
THESIS = {"r2_oos": 0.406, "rmse_oos": 5.108, "energy_spread": 1.2199}

SKLEARN_COMMON = dict(
    loss="squared_error",
    criterion="squared_error",
    learning_rate=0.03,
    n_estimators=300,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features=None,
    subsample=1.0,
    min_impurity_decrease=0.0,
    min_weight_fraction_leaf=0.0,
    ccp_alpha=0.0,
    n_iter_no_change=None,
    random_state=1,
)


def r2(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - sse / sst)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def assert_column_order(columns):
    """
    The design must follow the MATLAB ordering of FN_full: the base names first,
    then driver-major, entity-minor interactions, filtered by the K62 mask with
    the order preserved.
    """
    expected = load_winning_set()["Name"].tolist()
    assert list(columns) == expected, (
        "column order does not match the K62 mask applied to FN_full:\n"
        f"  first mismatch at position "
        f"{next(i for i, (a, b) in enumerate(zip(columns, expected)) if a != b)}"
    )

    canonical = load_feature_names()
    positions = [canonical.index(name) for name in expected]
    assert positions == sorted(positions), "K62 mask did not preserve the FN_full order"
    print(f"column order checked: {len(expected)} features in FN_full order")


def sklearn_shape(model):
    leaves, splits, depths = [], [], []
    for estimator in model.estimators_.ravel():
        tree = estimator.tree_
        leaves.append(int((tree.children_left == -1).sum()))
        splits.append(int((tree.children_left != -1).sum()))
        depths.append(int(tree.max_depth))
    return {
        "leaves": np.array(leaves),
        "splits": np.array(splits),
        "depth": np.array(depths),
    }


def sklearn_thresholds(model, column_index):
    collected = []
    for estimator in model.estimators_.ravel():
        tree = estimator.tree_
        collected.extend(tree.threshold[tree.feature == column_index].tolist())
    return np.array(collected)


def risk_free_path():
    """
    The scenario risk-free series, taken from the thesis predictions file.

    scenario_design_K62.csv carries no risk-free column, so it is joined here on
    Scenario x Component x Date x Entity. This is downstream of the model: the
    same predictions are compounded with and without it.
    """
    predictions = pd.read_csv(RESULTS / "scenario_monthly_predictions.csv")
    predictions["Date"] = pd.to_datetime(predictions["Date"], format="%d-%b-%Y")
    return predictions[["Scenario", "Component", "Date", "Entity", "RF"]]


def energy_row(predictions_frame, design, use_risk_free, risk_free):
    """Energy transition log-ratio at 2050, one value per scenario."""
    frame = design.copy()
    frame["yhat"] = predictions_frame

    if use_risk_free:
        frame = frame.merge(risk_free, on=["Scenario", "Component", "Date", "Entity"], how="left")
        frame = frame[frame["RF"].notna()].copy()
        frame["yhat"] = frame["yhat"] + frame["RF"]

    ratio = log_ratio(frame)
    energy = ratio[
        (ratio["Sector"] == "ENRG")
        & (ratio["Component"] == "transition")
        & (ratio["Date"] == END)
    ]
    return energy.set_index("Scenario")["LogRatio"]


def describe(name, shape, thresholds, metrics, energy_excess, energy_total):
    low, high = WINDOW
    inside = thresholds[(thresholds >= low) & (thresholds <= high)]

    print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}")
    print(f"  R2 in-sample      {metrics['r2_is']:.5f}      RMSE in-sample   {metrics['rmse_is']:.4f}")
    print(f"  R2 out-of-sample  {metrics['r2_oos']:.5f}      RMSE out-of-sample {metrics['rmse_oos']:.4f}")
    print(f"  gap               {metrics['r2_is'] - metrics['r2_oos']:.5f}")

    print("\n  tree shape over 300 trees          min    mean     max")
    for key in ("leaves", "splits", "depth"):
        values = shape[key]
        print(f"    {key:<28} {values.min():>4}  {values.mean():>6.2f}  {values.max():>6}")
    at_budget = int((shape["splits"] == 15).sum())
    print(f"    trees at the 15-split budget  {at_budget} / {len(shape['splits'])}")

    print(f"\n  splits on {TARGET_FEATURE} (WTI(t-1) x ENRG-Brown)")
    print(f"    total                         {len(thresholds)}")
    print(f"    inside the window {WINDOW}  {len(inside)}")
    if len(inside):
        print(f"    values                        {np.sort(inside).round(4).tolist()}")
    if len(thresholds):
        below = thresholds[thresholds < low]
        above = thresholds[thresholds > high]
        print(f"    nearest below                 {below.max():.4f}" if len(below) else "    none below")
        print(f"    nearest above                 {above.min():.4f}" if len(above) else "    none above")

    print("\n  Energy transition log-ratio, December 2050")
    print(f"    {'scenario':<44}{'excess only':>14}{'with RF':>12}")
    for scenario in THESIS_SCENARIOS:
        if scenario in energy_excess.index:
            print(f"    {scenario:<44}{energy_excess[scenario]:>14.4f}{energy_total[scenario]:>12.4f}")
    spread_excess = energy_excess.max() - energy_excess.min()
    spread_total = energy_total.max() - energy_total.min()
    print(f"    {'spread across scenarios':<44}{spread_excess:>14.4f}{spread_total:>12.4f}")
    print(f"    {'scenarios below zero':<44}{int((energy_excess < 0).sum()):>14}"
          f"{int((energy_total < 0).sum()):>12}")

    return {
        "model": name,
        "r2_is": metrics["r2_is"],
        "r2_oos": metrics["r2_oos"],
        "rmse_is": metrics["rmse_is"],
        "rmse_oos": metrics["rmse_oos"],
        "gap": metrics["r2_is"] - metrics["r2_oos"],
        "leaves_mean": float(shape["leaves"].mean()),
        "leaves_max": int(shape["leaves"].max()),
        "splits_mean": float(shape["splits"].mean()),
        "depth_max": int(shape["depth"].max()),
        "trees_at_budget": at_budget,
        "splits_on_target": len(thresholds),
        "splits_in_window": len(inside),
        "energy_spread_excess": float(spread_excess),
        "energy_spread_with_rf": float(spread_total),
        "energy_negative_scenarios": int((energy_total < 0).sum()),
    }


def main():
    print("validating the builder before use")
    check_equivalence_with_sklearn(depth=4)

    data = prepare_data()
    x_fit, y_fit = data["x_fit"], data["y_fit"]
    x_test, y_test = data["x_test"], data["y_test"]
    assert_column_order(x_fit.columns)

    columns = list(x_fit.columns)
    target_index = columns.index(TARGET_FEATURE)

    design = load_design()
    design_matrix = design[columns].to_numpy(dtype=float)
    risk_free = risk_free_path()

    summaries = []

    for name, kind, kwargs in [
        ("A  sklearn max_depth=4 (level-wise, depth capped)", "sklearn",
         dict(max_depth=4, max_leaf_nodes=None)),
        ("B  sklearn max_leaf_nodes=16 (best-first)", "sklearn",
         dict(max_depth=None, max_leaf_nodes=16)),
        ("C  numpy builder max_splits=15 (MATLAB policy)", "builder", {}),
    ]:
        started = time.time()
        if kind == "sklearn":
            model = GradientBoostingRegressor(**SKLEARN_COMMON, **kwargs).fit(x_fit, y_fit)
            assert abs(model.init_.constant_.ravel()[0] - float(np.mean(y_fit))) < 1e-9
            shape = sklearn_shape(model)
            thresholds = sklearn_thresholds(model, target_index)
            fitted = model.predict(x_fit)
            tested = model.predict(x_test)
            projected = model.predict(design[columns])
        else:
            model = MatlabPolicyGB(
                max_splits=15, min_leaf=10, min_parent=20,
                learning_rate=0.03, n_estimators=300,
            ).fit(x_fit.to_numpy(dtype=float), y_fit)
            assert abs(model.init_ - float(np.mean(y_fit))) < 1e-9
            shape = model.tree_shape()
            thresholds = model.thresholds_on(target_index)
            fitted = model.predict(x_fit.to_numpy(dtype=float))
            tested = model.predict(x_test.to_numpy(dtype=float))
            projected = model.predict(design_matrix)

        metrics = {
            "r2_is": r2(y_fit, fitted),
            "rmse_is": rmse(y_fit, fitted),
            "r2_oos": r2(y_test, tested),
            "rmse_oos": rmse(y_test, tested),
        }
        energy_excess = energy_row(projected, design, False, risk_free)
        energy_total = energy_row(projected, design, True, risk_free)

        summaries.append(describe(name, shape, thresholds, metrics, energy_excess, energy_total))
        print(f"\n  fitted in {time.time() - started:.1f}s")

    table = pd.DataFrame(summaries)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "growth_policy_comparison.csv"
    table.to_csv(out, index=False)

    print(f"\n{'=' * 78}\n  SUMMARY  (thesis reference: R2oos {THESIS['r2_oos']}, "
          f"Energy spread {THESIS['energy_spread']:.4f})\n{'=' * 78}")
    view = table[["model", "r2_oos", "rmse_oos", "splits_mean", "leaves_mean",
                  "splits_in_window", "energy_spread_with_rf", "energy_negative_scenarios"]]
    print(view.to_string(index=False))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
