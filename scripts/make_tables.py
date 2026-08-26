"""
Build the two tables in the README.

**Mean differences.** For every sector and scenario, the average over 2025-2050
of the monthly difference in predicted return between the Green and the Brown
portfolio of that sector, reported for the transition and the physical
component, with the range across scenarios in the last column. Everything in
that table comes from the thesis output
(results/scenario_monthly_predictions.csv), not from the Python replication. A
sanity check guards it: the range must be about 0.39 for Energy, about 0.31 for
Materials, and close to zero elsewhere, or the script stops instead of writing.

**Growth policies.** MaxNumSplits = 15 is a budget on the number of splits spent
breadth-first, and scikit-learn has no equivalent: `max_depth` bounds the depth
instead of the count, `max_leaf_nodes` switches to best-first growth. All three
are fitted here on the same design and scored on the same axes, so the reader
can see where the budget policy sits relative to the two approximations. This
half needs data_private/ and takes a couple of minutes, six fits in all; without
the private inputs it is skipped and the existing CSV is left alone.

Run from the repository root:  python scripts/make_tables.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

COMPONENTS = ["transition", "physical"]

SCENARIO_ORDER = [
    "Net Zero 2050",
    "Delayed transition",
    "Below 2°C",
    "Nationally Determined Contributions (NDCs)",
    "Fragmented World",
]

# Short column headers for the markdown version.
SCENARIO_SHORT = {
    "Net Zero 2050": "Net Zero",
    "Delayed transition": "Delayed",
    "Below 2°C": "Below 2C",
    "Nationally Determined Contributions (NDCs)": "NDCs",
    "Fragmented World": "Fragmented",
}

# Sanity bands for the range across scenarios (transition component).
SANITY = {"ENRG": (0.37, 0.41), "MATS": (0.29, 0.33)}
OTHER_SECTORS_MAX_RANGE = 0.15


def load_predictions():
    path = RESULTS / "scenario_monthly_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy the thesis exports into results/ first."
        )
    predictions = pd.read_csv(path)
    predictions["Sector"] = predictions["EntityLabel"].str.rsplit("_", n=1).str[0]
    return predictions


def mean_differences(predictions):
    """Average Green-minus-Brown monthly prediction, by sector and scenario."""
    rows = []
    for component in COMPONENTS:
        subset = predictions[predictions["Component"] == component]
        pivot = subset.pivot_table(
            index=["Sector", "Scenario"], columns="Green", values="yhat", aggfunc="mean"
        )
        difference = (pivot[1] - pivot[0]).unstack("Scenario")
        difference = difference[[s for s in SCENARIO_ORDER if s in difference.columns]]
        difference["Range"] = difference.max(axis=1) - difference.min(axis=1)
        difference = difference.reset_index()
        difference.insert(1, "Component", component)
        rows.append(difference)

    table = pd.concat(rows, ignore_index=True)
    table.columns.name = None
    return table


def sanity_check(table):
    """Stop the build if the scenario ranges do not look like the thesis ones."""
    transition = table[table["Component"] == "transition"].set_index("Sector")
    problems = []

    for sector, (low, high) in SANITY.items():
        if sector not in transition.index:
            problems.append(f"{sector}: missing from the table")
            continue
        value = float(transition.loc[sector, "Range"])
        if not (low <= value <= high):
            problems.append(f"{sector}: range {value:.4f}, expected between {low} and {high}")

    for sector, value in transition["Range"].items():
        if sector in SANITY:
            continue
        if float(value) > OTHER_SECTORS_MAX_RANGE:
            problems.append(
                f"{sector}: range {float(value):.4f}, expected below {OTHER_SECTORS_MAX_RANGE}"
            )

    if problems:
        raise ValueError(
            "Sanity check failed on the mean-differences table:\n  "
            + "\n  ".join(problems)
            + "\nStopping instead of publishing."
        )
    print("Sanity check passed: Energy and Materials in band, other sectors close to zero.")


def to_markdown(table, component):
    """Render one component of the table as a markdown block."""
    subset = table[table["Component"] == component].drop(columns="Component")
    subset = subset.rename(columns=SCENARIO_SHORT).set_index("Sector").round(3)

    header = "| Sector | " + " | ".join(subset.columns) + " |"
    separator = "|" + "---|" * (len(subset.columns) + 1)
    lines = [header, separator]
    for sector, row in subset.iterrows():
        lines.append(f"| {sector} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    return "\n".join(lines)


def growth_policy_table():
    """
    Fit the three growth policies and score them on the same axes.

    Each policy is fitted twice, on the 189-month sample that carries the
    out-of-sample metrics and on the 237-month sample that carries the
    projection, because the two answer different questions and the thesis keeps
    them apart.
    """
    import numpy as np

    from scenarios import load_design, log_ratio, load_risk_free
    from train_gb import VARIANTS, fit_model, prepare_fit_a, prepare_fit_b

    fit_a = prepare_fit_a()
    fit_b = prepare_fit_b()
    columns = list(fit_a["x_fit"].columns)
    # Every variant is fitted on arrays, so it is scored on arrays too; passing a
    # frame here makes scikit-learn warn about missing feature names.
    x_fit_values = fit_a["x_fit"].to_numpy(dtype=float)
    x_test_values = fit_a["x_test"].to_numpy(dtype=float)
    design = load_design()
    risk_free = load_risk_free()
    end = pd.Timestamp("2050-12-31")

    def r2(y_true, y_pred):
        sse = float(np.sum((y_true - y_pred) ** 2))
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - sse / sst

    # Bounded by depth, bounded by split count, bounded by leaf count chosen
    # best-first: the budget policy is listed between the two approximations
    # because that is where its behaviour falls, not because of any order of
    # discovery.
    order = ["sklearn_depth4", "matlab_policy", "sklearn_bestfirst"]

    rows = []
    for variant in order:
        description = VARIANTS[variant]
        print(f"  fitting {variant} ...", flush=True)

        model_a = fit_model(fit_a["x_fit"], fit_a["y_fit"], variant)
        model_b = fit_model(fit_b["x_fit"], fit_b["y_fit"], variant)

        if hasattr(model_a, "tree_shape"):
            shape = model_a.tree_shape()
            splits = float(shape["splits"].mean())
            leaves = float(shape["leaves"].mean())
        else:
            per_tree = [e.tree_ for e in model_a.estimators_.ravel()]
            splits = float(np.mean([(t.children_left != -1).sum() for t in per_tree]))
            leaves = float(np.mean([(t.children_left == -1).sum() for t in per_tree]))

        projected = design.copy()
        projected["yhat"] = model_b.predict(projected[columns].to_numpy(dtype=float))
        ratio = log_ratio(projected, risk_free)
        energy = ratio[
            (ratio["Sector"] == "ENRG")
            & (ratio["Component"] == "transition")
            & (ratio["Date"] == end)
        ]["LogRatio"]

        rows.append(
            {
                "Policy": variant,
                "Description": description,
                "R2_is_fitA": round(r2(fit_a["y_fit"], model_a.predict(x_fit_values)), 4),
                "R2_oos_fitA": round(r2(fit_a["y_test"], model_a.predict(x_test_values)), 4),
                "SplitsPerTree": round(splits, 2),
                "LeavesPerTree": round(leaves, 2),
                "EnergySpread2050_fitB": round(float(energy.max() - energy.min()), 4),
            }
        )

    return pd.DataFrame(rows)


def growth_policy_markdown(table):
    header = "| Growth policy | R² in-sample | R² out-of-sample | splits/tree | Energy 2050 spread |"
    lines = [header, "|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(
            f"| {row.Description} | {row.R2_is_fitA:.4f} | {row.R2_oos_fitA:.4f} | "
            f"{row.SplitsPerTree:.2f} | {row.EnergySpread2050_fitB:.4f} |"
        )
    return "\n".join(lines)


def main():
    predictions = load_predictions()
    table = mean_differences(predictions)
    sanity_check(table)

    out = RESULTS / "mean_differences_table.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(ROOT)}")

    for component in COMPONENTS:
        print(f"\n### {component}\n")
        print(to_markdown(table, component))

    print("\n### growth policies\n")
    if not (ROOT / "data_private" / "training_panel.csv").exists():
        print("data_private/ is not present, so the growth-policy table is skipped; "
              "results/growth_policy_table.csv is left as it is.")
        return

    policies = growth_policy_table()
    policy_out = RESULTS / "growth_policy_table.csv"
    policies.to_csv(policy_out, index=False)
    print(f"\nWrote {policy_out.relative_to(ROOT)}\n")
    print(growth_policy_markdown(policies))


if __name__ == "__main__":
    main()
