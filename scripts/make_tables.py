"""
Build the mean-differences table from the thesis scenario predictions.

For every sector and scenario, the average over 2025-2050 of the monthly
difference in predicted return between the Green and the Brown portfolio of
that sector, reported for the transition and the physical component, with the
range across scenarios in the last column.

Everything in this table comes from the thesis output
(results/scenario_monthly_predictions.csv), not from the Python replication.

A sanity check guards the numbers: the range must be about 0.39 for Energy,
about 0.31 for Materials, and close to zero for the other sectors. If it is
not, the script stops instead of writing.

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


if __name__ == "__main__":
    main()
