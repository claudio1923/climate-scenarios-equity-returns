"""
Feature engineering: from the 24 base drivers to the 552 candidate features,
then down to the 73 features of the winning set.

Design rule of the thesis:
    - 24 main effects   : the base drivers (8 variables x lags 0, 1, 2)
    - 528 interactions  : every driver multiplied by the dummy of every entity
                          (24 drivers x 22 portfolios)
    -> 552 candidates in total.

The candidate ordering is fixed and must match feature_names_full552.csv:
    index 1..24                      -> main effects, in panel column order
    index 24 + (j-1)*22 + e          -> driver j interacted with entity e

Running this file as a script performs the mandatory validation against
X_trval_K62_reference.csv, the design matrix built in MATLAB.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_private"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Columns of training_panel.csv that are keys/target rather than drivers.
KEY_COLS = ("Date", "Entity", "Green", "Sector", "Split", "Y")

TRAIN_SPLITS = ("tr70", "va10")  # the 80% used for fitting
TEST_SPLIT = "test"              # the sealed 2021-2024 block


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "The proprietary inputs live in data_private/ and are not distributed "
            "with this repository. See the data availability note in the README."
        )
    return path


def load_panel():
    """Load the monthly panel and parse dates. One row = one portfolio-month."""
    panel = pd.read_csv(_require(DATA / "training_panel.csv"))
    panel["Date"] = pd.to_datetime(panel["Date"], format="%d-%b-%Y")
    return panel


def load_entity_labels():
    """Entity code -> portfolio label (e.g. 5 -> ENRG_G)."""
    labels = pd.read_csv(_require(DATA / "entity_labels.csv"))
    return dict(zip(labels["Entity"], labels["EntityLabel"]))


def load_feature_names():
    """The 552 candidate names, in canonical order."""
    return pd.read_csv(_require(DATA / "feature_names_full552.csv"))["Name"].tolist()


def load_winning_set():
    """The 73 features selected in the thesis (index into the 552, plus name)."""
    return pd.read_csv(_require(DATA / "winning_set_K62.csv"))


def driver_columns(panel):
    """The 24 base drivers, in the order they appear in the panel."""
    return [c for c in panel.columns if c not in KEY_COLS]


def entity_codes(panel):
    return sorted(panel["Entity"].unique())


def build_full552(panel, drivers=None, entities=None):
    """
    Build the full 552-column candidate matrix for the rows of `panel`.

    An interaction column is the driver value where the row belongs to that
    entity and zero everywhere else, i.e. driver * 1{Entity == e}.
    """
    drivers = drivers or driver_columns(panel)
    entities = entities or entity_codes(panel)

    entity_values = panel["Entity"].to_numpy()
    columns = {}

    # Main effects.
    for driver in drivers:
        columns[driver] = panel[driver].to_numpy()

    # Interactions, driver-major then entity, matching the canonical index rule.
    for driver in drivers:
        values = panel[driver].to_numpy()
        for entity in entities:
            columns[f"{driver}_x_Entity_{entity}"] = values * (entity_values == entity)

    return pd.DataFrame(columns, index=panel.index)


def build_winning_matrix(panel):
    """Build the 552 candidates and keep only the 73 columns of the winning set."""
    full = build_full552(panel)
    names = load_feature_names()

    # Guard: the constructed names must be exactly the canonical 552.
    missing = [n for n in names if n not in full.columns]
    if missing:
        raise ValueError(f"{len(missing)} canonical feature names were not built, e.g. {missing[:5]}")

    winning = load_winning_set()["Name"].tolist()
    return full[winning]


def split_frames(panel):
    """
    Split the panel into fit sample (tr70 + va10) and sealed test block.

    The fit sample is ordered tr70 first, then va10: this is the row order of
    the MATLAB reference matrix, so the validation below compares like with like.
    """
    fit = pd.concat(
        [panel[panel["Split"] == s] for s in TRAIN_SPLITS], ignore_index=True
    )
    test = panel[panel["Split"] == TEST_SPLIT].reset_index(drop=True)
    return fit, test


def validate_against_reference(verbose=True):
    """
    Mandatory check: the Python design matrix on tr70 + va10 must reproduce the
    MATLAB one, column by column. Raises if any column diverges.
    """
    panel = load_panel()
    fit, _ = split_frames(panel)

    built = build_winning_matrix(fit)
    reference = pd.read_csv(_require(DATA / "X_trval_K62_reference.csv"))

    # Row alignment: the reference carries Y and Entity, use them as a key check.
    if not np.allclose(fit["Y"].to_numpy(), reference["Y"].to_numpy()):
        raise ValueError("Row order mismatch: Y does not align with the MATLAB reference.")
    if not (fit["Entity"].to_numpy() == reference["Entity"].to_numpy()).all():
        raise ValueError("Row order mismatch: Entity does not align with the MATLAB reference.")

    reference_features = reference[built.columns.tolist()]
    per_column = np.abs(built.to_numpy() - reference_features.to_numpy()).max(axis=0)
    tolerance = 1e-10
    diverging = [(c, m) for c, m in zip(built.columns, per_column) if m > tolerance]

    if diverging:
        report = "\n".join(f"  {c}: max abs diff {m:.3e}" for c, m in diverging)
        raise ValueError(
            f"Feature construction does not match the MATLAB reference in "
            f"{len(diverging)} column(s):\n{report}"
        )

    if verbose:
        print(f"Validation OK: {built.shape[1]} columns, {built.shape[0]} rows "
              f"(tr70 + va10), max abs difference vs MATLAB = {per_column.max():.3e}")
    return float(per_column.max())


if __name__ == "__main__":
    validate_against_reference()
