"""
Re-export the private CSVs from the MATLAB .mat files at full double precision.

Why
---
The CSVs in data_private/ were written with the default text precision, so
parsing them back gives values that differ from the MATLAB doubles by about
5e-14. That looks harmless and is not: feeding the 237-month fit from the CSVs
gives an Energy 2050 spread of 1.4631, feeding it from the .mat gives 1.219885.
The projection is sensitive enough that a perturbation in the fourteenth decimal
moves it by twenty per cent, so the exported precision has to be exact.

Two things are needed for an exact round-trip, and neither is sufficient alone:

  - writing with `%.17g`, since the original files carry too few digits (they
    are off by 5e-14 even when parsed exactly);
  - reading with `float_precision="round_trip"`, since the default pandas parser
    is fast rather than correctly rounded and reintroduces about 1e-14.

The check after each export verifies both, and the readers in src/ pass the same
option.

Inputs live outside the repository and are proprietary; see the data
availability note in the README.

Run from the repository root:  python scripts/reexport_data_private.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_private"
BACKUP = DATA / "_pre_highprecision"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import verify_237_months as mat  # noqa: E402  (reads the .mat inputs)

FLOAT_FORMAT = "%.17g"
DATE_FORMAT = "%d-%b-%Y"


def _write(frame, path, label):
    """Write with round-tripping precision, then prove it round-trips."""
    BACKUP.mkdir(exist_ok=True)
    if path.exists() and not (BACKUP / path.name).exists():
        shutil.copy2(path, BACKUP / path.name)

    frame.to_csv(path, index=False, float_format=FLOAT_FORMAT)

    # The default parser is fast but not correctly rounded; only "round_trip"
    # returns the bit pattern that was written.
    reread = pd.read_csv(path, float_precision="round_trip")
    numeric = frame.select_dtypes(include=[np.number]).columns
    worst = 0.0
    for column in numeric:
        gap = np.abs(frame[column].to_numpy() - reread[column].to_numpy()).max()
        worst = max(worst, float(gap))
    if worst != 0.0:
        raise ValueError(f"{label}: re-reading changed values by {worst:.3e}, expected 0")

    print(f"  {label:<28} {frame.shape[0]:>6} rows x {frame.shape[1]:>3} cols   round-trip exact")
    return worst


def reexport_training_panel(base):
    """
    The monthly panel. Keys and labels are taken from the existing file, since
    integers and date strings carry no precision loss; the 24 drivers and Y come
    from X_model / Y_model.
    """
    path = DATA / "training_panel.csv"
    current = pd.read_csv(path, float_precision="round_trip")
    drivers = [c for c in current.columns if c not in ("Date", "Entity", "Green", "Sector", "Split", "Y")]
    assert len(drivers) == 24, len(drivers)

    # The .mat panel is in the same row order as the CSV; the integer keys prove it.
    assert np.array_equal(base["entity"].astype(int), current["Entity"].to_numpy())
    assert np.array_equal(base["green"].astype(int), current["Green"].to_numpy())
    assert np.array_equal(base["sector"].astype(int), current["Sector"].to_numpy())

    rebuilt = current.copy()
    rebuilt["Y"] = base["y"]
    for position, name in enumerate(drivers):
        rebuilt[name] = base["x"][:, position]

    return _write(rebuilt, path, "training_panel.csv")


def reexport_reference_matrix(design):
    """The tr70 + va10 design matrix used to validate the feature construction."""
    path = DATA / "X_trval_K62_reference.csv"
    frame = pd.DataFrame(design["x_a"], columns=design["names"])
    frame["Y"] = design["y_a"]

    current = pd.read_csv(path, float_precision="round_trip")
    assert list(current.columns) == list(frame.columns) + ["Entity"], "column layout changed"
    frame["Entity"] = current["Entity"].to_numpy()

    return _write(frame, path, "X_trval_K62_reference.csv")


def reexport_scenario_design(design):
    """The 2025-2050 scenario design: keys plus the 73 winning features."""
    path = DATA / "scenario_design_K62.csv"
    x_scenario, keys = mat.load_scenario_design(design["mask"], design["fn_full"])

    current = pd.read_csv(path, float_precision="round_trip")
    current["Date"] = pd.to_datetime(current["Date"], format=DATE_FORMAT)

    frame = keys.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Green"] = frame["Green"].astype(int)

    # SSP is a label, not a measurement, so it is carried over from the old file.
    join_keys = ["Entity", "Scenario", "Component", "Date"]
    before = len(frame)
    frame = frame.merge(current[join_keys + ["SSP"]], on=join_keys, how="left")
    assert len(frame) == before, "the SSP join duplicated rows"
    assert frame["SSP"].notna().all(), "SSP missing for some scenario rows"

    features = pd.DataFrame(x_scenario, columns=design["names"])
    frame = pd.concat([frame, features], axis=1)
    frame = frame[list(current.columns)]
    frame["Date"] = frame["Date"].dt.strftime(DATE_FORMAT)

    return _write(frame, path, "scenario_design_K62.csv")


def main():
    print("reading the MATLAB inputs")
    design = mat.load_estimation_design()

    import h5py

    with h5py.File(mat.BASE_FILE, "r") as handle:
        base = {
            "x": np.array(handle["X_model"]).T,
            "y": np.array(handle["Y_model"]).ravel(),
            "entity": np.array(handle["Entity_model"]).ravel(),
            "green": np.array(handle["Green_model"]).ravel(),
            "sector": np.array(handle["Sector_model"]).ravel(),
        }

    print(f"\nre-exporting to {DATA.relative_to(ROOT)} at {FLOAT_FORMAT}")
    print(f"originals backed up to {BACKUP.relative_to(ROOT)}\n")

    reexport_training_panel(base)
    reexport_reference_matrix(design)
    reexport_scenario_design(design)

    print("\nthe remaining files in data_private/ are label tables with no float columns")


if __name__ == "__main__":
    main()
