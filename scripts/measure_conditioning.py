"""
How much of the scenario projection survives a perturbation of the inputs.

Re-exporting the private CSVs at full precision removed one particular
perturbation. It did not remove the sensitivity that made that perturbation
matter, and the sensitivity is the more interesting object: a relative change of
5e-14 in the estimation matrix moved the Energy 2050 spread by twenty per cent.

This script measures it. The 237-month estimation matrix is perturbed with
controlled relative noise at four magnitudes, the model is refitted from
scratch on each perturbed sample, and the projection is recomputed. What is
collected is not only the size of the spread but whether the qualitative
readings survive: the sign of each scenario, and their ordering.

Structural zeros stay zero. The interaction columns are zero wherever the row
belongs to another entity, and that is a property of the design rather than a
measured quantity, so relative noise leaves it untouched.

Every draw uses a fixed seed, so the whole table is reproducible.

Run from the repository root:  python scripts/measure_conditioning.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "conditioning"
sys.path.insert(0, str(ROOT / "src"))

from build_features import build_winning_matrix, load_panel  # noqa: E402
from matlab_policy_gb import MatlabPolicyGB  # noqa: E402
from scenarios import THESIS_SCENARIOS, load_design, load_risk_free, log_ratio  # noqa: E402

LEVELS = [1e-14, 1e-13, 1e-12, 1e-10]
DRAWS = 8
END = pd.Timestamp("2050-12-31")
TARGET = "R_WTI_L1_x_Entity_6"
WINDOW = (0.0399, 1.8447)

GB = dict(max_splits=15, min_leaf=10, min_parent=20, learning_rate=0.03, n_estimators=300)

# Reference reading, from the unperturbed fit: which scenarios are positive and
# in what order. Both are recomputed below rather than hard-coded.
THESIS_SPREAD = 1.219885


def fit_b_sample():
    """The 237-month estimation sample, in the MATLAB block order."""
    panel = load_panel()
    blocks = pd.concat(
        [panel[panel["Split"] == s] for s in ("tr70", "va10", "test")], ignore_index=True
    )
    x = build_winning_matrix(blocks)
    return x, blocks["Y"].to_numpy(dtype=float)


def energy_curve(model, design, columns, risk_free):
    """Energy transition log-ratio at 2050, one value per scenario."""
    frame = design.copy()
    frame["yhat"] = model.predict(frame[columns].to_numpy(dtype=float))
    ratio = log_ratio(frame, risk_free)
    energy = ratio[
        (ratio["Sector"] == "ENRG")
        & (ratio["Component"] == "transition")
        & (ratio["Date"] == END)
    ]
    return energy.set_index("Scenario")["LogRatio"].reindex(THESIS_SCENARIOS)


def summarise(model, curve, target_index):
    thresholds = model.thresholds_on(target_index)
    inside = thresholds[(thresholds >= WINDOW[0]) & (thresholds <= WINDOW[1])]
    return {
        "spread": float(curve.max() - curve.min()),
        "signs": "".join("+" if v > 0 else "-" for v in curve),
        "order": ">".join(curve.sort_values(ascending=False).index.str[:4]),
        "splits_on_target": int(thresholds.size),
        "splits_in_window": int(inside.size),
        **{f"s_{name[:4]}": float(value) for name, value in curve.items()},
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    x_frame, y = fit_b_sample()
    columns = list(x_frame.columns)
    target_index = columns.index(TARGET)
    x_base = x_frame.to_numpy(dtype=float)

    design = load_design()
    risk_free = load_risk_free()

    print(f"baseline fit on {x_base.shape[0]} rows x {x_base.shape[1]} columns")
    started = time.time()
    baseline_model = MatlabPolicyGB(**GB).fit(x_base, y)
    baseline_curve = energy_curve(baseline_model, design, columns, risk_free)
    baseline = summarise(baseline_model, baseline_curve, target_index)
    print(f"  spread {baseline['spread']:.6f} (thesis {THESIS_SPREAD})   "
          f"signs {baseline['signs']}   {time.time() - started:.0f}s")
    print(f"  order  {baseline['order']}")

    rows = [{"level": 0.0, "draw": -1, **baseline}]

    for level_index, level in enumerate(LEVELS):
        print(f"\nrelative noise {level:.0e}")
        for draw in range(DRAWS):
            seed = 10_000 * (level_index + 1) + draw
            generator = np.random.default_rng(seed)
            # Relative noise: structural zeros are left alone by construction.
            noise = generator.uniform(-1.0, 1.0, size=x_base.shape)
            perturbed = x_base * (1.0 + level * noise)

            model = MatlabPolicyGB(**GB).fit(perturbed, y)
            curve = energy_curve(model, design, columns, risk_free)
            record = summarise(model, curve, target_index)
            rows.append({"level": level, "draw": draw, "seed": seed, **record})
            print(f"  draw {draw}: spread {record['spread']:.6f}  signs {record['signs']}  "
                  f"in-window {record['splits_in_window']}")

    table = pd.DataFrame(rows)
    out = RESULTS / "perturbation_sensitivity.csv"
    table.to_csv(out, index=False)

    print(f"\n{'=' * 74}\n  SUMMARY\n{'=' * 74}")
    print(f"  unperturbed spread {baseline['spread']:.6f}, signs {baseline['signs']}\n")

    perturbed_rows = table[table["level"] > 0]
    header = f"  {'noise':>8} {'mean':>10} {'sd':>10} {'min':>10} {'max':>10} " \
             f"{'signs kept':>11} {'order kept':>11}"
    print(header)
    for level, block in perturbed_rows.groupby("level"):
        signs_kept = int((block["signs"] == baseline["signs"]).sum())
        order_kept = int((block["order"] == baseline["order"]).sum())
        print(f"  {level:>8.0e} {block['spread'].mean():>10.4f} {block['spread'].std():>10.4f} "
              f"{block['spread'].min():>10.4f} {block['spread'].max():>10.4f} "
              f"{signs_kept:>8}/{len(block)} {order_kept:>8}/{len(block)}")

    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
