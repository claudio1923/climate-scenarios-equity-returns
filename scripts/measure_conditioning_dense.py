"""
Dense version of the perturbation sweep, for the two magnitudes near the last
representable bit.

Eight draws per level are enough to see that the projection moves and nowhere
near enough to say by how much: they cannot separate 5/8 from 7/8, and they
cannot establish where the sensitivity saturates. This script runs the same
experiment at 1e-16 and 1e-15 with enough draws to put a confidence interval on
the fraction of draws that keep the reference sign pattern.

The fits are independent, so they run in parallel; every draw still has its own
fixed seed, and the seeds do not overlap with the coarse sweep in
measure_conditioning.py.

What is measured here is the numerical conditioning of the procedure: how much
the answer moves when the inputs are nudged at the level of floating-point
representation. It is not a statistical confidence interval for the estimate,
and the two must not be read as if they were the same quantity.

Run from the repository root:  python scripts/measure_conditioning_dense.py
"""

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "conditioning"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

LEVELS = [1e-16, 1e-15]
DRAWS = 150
SEED_BASE = 500_000
SEED_STEP = 10_000
WORKERS = 6

END = pd.Timestamp("2050-12-31")

_STATE = {}


def _initialise():
    """Load the sample and the scenario design once per worker process."""
    from build_features import build_winning_matrix, load_panel
    from scenarios import (
        REFERENCE_SCENARIOS,
        SCENARIO_CODES,
        THESIS_SCENARIOS,
        load_design,
        load_risk_free,
    )

    panel = load_panel()
    blocks = pd.concat(
        [panel[panel["Split"] == s] for s in ("tr70", "va10", "test")], ignore_index=True
    )
    x_frame = build_winning_matrix(blocks)
    columns = list(x_frame.columns)

    design = load_design()
    # Reference scenarios are dropped before compounding anyway; removing them
    # here keeps the per-draw prediction smaller.
    lowered = design["Scenario"].str.lower()
    keep = np.ones(len(design), dtype=bool)
    for token in REFERENCE_SCENARIOS:
        keep &= ~lowered.str.contains(token, regex=False).to_numpy()
    design = design.loc[keep]

    _STATE.update(
        x_base=x_frame.to_numpy(dtype=float),
        y=blocks["Y"].to_numpy(dtype=float),
        keys=design[["Scenario", "Component", "Date", "EntityLabel", "Green", "Sector"]]
        .reset_index(drop=True),
        x_scenario=design[columns].to_numpy(dtype=float),
        risk_free=load_risk_free(),
        scenarios=THESIS_SCENARIOS,
    )


def _one_draw(task):
    """Perturb, refit, reproject, and read off the Energy curve."""
    from matlab_policy_gb import MatlabPolicyGB
    from scenarios import log_ratio

    level, draw, seed = task
    if not _STATE:
        _initialise()

    generator = np.random.default_rng(seed)
    noise = generator.uniform(-1.0, 1.0, size=_STATE["x_base"].shape)
    model = MatlabPolicyGB(
        max_splits=15, min_leaf=10, min_parent=20, learning_rate=0.03, n_estimators=300
    ).fit(_STATE["x_base"] * (1.0 + level * noise), _STATE["y"])

    frame = _STATE["keys"].copy()
    frame["yhat"] = model.predict(_STATE["x_scenario"])
    ratio = log_ratio(frame, _STATE["risk_free"])

    energy = ratio[
        (ratio["Sector"] == "ENRG")
        & (ratio["Component"] == "transition")
        & (ratio["Date"] == END)
    ]
    curve = energy.set_index("Scenario")["LogRatio"].reindex(_STATE["scenarios"])

    return {
        "level": level,
        "draw": draw,
        "seed": seed,
        "spread": float(curve.max() - curve.min()),
        "signs": "".join("+" if v > 0 else "-" for v in curve),
        "order": ">".join(curve.sort_values(ascending=False).index.map(SCENARIO_CODES)),
        **{f"s_{SCENARIO_CODES[name]}": float(value) for name, value in curve.items()},
    }


def wilson_interval(successes, total, z=1.96):
    """
    Wilson score interval for a proportion.

    Preferred over the normal approximation here because the fractions are close
    to one and the counts are moderate, where the normal interval misbehaves.
    """
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("baseline fit")
    _initialise()
    from matlab_policy_gb import MatlabPolicyGB
    from scenarios import log_ratio

    baseline_model = MatlabPolicyGB(
        max_splits=15, min_leaf=10, min_parent=20, learning_rate=0.03, n_estimators=300
    ).fit(_STATE["x_base"], _STATE["y"])
    frame = _STATE["keys"].copy()
    frame["yhat"] = baseline_model.predict(_STATE["x_scenario"])
    ratio = log_ratio(frame, _STATE["risk_free"])
    energy = ratio[
        (ratio["Sector"] == "ENRG")
        & (ratio["Component"] == "transition")
        & (ratio["Date"] == END)
    ]
    curve = energy.set_index("Scenario")["LogRatio"].reindex(_STATE["scenarios"])
    reference_signs = "".join("+" if v > 0 else "-" for v in curve)
    reference_order = ">".join(curve.sort_values(ascending=False).index.map(SCENARIO_CODES))
    baseline_spread = float(curve.max() - curve.min())
    print(f"  spread {baseline_spread:.6f}  signs {reference_signs}")

    tasks = [
        (level, draw, SEED_BASE + SEED_STEP * index + draw)
        for index, level in enumerate(LEVELS)
        for draw in range(DRAWS)
    ]
    print(f"\n{len(tasks)} fits on {WORKERS} workers")

    started = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for done, record in enumerate(pool.map(_one_draw, tasks, chunksize=4), start=1):
            records.append(record)
            if done % 25 == 0:
                elapsed = time.time() - started
                rate = elapsed / done
                print(f"  {done}/{len(tasks)}  elapsed {elapsed / 60:.1f} min, "
                      f"eta {(len(tasks) - done) * rate / 60:.1f} min")

    table = pd.DataFrame(records)
    out = RESULTS / "perturbation_dense.csv"
    table.to_csv(out, index=False)

    print(f"\n{'=' * 88}\n  DENSE SWEEP, {DRAWS} draws per level\n{'=' * 88}")
    print(f"  unperturbed spread {baseline_spread:.6f}, signs {reference_signs}, "
          f"order {reference_order}\n")
    print(f"  {'noise':>8} {'mean':>9} {'sd':>9} {'min':>9} {'max':>9} "
          f"{'signs kept':>12} {'95% CI':>18} {'order kept':>12}")
    for level, block in table.groupby("level"):
        kept = int((block["signs"] == reference_signs).sum())
        order_kept = int((block["order"] == reference_order).sum())
        low, high = wilson_interval(kept, len(block))
        print(f"  {level:>8.0e} {block['spread'].mean():>9.4f} {block['spread'].std():>9.4f} "
              f"{block['spread'].min():>9.4f} {block['spread'].max():>9.4f} "
              f"{kept:>5}/{len(block)} {100 * kept / len(block):>5.1f}% "
              f"[{100 * low:>5.1f}%, {100 * high:>5.1f}%] {order_kept:>7}/{len(block)}")

    print("\n  sign of each scenario, share of draws matching the unperturbed fit")
    for name in _STATE["scenarios"]:
        column = f"s_{SCENARIO_CODES[name]}"
        for level, block in table.groupby("level"):
            same = int((np.sign(block[column]) == np.sign(curve[name])).sum())
            low, high = wilson_interval(same, len(block))
            print(f"    {name[:34]:<34} {level:.0e}  {100 * same / len(block):>5.1f}% "
                  f"[{100 * low:>5.1f}%, {100 * high:>5.1f}%]")

    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
