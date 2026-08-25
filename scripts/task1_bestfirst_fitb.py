"""
Task 1 - la cella mancante: FIT B con la politica best-first.

Una sola stima. FIT B (237 mesi, 5214 righe) con max_leaf_nodes=16 e
max_depth=None, tutto il resto invariato, poi la proiezione completa come
in 0.2.

Serve a sapere se anche la proiezione e' incapsulata come lo sono le
metriche di fit:

    best-first sopra 1.22  -> la politica intermedia ha un bersaglio dentro
                              cui cadere, il builder ha senso;
    best-first sotto 1.22  -> il budget di split non spiega il residuo e
                              c'e' una terza causa da cercare.

best-first NON e' la politica MATLAB. E' una sonda di capacita': spende
tutti e 15 gli split, quindi delimita l'estremo superiore del percorso.

Nessun tuning: l'unica variante rispetto a GB_PARAMS e' la politica di
crescita, che e' esattamente l'oggetto della misura.

Output: results/task1/.
Uso:  python scripts/task1_bestfirst_fitb.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1"

TARGET_FEATURE_ENTITY = "ENRG_B"
WINDOW = (0.0399, 1.8447)          # finestra visitata dallo scenario, da 0.2
SPREAD_DEPTH4 = 0.521270           # FIT B, max_depth=4
SPREAD_THESIS = 1.2198871205       # tesi, ricalcolato dall'export


def load_verifier():
    path = Path(__file__).with_name("verify_237_months.py")
    spec = importlib.util.spec_from_file_location("verify_237_months", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    v = load_verifier()
    if not v.check_inputs():
        print("\nMi fermo: senza i .mat non stimo nulla.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg)
        lines.append(str(msg))

    params = dict(v.GB_PARAMS)
    params["max_depth"] = None
    params["max_leaf_nodes"] = 16
    say("TASK 1 - FIT B con politica best-first (sonda, non correzione)")
    say(f"parametri: {params}")

    d = v.load_estimation_design()
    names = d["names"]
    x_scen, keys = v.load_scenario_design(d["mask"], d["fn_full"])
    drivers = v.load_risk_free()

    model = GradientBoostingRegressor(**params).fit(d["x_b"], d["y_b"])

    # --- capacita' e fit ---------------------------------------------------
    leaves = np.array(
        [int((t.tree_.children_left == -1).sum()) for t in model.estimators_.ravel()]
    )
    splits = leaves - 1
    r2_is, rmse_is = v.metrics(d["y_b"], model.predict(d["x_b"]))
    say("\n--- capacita' e fit in-sample (FIT B) ---")
    say(f"  foglie per albero: media={leaves.mean():.4f} min={leaves.min()} max={leaves.max()}")
    say(f"  alberi al tetto di 16 foglie: {int((leaves == 16).sum())}/300")
    say(f"  split per albero: media={splits.mean():.4f}")
    say(f"  R2 in-sample = {r2_is:.6f}   RMSE in-sample = {rmse_is:.6f}")

    # --- soglie sulla feature WTI -----------------------------------------
    entity_of = {}
    for label, code in zip(keys["EntityLabel"], keys["Entity"]):
        entity_of.setdefault(label, int(code))
    target = f"R_WTI_L1_x_Entity_{entity_of[TARGET_FEATURE_ENTITY]}"
    thr = np.array([t for _, t in v.thresholds_on(model, names, target)])
    inside = thr[(thr >= WINDOW[0]) & (thr <= WINDOW[1])]
    say(f"\n--- soglie su {target} ---")
    say(f"  split totali: {thr.size}")
    say(f"  dentro la finestra [{WINDOW[0]}, {WINDOW[1]}]: {inside.size}")
    if inside.size:
        say(f"  valori: {np.sort(inside).tolist()}")

    # --- proiezione --------------------------------------------------------
    frame, ratio = v.project(model, x_scen, keys, drivers)
    frame.to_csv(OUT / "scenario_monthly_fitB_bestfirst.csv", index=False)
    ratio.to_csv(OUT / "logratio_fitB_bestfirst.csv", index=False)

    spread, per_scenario = v.spread_2050(ratio, "transition")
    say("\n--- log-ratio ENRG dicembre 2050, componente transition ---")
    say(per_scenario.to_string(float_format=lambda x: f"{x:.6f}"))
    say(f"  spread = {spread:.6f}")
    for comp in ("combined", "physical"):
        s, _ = v.spread_2050(ratio, comp)
        say(f"  spread {comp} = {s:.6f}")

    # --- tabella dei tre spread -------------------------------------------
    table = pd.DataFrame(
        [
            {"politica": "max_depth=4", "spread_transition": SPREAD_DEPTH4},
            {"politica": "best-first (max_leaf_nodes=16)", "spread_transition": spread},
            {"politica": "tesi (MATLAB, budget 15)", "spread_transition": SPREAD_THESIS},
        ]
    ).set_index("politica")
    table.to_csv(OUT / "confronto_spread.csv")
    say("\n--- i tre spread affiancati (FIT B, 237 mesi) ---")
    say(table.to_string(float_format=lambda x: f"{x:.6f}"))

    incapsulato = SPREAD_DEPTH4 < SPREAD_THESIS < spread
    say(f"\n  incapsulamento della proiezione: {'SI' if incapsulato else 'NO'}")

    (OUT / "report_task1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nScritto in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
