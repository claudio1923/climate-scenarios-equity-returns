"""
Task 2 - la politica di crescita MATLAB applicata ai due campioni.

Il builder numpy vive gia' in src/matlab_policy_gb.py (crescita per livelli,
budget di 15 split, annullamento per livello, gain SSE, soglie a meta' fra
valori adiacenti). Questo script non lo riscrive: lo valida e lo usa sui dati
letti dai .mat, che e' il pezzo che mancava.

  1. test di equivalenza obbligatorio: con max_splits = 2**d - 1 su dati dove
     gli alberi riempiono il budget, il builder deve dare alberi identici a
     sklearn con max_depth = d;
  2. FIT A (189 mesi) -> metriche in-sample e out-of-sample, forma degli
     alberi, fingerprint delle importanze;
  3. FIT B (237 mesi) -> proiezione di scenario e spread Energy 2050;
  4. verifica delle quattro previsioni dichiarate prima di guardare i numeri.

Le importanze sono calcolate come quota della riduzione totale dell'errore
quadratico, che e' la definizione usata dalla tesi. Per confrontare like with
like la stessa quota viene ricalcolata anche per i modelli sklearn, invece di
usare feature_importances_ (che normalizza per albero e poi media).

Nessun tuning: gli iperparametri sono quelli della tesi.

Output: results/task2/.
Uso:  python scripts/task2_matlab_policy.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task2"

WINDOW = (0.0399, 1.8447)
SPREAD_THESIS = 1.2198871205
THESIS = dict(r2_is=0.7065, rmse_is=3.8182, r2_oos=0.406, rmse_oos=5.108)

# Previsioni dichiarate prima della stima, da verificare non da inseguire.
PREDICTIONS = dict(splits_mean=(13, 14), r2_is=0.7065, n_zero=1, exmkt_share=0.69)


def load_module(name):
    path = Path(__file__).resolve().parents[1] / "src" / f"{name}.py"
    if not path.exists():
        path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sklearn_sse_importance(model, n_features):
    """
    Quota della riduzione totale di SSE per colonna, per un modello sklearn.

    feature_importances_ normalizza ogni albero e poi media: definizione diversa
    da quella della tesi. Qui si somma la riduzione grezza, come fa il builder.
    """
    importance = np.zeros(n_features, dtype=float)
    for estimator in model.estimators_.ravel():
        t = estimator.tree_
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                continue
            left, right = t.children_left[node], t.children_right[node]
            gain = (
                t.weighted_n_node_samples[node] * t.impurity[node]
                - t.weighted_n_node_samples[left] * t.impurity[left]
                - t.weighted_n_node_samples[right] * t.impurity[right]
            )
            importance[t.feature[node]] += gain
    total = importance.sum()
    return importance / total if total > 0 else importance


def fingerprint(shares, names, say, label):
    series = pd.Series(shares, index=names)
    n_zero = int((series == 0).sum())
    aggregate = [n for n in names if "_x_Entity_" not in n]
    agg = float(series[aggregate].sum())
    exmkt = float(series.get("ExMkt_L0", 0.0))
    say(f"\n  fingerprint {label}")
    say(f"    predittori a importanza zero : {n_zero:>6}   (tesi 1)")
    say(f"    quota termini aggregati      : {agg:>6.1%}   (tesi ~77%)")
    say(f"    quota ExMkt_L0               : {exmkt:>6.1%}   (tesi ~69%)")
    if n_zero:
        say(f"    colonne a zero: {series[series == 0].index.tolist()}")
    return dict(n_zero=n_zero, aggregate_share=agg, exmkt_share=exmkt)


def main():
    v = load_module("verify_237_months")
    mp = load_module("matlab_policy_gb")

    if not v.check_inputs():
        print("\nMi fermo: senza i .mat non stimo nulla.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    # --- 1. test obbligatorio ---------------------------------------------
    say("=" * 70)
    say("1. TEST DI EQUIVALENZA CON SKLEARN (obbligatorio)")
    say("=" * 70)
    for depth in (2, 3, 4):
        for seed in (0, 1):
            mp.check_equivalence_with_sklearn(
                n_rows=2000, n_columns=5, depth=depth, seed=seed, verbose=False
            )
            say(f"  depth={depth} seed={seed}: builder identico a sklearn max_depth={depth}")
    say("  -> il builder e' validato: con budget pieno riproduce l'albero completo.")

    # --- 2. dati -----------------------------------------------------------
    d = v.load_estimation_design()
    names = d["names"]
    x_scen, keys = v.load_scenario_design(d["mask"], d["fn_full"])
    drivers = v.load_risk_free()

    entity_of = {}
    for label, code in zip(keys["EntityLabel"], keys["Entity"]):
        entity_of.setdefault(label, int(code))
    target = f"R_WTI_L1_x_Entity_{entity_of['ENRG_B']}"
    target_index = names.index(target)

    def builder():
        return mp.MatlabPolicyGB(
            max_splits=15, min_leaf=10, min_parent=20,
            learning_rate=0.03, n_estimators=300,
        )

    rows = []

    # --- 3. FIT A ----------------------------------------------------------
    say("\n" + "=" * 70)
    say("2. FIT A (189 mesi) con la politica MATLAB")
    say("=" * 70)
    model_a = builder().fit(d["x_a"], d["y_a"])
    shape_a = model_a.tree_shape()
    r2_is_a, rmse_is_a = v.metrics(d["y_a"], model_a.predict(d["x_a"]))
    r2_oos_a, rmse_oos_a = v.metrics(d["y_test"], model_a.predict(d["x_test"]))
    say(f"  split per albero : media={shape_a['splits'].mean():.4f} "
        f"min={shape_a['splits'].min()} max={shape_a['splits'].max()}")
    say(f"  foglie per albero: media={shape_a['leaves'].mean():.4f} "
        f"max={shape_a['leaves'].max()}")
    say(f"  profondita' max  : {shape_a['depth'].max()}  "
        f"(media {shape_a['depth'].mean():.2f})")
    say(f"  alberi al budget di 15 split: {int((shape_a['splits'] == 15).sum())}/300")
    say(f"\n  R2  in-sample = {r2_is_a:.6f}   (tesi {THESIS['r2_is']})")
    say(f"  RMSE in-sample = {rmse_is_a:.6f}   (tesi {THESIS['rmse_is']})")
    say(f"  R2  out-of-sample = {r2_oos_a:.6f}   (tesi {THESIS['r2_oos']})")
    say(f"  RMSE out-of-sample = {rmse_oos_a:.6f}   (tesi {THESIS['rmse_oos']})")

    shares_a = model_a.feature_importance(d["x_a"], d["y_a"])
    fp_a = fingerprint(shares_a, names, say, "FIT A (politica MATLAB)")

    thr_a = model_a.thresholds_on(target_index)
    inside_a = thr_a[(thr_a >= WINDOW[0]) & (thr_a <= WINDOW[1])]
    say(f"\n  soglie su {target}: {thr_a.size} split, "
        f"{inside_a.size} dentro [{WINDOW[0]}, {WINDOW[1]}]")

    # --- 4. FIT B ----------------------------------------------------------
    say("\n" + "=" * 70)
    say("3. FIT B (237 mesi) con la politica MATLAB, poi proiezione")
    say("=" * 70)
    model_b = builder().fit(d["x_b"], d["y_b"])
    shape_b = model_b.tree_shape()
    r2_is_b, rmse_is_b = v.metrics(d["y_b"], model_b.predict(d["x_b"]))
    say(f"  split per albero : media={shape_b['splits'].mean():.4f}")
    say(f"  foglie per albero: media={shape_b['leaves'].mean():.4f}")
    say(f"  profondita' max  : {shape_b['depth'].max()}")
    say(f"  R2 in-sample = {r2_is_b:.6f}   RMSE in-sample = {rmse_is_b:.6f}")

    shares_b = model_b.feature_importance(d["x_b"], d["y_b"])
    fp_b = fingerprint(shares_b, names, say, "FIT B (politica MATLAB)")

    thr_b = model_b.thresholds_on(target_index)
    inside_b = thr_b[(thr_b >= WINDOW[0]) & (thr_b <= WINDOW[1])]
    say(f"\n  soglie su {target}: {thr_b.size} split, "
        f"{inside_b.size} dentro [{WINDOW[0]}, {WINDOW[1]}]")
    if inside_b.size:
        say(f"    valori: {np.sort(inside_b).tolist()}")

    frame, ratio = v.project(model_b, x_scen, keys, drivers)
    frame.to_csv(OUT / "scenario_monthly_fitB_matlabpolicy.csv", index=False)
    ratio.to_csv(OUT / "logratio_fitB_matlabpolicy.csv", index=False)

    spread, per_scenario = v.spread_2050(ratio, "transition")
    say("\n  log-ratio ENRG dicembre 2050, componente transition:")
    say(per_scenario.to_string(float_format=lambda x: f"{x:.6f}"))
    say(f"  spread = {spread:.6f}   (tesi {SPREAD_THESIS:.6f})")
    for comp in ("combined", "physical"):
        s, _ = v.spread_2050(ratio, comp)
        say(f"  spread {comp} = {s:.6f}")

    # --- 5. tabella delle politiche ---------------------------------------
    say("\n" + "=" * 70)
    say("4. LE TRE POLITICHE AFFIANCATE")
    say("=" * 70)
    sk4 = GradientBoostingRegressor(**v.GB_PARAMS).fit(d["x_a"], d["y_a"])
    probe = dict(v.GB_PARAMS)
    probe["max_depth"] = None
    probe["max_leaf_nodes"] = 16
    skbf = GradientBoostingRegressor(**probe).fit(d["x_a"], d["y_a"])

    for tag, model in (("sklearn max_depth=4", sk4), ("sklearn best-first", skbf)):
        shares = sklearn_sse_importance(model, len(names))
        rows.append(
            {
                "politica": tag,
                "r2_is": v.metrics(d["y_a"], model.predict(d["x_a"]))[0],
                "rmse_is": v.metrics(d["y_a"], model.predict(d["x_a"]))[1],
                "r2_oos": v.metrics(d["y_test"], model.predict(d["x_test"]))[0],
                "rmse_oos": v.metrics(d["y_test"], model.predict(d["x_test"]))[1],
                "splits_mean": float(
                    np.mean([(t.tree_.children_left != -1).sum() for t in model.estimators_.ravel()])
                ),
                "n_zero": int((shares == 0).sum()),
                "exmkt_share": float(shares[names.index("ExMkt_L0")]),
            }
        )
    rows.append(
        {
            "politica": "numpy budget 15 (MATLAB)",
            "r2_is": r2_is_a, "rmse_is": rmse_is_a,
            "r2_oos": r2_oos_a, "rmse_oos": rmse_oos_a,
            "splits_mean": float(shape_a["splits"].mean()),
            "n_zero": fp_a["n_zero"], "exmkt_share": fp_a["exmkt_share"],
        }
    )
    rows.append(
        {
            "politica": "tesi (MATLAB)",
            "r2_is": THESIS["r2_is"], "rmse_is": THESIS["rmse_is"],
            "r2_oos": THESIS["r2_oos"], "rmse_oos": THESIS["rmse_oos"],
            "splits_mean": float("nan"), "n_zero": 1, "exmkt_share": 0.69,
        }
    )
    table = pd.DataFrame(rows).set_index("politica")
    table.to_csv(OUT / "confronto_politiche.csv")
    say(table.to_string(float_format=lambda x: f"{x:.4f}"))

    # --- 6. le quattro previsioni -----------------------------------------
    say("\n" + "=" * 70)
    say("5. LE QUATTRO PREVISIONI DICHIARATE IN ANTICIPO")
    say("=" * 70)
    lo, hi = PREDICTIONS["splits_mean"]
    checks = [
        ("split medi per albero", f"{shape_a['splits'].mean():.2f}", f"{lo}-{hi}",
         lo <= shape_a["splits"].mean() <= hi),
        ("R2 in-sample", f"{r2_is_a:.4f}", f"{PREDICTIONS['r2_is']}",
         abs(r2_is_a - PREDICTIONS["r2_is"]) < 0.005),
        ("predittori a importanza zero", f"{fp_a['n_zero']}", f"~{PREDICTIONS['n_zero']}",
         abs(fp_a["n_zero"] - PREDICTIONS["n_zero"]) <= 1),
        ("quota ExMkt_L0", f"{fp_a['exmkt_share']:.1%}", f"~{PREDICTIONS['exmkt_share']:.0%}",
         abs(fp_a["exmkt_share"] - PREDICTIONS["exmkt_share"]) < 0.02),
    ]
    for name, got, expected, ok in checks:
        say(f"  [{'CENTRATA' if ok else '  MANCATA'}] {name:<30} {got:>10}   atteso {expected}")
    say(f"\n  centrate: {sum(c[3] for c in checks)}/4")

    (OUT / "report_task2.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nScritto in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
