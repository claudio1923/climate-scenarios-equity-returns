"""
Fase 0 - diagnostica prima di scrivere un builder di alberi.

Cinque domande, nessuna delle quali richiede codice nuovo di stima:

  0.1  su quale Component e' calcolato lo 0.2546 della pipeline attuale;
  0.2  i numeri di proiezione del FIT B (237 mesi), che mancavano;
  0.3  fingerprint delle importanze contro le tre cifre riportate in tesi;
  0.4  sonda di capacita': max_leaf_nodes=16 spende tutti e 15 gli split.
       E' best-first, NON la politica MATLAB: serve solo a misurare quanto
       in-sample si recupera usando l'intero budget. Diagnostica, non correzione;
  0.5  quanta estrapolazione c'e' nella proiezione, colonna per colonna.

Riusa il caricamento .mat gia' validato in verify_237_months.py: stessi
assert su ordine delle colonne, maschera K62, allineamento Keys/X.

Nessun tuning. Gli iperparametri restano quelli della tesi; l'unica variante
e' la sonda 0.4, etichettata come tale.

Output: results/fase0/.
Uso:  python scripts/phase0_diagnostics.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fase0"

# Cifre strutturali riportate nella tesi (capitolo 3).
THESIS_N_ZERO_IMPORTANCE = 1          # un solo predittore su 73 a importanza nulla
THESIS_AGGREGATE_SHARE = 0.77         # quota dei termini aggregati
THESIS_EXMKT_L0_SHARE = 0.69          # quota del solo rendimento di mercato contemporaneo
THESIS_SPREAD_TRANSITION = 1.22


def load_verifier():
    """Importa verify_237_months.py senza eseguirne il main."""
    path = Path(__file__).with_name("verify_237_months.py")
    spec = importlib.util.spec_from_file_location("verify_237_months", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 0.1
# ---------------------------------------------------------------------------

def q01_component(say):
    """
    Su quale componente e' calcolato lo 0.2546 prodotto dalla pipeline attuale?

    Legge l'output gia' su disco (results/replication_logratio_2050_comparison.csv),
    senza ristimare: e' proprio quel file che ha generato il numero contestato.
    """
    say("\n" + "=" * 70)
    say("0.1  COMPONENTE SU CUI E' CALCOLATO LO 0.2546")
    say("=" * 70)

    path = ROOT / "results" / "replication_logratio_2050_comparison.csv"
    if not path.exists():
        say(f"  manca {path.relative_to(ROOT)} - salto")
        return None

    frame = pd.read_csv(path)
    energy = frame[frame["Sector"] == "ENRG"]
    rows = []
    for comp in ("transition", "combined", "physical"):
        sel = energy[energy["Component"] == comp].set_index("Scenario")
        rep = sel["LogRatioReplication"]
        thesis = sel["LogRatioThesis"]
        rows.append(
            {
                "Component": comp,
                "spread_replica": float(rep.max() - rep.min()),
                "spread_tesi": float(thesis.max() - thesis.min()),
            }
        )
    table = pd.DataFrame(rows).set_index("Component")
    say(table.to_string(float_format=lambda v: f"{v:.10f}"))

    target = 0.2546338788
    hits = [c for c, v in table["spread_replica"].items() if abs(v - target) < 1e-9]
    say(f"\n  Il valore {target} coincide con: {hits if hits else 'nessuna componente'}")
    say("  Nota: combined vale 0.2546586569, che arrotondato a 4 cifre da' lo stesso")
    say("  0.2547. Le due componenti si distinguono solo alle cifre successive.")
    return table


# ---------------------------------------------------------------------------
# 0.2 / 0.4 / 0.5 usano i modelli
# ---------------------------------------------------------------------------

def fit(params, x, y):
    return GradientBoostingRegressor(**params).fit(x, y)


def leaves_per_tree(model):
    return np.array(
        [int((t.tree_.children_left == -1).sum()) for t in model.estimators_.ravel()]
    )


def q02_fit_b(v, d, model_b, x_scen, keys, drivers, say):
    say("\n" + "=" * 70)
    say("0.2  FIT B (237 mesi): SOGLIE WTI E SPREAD DI PROIEZIONE")
    say("=" * 70)

    entity_of = {}
    for label, code in zip(keys["EntityLabel"], keys["Entity"]):
        entity_of.setdefault(label, int(code))
    enrg_b = entity_of["ENRG_B"]
    target = f"R_WTI_L1_x_Entity_{enrg_b}"
    names = d["names"]
    col = names.index(target)

    scen_rows = (keys["EntityLabel"] == "ENRG_B").to_numpy()
    window = (float(x_scen[scen_rows, col].min()), float(x_scen[scen_rows, col].max()))
    say(f"  feature: {target}   (ENRG-Brown = entita' {enrg_b})")
    say(f"  finestra visitata dallo scenario: [{window[0]:.4f}, {window[1]:.4f}]")
    say(f"  escursione nel campione di stima FIT B: "
        f"[{d['x_b'][:, col].min():.4f}, {d['x_b'][:, col].max():.4f}]")

    thr = np.array([t for _, t in v.thresholds_on(model_b, names, target)])
    inside = thr[(thr >= window[0]) & (thr <= window[1])]
    say(f"\n  split totali su questa feature: {thr.size}")
    say(f"  soglie dentro la finestra: {inside.size}  -> {np.sort(inside).tolist()}")
    say("\n  tutte le soglie ordinate:")
    say("  " + np.array2string(np.sort(thr), precision=4, max_line_width=100))

    _, ratio = v.project(model_b, x_scen, keys, drivers)
    say("\n  log-ratio ENRG a dicembre 2050, componente transition:")
    spread, per_scenario = v.spread_2050(ratio, "transition")
    say(per_scenario.to_string(float_format=lambda x: f"{x:.6f}"))
    say(f"  spread FIT B = {spread:.6f}   (tesi {THESIS_SPREAD_TRANSITION})")
    for comp in ("combined", "physical"):
        s, _ = v.spread_2050(ratio, comp)
        say(f"  spread {comp} = {s:.6f}")
    return ratio


def q03_importance(model, names, say, label):
    say("\n" + "=" * 70)
    say(f"0.3  FINGERPRINT DELLE IMPORTANZE - {label}")
    say("=" * 70)

    imp = pd.Series(model.feature_importances_, index=names)
    total = float(imp.sum())
    n_zero = int((imp == 0).sum())
    aggregate = [n for n in names if "_x_Entity_" not in n]
    agg_share = float(imp[aggregate].sum() / total)
    exmkt = float(imp.get("ExMkt_L0", 0.0) / total)

    say(f"  predittori a importanza esattamente zero : {n_zero:>6}   (tesi {THESIS_N_ZERO_IMPORTANCE})")
    say(f"  quota dei termini aggregati ({len(aggregate)} colonne): {agg_share:>6.1%}   "
        f"(tesi ~{THESIS_AGGREGATE_SHARE:.0%})")
    say(f"  quota di ExMkt_L0 da solo                : {exmkt:>6.1%}   "
        f"(tesi ~{THESIS_EXMKT_L0_SHARE:.0%})")
    say("\n  prime 10 feature per importanza normalizzata:")
    top = (imp / total).sort_values(ascending=False).head(10)
    say(top.to_string(float_format=lambda x: f"{x:.4%}"))
    if n_zero:
        say(f"\n  colonne a zero: {imp[imp == 0].index.tolist()}")
    return dict(n_zero=n_zero, aggregate_share=agg_share, exmkt_share=exmkt)


def q04_capacity(v, d, base_params, say):
    say("\n" + "=" * 70)
    say("0.4  SONDA DI CAPACITA' (DIAGNOSTICA, NON CORREZIONE)")
    say("=" * 70)
    say("  max_leaf_nodes=16, max_depth=None: best-first, quindi NON la politica")
    say("  MATLAB. Spende pero' tutti e 15 gli split, cosi' si misura quanto")
    say("  in-sample si recupera togliendo il taglio a profondita' 4.")

    probe_params = dict(base_params)
    probe_params.pop("max_depth")
    probe_params["max_depth"] = None
    probe_params["max_leaf_nodes"] = 16

    rows = []
    for tag, params in (("FIT A - max_depth=4", base_params),
                        ("FIT A - max_leaf_nodes=16", probe_params)):
        model = fit(params, d["x_a"], d["y_a"])
        r2_is, rmse_is = v.metrics(d["y_a"], model.predict(d["x_a"]))
        r2_oos, rmse_oos = v.metrics(d["y_test"], model.predict(d["x_test"]))
        leaves = leaves_per_tree(model)
        rows.append(
            {
                "modello": tag,
                "r2_is": r2_is,
                "rmse_is": rmse_is,
                "r2_oos": r2_oos,
                "rmse_oos": rmse_oos,
                "foglie_media": float(leaves.mean()),
                "foglie_max": int(leaves.max()),
                "alberi_al_tetto": int((leaves == 16).sum()),
            }
        )
    rows.append(
        {
            "modello": "tesi (MATLAB)",
            "r2_is": 0.7065,
            "rmse_is": 3.8182,
            "r2_oos": 0.406,
            "rmse_oos": 5.108,
            "foglie_media": float("nan"),
            "foglie_max": float("nan"),
            "alberi_al_tetto": float("nan"),
        }
    )
    table = pd.DataFrame(rows).set_index("modello")
    say("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))
    return table


def q05_extrapolation(d, x_scen, names, say):
    say("\n" + "=" * 70)
    say("0.5  ESTRAPOLAZIONE: RIGHE DI SCENARIO FUORI DALL'INTERVALLO DI STIMA")
    say("=" * 70)
    say(f"  su {x_scen.shape[0]} righe di scenario, per ciascuna delle 73 colonne.")

    rows = []
    for tag, x_fit in (("A", d["x_a"]), ("B", d["x_b"])):
        lo = x_fit.min(axis=0)
        hi = x_fit.max(axis=0)
        below = (x_scen < lo).sum(axis=0)
        above = (x_scen > hi).sum(axis=0)
        rows.append(
            pd.DataFrame(
                {
                    "feature": names,
                    f"sotto_{tag}": below,
                    f"sopra_{tag}": above,
                    f"fuori_{tag}": below + above,
                }
            ).set_index("feature")
        )
    table = pd.concat(rows, axis=1)
    n_cols_out = {t: int((table[f"fuori_{t}"] > 0).sum()) for t in ("A", "B")}
    say(f"\n  colonne con almeno una riga fuori intervallo: "
        f"FIT A = {n_cols_out['A']}/73, FIT B = {n_cols_out['B']}/73")
    say(f"  righe-colonna fuori intervallo in totale: "
        f"FIT A = {int(table['fuori_A'].sum())}, FIT B = {int(table['fuori_B'].sum())}")

    worst = table.sort_values("fuori_A", ascending=False)
    worst = worst[worst[["fuori_A", "fuori_B"]].max(axis=1) > 0]
    if len(worst):
        say("\n  colonne interessate:")
        say(worst.to_string())
    else:
        say("\n  nessuna colonna estrapola: lo scenario resta dentro il supporto di stima.")
    return table


# ---------------------------------------------------------------------------

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

    say("FASE 0 - diagnostica")
    say(f"iperparametri: {v.GB_PARAMS}")

    q01_component(say)

    d = v.load_estimation_design()
    names = d["names"]
    x_scen, keys = v.load_scenario_design(d["mask"], d["fn_full"])
    drivers = v.load_risk_free()

    model_a = fit(v.GB_PARAMS, d["x_a"], d["y_a"])
    model_b = fit(v.GB_PARAMS, d["x_b"], d["y_b"])

    ratio_b = q02_fit_b(v, d, model_b, x_scen, keys, drivers, say)
    ratio_b.to_csv(OUT / "logratio_fitB.csv", index=False)

    fp_a = q03_importance(model_a, names, say, "FIT A (189 mesi)")
    fp_b = q03_importance(model_b, names, say, "FIT B (237 mesi)")
    pd.DataFrame([dict(fit="A", **fp_a), dict(fit="B", **fp_b)]).to_csv(
        OUT / "fingerprint_importanze.csv", index=False
    )

    q04_capacity(v, d, v.GB_PARAMS, say).to_csv(OUT / "sonda_capacita.csv")
    q05_extrapolation(d, x_scen, names, say).to_csv(OUT / "estrapolazione.csv")

    (OUT / "report_fase0.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nScritto in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
