"""
Verifica: il refit sul campione pieno (237 mesi) fa comparire soglie nella
finestra WTI?

Contesto. In MATLAB ci sono due fit distinti, stessi iperparametri, campioni
diversi:

    refit_finale_K62.m   -> refit su train+val (189 mesi, 4158 righe).
                            Serve solo alle metriche sul test sigillato.
    s3_K62_leaf10.m      -> refit su train+val+test (237 mesi, 5214 righe),
                            "Training refit 100%". E' il modello che produce
                            la proiezione di scenario 2025-2050.

La replica Python corrente usa 189 mesi anche per proiettare. Questo script
stima entrambi i modelli, identici negli iperparametri, e li confronta sulla
diagnostica delle soglie e sulla proiezione.

Lettura dell'esito, decisa prima di guardare i numeri:
    soglie WTI presenti nel FIT B  -> causa identificata;
    soglie ancora a zero           -> ipotesi caduta, il problema e' a monte.

Nessun tuning, nessuna grid search: gli iperparametri sono fissi e identici
per i due fit.

Input: i .mat originali in C:\\Users\\Claud\\Desktop\\modello (v7.3 = HDF5).
Output: results/verifica_237_mesi/.

Uso:  python scripts/verify_237_months.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("Serve h5py per leggere i .mat v7.3:  pip install h5py")

MATLAB_DIR = Path(r"C:\Users\Claud\Desktop\modello")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verifica_237_mesi"

BASE_FILE = MATLAB_DIR / "Pipeline_Base.mat"
SCREEN_FILE = MATLAB_DIR / "Pipeline_GB_Screening.mat"
DESIGN_FILE = MATLAB_DIR / "Scenario_Design_Full552_2005_2024.mat"
DRIVERS_FILE = MATLAB_DIR / "Scenario_Drivers_2005_2024.mat"

# Iperparametri del capitolo 3, identici per i due fit. Non toccare.
#   MaxNumSplits=15  ~ max_depth=4   (differenza di politica, vedi report)
#   MinLeafSize=10   -> min_samples_leaf=10
#   MinParentSize = max(MinParentSize, 2*MinLeafSize) = 20 -> min_samples_split=20
#   NumVariablesToSample='all' -> max_features=None
#   subsample=1.0 -> deterministico, random_state ininfluente
GB_PARAMS = dict(
    max_depth=4,
    min_samples_leaf=10,
    min_samples_split=20,
    learning_rate=0.03,
    n_estimators=300,
    subsample=1.0,
    max_features=None,
    random_state=42,
)

LEAF_BUDGET = 2 ** GB_PARAMS["max_depth"]          # 16
SPLIT_BUDGET = LEAF_BUDGET - 1                     # 15

REFERENCE_SCENARIOS = ("current policies", "baseline")
THESIS_SPREAD_ENERGY_2050 = 1.22
ENERGY_SECTOR = "ENRG"


# ---------------------------------------------------------------------------
# lettura .mat v7.3
# ---------------------------------------------------------------------------

def mclass(obj) -> str:
    c = obj.attrs.get("MATLAB_class", b"")
    return c.decode() if isinstance(c, bytes) else str(c)


def decode_string_array(blob):
    """
    Decodifica un array `string` MATLAB serializzato come blob uint64.

    Layout: [1, ndims, dims..., lunghezze per elemento..., UTF-16 impacchettato].
    Restituisce None se il blob non ha questa forma.
    """
    d = np.asarray(blob).ravel().astype(np.uint64)
    if d.size < 4 or d[0] != 1:
        return None
    ndims = int(d[1])
    dims = [int(x) for x in d[2:2 + ndims]]
    n = int(np.prod(dims))
    pos = 2 + ndims
    if pos + n > d.size:
        return None
    lengths = [int(x) for x in d[pos:pos + n]]
    pos += n
    chars = d[pos:].view(np.uint16)
    if sum(lengths) > chars.size:
        return None
    out, q = [], 0
    for length in lengths:
        out.append("".join(chr(c) for c in chars[q:q + length]))
        q += length
    return out


def string_arrays(refs, n_elements=None):
    """Tutti i blob di #refs# decodificabili come string array, per chiave."""
    found = {}
    for key in refs:
        obj = refs[key]
        if not isinstance(obj, h5py.Dataset) or mclass(obj) != "uint64":
            continue
        values = decode_string_array(obj)
        if values is None:
            continue
        if n_elements is not None and len(values) != n_elements:
            continue
        found[key] = values
    return found


def pick_string_column(refs, n_elements, must_contain, label):
    """
    Individua per contenuto una colonna string di una tabella MATLAB.

    Le colonne non numeriche di una tabella sono handle MCOS opachi: non si
    risolvono con h5py. I dati pero' sono nel file, quindi li si identifica
    per contenuto, con l'assert che il candidato sia unico.
    """
    candidates = {
        k: v
        for k, v in string_arrays(refs, n_elements).items()
        if must_contain in set(v)
    }
    if len(candidates) != 1:
        raise AssertionError(
            f"{label}: attesa 1 colonna string con {n_elements} elementi "
            f"contenente {must_contain!r}, trovate {len(candidates)} "
            f"({sorted(candidates)})"
        )
    return next(iter(candidates.values()))


def pick_canonical_552(refs, label):
    """
    Sceglie FN_full fra i blob da 552 nomi.

    Il file ne contiene due, FN_full e FN_sorted (ordinato per importanza).
    Solo FN_full ha la struttura canonica [24 effetti principali ; 528
    interazioni], quindi il criterio e' strutturale, non posizionale.
    """
    candidates = {
        k: v
        for k, v in string_arrays(refs, 552).items()
        if not any("_x_Entity_" in n for n in v[:24])
        and all("_x_Entity_" in n for n in v[24:])
    }
    if len(candidates) != 1:
        raise AssertionError(
            f"{label}: atteso 1 elenco canonico di 552 nomi, "
            f"trovati {len(candidates)} ({sorted(candidates)})"
        )
    return next(iter(candidates.values()))


def cell_of_refs(f, dataset):
    """Dereferenzia una cell MATLAB, restituendo la lista di oggetti puntati."""
    return [f[ref] for ref in np.array(dataset).ravel()]


def find_table_columns(f, varnames_wanted):
    """
    Trova la coppia (cell dei nomi variabile, cell dei dati) di una tabella e
    restituisce nome -> dataset. Le colonne opache restano handle uint32.
    """
    refs = f["#refs#"]
    names_key = None
    for key in refs:
        obj = refs[key]
        if not isinstance(obj, h5py.Dataset) or mclass(obj) != "cell":
            continue
        try:
            items = cell_of_refs(f, obj)
        except Exception:  # noqa: BLE001
            continue
        if not items or not all(mclass(o) == "char" for o in items):
            continue
        names = ["".join(chr(x) for x in np.array(o).ravel()) for o in items]
        if varnames_wanted.issubset(set(names)):
            names_key = (key, names)
            break
    if names_key is None:
        raise AssertionError(f"nessuna cell di nomi variabile contiene {varnames_wanted}")

    key, names = names_key
    size = len(names)
    for other in refs:
        obj = refs[other]
        if other == key or not isinstance(obj, h5py.Dataset):
            continue
        if mclass(obj) != "cell" or obj.size != size:
            continue
        items = cell_of_refs(f, obj)
        if all(mclass(o) == "char" for o in items):
            continue
        return dict(zip(names, items))
    raise AssertionError("cell dei dati della tabella non trovata")


def unreferenced_double(f, referenced, n_elements, label):
    """
    La colonna datetime di una tabella e' un handle opaco; i millisecondi sono
    un double della stessa lunghezza che nessun'altra colonna referenzia.
    """
    used = {o.name for o in referenced}
    refs = f["#refs#"]
    hits = [
        k
        for k in refs
        if isinstance(refs[k], h5py.Dataset)
        and mclass(refs[k]) == "double"
        and refs[k].size == n_elements
        and refs[k].name not in used
    ]
    if len(hits) != 1:
        raise AssertionError(
            f"{label}: atteso 1 double non referenziato di {n_elements} elementi, "
            f"trovati {len(hits)} ({hits})"
        )
    return np.array(refs[hits[0]]).ravel()


def as_dates(millis):
    return pd.to_datetime(np.asarray(millis), unit="ms")


# ---------------------------------------------------------------------------
# controllo di accessibilita' degli input
# ---------------------------------------------------------------------------

def check_inputs():
    files = (BASE_FILE, SCREEN_FILE, DESIGN_FILE, DRIVERS_FILE)
    missing = [p for p in files if not p.exists()]
    if missing:
        print("INPUT MANCANTI - non stimo nulla:")
        for p in missing:
            print(f"  {p}")
        return False
    unreadable = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                _ = list(f.keys())
        except Exception as exc:  # noqa: BLE001
            unreadable.append((p, exc))
    if unreadable:
        print("INPUT NON LEGGIBILI - non stimo nulla:")
        for p, exc in unreadable:
            print(f"  {p}: {exc}")
        return False
    print("Input: tutti e quattro i .mat presenti e leggibili (MATLAB v7.3 / HDF5).")
    return True


# ---------------------------------------------------------------------------
# costruzione del design di stima
# ---------------------------------------------------------------------------

def unique_stable(values):
    """unique(x,'stable') di MATLAB: ordine di prima apparizione."""
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return np.array(out)


def build_interactions(x, entity, unique_ent, base_names):
    """
    buildInteractions di MATLAB, ordine delle colonne incluso.

        for f = 1:nF, for e = 1:nE   ->  driver esterno, entita' interna
        Xout = [Xin, Xint]

    L'ordine e' critico: la maschera K62 indicizza posizioni, non nomi.
    """
    n_rows, n_feat = x.shape
    n_ent = len(unique_ent)
    dummies = np.stack([(entity == e).astype(float) for e in unique_ent], axis=1)
    inter = np.empty((n_rows, n_feat * n_ent), dtype=float)
    names = []
    col = 0
    for j in range(n_feat):
        for e in range(n_ent):
            inter[:, col] = x[:, j] * dummies[:, e]
            names.append(f"{base_names[j]}_x_Entity_{int(unique_ent[e])}")
            col += 1
    return np.hstack([x, inter]), list(base_names) + names


def load_estimation_design():
    with h5py.File(SCREEN_FILE, "r") as f:
        fn_full = pick_canonical_552(f["#refs#"], "FN_full")
        mask = np.array(f["Sets"]["K62"]["mask"]).ravel().astype(bool)
        k62_blobs = list(string_arrays(f["#refs#"], int(mask.sum())).values())
        k62_k = float(np.array(f["Sets"]["K62"]["K"]).ravel()[0])
        k62_s = float(np.array(f["Sets"]["K62"]["S"]).ravel()[0])

    assert len(fn_full) == 552, len(fn_full)
    assert mask.size == 552, mask.size
    assert int(mask.sum()) == 73, int(mask.sum())
    assert (k62_k, k62_s) == (62.0, 73.0), (k62_k, k62_s)

    with h5py.File(BASE_FILE, "r") as f:
        # HDF5 conserva i dati trasposti rispetto a MATLAB.
        x_tr = np.array(f["X_tr70"]).T
        x_va = np.array(f["X_va10"]).T
        x_te = np.array(f["X_test"]).T
        y_tr = np.array(f["Y_tr70"]).ravel()
        y_va = np.array(f["Y_va10"]).ravel()
        y_te = np.array(f["Y_test"]).ravel()
        e_tr = np.array(f["Entity_tr70"]).ravel()
        e_va = np.array(f["Entity_va10"]).ravel()
        e_te = np.array(f["Entity_test"]).ravel()
        e_train = np.array(f["Entity_train"]).ravel()

    base_names = fn_full[:24]
    unique_ent = unique_stable(e_train)
    assert len(unique_ent) == 22, len(unique_ent)

    blocks, names = [], None
    for x, e in ((x_tr, e_tr), (x_va, e_va), (x_te, e_te)):
        assert x.shape[1] == 24, x.shape
        full, names = build_interactions(x, e, unique_ent, base_names)
        assert full.shape[1] == 552, full.shape
        blocks.append(full)

    assert names == fn_full, "ordine delle interazioni != FN_full"

    kept = [n for n, m in zip(fn_full, mask) if m]
    assert len(kept) == 73, len(kept)
    # Sets.K62.names e' in ordine di importanza (deriva da FN_sorted), la
    # maschera preserva l'ordine canonico: stesso insieme, ordine diverso.
    # MATLAB indicizza con X(:,mask), quindi vale l'ordine canonico.
    assert any(set(kept) == set(blob) for blob in k62_blobs), \
        "i 73 nomi mascherati non coincidono con Sets.K62.names"

    x_tr_k, x_va_k, x_te_k = (b[:, mask] for b in blocks)
    assert x_tr_k.shape == (3630, 73), x_tr_k.shape
    assert x_va_k.shape == (528, 73), x_va_k.shape
    assert x_te_k.shape == (1056, 73), x_te_k.shape

    x_a = np.vstack([x_tr_k, x_va_k])
    y_a = np.concatenate([y_tr, y_va])
    x_b = np.vstack([x_tr_k, x_va_k, x_te_k])
    y_b = np.concatenate([y_tr, y_va, y_te])
    assert x_a.shape == (4158, 73), x_a.shape
    assert x_b.shape == (5214, 73), x_b.shape

    return dict(
        fn_full=fn_full,
        mask=mask,
        names=kept,
        x_a=x_a, y_a=y_a,
        x_b=x_b, y_b=y_b,
        x_test=x_te_k, y_test=y_te,
    )


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------

def load_scenario_design(mask, fn_full):
    idx = np.flatnonzero(mask)
    with h5py.File(DESIGN_FILE, "r") as f:
        n_rows = f["X"].shape[1]
        design_names = pick_canonical_552(f["#refs#"], "FeatureNames scenario")
        assert design_names == fn_full, "FeatureNames scenario != FN_full"

        x = f["X"][idx, :].T
        cols = find_table_columns(
            f, {"Entity", "EntityLabel", "Green", "Scenario", "Component", "Date"}
        )
        entity = np.array(cols["Entity"]).ravel()
        green = np.array(cols["Green"]).ravel().astype(bool)
        date_ms = unreferenced_double(f, cols.values(), n_rows, "Keys.Date")
        refs = f["#refs#"]
        keys = pd.DataFrame(
            {
                "Entity": entity.astype(int),
                "EntityLabel": pick_string_column(refs, n_rows, "ENRG_B", "Keys.EntityLabel"),
                "Green": green,
                "Scenario": pick_string_column(refs, n_rows, "Net Zero 2050", "Keys.Scenario"),
                "Component": pick_string_column(refs, n_rows, "transition", "Keys.Component"),
                "Date": as_dates(date_ms),
            }
        )
    assert x.shape == (n_rows, 73), x.shape
    assert len(keys) == n_rows, (len(keys), n_rows)
    return x, keys


def load_risk_free():
    with h5py.File(DRIVERS_FILE, "r") as f:
        cols = find_table_columns(f, {"Scenario", "Component", "Date", "RF"})
        rf = np.array(cols["RF"]).ravel()
        n_rows = rf.size
        date_ms = unreferenced_double(f, cols.values(), n_rows, "Drivers.Date")
        refs = f["#refs#"]
        drivers = pd.DataFrame(
            {
                "Scenario": pick_string_column(refs, n_rows, "Net Zero 2050", "Drivers.Scenario"),
                "Component": pick_string_column(refs, n_rows, "transition", "Drivers.Component"),
                "Date": as_dates(date_ms),
                "RF": rf,
            }
        )
    duplicated = drivers.duplicated(["Scenario", "Component", "Date"]).any()
    assert not duplicated, "chiavi (Scenario, Component, Date) non uniche nella tabella driver"
    return drivers


def project(model, x_scen, keys, drivers):
    """
    Proiezione secondo s3_K62_leaf10.m + s4_cumulate_logratio.m:
      yhat -> togli gli scenari di riferimento -> join RF -> total = yhat + RF
      -> cumulato composto prod(1 + r/100) -> log-ratio Green/Brown per settore.
    """
    frame = keys.copy()
    frame["yhat"] = model.predict(x_scen)

    lowered = frame["Scenario"].str.lower()
    is_reference = np.zeros(len(frame), dtype=bool)
    for token in REFERENCE_SCENARIOS:
        is_reference |= lowered.str.contains(token, regex=False).to_numpy()
    frame = frame.loc[~is_reference].copy()

    before = len(frame)
    frame = frame.merge(drivers, on=["Scenario", "Component", "Date"], how="left")
    assert len(frame) == before, "il join di RF ha duplicato righe"
    assert frame["RF"].notna().all(), "RF mancante per alcune combinazioni"

    frame = frame.sort_values(
        ["Scenario", "Component", "EntityLabel", "Date"]
    ).reset_index(drop=True)
    frame["total"] = frame["yhat"] + frame["RF"]
    frame["Sector"] = frame["EntityLabel"].str.rsplit("_", n=1).str[0]
    frame["Leg"] = frame["EntityLabel"].str.rsplit("_", n=1).str[1]

    group = ["Scenario", "Component", "EntityLabel"]
    frame["Cum"] = frame.groupby(group)["total"].transform(
        lambda s: (1 + s / 100).cumprod()
    )

    keycols = ["Scenario", "Component", "Sector", "Date"]
    green = frame.loc[frame["Leg"] == "G"].set_index(keycols)["Cum"]
    brown = frame.loc[frame["Leg"] == "B"].set_index(keycols)["Cum"]
    ratio = (np.log(green) - np.log(brown)).rename("LogRatio").reset_index()
    return frame, ratio.sort_values(keycols).reset_index(drop=True)


def spread_2050(ratio, component):
    end = ratio["Date"].max()
    sel = ratio[
        (ratio["Date"] == end)
        & (ratio["Sector"] == ENERGY_SECTOR)
        & (ratio["Component"] == component)
    ]
    per_scenario = sel.set_index("Scenario")["LogRatio"].sort_index()
    if not len(per_scenario):
        return float("nan"), per_scenario
    return float(per_scenario.max() - per_scenario.min()), per_scenario


# ---------------------------------------------------------------------------
# diagnostica sugli alberi
# ---------------------------------------------------------------------------

def tree_stats(model):
    leaves, splits = [], []
    for tree in model.estimators_.ravel():
        t = tree.tree_
        is_leaf = t.children_left == -1
        leaves.append(int(is_leaf.sum()))
        splits.append(int((~is_leaf).sum()))
    leaves, splits = np.array(leaves), np.array(splits)
    return dict(
        leaves_min=int(leaves.min()),
        leaves_mean=float(leaves.mean()),
        leaves_max=int(leaves.max()),
        trees_at_budget=int((leaves == LEAF_BUDGET).sum()),
        splits_mean=float(splits.mean()),
        splits_max=int(splits.max()),
    )


def thresholds_on(model, feature_names, target):
    col = feature_names.index(target)
    out = []
    for k, tree in enumerate(model.estimators_.ravel()):
        t = tree.tree_
        for node in np.flatnonzero(t.feature == col):
            out.append((k, float(t.threshold[node])))
    return out


def thesis_energy_2050():
    """Log-ratio ENRG a dicembre 2050 come esportato dalla tesi (MATLAB)."""
    path = ROOT / "results" / "logratio_green_brown.csv"
    if not path.exists():
        return None
    wide = pd.read_csv(path)
    sel = wide[wide["Sector"] == ENERGY_SECTOR]
    return sel.set_index(["Component", "Scenario"])["d2050_12"]


def secondary_checks(d, x_scen, keys, names, say):
    """
    Controlli di contorno, non decisivi per l'ipotesi ma necessari a
    interpretare i numeri.
    """
    say("\n================ CONTROLLI DI CONTORNO ================")

    say("\n[a] Colonne Wind_ANOM nel set K62 e loro escursione in proiezione")
    wind = [n for n in names if n.startswith("Wind_ANOM")]
    say(f"    {len(wind)} colonne su 73: {wind}")
    for n in wind:
        c = names.index(n)
        v = x_scen[:, c]
        say(f"    {n:<26} scenario std={v.std():.6g}  stima(237m) std={d['x_b'][:, c].std():.4g}")
    inert = all(x_scen[:, names.index(n)].std() == 0 for n in wind)
    say(f"    -> in proiezione sono {'inerti (varianza nulla)' if inert else 'attive'}.")

    say("\n[b] Ampiezza della finestra WTI: stima contro scenario")
    for n in ("R_WTI_L0", "R_WTI_L1", "R_WTI_L2"):
        if n not in names:
            continue
        c = names.index(n)
        v, e = x_scen[:, c], d["x_b"][:, c]
        say(f"    {n}: scenario [{v.min():.4f}, {v.max():.4f}]   "
            f"stima 237m [{e.min():.4f}, {e.max():.4f}]")

    say("\n[c] Componente physical: il design distingue ENRG_G da ENRG_B?")
    for comp in ("physical", "transition"):
        sel = ((keys["Component"] == comp) & (keys["Scenario"] == "Net Zero 2050")).to_numpy()
        g = x_scen[sel & (keys["EntityLabel"] == "ENRG_G").to_numpy()]
        b = x_scen[sel & (keys["EntityLabel"] == "ENRG_B").to_numpy()]
        diff = np.abs(g - b).max(axis=0)
        say(f"    {comp:<11}: {int((diff > 1e-12).sum())} colonne su 73 differiscono "
            f"(max diff = {diff.max():.6f})")


def metrics(y_true, y_pred):
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - sse / sst, float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------

def main():
    if not check_inputs():
        print("\nMi fermo qui: il codice e' scritto, ma senza gli input non stimo nulla.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg)
        lines.append(msg)

    d = load_estimation_design()
    names = d["names"]
    say(f"Design: 552 candidate -> maschera K62 -> {len(names)} colonne. "
        "Ordine e nomi verificati contro FN_full e Sets.K62.names.")
    say(f"FIT A: {d['x_a'].shape[0]} righe (189 mesi)    "
        f"FIT B: {d['x_b'].shape[0]} righe (237 mesi)")

    model_a = GradientBoostingRegressor(**GB_PARAMS).fit(d["x_a"], d["y_a"])
    model_b = GradientBoostingRegressor(**GB_PARAMS).fit(d["x_b"], d["y_b"])

    f0_a = float(np.ravel(model_a.init_.constant_)[0])
    f0_b = float(np.ravel(model_b.init_.constant_)[0])
    assert np.isclose(f0_a, d["y_a"].mean()), (f0_a, d["y_a"].mean())
    assert np.isclose(f0_b, d["y_b"].mean()), (f0_b, d["y_b"].mean())
    say(f"F0 (media di y): FIT A = {f0_a:.6f}   FIT B = {f0_b:.6f}")

    r2_a, rmse_a = metrics(d["y_test"], model_a.predict(d["x_test"]))
    say(f"FIT A sul test sigillato: R2 = {r2_a:.5f}   RMSE = {rmse_a:.5f}")

    x_scen, keys = load_scenario_design(d["mask"], d["fn_full"])
    entity_of = {}
    for label, code in zip(keys["EntityLabel"], keys["Entity"]):
        entity_of.setdefault(label, int(code))
    enrg_b = entity_of["ENRG_B"]
    target = f"R_WTI_L1_x_Entity_{enrg_b}"
    say(f"\nENRG-Brown = entita' {enrg_b}; feature osservata: {target}")
    assert target in names, f"{target} non e' nel set K62"

    col = names.index(target)
    scen_rows = (keys["EntityLabel"] == "ENRG_B").to_numpy()
    window = (float(x_scen[scen_rows, col].min()), float(x_scen[scen_rows, col].max()))
    range_a = (float(d["x_a"][:, col].min()), float(d["x_a"][:, col].max()))
    range_b = (float(d["x_b"][:, col].min()), float(d["x_b"][:, col].max()))
    say(f"  escursione in stima FIT A (189m): [{range_a[0]:.4f}, {range_a[1]:.4f}]")
    say(f"  escursione in stima FIT B (237m): [{range_b[0]:.4f}, {range_b[1]:.4f}]")
    say(f"  finestra di scenario (righe ENRG_B): [{window[0]:.4f}, {window[1]:.4f}]")

    drivers = load_risk_free()
    say(f"\nDriver di scenario: {len(drivers)} righe; "
        "RF agganciato su [Scenario, Component, Date].")

    rows, detail = [], {}
    for tag, model, x_fit in (("A", model_a, d["x_a"]), ("B", model_b, d["x_b"])):
        stats = tree_stats(model)
        thr = thresholds_on(model, names, target)
        values = np.array([t for _, t in thr]) if thr else np.array([])
        in_window = int(((values >= window[0]) & (values <= window[1])).sum()) if values.size else 0
        beyond_a = int((values > range_a[1]).sum()) if values.size else 0

        frame, ratio = project(model, x_scen, keys, drivers)
        frame.to_csv(OUT / f"scenario_monthly_fit{tag}.csv", index=False)
        ratio.to_csv(OUT / f"logratio_green_brown_fit{tag}.csv", index=False)

        entry = dict(
            fit=tag,
            n_rows=int(x_fit.shape[0]),
            **stats,
            n_splits_wti=len(thr),
            n_soglie_in_finestra=in_window,
            n_soglie_oltre_max_A=beyond_a,
        )
        for comp in ("transition", "combined", "physical"):
            spread, per_scenario = spread_2050(ratio, comp)
            entry[f"spread_ENRG_2050_{comp}"] = spread
            detail[(tag, comp)] = per_scenario
        entry["r2_oos"] = r2_a if tag == "A" else float("nan")
        entry["rmse_oos"] = rmse_a if tag == "A" else float("nan")
        rows.append(entry)
        detail[(tag, "thresholds")] = values

    table = pd.DataFrame(rows).set_index("fit").T
    table.to_csv(OUT / "confronto_fitA_fitB.csv")

    say("\n================ CONFRONTO FIT A / FIT B ================")
    say(table.to_string())

    for tag in ("A", "B"):
        values = detail[(tag, "thresholds")]
        say(f"\nSoglie su {target} - FIT {tag} ({values.size} split):")
        if values.size:
            say("  " + np.array2string(np.sort(values), precision=4, max_line_width=100))
        else:
            say("  nessuna")

    thesis = thesis_energy_2050()
    say("\n---- log-ratio ENRG a dicembre 2050, per scenario ----")
    for comp in ("transition", "combined", "physical"):
        say(f"\n[{comp}]")
        merged = pd.DataFrame({f"FIT {t}": detail[(t, comp)] for t in ("A", "B")})
        if thesis is not None and comp in thesis.index.get_level_values(0):
            merged["Tesi"] = thesis.loc[comp]
        say(merged.to_string())
        for t in ("A", "B"):
            s = detail[(t, comp)]
            if len(s):
                say(f"  spread FIT {t} = {float(s.max() - s.min()):.6f}")
        if "Tesi" in merged:
            say(f"  spread tesi  = {float(merged['Tesi'].max() - merged['Tesi'].min()):.6f}")
    say(f"\nRiferimento dichiarato nella tesi (transition): {THESIS_SPREAD_ENERGY_2050}")

    secondary_checks(d, x_scen, keys, names, say)

    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nScritto in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
