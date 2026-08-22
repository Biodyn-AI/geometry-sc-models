"""WHAT MAKES A TISSUE CARRY CHROMOSOME-SCALE CO-EXPRESSION?

The retraction (see coocc_final.py) left one genuinely new fact standing: a shallow factorisation of fetal gut
recovers chromosome identity at 0.692, while adult Tabula Sapiens kidney sits at 0.081. The obvious reading is
"developing tissue". But fetal tissue is also intensely PROLIFERATIVE, and replication-timing domains are
megabase-scale and co-express through S phase -- so development and proliferation are confounded in those two
panels, and picking the wrong one would repeat the error that produced the retraction.

Two panels break the confound:
  * gut_setty_schema      -- ADULT GUT. Same tissue as fetal gut, different developmental stage.
                             Isolates development at fixed tissue identity.
  * setty19_cd34_bm       -- ADULT CD34+ HSPCs. Adult, but among the most proliferative normal cells there are.
                             Isolates proliferation at fixed developmental stage.

Predictions that distinguish the hypotheses:
  DEVELOPMENT drives it -> fetal gut HIGH; adult gut, CD34, lung, kidney, aging all LOW.
  PROLIFERATION drives it -> fetal gut HIGH and CD34 HIGH; the quiescent adult panels LOW.

Every panel is matched on cell count, top-K, dimensionality, probe and split, so none of those can explain a
difference. A per-panel proliferation index (detection rate of a canonical cell-cycle gene set) is computed so
the relationship can be reported as a correlation rather than an anecdote.

Run: ../../.venv/bin/python -u panel_sweep.py
Out: results/panel_sweep.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
from scipy import sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from model_scale import BLOCK
from genome_wide import coords, AUTOSOMES
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

N_CELLS, TOPK, DIM, SEED = 3500, 2048, 256, 0
DATA = f"{_DATA}"
CYCLE = ["MKI67", "TOP2A", "PCNA", "CCNB1", "CDK1", "AURKB", "BIRC5", "TYMS", "RRM2", "MCM2",
         "CCNA2", "UBE2C", "NUSAP1", "TPX2", "CENPF"]

PANELS = [
    ("fetal gut",            f"{DATA}/pancreas/fetal_gut.h5ad",                   "fetal",  "proliferative"),
    ("adult gut",            f"{DATA}/pancreas/gut_setty_schema.h5ad",            "adult",  "quiescent"),
    ("adult CD34+ marrow",   f"{DATA}/hematopoiesis/setty19_cd34_bm.h5ad",        "adult",  "proliferative"),
    ("adult lung airway",    f"{DATA}/pancreas/lung_airway_setty_schema.h5ad",    "adult",  "quiescent"),
    ("aged marrow",          f"{DATA}/aging/aging_setty_schema.h5ad",          "aged",   "quiescent"),
    ("TS kidney",            f"{G.TS_RAW}/tabula_sapiens_kidney.h5ad",            "adult",  "quiescent"),
    ("TS lung",              f"{G.TS_RAW}/tabula_sapiens_lung.h5ad",              "adult",  "quiescent"),
]


def symbols(f):
    """gene symbols, handling both categorical var/feature_name and a plain var index dataset."""
    v = f["var"]
    if "feature_name" in v:
        fnm = v["feature_name"]
        if isinstance(fnm, h5py.Group):
            return np.char.upper(G._dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]].astype(str))
        return np.char.upper(G._dec(fnm[:]).astype(str))
    for k in ("_index", "index"):
        if k in v:
            return np.char.upper(G._dec(v[k][:]).astype(str))
    raise KeyError("no symbol column")


def stream_topk(path, want, n_cells, rng):
    """binary cell x gene matrix over the top-K expressed genes per cell, plus a proliferation index."""
    with h5py.File(path, "r") as f:
        X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
        syms = symbols(f)
        col = {s: i for i, s in enumerate(want)}
        keepcol = np.full(g, -1, np.int64)
        for j, s in enumerate(syms):
            if s in col:
                keepcol[j] = col[s]
        cyc = np.zeros(g, bool)
        cyc_ok = [c for c in CYCLE if c in col]   # only cycle genes in the COMMON universe
        for j, s in enumerate(syms):
            if s in cyc_ok:
                cyc[j] = True
        sel = np.arange(n) if n_cells >= n else np.sort(rng.choice(n, n_cells, replace=False))
        indptr = X["indptr"][:]; data = X["data"]; idx = X["indices"]
        rows, cols, cyc_frac = [], [], []
        for c, r in enumerate(sel):
            s0, e0 = int(indptr[r]), int(indptr[r + 1])
            if e0 <= s0:
                cyc_frac.append(0.0); continue
            ii = idx[s0:e0]; vv = data[s0:e0]
            cyc_frac.append(float(cyc[ii].sum()) / max(1, int(cyc.sum())))
            m = keepcol[ii] >= 0
            ii2, vv2 = keepcol[ii[m]], vv[m]
            if len(ii2) > TOPK:
                p = np.argpartition(-vv2, TOPK)[:TOPK]; ii2 = ii2[p]
            rows.append(np.full(len(ii2), c, np.int32)); cols.append(ii2.astype(np.int32))
        r = np.concatenate(rows); k = np.concatenate(cols)
        B = sp.csr_matrix((np.ones(len(r), np.float32), (r, k)), shape=(len(sel), len(want)))
    return B, float(np.mean(cyc_frac)), len(sel)


def main():
    C = coords()
    auto = set(C.index[C.chromosome.isin(AUTOSOMES)])
    # PASS 1 -- common gene universe. Panels use different annotation vintages (the CD34 file predates 2019
    # symbols), so without this each panel is scored on its own gene set and the comparison is meaningless.
    sets = {}
    for name, path, _, _ in PANELS:
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            sets[name] = set(symbols(f)) & auto
        print(f"  [pass1] {name:<22} {len(sets[name]):>6} autosomal symbols")
    want = sorted(set.intersection(*sets.values()))
    print(f"  [pass1] COMMON universe across {len(sets)} panels: {len(want)} genes\n")
    y_all = np.array([C.chromosome[q] for q in want])
    st_all = C.loc[want, "start"].values.astype(float)
    rng = np.random.default_rng(SEED)
    res = {"n_cells": N_CELLS, "topk": TOPK, "dim": DIM, "chance": 1 / 22, "panels": {}}

    print(f"MATCHED PANEL SWEEP  |  {N_CELLS} cells, top-{TOPK}, LSA-{DIM}, 10-Mb group split")
    print(f"chance = {1/22:.3f}\n")
    print(f"{'panel':<22} {'stage':<7} {'cycling':<14} {'cells':>6} {'genes':>6} {'random':>8} {'group':>8} {'prolif':>7}")
    print("-" * 90)

    for name, path, stage, prolif in PANELS:
        if not os.path.exists(path):
            print(f"{name:<22} MISSING"); continue
        try:
            B, cyc, ncell = stream_topk(path, want, N_CELLS, rng)
            det = np.asarray(B.sum(0)).ravel()
            keep = det >= 5
            Bk = B[:, keep]; yk = y_all[keep]; stk = st_all[keep]
            Z = TruncatedSVD(n_components=min(DIM, Bk.shape[1] - 1), random_state=SEED
                             ).fit_transform(Bk.T.tocsr()).astype(np.float32)
            groups = np.array([f"{c}_{int(s // BLOCK)}" for c, s in zip(yk, stk)])
            out = {}
            for sname, folds in [("random", list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, yk))),
                                 ("group", list(GroupKFold(5).split(Z, yk, groups=groups)))]:
                pred = np.empty(len(yk), dtype=object)
                for tr, te in folds:
                    sc = StandardScaler().fit(Z[tr])
                    pred[te] = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1
                                                  ).fit(sc.transform(Z[tr]), yk[tr]).predict(sc.transform(Z[te]))
                out[sname] = float(balanced_accuracy_score(yk, pred.astype(str)))
            out.update(stage=stage, cycling=prolif, n_cells=ncell, n_genes=int(keep.sum()), prolif_index=cyc)
            res["panels"][name] = out
            print(f"{name:<22} {stage:<7} {prolif:<14} {ncell:>6} {int(keep.sum()):>6} "
                  f"{out['random']:>8.3f} {out['group']:>8.3f} {cyc:>7.3f}", flush=True)
        except Exception as e:
            print(f"{name:<22} FAILED {repr(e)[:52]}", flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "panel_sweep.json"), "w"), indent=1)

    P = res["panels"]
    if len(P) >= 4:
        print("\n=== WHICH HYPOTHESIS SURVIVES? ===")
        print("  per-panel, sorted (the buckets below hide bimodality -- read this first):")
        for k, v in sorted(P.items(), key=lambda x: -x[1]["group"]):
            print(f"    {k:<22} {v['group']:.3f}   [{v['stage']}, {v['cycling']}]")
        dev = [v["group"] for v in P.values() if v["stage"] == "fetal"]
        adult_pro = [v["group"] for v in P.values() if v["stage"] != "fetal" and v["cycling"] == "proliferative"]
        adult_qui = [v["group"] for v in P.values() if v["stage"] != "fetal" and v["cycling"] == "quiescent"]
        f = lambda a: f"{np.mean(a):.3f}" if a else "n/a"
        print(f"  fetal / proliferative      : {f(dev)}")
        print(f"  ADULT but proliferative    : {f(adult_pro)}   <- the discriminator")
        print(f"  adult, quiescent           : {f(adult_qui)}")
        if adult_pro and adult_qui and dev:
            if np.mean(adult_pro) > (np.mean(adult_qui) + np.mean(dev)) / 2:
                print("  -> PROLIFERATION: adult proliferative tissue behaves like fetal tissue.")
            elif np.mean(adult_pro) < np.mean(adult_qui) * 1.5:
                print("  -> DEVELOPMENT: adult proliferative tissue behaves like quiescent adult tissue.")
            else:
                print("  -> INTERMEDIATE: neither hypothesis is cleanly supported; report as such.")
        g = np.array([v["group"] for v in P.values()]); pi = np.array([v["prolif_index"] for v in P.values()])
        if len(g) > 3 and pi.std() > 0:
            print(f"\n  corr(proliferation index, group-split accuracy) = {np.corrcoef(pi, g)[0,1]:+.2f} "
                  f"over {len(g)} panels")
    print("\n[done] -> results/panel_sweep.json")


if __name__ == "__main__":
    main()
