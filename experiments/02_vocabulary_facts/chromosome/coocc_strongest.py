"""THE STRONGEST SHALLOW BASELINE WE CAN BUILD -- does MaxToki-1B's surviving chromosome claim survive it?

Why this is needed. coocc_bestprobe.py concluded that MaxToki-1B beats a shallow factorisation by +0.161 and
described that baseline as having been given "its best panel". An audit of results/panel_sweep.json (built by a
parallel session, all panels matched at 3,500 cells / top-2048 / LSA-256 / same split) shows that was FALSE:

    adult gut 0.524 > fetal gut 0.413 > aged marrow 0.298 > CD34 0.113 > lung 0.099 > TS kidney 0.086

Fetal gut was chosen for being the LARGEST file (62,849 cells vs adult gut's 4,269), not the strongest substrate.
Adult gut is ~27% better per cell and was only ever run at 3,500. Two consequences:
  * the "developing tissue" interpretation is dead -- the strongest panel is adult and quiescent;
  * the surviving 1B margin was measured against a baseline that was NOT at its best, so it is not yet safe.

This script builds the strongest baseline available and re-runs the comparison:
  A  POOLED corpus -- every panel concatenated (~96k cells, diverse tissues). This is also the construction
     closest to how the model was actually trained (a large multi-tissue corpus), so it is the honest analogue,
     not merely the strongest. Pooling could instead DILUTE the signal, which is itself informative.
  B  each panel alone at its full size, so per-panel scaling is visible rather than assumed.
  C  a dimension sweep BRACKETED FROM BELOW (128/256/512/1024). The earlier "peaks at 256" defence was thin:
     at full corpus we never tested below 256, and under LinearSVC 512 (0.711) beat 256 (0.688).
  D  position with the chromosome set MATCHED across representations (the earlier run averaged the models over
     22 chromosomes and LSA over 21, so the headline +0.776 vs +0.626 was across different denominators).

Every representation gets its own best probe from the same fixed family.

Out: results/coocc_strongest.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, gc, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK, MINCHR, dedup, ALPHAS, SEED as MS_SEED
from table_grid import load as load_mt
from shallow_coocc_baseline import lsa
from coocc_fair import stream_binary
from final_probe_grid import mlp_for, evaluate
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr

SEED = 0
DIMS = [128, 256, 512, 1024]
DATA = f"{_DATA}"
PANELS = [
    ("fetal gut",          f"{DATA}/pancreas/fetal_gut.h5ad"),
    ("adult gut",          f"{DATA}/pancreas/gut_setty_schema.h5ad"),
    ("aged marrow",        f"{DATA}/aging/aging_setty_schema.h5ad"),
    ("adult CD34+ marrow", f"{DATA}/hematopoiesis/setty19_cd34_bm.h5ad"),
    ("adult lung airway",  f"{DATA}/pancreas/lung_airway_setty_schema.h5ad"),
]


def symbols_of(f):
    """Gene symbols across both h5ad schemas in this project: categorical var/feature_name (Tabula Sapiens,
    fetal gut) and a plain var/index dataset (the *_setty_schema files). Mirrors panel_sweep.symbols."""
    v = f["var"]
    if "feature_name" in v:
        fnm = v["feature_name"]
        if isinstance(fnm, h5py.Group):
            return np.char.upper(G._dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]].astype(str))
        return np.char.upper(G._dec(fnm[:]).astype(str))
    for k in ("_index", "index"):
        if k in v:
            return np.char.upper(G._dec(v[k][:]).astype(str))
    raise KeyError("no gene-symbol field in var")


def stream_panel(path, want, topk=2048):
    """Binary top-K cell x gene matrix streamed from a CSR h5ad, schema-tolerant.

    Replaces coocc_fair.stream_binary, which assumed var/feature_name and therefore silently dropped four of the
    five panels in the first run of this script -- the 'pooled' corpus was then just fetal gut under another name.
    """
    with h5py.File(path, "r") as f:
        X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
        syms = symbols_of(f)
        col = {s: i for i, s in enumerate(want)}
        keepcol = np.full(g, -1, np.int64)
        for j, s in enumerate(syms):
            if s in col:
                keepcol[j] = col[s]
        if (keepcol >= 0).sum() < 0.2 * len(want):
            raise ValueError(f"only {(keepcol>=0).sum()} of {len(want)} genes matched -- wrong symbol field?")
        indptr = X["indptr"][:]; data = X["data"]; idx = X["indices"]
        rows, cols = [], []
        for r in range(n):
            s0, e0 = int(indptr[r]), int(indptr[r + 1])
            if e0 <= s0:
                continue
            ii = idx[s0:e0]; vv = data[s0:e0]
            m = keepcol[ii] >= 0
            ii, vv = keepcol[ii[m]], vv[m]
            if len(ii) > topk:
                ii = ii[np.argpartition(-vv, topk)[:topk]]
            rows.append(np.full(len(ii), r, np.int32)); cols.append(ii.astype(np.int32))
        rr = np.concatenate(rows); kk = np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(rr), np.float32), (rr, kk)), shape=(n, len(want)))


def probes(X):
    hl = mlp_for(X.shape[1])
    return [("logistic C=0.1", lambda: LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1)),
            ("LinearSVC C=0.01", lambda: LinearSVC(C=0.01, max_iter=4000)),
            (f"MLP {hl}", lambda: MLPClassifier(hidden_layer_sizes=hl, alpha=1.0, max_iter=600,
                                                early_stopping=True, random_state=SEED))]


def best_probe(X, y, fr, fg, tag):
    out = {"per_probe": {}}
    for pn, make in probes(X):
        t0 = time.time()
        try:
            r = evaluate(make, X, y, fr); g = evaluate(make, X, y, fg)
            out["per_probe"][pn] = {"random": r, "group": g}
            print(f"    {tag:<30} {pn:<18} r {r:.3f}  g {g:.3f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"    {tag:<30} {pn:<18} FAILED {repr(e)[:30]}", flush=True)
    if out["per_probe"]:
        bp = max(out["per_probe"].items(), key=lambda kv: kv[1]["group"])
        out.update(best_probe=bp[0], group=bp[1]["group"], random=bp[1]["random"], dim=int(X.shape[1]))
    return out


def position_rho_on(M, syms, C, chroms):
    """model_scale.position_rho but restricted to a FIXED chromosome list, so denominators match."""
    pi = {s: i for i, s in enumerate(syms)}
    rr = {}
    for c in chroms:
        g = [s for s in C.index[C.chromosome == c] if s in pi]
        if len(g) < MINCHR:
            continue
        Xf = M[[pi[s] for s in g]]; start = C.loc[g, "start"].values.astype(float)
        keep = dedup(Xf, start)
        if keep.sum() < 120:
            continue
        X, yv = Xf[keep], start[keep]
        P = np.zeros(len(yv))
        for tr, te in KFold(5, shuffle=True, random_state=MS_SEED).split(X):
            sc = StandardScaler().fit(X[tr])
            P[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), yv[tr]).predict(sc.transform(X[te]))
        r = spearmanr(P, yv).statistic
        rr[c] = 0.0 if not np.isfinite(r) else float(r)
    return rr


def main():
    C = coords()
    tabs = {nm: load_mt(w, "output") for nm, w in [("MaxToki-217M", "217M"), ("MaxToki-1B", "1B")]}
    _, sd0 = G.basis("coexpr_devel")
    sd = [s for s in sd0 if s in C.index and C.chromosome[s] in AUTOSOMES]

    common = sorted(set.intersection(*[set(s) for _, s in tabs.values()], set(sd))
                    & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    y = np.array([C.chromosome[q] for q in common])
    st = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])
    fr = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    fg = list(GroupKFold(5).split(np.zeros(len(y)), y, groups=groups))
    print(f"{len(common)} matched genes | chance {1/22:.3f}\n")

    def sub(M, syms):
        pi = {q: i for i, q in enumerate(syms)}
        return M[[pi[q] for q in common]]

    res = {"n_genes": len(common), "chance": 1 / 22, "models": {}, "panels": {}, "pooled": {}}

    print("=== models (native width, best probe) ===", flush=True)
    for nm, (M, s) in tabs.items():
        res["models"][nm] = best_probe(sub(M.astype(np.float32), s), y, fr, fg, nm)
        gc.collect()
    json.dump(res, open(os.path.join(HERE, "results", "coocc_strongest.json"), "w"), indent=1)

    print("\n=== B: each panel alone, FULL size, LSA-256 ===", flush=True)
    mats = {}; failed = []
    for nm, path in PANELS:
        try:
            B = stream_panel(path, sd)
            mats[nm] = B
            E = sub(lsa(B, dims=256), sd)
            res["panels"][nm] = dict(n_cells=int(B.shape[0]), nnz=int(B.nnz), **best_probe(E, y, fr, fg, nm))
            del E; gc.collect()
        except Exception as e:
            print(f"    {nm:<30} FAILED {repr(e)[:60]}", flush=True)
            failed.append(nm)
        json.dump(res, open(os.path.join(HERE, "results", "coocc_strongest.json"), "w"), indent=1)

    print("\n=== A+C: POOLED corpus, dimension bracketed from below ===", flush=True)
    if failed:
        raise SystemExit(f"[abort] {len(failed)} panel(s) failed to load: {failed}. "
                         "Pooling a subset would silently mislabel one panel as the pooled corpus.")
    P = sp.vstack([mats[k] for k in mats], format="csr")
    for k in list(mats):
        del mats[k]
    gc.collect()
    print(f"  pooled: {P.shape[0]} cells x {P.shape[1]} genes, {P.nnz/1e6:.0f}M nonzeros", flush=True)
    res["pooled"]["n_cells"] = int(P.shape[0])
    for d in DIMS:
        E = sub(lsa(P, dims=d), sd)
        res["pooled"][f"LSA-{d}"] = best_probe(E, y, fr, fg, f"pooled LSA-{d}")
        del E; gc.collect()
        json.dump(res, open(os.path.join(HERE, "results", "coocc_strongest.json"), "w"), indent=1)

    print("\n=== D: position, chromosome set MATCHED ===", flush=True)
    Ebest = sub(lsa(P, dims=512), sd)
    del P; gc.collect()
    rr = {}
    for nm, (M, s) in tabs.items():
        rr[nm] = position_rho_on(sub(M.astype(np.float32), s), common, C, AUTOSOMES)
    rr["pooled LSA-512"] = position_rho_on(Ebest, common, C, AUTOSOMES)
    shared = sorted(set.intersection(*[set(v) for v in rr.values()]))
    res["position_matched"] = {"n_chrom": len(shared),
                               **{k: float(np.mean([v[c] for c in shared])) for k, v in rr.items()}}
    print(f"  on the {len(shared)} chromosomes ALL representations retain:", flush=True)
    for k in rr:
        print(f"    {k:<20} rho {res['position_matched'][k]:+.3f}", flush=True)

    json.dump(res, open(os.path.join(HERE, "results", "coocc_strongest.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    cands = [(f"pooled {k}", v["group"]) for k, v in res["pooled"].items() if isinstance(v, dict)]
    cands += [(k, v["group"]) for k, v in res["panels"].items() if "group" in v]
    bl_nm, bl = max(cands, key=lambda kv: kv[1])
    print(f"  strongest shallow baseline: {bl_nm} = {bl:.3f}")
    for nm in tabs:
        m = res["models"][nm]["group"]
        print(f"  {nm:<14} {m:.3f} -> {'BEATS by %+.3f' % (m - bl) if m > bl else 'LOSES by %+.3f' % (m - bl)}")
    print("\n[done] -> results/coocc_strongest.json")


if __name__ == "__main__":
    main()
