"""IS CHROMOSOME-SCALE CO-EXPRESSION MANUFACTURED BY CELL-TYPE HETEROGENEITY?

The unexplained fact. A training-free factorisation of expression data decodes a gene's chromosome at 0.720, and
that number swings enormously across panels with no known cause: fetal gut 0.720, adult gut 0.497, aged marrow
0.470, CD34+ marrow 0.080, lung airway 0.077, Tabula Sapiens kidney/lung ~0.08. Neither developmental stage nor
proliferation orders these (an earlier "developing tissue" reading was falsified by a matched panel sweep). The
one untested regularity: the two near-chance panels are PURIFIED populations (sorted CD34+ progenitors, sorted
airway epithelium) while the strong ones are whole heterogeneous tissues.

Why it matters more than any model experiment. If chromosome-scale co-expression is manufactured by cell-type
heterogeneity, then the model's +0.161 margin becomes a claim about heterogeneity-INDEPENDENT structure, which
is both more interesting and more likely to survive review. If instead heterogeneity explains everything, the
regional-programme story is much smaller than it looks. Either way the result stands alone without the model.

DESIGN. Fetal gut carries 21 annotated cell types over 62,849 cells, so diversity can be varied with cell count
held EXACTLY fixed -- which the cross-panel comparison could never do, since panel size and panel purity were
confounded there (fetal gut 62,849 cells vs lung airway 3,600).

  A  DIVERSITY LADDER (primary). Fix total cells N; draw them from k cell types, N/k each, for k = 1..12, with
     several independent draws of WHICH types. Chromosome decoding as a function of k at fixed N isolates
     diversity from depth. Prediction under the hypothesis: strongly increasing in k.
  B  SPECIFICITY CONTROL (the control that makes A interpretable). More diverse cells plausibly make the
     co-occurrence matrix more informative about EVERYTHING, which would raise any 22-class gene property, not
     chromosome in particular. So every ladder rung is also scored on a matched-difficulty NON-GENOMIC target:
     22 gene clusters obtained by k-means on ESM2 protein embeddings. If chromosome and ESM2-cluster rise
     together, diversity is a generic quality effect and the chromosome reading is wrong. If chromosome rises
     and ESM2-cluster does not, heterogeneity specifically manufactures genomic structure.
  C  PURIFIED-PANEL RESCUE. Take the two near-chance purified panels (CD34+ marrow, lung airway) and pool them
     with each other and with unrelated tissue at matched total N. Does decoding rise from ~0.08?

Out: results/purity_decomposition.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from shallow_coocc_baseline import lsa
from coocc_strongest import symbols_of, stream_panel
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

SEED = 0
N_TOTAL = 6000                     # cells held fixed across every rung of the ladder
K_LADDER = [1, 2, 3, 5, 8, 12]
N_DRAWS = 3                        # independent draws of WHICH cell types, per rung
DIMS = 256
TOPK = 2048
DATA = f"{_DATA}"
FETAL = f"{DATA}/pancreas/fetal_gut.h5ad"


def decode(E, y, groups):
    """22-class balanced accuracy under the 10-Mb group split (the honest split used throughout)."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(E, y, groups=groups):
        sc = StandardScaler().fit(E[tr])
        pred[te] = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1).fit(
            sc.transform(E[tr]), y[tr]).predict(sc.transform(E[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def decode_random_split(E, y):
    """For the ESM2-cluster control there is no genomic neighbourhood to hold out, so use stratified folds."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(E, y):
        sc = StandardScaler().fit(E[tr])
        pred[te] = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1).fit(
            sc.transform(E[tr]), y[tr]).predict(sc.transform(E[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def build_from_rows(path, want, rows, topk=TOPK):
    """Binary top-K cell x gene matrix for an explicit set of cell row indices."""
    with h5py.File(path, "r") as f:
        X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
        syms = symbols_of(f)
        col = {s: i for i, s in enumerate(want)}
        keepcol = np.full(g, -1, np.int64)
        for j, s in enumerate(syms):
            if s in col:
                keepcol[j] = col[s]
        indptr = X["indptr"][:]; data = X["data"]; idx = X["indices"]
        rr, cc = [], []
        for c, r in enumerate(sorted(int(v) for v in rows)):
            s0, e0 = int(indptr[r]), int(indptr[r + 1])
            if e0 <= s0:
                continue
            ii = idx[s0:e0]; vv = data[s0:e0]
            m = keepcol[ii] >= 0
            ii, vv = keepcol[ii[m]], vv[m]
            if len(ii) > topk:
                ii = ii[np.argpartition(-vv, topk)[:topk]]
            rr.append(np.full(len(ii), c, np.int32)); cc.append(ii.astype(np.int32))
        a = np.concatenate(rr); b = np.concatenate(cc)
        return sp.csr_matrix((np.ones(len(a), np.float32), (a, b)), shape=(len(rows), len(want)))


def main():
    C = coords()
    _, sd0 = G.basis("coexpr_devel")
    sd = [s for s in sd0 if s in C.index and C.chromosome[s] in AUTOSOMES]

    # gene-level targets, shared by every rung
    y_chr = np.array([C.chromosome[s] for s in sd])
    st = C.loc[list(sd), "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y_chr, st)])

    Me, se = G.basis("esm2")
    pe = {s: i for i, s in enumerate(se)}
    have = np.array([s in pe for s in sd])
    Xe = np.zeros((len(sd), Me.shape[1]), np.float32)
    Xe[have] = Me[[pe[s] for s in np.array(sd)[have]]]
    y_esm = KMeans(n_clusters=22, n_init=4, random_state=SEED).fit_predict(Xe).astype(str)
    y_esm[~have] = "na"
    del Me, Xe; gc.collect()
    print(f"{len(sd)} genes | chance {1/22:.3f} | ESM2 control clusters built "
          f"({have.sum()} genes with a protein embedding)\n", flush=True)

    with h5py.File(FETAL, "r") as f:
        ct = f["obs"]["cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ct["categories"][:]]
        codes = ct["codes"][:]
    counts = {i: int((codes == i).sum()) for i in range(len(cats))}
    usable = [i for i, n in counts.items() if n >= N_TOTAL // max(K_LADDER)]
    usable.sort(key=lambda i: -counts[i])
    print("cell types usable for the ladder:")
    for i in usable[:14]:
        print(f"   {cats[i]:<38} {counts[i]:>6} cells")
    print(flush=True)

    res = {"n_total": N_TOTAL, "dims": DIMS, "chance": 1 / 22, "n_genes": len(sd), "ladder": {}, "rescue": {}}
    rng = np.random.default_rng(SEED)

    print("=== A+B: DIVERSITY LADDER at fixed N = %d cells ===" % N_TOTAL, flush=True)
    print(f"{'k types':<9} {'draw':<6} {'chromosome':<13} {'ESM2-cluster (control)':<24} types")
    print("-" * 90)
    for k in K_LADDER:
        rows_k = []
        for d in range(N_DRAWS if k < len(usable) else 1):
            pick = list(rng.choice(usable, size=min(k, len(usable)), replace=False))
            per = N_TOTAL // len(pick)
            sel = []
            for t in pick:
                idx = np.nonzero(codes == t)[0]
                sel.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
            sel = np.concatenate(sel)
            B = build_from_rows(FETAL, sd, sel)
            E = lsa(B, dims=DIMS); del B; gc.collect()
            a_chr = decode(E, y_chr, groups)
            a_esm = decode_random_split(E[have], y_esm[have])
            del E; gc.collect()
            rows_k.append({"chromosome": a_chr, "esm2_control": a_esm, "n_cells": int(len(sel)),
                           "types": [cats[t] for t in pick]})
            print(f"{k:<9} {d:<6} {a_chr:<13.3f} {a_esm:<24.3f} {', '.join(cats[t] for t in pick)[:44]}",
                  flush=True)
        res["ladder"][str(k)] = rows_k
        json.dump(res, open(os.path.join(HERE, "results", "purity_decomposition.json"), "w"), indent=1)

    print("\n=== C: PURIFIED-PANEL RESCUE (matched total N) ===", flush=True)
    pur = [("CD34+ marrow", f"{DATA}/hematopoiesis/setty19_cd34_bm.h5ad"),
           ("lung airway", f"{DATA}/pancreas/lung_airway_setty_schema.h5ad")]
    mats = {}
    for nm, p in pur:
        try:
            mats[nm] = stream_panel(p, sd)
            n = mats[nm].shape[0]
            sub = mats[nm][rng.choice(n, size=min(3000, n), replace=False)]
            E = lsa(sub, dims=DIMS)
            res["rescue"][nm] = {"n_cells": int(sub.shape[0]), "chromosome": decode(E, y_chr, groups)}
            print(f"  {nm:<28} alone  {res['rescue'][nm]['chromosome']:.3f}", flush=True)
            del E, sub; gc.collect()
        except Exception as e:
            print(f"  {nm:<28} FAILED {repr(e)[:50]}", flush=True)
    if len(mats) == 2:
        parts = [m[rng.choice(m.shape[0], size=min(3000, m.shape[0]), replace=False)] for m in mats.values()]
        Bm = sp.vstack(parts, format="csr")
        E = lsa(Bm, dims=DIMS)
        res["rescue"]["CD34+ marrow + lung airway pooled"] = {
            "n_cells": int(Bm.shape[0]), "chromosome": decode(E, y_chr, groups)}
        print(f"  {'two purified panels pooled':<28} {res['rescue']['CD34+ marrow + lung airway pooled']['chromosome']:.3f}",
              flush=True)
        del E, Bm; gc.collect()

    json.dump(res, open(os.path.join(HERE, "results", "purity_decomposition.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    mk = {k: float(np.mean([r["chromosome"] for r in v])) for k, v in res["ladder"].items()}
    me = {k: float(np.mean([r["esm2_control"] for r in v])) for k, v in res["ladder"].items()}
    print("  chromosome by k: " + "  ".join(f"{k}:{v:.3f}" for k, v in mk.items()))
    print("  ESM2 control   : " + "  ".join(f"{k}:{v:.3f}" for k, v in me.items()))
    lo, hi = mk[str(K_LADDER[0])], mk[str(K_LADDER[-1])]
    elo, ehi = me[str(K_LADDER[0])], me[str(K_LADDER[-1])]
    print(f"  chromosome rises {lo:.3f} -> {hi:.3f} ({hi-lo:+.3f}); control {elo:.3f} -> {ehi:.3f} ({ehi-elo:+.3f})")
    if hi - lo > 0.15 and (hi - lo) > 2 * abs(ehi - elo):
        print("  -> HETEROGENEITY MANUFACTURES CHROMOSOME STRUCTURE, specifically (control flat).")
    elif hi - lo > 0.15:
        print("  -> diversity raises BOTH targets: a generic informativeness effect, NOT chromosome-specific.")
    else:
        print("  -> cell-type diversity does NOT explain the panel ordering; the cause is still unknown.")
    print("\n[done] -> results/purity_decomposition.json")


if __name__ == "__main__":
    main()
