"""INTERPRETING THE HOX ANALOGY AXES (Ihor, 2026-07-21).

The analogy HOXA9 - HOXA1 + HOXB1 ~ HOXB9 works (hox_analogy_null.py). If it works, the grid must decompose into
two consistent directions: a PARALOG (anterior-posterior body-axis, 1..13) offset and a CLUSTER (genomic locus,
A/B/C/D) offset, each roughly the SAME vector wherever it is applied (parallel transport). This script measures
that directly, three ways, per basis:

  1. ADDITIVE R^2: fit  W[g] ~= mu + cluster_effect[c(g)] + paralog_effect[p(g)]  (two-way, no interaction) and
     report the fraction of embedding variance it explains -- how "grid-like" (bilinear) the geometry is. Also
     cluster-only and paralog-only R^2.
  2. PARALOG-DIRECTION CROSS-CLUSTER TRANSFER: leave-one-cluster-out ridge regressing paralog number on the
     embedding, trained on 3 clusters, Spearman rho on the held-out 4th. High => ONE shared body-axis direction,
     not four private ones.
  3. OFFSET PARALLEL-TRANSPORT: mean cosine between the same paralog-shift offset computed in different clusters
     (does p1->p2 point the same way in cluster A as in cluster B?), and the same cluster-shift offset computed
     at different paralogs. This is exactly what 3CosAdd relies on.

Run: ../../.venv/bin/python -u hox_axes.py
Out: results/hox_axes.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, warnings, itertools; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gm_lib as G, gene_sets as S
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
h = S.H["hox_grid"]
GENES = np.array([g.upper() for g in h["genes"]])
PAR = np.asarray(h["coord"][0], float)   # paralog number 1..13
CLU = np.asarray(h["coord"][1])          # cluster label A/B/C/D
BASES = ["maxtoki1b_lmhead", "maxtoki_lmhead", "maxtoki_we", "scgpt_we", "coexpr_devel", "esm2"]


def load_1b():
    """Register the MaxToki-1B lm_head table into the basis cache (same recipe as hox_steer.py)."""
    import pickle
    MT = f"{_DATA}/maxtoki/setup"
    R = G.ST_Reader(f"{MT}/MaxToki-1B-HF/model.safetensors")
    M = R.get("lm_head.weight")
    tok = json.load(open(f"{MT}/token_dictionary.json"))
    e2s = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    rows, syms = [], []
    for ens, r in tok.items():
        s = e2s.get(ens)
        if s is not None and r < M.shape[0]:
            rows.append(r); syms.append(s)
    o = np.argsort(syms); rows, syms = np.array(rows)[o], np.array(syms)[o]
    _, k = np.unique(syms, return_index=True)
    G._cache["maxtoki1b_lmhead"] = (np.asarray(M)[rows[k]], syms[k])


def additive_r2(X, clu, par):
    """Variance explained by mu + cluster_effect + paralog_effect (two-way additive, centred)."""
    mu = X.mean(0)
    def eff(labels):
        e = np.zeros_like(X)
        for lab in np.unique(labels):
            m = labels == lab
            e[m] = X[m].mean(0) - mu
        return e
    ce, pe = eff(clu), eff(par)
    ss_tot = ((X - mu) ** 2).sum()
    r2 = lambda pred: 1 - ((X - pred) ** 2).sum() / ss_tot
    return dict(cluster_only=float(r2(mu + ce)),
                paralog_only=float(r2(mu + pe)),
                additive_both=float(r2(mu + ce + pe)))


def paralog_transfer(X, clu, par):
    """Leave-one-cluster-out: ridge paralog-number direction from 3 clusters -> Spearman on held-out cluster."""
    rhos = []
    for held in np.unique(clu):
        tr, te = clu != held, clu == held
        if te.sum() < 3 or len(np.unique(par[tr])) < 2:
            continue
        r = Ridge(alpha=10.0).fit(X[tr], par[tr])
        pred = r.predict(X[te])
        if np.std(pred) > 0 and len(np.unique(par[te])) > 1:
            rhos.append(spearmanr(pred, par[te]).correlation)
    return float(np.nanmean(rhos)) if rhos else float("nan")


def offset_transport(X, clu, par):
    """Cosine of the SAME shift offset computed in different contexts (parallel transport)."""
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan

    # PARALOG offset p1->p2 computed within each cluster, compared across cluster pairs
    par_cos = []
    clusters = np.unique(clu)
    for p1, p2 in itertools.combinations(sorted(np.unique(par)), 2):
        offs = {}
        for c in clusters:
            a = X[(clu == c) & (par == p1)]; b = X[(clu == c) & (par == p2)]
            if len(a) and len(b):
                offs[c] = b.mean(0) - a.mean(0)
        for c1, c2 in itertools.combinations(list(offs), 2):
            par_cos.append(cos(offs[c1], offs[c2]))

    # CLUSTER offset c1->c2 computed at each paralog, compared across paralog pairs
    clu_cos = []
    for c1, c2 in itertools.combinations(clusters, 2):
        offs = {}
        for p in np.unique(par):
            a = X[(clu == c1) & (par == p)]; b = X[(clu == c2) & (par == p)]
            if len(a) and len(b):
                offs[p] = b.mean(0) - a.mean(0)
        for p1, p2 in itertools.combinations(list(offs), 2):
            clu_cos.append(cos(offs[p1], offs[p2]))

    return (float(np.nanmean(par_cos)) if par_cos else float("nan"),
            float(np.nanmean(clu_cos)) if clu_cos else float("nan"))


def main():
    print(f"{'basis':<18}{'add-R2':>8}{'clu-R2':>8}{'par-R2':>8}"
          f"{'par-transfer':>14}{'par-offset-cos':>16}{'clu-offset-cos':>16}")
    print("-" * 88)
    load_1b()
    out = {}
    for b in BASES:
        M, ok = G.subset(b, list(GENES))
        ok = np.asarray(ok)
        if ok.sum() < 20:
            print(f"{b:<18} n={int(ok.sum())} skip"); continue
        X = np.asarray(M, np.float64)          # G.subset already returns matched rows, aligned to GENES[ok]
        clu, par = CLU[ok], PAR[ok]
        r2 = additive_r2(X, clu, par)
        tr = paralog_transfer(X, clu, par)
        pcos, ccos = offset_transport(X, clu, par)
        out[b] = dict(n=int(ok.sum()), **r2, paralog_transfer=tr,
                      paralog_offset_cos=pcos, cluster_offset_cos=ccos)
        print(f"{b:<18}{r2['additive_both']:>8.3f}{r2['cluster_only']:>8.3f}{r2['paralog_only']:>8.3f}"
              f"{tr:>14.3f}{pcos:>16.3f}{ccos:>16.3f}")
    json.dump(out, open(os.path.join(HERE, "results", "hox_axes.json"), "w"), indent=1)
    print("\nadd-R2  = variance explained by mu + cluster + paralog (additive 2-way); >0.5 => grid-like")
    print("par-transfer   = leave-one-cluster-out Spearman on paralog number (shared body-axis direction)")
    print("*-offset-cos   = cosine of the same shift computed in different contexts (parallel transport)")
    print("[done] -> results/hox_axes.json")


if __name__ == "__main__":
    main()
