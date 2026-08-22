"""SUPERVISED CROSS-VALIDATED PROBES — the reframing the PCA battery needed.

WHY. run_contextfree/run_scale asked "does the annotated order appear in the top 3 PCs of the gene subset?"
That is UNSUPERVISED: it throws the annotation away, demands the structure occupy a top principal plane, and it
had no power at n=46 (every margin CI spanned +-0.4). Ihor's cosine objection exposed the same disease in the
antipodal test -- a full-space measure was dominated by shared mass, and the hypothesis came back to life the
moment we asked the question in the right subspace with a MARKER-DEFINED (i.e. supervised) axis.

Generalise that. The right question is not "is the order in PC1-3" but:

    IS THE ANNOTATION LINEARLY DECODABLE FROM THE GENE EMBEDDING, and does it GENERALISE to held-out genes?

A cross-validated ridge probe uses all d dimensions and the labels, finds the best direction FOR the question,
and stays honest because every number is out-of-fold. It is strictly more sensitive than PCA-max, and at
n=20-50 with strong regularisation + CV it is the standard tool. This is what should have been run first.

  order  -> Ridge predicts the index; score = out-of-fold Spearman
  circle -> Ridge predicts (cos phi, sin phi); score = out-of-fold circular correlation of atan2(pred)
            (the cell-cycle analogue of route_cellcycle/cc_benchmark.order_circ_r2, applied to GENES)
  grid   -> one probe per coordinate; score = min of the two (a grid must encode BOTH)

All bases share the SAME CV folds, so model-vs-reference is PAIRED and comparable. The verdict rule is
unchanged and non-negotiable: a model basis must beat coexpr (data) AND esm2 (sequence).

Run: ../../.venv/bin/python -u run_probe.py
Out: results/probe.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

BASES = ["scgpt_we", "maxtoki_we", "maxtoki_lmhead", "coexpr", "coexpr_k562", "esm2"]
MODEL_BASES = ["scgpt_we", "maxtoki_we", "maxtoki_lmhead"]
REFS = ["coexpr", "coexpr_k562", "esm2"]
ALPHAS = np.logspace(0, 5, 12)          # d >> n here, so the useful range is heavily regularised
N_PERM = 1000
SEED = 0


def _folds(n, k=5):
    return list(KFold(min(k, n), shuffle=True, random_state=SEED).split(np.arange(n)))


def oof_linear(X, y, folds):
    P = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        P[te] = m.predict(sc.transform(X[te]))
    return P


def probe_order(X, y, folds):
    P = oof_linear(X, y, folds)
    r = spearmanr(P, y).statistic
    return 0.0 if not np.isfinite(r) else abs(float(r))


def probe_circle(X, phi, folds):
    """Out-of-fold circular correlation: predict (cos, sin), recombine with atan2."""
    C = oof_linear(X, np.cos(phi), folds)
    Sn = oof_linear(X, np.sin(phi), folds)
    return abs(G._circ_corr(np.arctan2(Sn, C), phi))


def score(kind, X, coord, folds):
    if kind == "circle":
        return probe_circle(X, coord, folds)
    if kind == "grid":
        return min(probe_order(X, coord[0], folds), probe_order(X, coord[1], folds))
    return probe_order(X, coord, folds)


def perm_p(kind, X, coord, folds, obs, n_perm=N_PERM):
    """Null: permute the annotation over the SAME genes and re-probe. Holds the gene set, its co-expression
    structure and its abundance prior fixed -- only the label-to-gene assignment moves."""
    rng = np.random.default_rng(SEED)
    cnt = 0
    n_run = max(50, n_perm // 10)         # probes are ~100x costlier than a Spearman; 100-200 perms is enough
    for _ in range(n_run):
        if kind == "grid":
            c = (rng.permutation(coord[0]), rng.permutation(coord[1]))
        else:
            c = rng.permutation(coord)
        if score(kind, X, c, folds) >= obs:
            cnt += 1
    return (cnt + 1) / (n_run + 1)


def main():
    res = {}
    for name, h in S.H.items():
        if h["kind"] == "antipodal":
            continue                       # handled by antipodal_subspace.py
        print(f"\n=== {name} ({h['kind']}, role={h['role']}) ===", flush=True)
        rec = {"kind": h["kind"], "role": h["role"], "by_basis": {}}
        for b in BASES:
            M, ok = G.subset(b, h["genes"])
            if ok.sum() < 8:
                continue
            if h["kind"] == "grid":
                coord = (np.asarray(h["coord"][0], float)[ok], np.asarray(h["coord"][1], float)[ok])
            else:
                coord = np.asarray(h["coord"], float)[ok]
            folds = _folds(int(ok.sum()))
            s = score(h["kind"], M, coord, folds)
            p = perm_p(h["kind"], M, coord, folds, s)
            rec["by_basis"][b] = dict(stat=float(s), p=float(p), n=int(ok.sum()))
            print(f"  {b:<15} n={int(ok.sum()):<3} oof={s:.3f}  p={p:.3f}", flush=True)
        res[name] = rec

    print("\n" + "=" * 104)
    print("VERDICT (supervised probe): model must beat EVERY reference basis that has the genes, and be p<0.05")
    print("=" * 104)
    for name, rec in res.items():
        bb = rec["by_basis"]
        mods = {b: bb[b]["stat"] for b in MODEL_BASES if b in bb}
        refs = {b: bb[b]["stat"] for b in REFS if b in bb}
        if not mods or not refs:
            print(f"  {name:<24} INCOMPLETE"); continue
        mb = max(mods, key=mods.get)
        wins = all(mods[mb] > v for v in refs.values()) and bb[mb]["p"] < 0.05
        rec["verdict"] = "MODEL" if wins else "no"
        rec["best_model_basis"] = mb
        rstr = " ".join(f"{r.split('_')[0][:6]}:{v:.2f}" for r, v in refs.items())
        print(f"  {name:<24} {mb:<15} {mods[mb]:.3f} (p={bb[mb]['p']:.3f})  refs[{rstr}]  "
              f"{'** MODEL **' if wins else ''}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "probe.json"), "w"), indent=1)
    print("\n[done] -> results/probe.json")


if __name__ == "__main__":
    main()
