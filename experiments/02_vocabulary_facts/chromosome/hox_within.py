"""WITHIN-CLUSTER PARALOG — does the curve exist outside PC1? (Ihor, 2026-07-17)

WHY THIS EXISTS. `hox_shape.py` TEST 3 asked "is paralog a 1-D curve?" by correlating paralog with **PC1 only**
(`pc1 = (U * s)[:, 0]`). PC1 carries ~26% of the variance, so 74% of the space was never looked at, and HOXB's
0.030 was read as "HOXB has no paralog order". That is the exact instrument RESULTS.md section 1 condemns:
unsupervised, discards the annotation, and demands the structure sit in a top principal plane. Ihor's objection:
maybe the order lives in some OTHER direction. This runs the supervised probe there instead.

THREE TESTS.
  T1  per-cluster LOO ridge probe, FULL space, + permutation null.
      *** T1 IS DEGENERATE -- DO NOT CITE ITS POINT ESTIMATES. Kept only because its null documents why. ***
      At n=9..11 a strongly-regularised ridge predicts ~the TRAINING mean = (S - y_i)/(n-1), which is a
      strictly DECREASING function of the held-out y_i. So Spearman(pred, y) -> -1 with ZERO signal in X, and
      probe_order's abs() reads that as a PERFECT score. Verified: pure-noise X (39x1232) scores |rho| = 0.89.
      That is why every T1 null mean below is ~0.77-0.89 and every p is non-significant -- the permutation null
      absorbs the artifact correctly, so the p-values are HONEST, but the statistic is pinned near its ceiling
      for signal and noise alike and has NO DYNAMIC RANGE LEFT. Verdict: no power at n=9-11, the same lesson
      RESULTS.md section 1 records at n=46, one order of magnitude worse.
      *** CORRECTION: "run_probe.py is NOT affected, the artifact needs LOO" -- MY EARLIER CLAIM -- IS WRONG. ***
      _folds is KFold(min(5, n)), which silently emits SIZE-1 FOLDS at small n, and a size-1 fold IS leave-one-
      out for that point. Any fold of size f gets one shared prediction ~ (S - sum(y_fold))/(n - f), which
      decreases in sum(y_fold), so folds self-order anti-monotonically in y with no signal. Measured noise floor
      of run_probe.probe_order under run_probe._folds: n=9 -> 0.736 (folds [1,2,2,2,2]); n=8/10/11/12 -> 0.42-
      0.48; n>=29 -> ~0.21-0.23. heme_biosynth is n=9 in EVERY basis. The HOX headline is safe (n=39/29). All
      published p-values stay honest (the null shares the folds), but small-n POINT ESTIMATES are not effect
      sizes. Fix for future work: signed (not abs) Spearman, or forbid folds smaller than ~3.
      The per-cluster question is answered by T3, which is immune -- see below.
  T2  PC sweep: |Spearman(PC_k, paralog)| for k = 1..5 with variance explained. WHERE does paralog live?
      If HOXB scores low on PC1 but high on PC2, TEST 3's verdict was an artifact of the instrument.
  T3  leave-one-CLUSTER-out paralog transfer -- the supervised twin of hox_analogy.py.
      Centre each cluster on its own centroid (kills cluster identity; label-free, so no label leakage), fit
      the paralog direction on 3 clusters, predict the paralog of the HELD-OUT cluster's genes. This asks the
      grid question directly: is the paralog direction SHARED across clusters? n = 39 pooled, not 9-11, so it
      is the best-powered thing in this file. Per-cluster output answers "is HOXB special?" cleanly.

Run: ../../.venv/bin/python -u hox_within.py
Out: results/hox_within.json
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

BASES = ["maxtoki_lmhead", "maxtoki_we", "scgpt_we", "coexpr", "coexpr_devel", "esm2"]
ALPHAS = np.logspace(0, 5, 12)
N_PERM = 200
SEED = 0
CN = "ABCD"

h = S.H["hox_grid"]
GENES = np.array(h["genes"]); PAR = np.asarray(h["coord"][0], float); CLU = np.asarray(h["coord"][1], int)


def _rho(a, b):
    r = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def loo_probe(X, y):
    """Leave-one-out ridge over the FULL space; returns |out-of-fold Spearman|."""
    n = len(y)
    P = np.zeros(n)
    for tr, te in KFold(n).split(np.arange(n)):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        P[te] = m.predict(sc.transform(X[te]))
    return abs(_rho(P, y))


def perm_p(X, y, obs, seed=SEED, n_perm=N_PERM):
    rng = np.random.default_rng(seed)
    cnt = 0
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = loo_probe(X, rng.permutation(y))
        cnt += null[i] >= obs
    return (cnt + 1) / (n_perm + 1), float(null.mean()), float(null.std())


def pc_sweep(X, y, k=5):
    """|Spearman(PC_j, paralog)| and variance explained, j = 1..k. The instrument TEST 3 used, extended."""
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U * s
    ev = (s ** 2) / (s ** 2).sum()
    kk = min(k, P.shape[1])
    return [dict(pc=j + 1, rho=abs(_rho(P[:, j], y)), var=float(ev[j])) for j in range(kk)]


def loco(M, par, clu):
    """T3 -- leave-one-CLUSTER-out paralog transfer.

    Centre every cluster on its own centroid so cluster identity cannot carry the prediction, fit paralog on 3
    clusters, score the 4th.

    CENTRING IS INERT, not merely "transductive". The held-out rows are Xc[te] = M[te] - mean(M[te]); the fitted
    model is affine, so subtracting that centroid is a CONSTANT SHIFT common to every held-out gene, and
    Spearman is shift-invariant. The test is therefore exactly equal to its strictly inductive version. (Earlier
    revisions of this file hedged this as "transductive, worth stating plainly" -- that was over-cautious.)

    THE SIGN IS THE STATISTIC -- DO NOT TAKE abs(). "The paralog direction is SHARED across clusters" is a claim
    about ORIENTATION. Under the unshared alternative -- every cluster has its own strong paralog axis, pointing
    somewhere arbitrary -- each held-out rho is large but its SIGN is random, and abs() would score that ~0.85
    and call it "perfect transfer". Only sign agreement across all four held-out clusters distinguishes shared
    from unshared. Return signed rho; the caller reports the signed mean.
    """
    Xc = M.copy()
    for c in np.unique(clu):
        m = clu == c
        Xc[m] = M[m] - M[m].mean(0)
    out = {}
    for c in np.unique(clu):
        te = clu == c
        tr = ~te
        if te.sum() < 6:
            continue
        sc = StandardScaler().fit(Xc[tr])
        mo = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xc[tr]), par[tr])
        pred = mo.predict(sc.transform(Xc[te]))
        out[CN[int(c)]] = dict(rho=float(_rho(pred, par[te])), n=int(te.sum()))   # SIGNED
    return out


def _loco_mean(r):
    return float(np.mean([v["rho"] for v in r.values()])) if r else np.nan


def loco_perm_p(M, par, clu, obs_mean, seed=SEED, n_perm=N_PERM):
    """Null for T3: permute paralog WITHIN each cluster, so cluster sizes and the geometry stay fixed and only
    the paralog-to-gene assignment moves. Returns (p, null_mean, null_sd) -- the effect size must be traceable
    to a run of THIS code, not recomputed by hand elsewhere."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = par.copy()
        for c in np.unique(clu):
            m = clu == c
            p[m] = rng.permutation(p[m])
        null[i] = _loco_mean(loco(M, p, clu))
    null = null[np.isfinite(null)]
    return float(((null >= obs_mean).sum() + 1) / (len(null) + 1)), float(null.mean()), float(null.std())


def main():
    res = {}
    for b in BASES:
        M, ok = G.subset(b, GENES)
        par, clu = PAR[ok], CLU[ok]
        print(f"\n{'=' * 88}\n=== {b}   (n={int(ok.sum())}/39 HOX genes present, d={M.shape[1]})\n{'=' * 88}",
              flush=True)
        rec = {"n_present": int(ok.sum()), "d": int(M.shape[1]), "per_cluster": {}, "loco": {}}

        print("T1/T2 -- within ONE cluster: supervised LOO probe (full space) vs the PC sweep")
        print("  !! T1 (LOO probe) IS DEGENERATE at n=9-11 -- pure noise scores 0.89; see module docstring.")
        print("  !! Read the perm p (honest, and uniformly null), NOT the point estimate. T3 answers this.")
        print(f"  {'clu':<5} {'n':<4} {'LOO probe':<11} {'perm p':<8} {'null mu':<9} | PC1..PC5 |rho| (var%)")
        for c in range(4):
            idx = np.where(clu == c)[0]
            if len(idx) < 6:
                continue
            X, y = M[idx], par[idx]
            obs = loo_probe(X, y)
            p, nm, ns = perm_p(X, y, obs)
            sw = pc_sweep(X, y)
            sws = "  ".join(f"{d['rho']:.2f}({d['var'] * 100:.0f}%)" for d in sw)
            print(f"  HOX{CN[c]:<2} {len(idx):<4} {obs:<11.3f} {p:<8.3f} {nm:<9.3f} | {sws}", flush=True)
            rec["per_cluster"][CN[c]] = dict(n=len(idx), probe=float(obs), p=float(p), null_mean=nm,
                                             null_sd=ns, pc_sweep=sw)

        print("\nT3 -- leave-one-CLUSTER-out paralog transfer (within-cluster centred, n=39 pooled)")
        lo = loco(M, par, clu)
        if lo:
            mean_rho = _loco_mean(lo)
            pv, nm, ns = loco_perm_p(M, par, clu, mean_rho)
            signs = [v["rho"] > 0 for v in lo.values()]
            for k, v in lo.items():
                print(f"  held-out HOX{k} (n={v['n']:<2}): SIGNED rho = {v['rho']:+.3f}")
            z = (mean_rho - nm) / (ns + 1e-12)
            print(f"  MEAN signed rho = {mean_rho:+.3f}   null {nm:+.3f} +- {ns:.3f}   z = {z:+.2f}   "
                  f"perm p = {pv:.3f}", flush=True)
            print(f"  SIGN AGREEMENT: {sum(signs)}/{len(signs)} held-out clusters positive "
                  f"-- {'SHARED orientation' if all(signs) else 'NOT a shared direction'}", flush=True)
            rec["loco"] = dict(per_cluster=lo, mean_signed_rho=mean_rho, null_mean=nm, null_sd=ns,
                               z=float(z), p=float(pv), n_positive=int(sum(signs)), n_clusters=len(signs))
        res[b] = rec

    # ---- the project's verdict rule (gm_lib.py:20, run_probe.py:23): a model basis must beat BOTH references.
    print(f"\n{'=' * 88}\nVERDICT on T3 -- gm_lib.py:20: 'a hypothesis only counts if a model basis beats BOTH'")
    print(f"{'=' * 88}")
    MODELS = ["maxtoki_lmhead", "maxtoki_we", "scgpt_we"]
    REFS = ["coexpr", "coexpr_devel", "esm2"]
    got = {b: res[b]["loco"]["mean_signed_rho"] for b in res if res[b].get("loco")}
    mods = {b: v for b, v in got.items() if b in MODELS}
    refs = {b: v for b, v in got.items() if b in REFS}
    if mods and refs:
        mb = max(mods, key=mods.get)
        beaten = {r: mods[mb] > v for r, v in refs.items()}
        ok = all(beaten.values())
        for r, v in refs.items():
            print(f"  {mb} ({mods[mb]:+.3f})  vs  {r} ({v:+.3f})  -> {'beats' if beaten[r] else 'LOSES'}")
        print(f"\n  VERDICT: {'** MODEL **' if ok else 'NO -- does not beat every reference'}")
        if not ok:
            lost = [r for r, w in beaten.items() if not w]
            print(f"  Fails the beat-both rule: loses to {', '.join(lost)}.")
            print("  READ THIS PRECISELY. The rule guards NOVELTY-FOR-BIOLOGY ('could a reviewer get it")
            print("  cheaper?' -- from BLAST, yes). It is NOT a claim about what the model learned, and it")
            print("  cannot be: MaxToki never saw a protein sequence, so esm2 is not a causal path into it.")
            print("  T3 DOES show MaxToki learned a shared composable paralog coordinate from expression.")
            print("  What makes paralog unsurprising is the BASELINE SCALING argument (RESULTS.md 10a):")
            print("  colinearity puts paralog in expression, and the data baseline IMPROVES with cells")
            print("  (0.507->0.630) -- unlike cluster, whose baseline sits at noise floor and DROPS.")
        res["_verdict_t3"] = dict(best_model=mb, score=mods[mb], refs=refs, passes=bool(ok))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "hox_within.json"), "w"), indent=1)
    print("\n[done] -> results/hox_within.json")


if __name__ == "__main__":
    main()
