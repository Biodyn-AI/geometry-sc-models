"""HOW is sub-chromosomal position encoded? — geometry of the position signal (for the paper + a figure).

genome_position2.py established THAT continuous within-chromosome position is decodable, model-specifically,
leakage-controlled (null-corrected excess: MaxToki +0.396 > esm2 +0.253 > coexpr +0.063). This characterises HOW:

  (1) LINEAR & leakage-clean per-gene evidence for the FIGURE. The 10-Mb group-blocked OOF statistic is corrupted
      by a mean-reversion artifact (a held-out contiguous block is predicted ~ the training mean, anti-correlated
      with its own position; this is the artifact genome_position2 corrects with a shuffle null). To get an
      artifact-free per-gene predicted-vs-true scatter we instead DE-DUPLICATE (drop any gene whose nearest
      same-chromosome neighbour is < DEDUP_MB away, killing tandem-duplicate leakage) and use RANDOM folds
      (which span the whole chromosome, so no mean-reversion). Clean signed Spearman per chromosome results.
  (2) SHARED AXIS vs per-chromosome. After removing chromosome identity (centre each chromosome), fit ONE
      position direction on all-but-one chromosome (target = within-chromosome percentile) and predict the
      held-out chromosome's percentile. Transfer > 0 -> a single genome-wide "position-along-the-chromosome"
      direction; transfer ~ 0 -> position is encoded chromosome-by-chromosome.

Basis: MaxToki lm_head (the model), ESM2 as the sequence contrast.

Run: ../../.venv/bin/python -u genome_position_geometry.py
Out: results/genome_position_geometry.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, rankdata

BASES = ["maxtoki_lmhead", "esm2"]
ALPHAS = np.logspace(0, 5, 12)
MINCHR = 200
DEDUP_MB = 2.0           # a gene is a "duplicate" only if a genomically-close (< this) gene is embedding-similar
DEDUP_COS = 0.85         # ...with cosine >= this. Targets near-identical tandem-duplicate embeddings only.
SEED = 0


def dedup(X, start, mb=DEDUP_MB, cos=DEDUP_COS):
    """Greedy removal of TANDEM-DUPLICATE leakage: keep a gene unless an already-kept gene is both genomically
    close (< mb) AND embedding-near-identical (cosine >= cos). Removes duplicate leakage without discarding
    ordinary neighbours, so the residual random-fold probe reflects genuine position encoding."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    o = np.argsort(start)
    kept_idx, keep = [], np.zeros(len(start), bool)
    for i in o:
        dup = False
        for j in kept_idx:
            if abs(start[i] - start[j]) < mb * 1e6 and float(Xn[i] @ Xn[j]) >= cos:
                dup = True; break
        if not dup:
            kept_idx.append(i); keep[i] = True
    return keep


def oof_random(X, y, alphas=ALPHAS, k=5):
    P = np.zeros(len(y))
    for tr, te in KFold(min(k, len(y)), shuffle=True, random_state=SEED).split(X):
        sc = StandardScaler().fit(X[tr])
        P[te] = RidgeCV(alphas=alphas).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return P


def main():
    C = coords()
    res = {}
    for b in BASES:
        M, syms = G.basis(b)
        pos_i = {s: i for i, s in enumerate(syms)}
        per_chr, scatter_true, scatter_pred = {}, [], []
        for c in AUTOSOMES:
            g = [s for s in C.index[C.chromosome == c] if s in pos_i]
            if len(g) < MINCHR:
                continue
            start = C.loc[g, "start"].values.astype(float)
            Xfull = M[[pos_i[s] for s in g]]
            keep = dedup(Xfull, start)
            if keep.sum() < 100:
                continue
            X = Xfull[keep]; y = start[keep]
            P = oof_random(X, y)                              # de-dup + random folds = leakage-clean, artifact-free
            rho = float(spearmanr(P, y).statistic)
            per_chr[c] = dict(rho=rho, n=int(keep.sum()))
            tr_pct = rankdata(y) / len(y); pr_pct = rankdata(P) / len(P)
            scatter_true += tr_pct.tolist(); scatter_pred += pr_pct.tolist()
        rhos = [v["rho"] for v in per_chr.values()]
        res[b] = dict(per_chr=per_chr, mean_rho=float(np.mean(rhos)), median_rho=float(np.median(rhos)),
                      n_chr=len(per_chr), scatter_true=scatter_true, scatter_pred=scatter_pred)
        print(f"[{b}] (1) de-dup + random-fold per-chr position (leakage-clean, artifact-free): "
              f"mean signed rho {np.mean(rhos):+.3f}, median {np.median(rhos):+.3f}, "
              f"{len(per_chr)} chromosomes", flush=True)

    # ---- (2) shared position axis: leave-one-chromosome-out percentile transfer, identity removed ----
    # centre each chromosome (removes chromosome identity), target = within-chr percentile, fit on
    # all-but-one chromosome's genes, predict held-out chromosome's percentile.
    M, syms = G.basis("maxtoki_lmhead"); pos_i = {s: i for i, s in enumerate(syms)}
    chroms = [c for c in AUTOSOMES if sum((C.chromosome == c) & C.index.isin(pos_i)) >= MINCHR]
    rows, symv, pctv, cv = [], [], [], []
    for c in chroms:
        g = [s for s in C.index[C.chromosome == c] if s in pos_i]
        Xc = M[[pos_i[s] for s in g]]; Xc = Xc - Xc.mean(0)   # identity removed
        pct = rankdata(C.loc[g, "start"].values.astype(float)) / len(g)
        rows.append(Xc); pctv.append(pct); cv.append(np.full(len(g), c))
    Xall = np.vstack(rows); pctall = np.concatenate(pctv); call = np.concatenate(cv)
    transfer = {}
    for c in chroms:
        te = call == c; tr = ~te
        sc = StandardScaler().fit(Xall[tr])
        mo = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xall[tr]), pctall[tr])
        pred = mo.predict(sc.transform(Xall[te]))
        transfer[c] = float(spearmanr(pred, pctall[te]).statistic)
    mt = float(np.mean(list(transfer.values())))
    res["shared_axis"] = dict(per_chr=transfer, mean_rho=mt, n_chr=len(transfer))
    print(f"\n(2) SHARED position axis (leave-one-chr-out, identity removed): mean transfer rho {mt:+.3f} "
          f"over {len(transfer)} chromosomes", flush=True)
    print("    -> if >0, ONE genome-wide 'position-along-the-chromosome' direction predicts position on a "
          "chromosome never used to fit it.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_position_geometry.json"), "w"))
    print("\n[done] -> results/genome_position_geometry.json")


if __name__ == "__main__":
    main()
