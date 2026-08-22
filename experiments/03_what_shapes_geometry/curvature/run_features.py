"""Route Q #1 — fit the supervised low-rank Q on the NAMED linear-atlas feature space.

Motivation. In RESULTS.md, Q fitted on the raw residual x is redundant with the atlas (R^2_SAE=0.79) but
its eigen-directions are NOT identifiable (gene lists change across equally-good fits). The fix: fit the
same Q on the atlas FEATURE activations h(x) instead of x. Then:
  - the linear baseline IS the atlas's own linear predictor, so Delta_Q directly measures
    "does a PRODUCT of atlas atoms predict pseudotime beyond the linear atlas?" -- the missing-product test;
  - Q's directions are combinations of SPARSE, NAMED features, so the interaction is identifiable and
    can be read as "feature A x feature B" with A, B already annotated.

Uses h_bar (atlas encoded per gene TOKEN, mean-pooled per cell) from extract_atlas_feats.py -- the
domain-correct, in-distribution feature representation (VE 0.95), NOT the crippled cell-embedding encoding.

Outputs:
  results/feat_q_scgpt_setty.json   Delta_Q, CI, rank sweep, top named eigen-directions, top named products
  results/feat_q_scgpt_setty.npz    Q_full (in ALIVE-feature space), alive_idx, per-fold Q
Usage: ../../.venv/bin/python run_features.py
"""
import os, sys, json, time
import numpy as np
import torch
from collections import Counter

import qfit, feat_naming as FN

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")


def load_feats(model="scgpt", dataset="setty"):
    X, y = qfit.load(model, dataset)
    F = np.load(os.path.join(RES, f"atlas_feats_{model}_{dataset}.npz"))
    h = F["h_bar"].astype(np.float64)
    alive = (np.abs(h) > 0).any(0)
    return h[:, alive], np.where(alive)[0], y


def named_products(Q, alive_idx, topn=25):
    """Rank the largest off-diagonal |Q_ij| -- the product of atlas features i and j."""
    A = np.abs(Q).copy()
    np.fill_diagonal(A, 0.0)
    n = Q.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = A[iu]
    order = np.argsort(-vals)[:topn]
    out = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        fi, fj = int(alive_idx[i]), int(alive_idx[j])
        out.append(dict(feat_i=fi, feat_j=fj, q_ij=float(Q[i, j]),
                        label_i=FN.short(fi), label_j=FN.short(fj)))
    return out


def named_eigdir(Q, alive_idx, v, topn=8):
    load = v
    order = np.argsort(-np.abs(load))[:topn]
    return [dict(feat=int(alive_idx[o]), loading=float(load[o]), label=FN.short(int(alive_idx[o])))
            for o in order]


def rank_sweep(H, y, lam):
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    n = len(y); out = {}
    for r in qfit.RANKS:
        pl, pq = np.zeros(n), np.zeros(n)
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=qfit.SEED).split(np.arange(n))):
            plq = qfit._fold_scores(H, y, tr, te, r, lam, qfit.SEED + k, True)
            pl[te], pq[te] = plq[0], plq[1]
        out[r] = float(r2_score(y, pq) - r2_score(y, pl))
    return out


def main(model="scgpt", dataset="setty"):
    t0 = time.time()
    H, alive_idx, y = load_feats(model, dataset)
    print(f"[{model}/{dataset}] feature-space Q: n={len(y)} alive_feats={H.shape[1]}", flush=True)

    cv = qfit.outer_cv(H, y, return_models=True)
    lo, hi = qfit.bootstrap_delta(y, cv["p_lin"], cv["p_quad"])
    modal_r, modal_lam = Counter([tuple(c) for c in cv["cfgs"]]).most_common(1)[0][0]
    print(f"  R2_lin(atlas)={cv['r2_lin']:.4f}  R2_quad={cv['r2_quad']:.4f}  "
          f"delta_Q={cv['delta_q']:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  modal=({modal_r},{modal_lam})", flush=True)

    sweep = rank_sweep(H, y, modal_lam)
    print("  rank sweep dQ: " + "  ".join(f"r={r}:{v:+.4f}" for r, v in sweep.items()), flush=True)

    # Q on all cells for reading
    xh, c, _, _ = qfit.preprocess(H, H, True)
    lin = qfit.LinearPart(xh, c, y)
    Qm = qfit.fit_q(lin, xh, y, modal_r, modal_lam, seed=qfit.SEED)
    Q = Qm.Q()
    w, V = np.linalg.eigh(Q)
    order = np.argsort(-np.abs(w))
    w, V = w[order], V[:, order]

    eigs = []
    for rank_i in range(min(4, len(w))):
        eigs.append(dict(rank=rank_i + 1, eigenvalue=float(w[rank_i]),
                         top_features=named_eigdir(Q, alive_idx, V[:, rank_i])))
    prods = named_products(Q, alive_idx, topn=25)

    print("\n  TOP NAMED PRODUCTS (feature_i x feature_j, by |Q_ij|):")
    for p in prods[:12]:
        print(f"    {p['q_ij']:+.3f}  {p['label_i']}  ×  {p['label_j']}", flush=True)

    print("\n  TOP EIGEN-DIRECTION (lambda={:+.3f}) features:".format(w[0]))
    for f in eigs[0]["top_features"]:
        print(f"    {f['loading']:+.3f}  {f['label']}", flush=True)

    np.savez(os.path.join(RES, f"feat_q_{model}_{dataset}.npz"),
             Q_full=Q, alive_idx=alive_idx, fold_Qs=np.stack([m["Q"] for m in cv["models"]]),
             p_lin=cv["p_lin"], p_quad=cv["p_quad"], y=y)
    out = dict(model=model, dataset=dataset, n=int(len(y)), n_alive_feats=int(H.shape[1]),
               r2_lin_atlas=cv["r2_lin"], r2_quad=cv["r2_quad"], delta_q=cv["delta_q"],
               delta_q_ci=[lo, hi], modal_r=int(modal_r), modal_lam=float(modal_lam),
               cfgs=cv["cfgs"], rank_sweep={str(k): v for k, v in sweep.items()},
               top_eigendirections=eigs, top_products=prods, seconds=time.time() - t0)
    json.dump(out, open(os.path.join(RES, f"feat_q_{model}_{dataset}.json"), "w"), indent=1)
    print(f"\n  -> results/feat_q_{model}_{dataset}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
