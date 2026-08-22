"""Route Q — how many interaction directions actually carry developmental time?

The rank sweep in run_substrate.py fits an independent Q per rank, so its participation ratio is
partly *forced* by the rank constraint (a rank-8 fit trivially has PR<=8). The honest question is:
fit Q at a generous rank (r=32), eigendecompose it, and ask how much held-out curvature survives when
we keep only the top-j eigen-directions.

Per fold: fit Q(r=32) on train -> eigendecompose -> truncate to top-j -> refit the linear part on
train with the truncated quad feature -> score the held-out fold. Fully out-of-fold.

Also reports the eigenmass profile of the r=32 Q (predictive share vs eigenvalue share can differ,
because directions have different variance of (v.x_hat)^2).

Usage: ../../.venv/bin/python spectrum.py scgpt setty
Out:   results/spectrum_<model>_<dataset>.json
"""
import os, sys, json
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import qfit

torch.set_num_threads(6)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

R_BIG = 32
JS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def truncated_q(Q, j):
    w, V = np.linalg.eigh(Q)
    o = np.argsort(-np.abs(w))
    w, V = w[o][:j], V[:, o][:, :j]
    return (V * w) @ V.T


def main(model, dataset):
    X, y = qfit.load(model, dataset)
    prim = json.load(open(os.path.join(RES, f"q_{model}_{dataset}.json")))
    lam = prim["modal_lam"]
    n = len(y)

    p_lin = np.zeros(n)
    p_j = {j: np.zeros(n) for j in JS}
    masses, prs, sign_splits = [], [], []

    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=qfit.SEED).split(np.arange(n))):
        xh_tr, c_tr, xh_te, c_te = qfit.preprocess(X[tr], X[te], True)
        lin = qfit.LinearPart(xh_tr, c_tr, y[tr])
        p_lin[te] = lin.predict_linear(xh_te, c_te)

        m = qfit.fit_q(lin, xh_tr, y[tr], R_BIG, lam, seed=qfit.SEED + k)
        Q = m.Q()

        w = np.linalg.eigvalsh(Q)
        aw = np.sort(np.abs(w))[::-1]
        tot = aw.sum()
        masses.append((aw[:8] / tot).tolist())
        prs.append(float((aw.sum() ** 2) / (w ** 2).sum()))
        wk = w[np.abs(w) > 1e-6 * np.abs(w).max()]
        sign_splits.append((int((wk > 0).sum()), int((wk < 0).sum())))

        for j in JS:
            Qj = truncated_q(Q, j)
            q_tr = np.einsum("ij,jk,ik->i", xh_tr, Qj, xh_tr)
            q_te = np.einsum("ij,jk,ik->i", xh_te, Qj, xh_te)
            lin.refit_with_quad(y[tr], q_tr)
            p_j[j][te] = lin.predict_quad(xh_te, c_te, q_te)

    r2l = r2_score(y, p_lin)
    curve = {str(j): float(r2_score(y, p_j[j]) - r2l) for j in JS}
    full = curve[str(R_BIG)]

    print(f"[{model}/{dataset}] r2_lin={r2l:.4f}  delta_Q(r=32,full spectrum)={full:+.4f}")
    print("  eigen-truncation curve (out-of-fold delta_Q, and % of full):")
    for j in JS:
        v = curve[str(j)]
        print(f"    top-{j:<2d}  dQ={v:+.4f}   {100*v/full if full else 0:5.1f}%")
    mm = np.mean(masses, 0)
    print(f"  eigenmass (mean over folds), top-8: {np.array2string(mm, precision=3)}")
    print(f"  top-1 mass={mm[0]:.3f}  top-2 mass={mm[:2].sum():.3f}  PR={np.mean(prs):.2f}")
    print(f"  sign split (pos,neg) per fold: {sign_splits}")

    out = dict(model=model, dataset=dataset, r_big=R_BIG, lam=lam, r2_lin=float(r2l),
               delta_q_full=full, curve=curve,
               eigenmass_top8_mean=mm.tolist(),
               eigmass_top1=float(mm[0]), eigmass_top2=float(mm[:2].sum()),
               participation_ratio=float(np.mean(prs)), sign_splits=sign_splits)
    json.dump(out, open(os.path.join(RES, f"spectrum_{model}_{dataset}.json"), "w"), indent=1)
    print(f"  -> results/spectrum_{model}_{dataset}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
