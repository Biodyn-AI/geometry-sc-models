"""Route Q — read the saddle. (prereg §4 Q2 + §7)

Finding that motivates this file: eigenvalue-ordered truncation of Q is NOT a valid decomposition here.
Because x_hat is unit-norm, the squared features z_i = (v_i . x_hat)^2 are strongly coupled (they sum to
the projected norm), so any partial sum lambda_1 z_1 carries a large common "how much of x_hat lies in
this subspace" component that the LINEAR part cannot remove (it is quadratic). Empirically, truncating
the rank-8 Q to its top eigen-direction scores -0.0200 (WORSE than linear) while a directly-fit rank-1 Q
scores +0.0101. The predictive signal is a CONTRAST between opposite-sign directions.

So the interpretable object is the signed split (prereg §7):
    Q = Q+ - Q-,   Q+ = sum_{l_i>0} l_i v_i v_i^T,   Q- = sum_{l_i<0} |l_i| v_i v_i^T
    x^T Q x = ||A x||^2 - ||B x||^2      (a difference of two squared projections)

We score, out-of-fold: full Q, Q+ alone, Q- alone. If neither arm alone works but the contrast does,
the developmental nonlinearity is irreducibly a saddle -- it cannot be written as a few additive
squared features, i.e. it is not decomposable into independent bilinear-SAE-style latents.

Usage: ../../.venv/bin/python arms.py scgpt setty
Out:   results/arms_<model>_<dataset>.json  (+ .npz with the full-data arm bases)
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


def split_arms(Q, tol=1e-6):
    w, V = np.linalg.eigh(Q)
    keep = np.abs(w) > tol * np.abs(w).max()
    w, V = w[keep], V[:, keep]
    pos, neg = w > 0, w < 0
    Qp = (V[:, pos] * w[pos]) @ V[:, pos].T
    Qn = (V[:, neg] * (-w[neg])) @ V[:, neg].T      # positive semidefinite
    return Qp, Qn, w, V, pos, neg


def qf(xh, Q):
    return np.einsum("ij,jk,ik->i", xh, Q, xh)


def main(model, dataset):
    X, y = qfit.load(model, dataset)
    prim = json.load(open(os.path.join(RES, f"q_{model}_{dataset}.json")))
    r, lam = prim["modal_r"], prim["modal_lam"]
    n = len(y)

    variants = ["full", "pos_only", "neg_only", "rank1"]
    P = {v: np.zeros(n) for v in variants}
    p_lin = np.zeros(n)
    arm_dims = []

    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=qfit.SEED).split(np.arange(n))):
        xh_tr, c_tr, xh_te, c_te = qfit.preprocess(X[tr], X[te], True)
        lin = qfit.LinearPart(xh_tr, c_tr, y[tr])
        p_lin[te] = lin.predict_linear(xh_te, c_te)

        Q = qfit.fit_q(lin, xh_tr, y[tr], r, lam, seed=qfit.SEED + k).Q()
        Qp, Qn, w, V, pos, neg = split_arms(Q)
        arm_dims.append((int(pos.sum()), int(neg.sum())))

        Q1 = qfit.fit_q(lin, xh_tr, y[tr], 1, lam, seed=qfit.SEED + k).Q()

        for name, QQ in [("full", Q), ("pos_only", Qp), ("neg_only", -Qn), ("rank1", Q1)]:
            q_tr, q_te = qf(xh_tr, QQ), qf(xh_te, QQ)
            lin.refit_with_quad(y[tr], q_tr)
            P[name][te] = lin.predict_quad(xh_te, c_te, q_te)

    r2l = r2_score(y, p_lin)
    d = {v: float(r2_score(y, P[v]) - r2l) for v in variants}
    print(f"[{model}/{dataset}] r2_lin={r2l:.4f}   rank={r} lam={lam}")
    print(f"  delta_Q  full (saddle)      = {d['full']:+.4f}")
    print(f"  delta_Q  positive arm only  = {d['pos_only']:+.4f}   ({100*d['pos_only']/d['full']:.0f}% of full)")
    print(f"  delta_Q  negative arm only  = {d['neg_only']:+.4f}   ({100*d['neg_only']/d['full']:.0f}% of full)")
    print(f"  delta_Q  directly-fit rank1 = {d['rank1']:+.4f}   ({100*d['rank1']/d['full']:.0f}% of full)")
    print(f"  arm dims (pos,neg) per fold: {arm_dims}")

    # ---- full-data arms + rank-1 direction, for gene decoding ----
    xh, c, _, _ = qfit.preprocess(X, X, True)
    lin = qfit.LinearPart(xh, c, y)
    Q = qfit.fit_q(lin, xh, y, r, lam, seed=qfit.SEED).Q()
    Qp, Qn, w, V, pos, neg = split_arms(Q)
    Q1 = qfit.fit_q(lin, xh, y, 1, lam, seed=qfit.SEED).Q()
    w1, V1 = np.linalg.eigh(Q1)
    v1 = V1[:, np.argmax(np.abs(w1))]

    b = lin.b[:xh.shape[1]]
    b = b / (np.linalg.norm(b) + 1e-12)
    cos_v1_blin = float(abs(v1 @ b))
    # how much of the linear pseudotime direction lies in each arm's span
    cos_b_pos = float(np.linalg.norm(V[:, pos].T @ b))
    cos_b_neg = float(np.linalg.norm(V[:, neg].T @ b))
    print(f"  |cos(rank1 dir, linear pseudotime dir)| = {cos_v1_blin:.3f}")
    print(f"  ||proj of linear dir onto pos arm|| = {cos_b_pos:.3f}   onto neg arm = {cos_b_neg:.3f}")

    np.savez(os.path.join(RES, f"arms_{model}_{dataset}.npz"),
             Q=Q, V=V, w=w, pos=pos, neg=neg, v1=v1, b_lin=b,
             mu=X.mean(0), sd=np.where(X.std(0) < 1e-8, 1.0, X.std(0)))

    out = dict(model=model, dataset=dataset, r=int(r), lam=float(lam), r2_lin=float(r2l),
               delta_q=d, arm_dims=arm_dims,
               eigenvalues=w[np.argsort(-np.abs(w))].tolist(),
               cos_rank1_vs_linear_dir=cos_v1_blin,
               proj_lineardir_pos_arm=cos_b_pos, proj_lineardir_neg_arm=cos_b_neg)
    json.dump(out, open(os.path.join(RES, f"arms_{model}_{dataset}.json"), "w"), indent=1)
    print(f"  -> results/arms_{model}_{dataset}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
