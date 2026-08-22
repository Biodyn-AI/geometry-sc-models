"""Route Q — the pre-registered redundancy discriminator applied to Q's interaction subspace.

Prereg §4 Q3 (thresholds verbatim from route_b/PREREGISTRATION.md §3):
    R^2_x   : out-of-sample R^2 predicting the latent from a LINEAR ridge on the raw centered x
    R^2_SAE : out-of-sample R^2 predicting the latent from a LINEAR ridge on linear-atlas features h(x)
    genuinely non-linear  iff  R^2_x   < 0.25
    novel (not in atlas)  iff  R^2_SAE < 0.50
    headline              iff  both

Latents tested:
  (i)  z_i = (v_i . x_hat)^2 for the top eigen-directions of the fitted rank-r Q   [prereg-specified]
  (ii) q(x) = x_hat^T Q x_hat, the whole saddle feature                            [extension: arms.py
       showed the eigen-directions are NOT separately meaningful -- the predictive object is the
       indefinite contrast, so it must be tested as a latent in its own right]
  (iii) z_1 = (v_1 . x_hat)^2 of the directly-fit RANK-1 Q (the one interpretable squared direction)

Atlas baselines (prereg §5.5):
  h_bar : atlas encoded per gene TOKEN, mean-pooled per cell   <- in-distribution, INSTRUMENT OF RECORD
  h_hat : atlas encoded on the cell embedding                  <- out-of-distribution, reported for continuity
  shuffled h_bar                                               <- null, must give R^2 ~ 0

Usage: ../../.venv/bin/python discriminator.py scgpt setty
Out:   results/discriminator_<model>_<dataset>.json
"""
import os, sys, json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import qfit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ALPHAS = np.logspace(-2, 5, 12)


def cv_r2(P, z, seed=qfit.SEED):
    """Out-of-fold R^2 of predicting z from features P with a ridge (alpha picked on train by inner CV)."""
    from sklearn.linear_model import RidgeCV
    n = len(z)
    pred = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        m = RidgeCV(alphas=ALPHAS).fit(P[tr], z[tr])
        pred[te] = m.predict(P[te])
    return float(r2_score(z, pred))


def main(model, dataset):
    X, y = qfit.load(model, dataset)
    A = np.load(os.path.join(RES, f"arms_{model}_{dataset}.npz"))
    Q, V, w, v1 = A["Q"], A["V"], A["w"], A["v1"]

    feats = np.load(os.path.join(RES, f"atlas_feats_{model}_{dataset}.npz"))
    h_bar, h_hat = feats["h_bar"].astype(np.float64), feats["h_hat"].astype(np.float64)

    # x_hat exactly as the Q was fit (full-data preprocessing, prereg §2)
    xh, _, _, _ = qfit.preprocess(X, X, True)
    x_raw = X - X.mean(0)

    # keep only atlas features that ever fire (dead columns carry no information and slow the ridge)
    alive_bar = (np.abs(h_bar) > 0).any(0)
    alive_hat = (np.abs(h_hat) > 0).any(0)
    Hb, Hh = h_bar[:, alive_bar], h_hat[:, alive_hat]
    print(f"[{model}/{dataset}] alive atlas feats: h_bar {alive_bar.sum()}  h_hat {alive_hat.sum()}")

    rng = np.random.default_rng(qfit.SEED)
    Hb_shuf = Hb[rng.permutation(len(Hb))]

    order = np.argsort(-np.abs(w))
    latents = {}
    for rank_i, i in enumerate(order[:8], start=1):
        latents[f"eig{rank_i}(lam={w[i]:+.3f})"] = (xh @ V[:, i]) ** 2
    latents["q_saddle (x'Qx)"] = np.einsum("ij,jk,ik->i", xh, Q, xh)
    latents["rank1 (v1.x)^2"] = (xh @ v1) ** 2

    rows = []
    for name, z in latents.items():
        r2x = cv_r2(x_raw, z)
        r2b = cv_r2(Hb, z)
        r2h = cv_r2(Hh, z)
        r2n = cv_r2(Hb_shuf, z)
        nonlin = r2x < 0.25
        novel = r2b < 0.50
        rows.append(dict(latent=name, r2_x=r2x, r2_sae_hbar=r2b, r2_sae_hhat=r2h, r2_sae_shuffnull=r2n,
                         genuinely_nonlinear=bool(nonlin), novel=bool(novel), headline=bool(nonlin and novel)))
        print(f"  {name:24s} R2_x={r2x:6.3f}  R2_SAE(h_bar)={r2b:6.3f}  "
              f"R2_SAE(h_hat)={r2h:6.3f}  null={r2n:+6.3f}   "
              f"{'NONLIN' if nonlin else 'linear':>6s} {'NOVEL' if novel else 'redundant':>9s}")

    out = dict(model=model, dataset=dataset, n=int(len(y)),
               n_alive_hbar=int(alive_bar.sum()), n_alive_hhat=int(alive_hat.sum()),
               thresholds=dict(r2_x=0.25, r2_sae=0.50), latents=rows)
    json.dump(out, open(os.path.join(RES, f"discriminator_{model}_{dataset}.json"), "w"), indent=1)
    print(f"  -> results/discriminator_{model}_{dataset}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
