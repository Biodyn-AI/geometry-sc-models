"""Route Q — fit the supervised low-rank Q on one (model, dataset) substrate.

Prereg §1/§3/§4/§5.  Produces:
  results/q_<model>_<dataset>.json   metrics: r2_lin, r2_quad, delta_q, bootstrap CI, rank sweep,
                                     abundance-off variant, chosen (r, lam) per fold
  results/q_<model>_<dataset>.npz    Q_full (fit on all cells) + per-fold Q matrices (stability)

Usage:  ../../.venv/bin/python run_substrate.py scgpt setty
"""
import os, sys, json, time
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import r2_score

import qfit

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)


def rank_sweep(X, y, lam, seed=qfit.SEED):
    """Descriptive: out-of-fold delta_q as a function of rank, at a FIXED lam (the modal one).

    Not the nested headline -- it answers 'is the nonlinearity low-rank?' by showing the R^2(r) curve.
    """
    from sklearn.model_selection import KFold
    n = len(y)
    out = {}
    for r in qfit.RANKS:
        p_lin, p_quad = np.zeros(n), np.zeros(n)
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
            pl, pq, _, _, _ = qfit._fold_scores(X, y, tr, te, r, lam, seed + k, True)
            p_lin[te], p_quad[te] = pl, pq
        out[r] = float(r2_score(y, p_quad) - r2_score(y, p_lin))
    return out


def full_data_Q(X, y, r, lam, seed=qfit.SEED):
    """Fit Q on ALL cells (for eigendecomposition). R^2 claims never come from this fit."""
    xh, c, _, _ = qfit.preprocess(X, X, True)
    lin = qfit.LinearPart(xh, c, y)
    m = qfit.fit_q(lin, xh, y, r, lam, seed=seed)
    return m.Q(), xh, c, lin


def main(model, dataset, quick=False):
    t0 = time.time()
    X, y = qfit.load(model, dataset)
    if X is None:
        print(f"[skip] {model}/{dataset}: no cache"); return
    print(f"[{model}/{dataset}] n={len(y)} d={X.shape[1]}{' [quick]' if quick else ''}", flush=True)

    # ---- headline: nested CV (prereg §3) ----
    cv = qfit.outer_cv(X, y, return_models=True)
    lo, hi = qfit.bootstrap_delta(y, cv["p_lin"], cv["p_quad"])
    modal_r, modal_lam = Counter([tuple(c) for c in cv["cfgs"]]).most_common(1)[0][0]
    print(f"  R2_lin={cv['r2_lin']:.4f}  R2_quad={cv['r2_quad']:.4f}  "
          f"delta_Q={cv['delta_q']:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  cfgs={cv['cfgs']}", flush=True)

    # quick mode (generality substrates): headline CV only. The abundance guard and rank sweep are
    # descriptive extras and are run on the primary + the mouse-pancreas negative control only.
    cv_noab, sweep = None, None
    if not quick:
        # ---- abundance guard (prereg §5.4) ----
        cv_noab = qfit.outer_cv(X, y, use_abundance=False)
        print(f"  abundance-off: delta_Q={cv_noab['delta_q']:+.4f} (R2_lin={cv_noab['r2_lin']:.4f})", flush=True)

        # ---- rank sweep (is it low-rank?) ----
        sweep = rank_sweep(X, y, modal_lam)
        print("  rank sweep dQ: " + "  ".join(f"r={r}:{v:+.4f}" for r, v in sweep.items()), flush=True)

    # ---- Q for eigendecomposition + per-fold stability ----
    Q_full, xh_all, c_all, _ = full_data_Q(X, y, modal_r, modal_lam)
    fold_Qs = np.stack([m["Q"] for m in cv["models"]])

    np.savez(os.path.join(RES, f"q_{model}_{dataset}.npz"),
             Q_full=Q_full, fold_Qs=fold_Qs, y=y,
             p_lin=cv["p_lin"], p_quad=cv["p_quad"])

    out = dict(model=model, dataset=dataset, n=int(len(y)), d=int(X.shape[1]), quick=bool(quick),
               r2_lin=cv["r2_lin"], r2_quad=cv["r2_quad"], delta_q=cv["delta_q"],
               delta_q_ci=[lo, hi], cfgs=cv["cfgs"], modal_r=int(modal_r), modal_lam=float(modal_lam),
               delta_q_no_abundance=cv_noab["delta_q"] if cv_noab else None,
               r2_lin_no_abundance=cv_noab["r2_lin"] if cv_noab else None,
               rank_sweep={str(k): v for k, v in sweep.items()} if sweep else None,
               seconds=time.time() - t0)
    json.dump(out, open(os.path.join(RES, f"q_{model}_{dataset}.json"), "w"), indent=1)
    print(f"  -> results/q_{model}_{dataset}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], quick="--quick" in sys.argv)
