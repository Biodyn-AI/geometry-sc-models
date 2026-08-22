"""Route Q — identifiability control for the arm split (prereg deviation D7).

Held-out R^2 identifies Q's predictive SUBSPACE, not Q. This script re-fits Q at several (r, lambda)
settings with statistically indistinguishable held-out Delta_Q, and asks whether the sign-split
("positive arm" vs "negative arm") reading is stable. It is not: arms flip from individually-harmful
to individually-useful as lambda grows, in BOTH scGPT and Geneformer.

Out: results/arms_identifiability.json
"""
import os, json
import numpy as np, torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import qfit, arms

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
CFGS = [("scgpt", "setty", 8, 1e-3), ("scgpt", "setty", 32, 1e-3), ("scgpt", "setty", 32, 1e-2),
        ("geneformer", "setty", 8, 1e-3), ("geneformer", "setty", 8, 1e-2), ("geneformer", "setty", 32, 1e-2)]


def run(model, ds, r, lam):
    X, y = qfit.load(model, ds); n = len(y)
    P = {k: np.zeros(n) for k in ("full", "pos", "neg")}; pl = np.zeros(n)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=qfit.SEED).split(np.arange(n))):
        a, c, b, ce = qfit.preprocess(X[tr], X[te], True)
        L = qfit.LinearPart(a, c, y[tr]); pl[te] = L.predict_linear(b, ce)
        Q = qfit.fit_q(L, a, y[tr], r, lam, seed=qfit.SEED + k).Q()
        Qp, Qn, *_ = arms.split_arms(Q)
        for nm, QQ in [("full", Q), ("pos", Qp), ("neg", -Qn)]:
            L.refit_with_quad(y[tr], arms.qf(a, QQ))
            P[nm][te] = L.predict_quad(b, ce, arms.qf(b, QQ))
    r2l = r2_score(y, pl)
    d = {k: float(r2_score(y, P[k]) - r2l) for k in P}
    print(f"{model:11s} {ds:6s} r={r:<2d} lam={lam:<6g}  full={d['full']:+.4f}  "
          f"pos={d['pos']:+.4f} ({100*d['pos']/d['full']:4.0f}%)  "
          f"neg={d['neg']:+.4f} ({100*d['neg']/d['full']:4.0f}%)", flush=True)
    return dict(model=model, dataset=ds, r=r, lam=lam, r2_lin=float(r2l), **d)


if __name__ == "__main__":
    out = [run(*c) for c in CFGS]
    json.dump(out, open(os.path.join(RES, "arms_identifiability.json"), "w"), indent=1)
    print("-> results/arms_identifiability.json")
