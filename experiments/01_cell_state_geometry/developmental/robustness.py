"""Stress-test the load-bearing claim in baseline.py before letting it stand.

The whole reframing rests on ONE comparison: on blood, expression-PCA out-steers every foundation model. If that
is a single-seed fluke, or an artefact of picking 2000 HVGs, the reframing is wrong. route_steering's own caveats
name exactly these two knobs ("Single train/test split (SEED=0)"), so they are the ones to turn.

Two knobs, crossed:
  * train/test split seed   -> 0, 1, 2, 3, 4     (the steering split, and the source-cell draw)
  * expression HVG count    -> 1000, 2000, 5000  (how the baseline space is built; the models are unaffected)

Reported: constrained advance at tau=1.3 for `linear_proj` (the method), expression-PCA vs each model, on Setty.
The claim survives iff expression-PCA beats scGPT in the large majority of cells, across both knobs.

Run:  ../../.venv/bin/python robustness.py
Out:  results/robustness_setty.json
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from common import RESULTS, load_expression                                                  # noqa: E402
from tangent_diagnostic import path as npz_path, prep, zscore, unit, steer, BUDGETS, T_STAR  # noqa: E402
from local_steering import project_constant, constrained_advance                             # noqa: E402
from baseline import shared_cells, MODELS                                                    # noqa: E402

TISSUE = "setty"
SEEDS = [0, 1, 2, 3, 4]
HVGS = [1000, 2000, 5000]


def ca13(Xz, y, seed):
    """Constrained advance @tau=1.3 for `linear_proj` — the method — at a given split seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    Xtr, ytr, Xte, yte = Xz[tr], y[tr], Xz[te], y[te]
    w = unit(Ridge(alpha=10.0).fit(Xtr, ytr).coef_)
    judge = KNeighborsRegressor(15).fit(Xtr, ytr)
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]))
    src = Xte[yte <= np.quantile(yte, 1 / 3)]
    if len(src) > 300:
        src = src[rng.choice(len(src), 300, replace=False)]
    out = {}
    for name, fn in [("linear", lambda x: np.tile(w, (len(x), 1))),
                     ("linear_proj", project_constant(Xtr, w))]:
        pts = steer(src, fn, 0.25 * d0, BUDGETS)
        base = judge.predict(pts[0])
        rows = {T: dict(off_manifold_ratio=float(nn1.kneighbors(pts[T])[0][:, 0].mean() / d0),
                        advance=float((judge.predict(pts[T]) - base).mean())) for T in BUDGETS}
        out[name] = constrained_advance(rows, 1.3)
    return out["linear_proj"]


def main():
    cells, _ = shared_cells(TISSUE)
    E, _, _ = load_expression(TISSUE, cells)
    var = E.var(0)
    z0 = np.load(npz_path("scgptbin", TISSUE), allow_pickle=True)
    pos = {c: i for i, c in enumerate(z0["cell_idx"].astype(int))}
    y = zscore(z0["pseudotime"].astype(np.float64)[[pos[c] for c in cells]])

    model_space = {}
    for m in MODELS:
        z = np.load(npz_path(m, TISSUE), allow_pickle=True)
        p = {c: i for i, c in enumerate(z["cell_idx"].astype(int))}
        model_space[m] = prep(z["emb"].astype(np.float64)[[p[c] for c in cells]])

    print(f"\n{'='*96}\n=== Robustness of the load-bearing claim: does expression-PCA really out-steer the models?")
    print(f"=== Setty, n={len(cells)}. Metric: constrained advance @tau=1.3, rule = linear_proj (the method).\n")
    print(f"  {'seed':>4} {'expr(1k)':>9} {'expr(2k)':>9} {'expr(5k)':>9} | "
          + " ".join(f"{m:>10}" for m in MODELS))
    rows, beats = [], {m: 0 for m in MODELS}
    n_cells = 0
    for seed in SEEDS:
        ex = {}
        for h in HVGS:
            ex[h] = ca13(prep(E[:, np.argsort(-var)[:h]].astype(np.float64)), y, seed)
        mv = {m: ca13(model_space[m], y, seed) for m in MODELS}
        rows.append(dict(seed=seed, expr={str(h): ex[h] for h in HVGS}, models=mv))
        print(f"  {seed:>4} " + " ".join(f"{ex[h]:>9.2f}" for h in HVGS) + " | "
              + " ".join(f"{mv[m]:>10.2f}" for m in MODELS))
        for h in HVGS:
            for m in MODELS:
                n_cells += 1
                if ex[h] > mv[m]:
                    beats[m] += 1

    per_model_cells = len(SEEDS) * len(HVGS)
    print(f"\n  expression-PCA beats the model, over {len(SEEDS)} seeds x {len(HVGS)} HVG settings:")
    for m in MODELS:
        print(f"    vs {m:<12} {beats[m]:>2}/{per_model_cells}")
    tot = sum(beats.values())
    print(f"\n  overall: expression-PCA wins {tot}/{n_cells} model x seed x HVG cells "
          f"({100*tot/n_cells:.0f}%)")
    verdict = ("HOLDS — the baseline result is not a seed or HVG artefact"
               if tot >= 0.75 * n_cells else
               "FRAGILE — the baseline advantage depends on the knobs; do not lean on it")
    print(f"  -> {verdict}")
    json.dump(dict(tissue=TISSUE, seeds=SEEDS, hvgs=HVGS, rows=rows, beats=beats,
                   n_cells=n_cells, frac=tot / n_cells, verdict=verdict),
              open(os.path.join(RESULTS, "robustness_setty.json"), "w"), indent=1)
    print(f"\n[done] -> results/robustness_setty.json")


if __name__ == "__main__":
    main()
