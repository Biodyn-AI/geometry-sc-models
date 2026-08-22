"""THE MISSING CONTROL: does the foundation model contribute anything, or is the geometry just the data?

Every number in route_steering is computed inside SOME model's embedding space. `random_proj` (NATIVE_DECODE)
shows the DIRECTION matters within scGPT's space. Nothing anywhere shows that scGPT's SPACE matters. The
baseline that settles it is the obvious one and it was never run:

    take log1p-CP10k expression -> top-2000 HVG -> the SAME prep() (20-D PCA + whiten) -> the SAME everything.

If expression-PCA steers as well as the foundation models, then "you can steer a cell to a chosen fate" is a
statement about single-cell data lying on a manifold, not about foundation models, and the headline must change.

This is not a formality. The project's own results already point BOTH ways:
  * theta_far reproduces 16/16 across four architecturally unrelated models (scGPT masked-expression, Geneformer
    MLM, STATE-SE, MaxToki Llama-CLM) with different objectives. Four models that share nothing but their
    TRAINING DATA agreeing that precisely is what you'd expect if you were measuring the DATA.
  * route_benchmark (2026-07-13) finds geometry BUILDS WITH DEPTH, peaking at scGPT L11 / MaxToki L8. Pure data
    geometry should not build with depth.
Both cannot be fully right. This adjudicates, and it wins either way: if expression matches, the program gets a
sharper negative that fits its own throughline ("the geometry was in the data all along"); if the models win,
the positive becomes attributable to PRETRAINING and gets much stronger.

PRE-STATED criterion (written before any number below was computed):
  The foundation-model embedding EARNS its place iff, pooled over the 4 tissues, it beats expression-PCA on
  BOTH (i) rotation excess (theta_far vs NULL C) and (ii) constrained advance at tau=1.3 for linear_proj,
  in >= 3/4 tissues, for a majority of models. Otherwise the honest reading is that the manifold geometry is a
  property of the data that the models merely preserve.

Every cell, every space, every null, every steering rule is shared with route_steering: this file imports its
functions rather than reimplementing them, and restricts all models AND expression to the SAME cells.

Run:  ../../.venv/bin/python baseline.py [tissue ...]        (default: all four)
Out:  results/baseline_<tissue>.json
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from common import RESULTS, TISSUES, FATE_CLUSTERS, ROOT_CLUSTERS, load_expression, N_HVG  # noqa: E402

# route_steering's own code — imported, not reimplemented, so the comparison is exact
from tangent_diagnostic import (path as npz_path, prep, zscore, unit, steer, cv_linear_r2,   # noqa: E402
                                tangent_rotation, null_resid_perm_stats, sig, oracle_field,
                                local_tangent_alignment, CAP, SEED, BUDGETS, T_STAR)
from local_steering import local_field, project_constant, constrained_advance                # noqa: E402

MODELS = ["scgptbin", "geneformer", "state", "maxtoki"]
BASE = "expr"          # the baseline "model": 20-D whitened PCA of log1p-CP10k expression


def load_space(name, tissue, cells):
    """Return the raw feature matrix + pseudotime for `name` on exactly the h5ad cells `cells` (cell_idx values,
    sorted). Spaces have DIFFERENT row counts (Geneformer keeps 2519 of Setty's 5780), so everything is selected
    by cell_idx, never by row position."""
    if name == BASE:
        E, _, _ = load_expression(tissue, cells)                          # h5ad rows, in `cells` order
        hvg = np.argsort(-E.var(0))[:N_HVG]                               # unsupervised; no label leakage
        z = np.load(npz_path("scgptbin", tissue), allow_pickle=True)      # borrow pseudotime only
        pos = {c: i for i, c in enumerate(z["cell_idx"].astype(int))}
        y = z["pseudotime"].astype(np.float64)[[pos[c] for c in cells]]
        return E[:, hvg].astype(np.float64), y
    p = npz_path(name, tissue)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    pos = {c: i for i, c in enumerate(z["cell_idx"].astype(int))}
    rows = [pos[c] for c in cells]
    return z["emb"].astype(np.float64)[rows], z["pseudotime"].astype(np.float64)[rows]


def shared_cells(tissue):
    """The h5ad cells present in EVERY space with a finite pseudotime, subsampled to CAP with the project seed —
    so expression-PCA and the models are judged on literally the same cells. Returns (cells, clusters)."""
    keep, clu = None, {}
    for m in ["scgptbin"] + MODELS:
        p = npz_path(m, tissue)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        ci = z["cell_idx"].astype(int)
        good = set(ci[np.isfinite(z["pseudotime"])].tolist())
        keep = good if keep is None else (keep & good)
        if m == "scgptbin":
            clu = dict(zip(ci.tolist(), z["clusters"].tolist()))
    cells = np.array(sorted(keep))
    if len(cells) > CAP:
        cells = np.sort(np.random.default_rng(SEED).choice(cells, CAP, replace=False))
    return cells, np.array([clu[c] for c in cells])


def geometry(Xz, y):
    """theta_far + NULL C (heteroscedastic residual permutation) — the headroom-free rotation statistic."""
    rot, _, _ = tangent_rotation(Xz, y)                     # (stats, bin dirs, global dir)
    thn, _ = null_resid_perm_stats(Xz, y)                   # NULL C draws: theta_far, monotonicity
    thn = np.asarray(thn, dtype=float)
    s = sig(rot["theta_far"], thn)
    return dict(theta_far=rot["theta_far"], monotonicity=rot["monotonicity"],
                null_mean=s["null_mean"], null_sd=s["null_sd"],
                excess=float(rot["theta_far"] - s["null_mean"]), z=s["z"], p=s["p_value"],
                lin_r2_20pc=cv_linear_r2(Xz, y))


def steering(Xz, y, clusters, tissue):
    """The route_steering ladder, restricted to the contrasts that matter here, + held-out endpoint purity."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    Xtr, ytr, Xte, yte = Xz[tr], y[tr], Xz[te], y[te]
    clu_tr = clusters[tr]

    w_lin = unit(Ridge(alpha=10.0).fit(Xtr, ytr).coef_)
    judge = KNeighborsRegressor(15).fit(Xtr, ytr)
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]))
    h = 0.25 * d0

    src = Xte[yte <= np.quantile(yte, 1.0 / 3.0)]
    if len(src) > 300:
        src = src[rng.choice(len(src), 300, replace=False)]

    fields = {"linear": lambda x: np.tile(w_lin, (len(x), 1)),
              "linear_proj": project_constant(Xtr, w_lin),
              "local_proj": local_field(Xtr, ytr, project_m=10),
              "oracle_knn": oracle_field(Xtr, ytr)}

    out = {}
    for name, fn in fields.items():
        pts = steer(src, fn, h, BUDGETS)
        base = judge.predict(pts[0])
        rows = {T: dict(off_manifold_ratio=float(nn1.kneighbors(pts[T])[0][:, 0].mean() / d0),
                        advance=float((judge.predict(pts[T]) - base).mean())) for T in BUDGETS}
        out[name] = dict(ca13=constrained_advance(rows, 1.3),
                         off_ratio_T8=rows[T_STAR]["off_manifold_ratio"],
                         advance_T8=rows[T_STAR]["advance"],
                         align_T8=local_tangent_alignment(Xtr, pts[T_STAR], fn(pts[T_STAR])))

    # ---- held-out endpoint purity: aim the projected steer at each terminal fate; at T=8, which REAL cluster is
    # the nearest train cell in? Purity = fraction landing in the intended fate. Judged in the same space, so it
    # is comparable across spaces. This is the "does the space steer you to the RIGHT PLACE" test.
    roots = np.isin(clu_tr, ROOT_CLUSTERS[tissue])
    src_c = Xtr[roots].mean(0) if roots.sum() >= 20 else src.mean(0)
    pur = {}
    for fate, cls in FATE_CLUSTERS[tissue].items():
        tgt = np.isin(clu_tr, cls)
        if tgt.sum() < 20:
            continue
        w_f = unit(Xtr[tgt].mean(0) - src_c)
        pts = steer(src, project_constant(Xtr, w_f), h, BUDGETS)
        nb = nn1.kneighbors(pts[T_STAR])[1][:, 0]
        pur[fate] = float(np.isin(clu_tr[nb], cls).mean())
    out["endpoint_purity"] = pur
    out["endpoint_purity_mean"] = float(np.mean(list(pur.values()))) if pur else float("nan")
    return out


def main():
    tissues = sys.argv[1:] or TISSUES
    allres = {}
    for tissue in tissues:
        cells, clusters = shared_cells(tissue)
        print(f"\n{'='*104}\n=== {tissue}  (n={len(cells)} shared cells, identical across every space) ===")
        res = {}
        for name in [BASE] + MODELS:
            got = load_space(name, tissue, cells)
            if got is None:
                print(f"  [skip] {name}"); continue
            X, y = got
            Xz, yz = prep(X), zscore(y)
            g = geometry(Xz, yz)
            s = steering(Xz, yz, clusters, tissue)
            res[name] = dict(geometry=g, steering=s, d_raw=int(X.shape[1]))

        print(f"\n  {'space':<12} {'d_raw':>6} | {'theta':>6} {'nullC':>6} {'excess':>7} {'z':>6} {'p':>6}"
              f" {'linR2':>6} | {'CA linear':>9} {'CA lin_proj':>11} {'gain':>6} {'off@T8':>7} {'align':>6}"
              f" | {'endpt purity':>12}")
        for name, r in res.items():
            g, s = r["geometry"], r["steering"]
            tag = "  <<BASELINE" if name == BASE else ""
            print(f"  {name:<12} {r['d_raw']:>6} | {g['theta_far']:>6.1f} {g['null_mean']:>6.1f}"
                  f" {g['excess']:>+7.1f} {g['z']:>+6.1f} {g['p']:>6.3f} {g['lin_r2_20pc']:>6.3f}"
                  f" | {s['linear']['ca13']:>9.2f} {s['linear_proj']['ca13']:>11.2f}"
                  f" {s['linear_proj']['ca13'] - s['linear']['ca13']:>+6.2f}"
                  f" {s['linear_proj']['off_ratio_T8']:>7.2f} {s['linear_proj']['align_T8']:>6.2f}"
                  f" | {s['endpoint_purity_mean']:>12.2f}{tag}")

        # the verdict, per tissue: does ANY model beat expression-PCA on both axes?
        b = res[BASE]
        wins = {m: dict(rotation=res[m]["geometry"]["excess"] > b["geometry"]["excess"],
                        steering=res[m]["steering"]["linear_proj"]["ca13"] > b["steering"]["linear_proj"]["ca13"],
                        purity=res[m]["steering"]["endpoint_purity_mean"] > b["steering"]["endpoint_purity_mean"])
                for m in res if m != BASE}
        nb = sum(w["rotation"] and w["steering"] for w in wins.values())
        print(f"\n  models beating expression-PCA on BOTH rotation-excess and CA@1.3: {nb}/{len(wins)}"
              f"   | on endpoint purity: {sum(w['purity'] for w in wins.values())}/{len(wins)}")
        allres[tissue] = dict(spaces=res, wins=wins, n=int(len(cells)))
        json.dump(allres[tissue], open(os.path.join(RESULTS, f"baseline_{tissue}.json"), "w"), indent=1)

    # ------------------------------------------------------------------ pooled verdict against the pre-stated bar
    print(f"\n\n{'='*104}\n================ POOLED VERDICT — does the foundation model earn its place? ================")
    print(f"  {'model':<12} " + " ".join(f"{t:>22}" for t in tissues))
    print(f"  {'':<12} " + " ".join(f"{'rot / CA@1.3 / purity':>22}" for _ in tissues))
    for m in MODELS:
        cells = []
        for t in tissues:
            if m not in allres[t]["spaces"]:
                cells.append(f"{'--':>22}"); continue
            w = allres[t]["wins"][m]
            cells.append(f"{('W' if w['rotation'] else '.') + ' / ' + ('W' if w['steering'] else '.') + ' / ' + ('W' if w['purity'] else '.'):>22}")
        print(f"  {m:<12} " + " ".join(cells))
    tot = {ax: sum(allres[t]["wins"][m][ax] for t in tissues for m in MODELS if m in allres[t]["spaces"])
           for ax in ("rotation", "steering", "purity")}
    n_cells = sum(1 for t in tissues for m in MODELS if m in allres[t]["spaces"])
    print(f"\n  model beats expression-PCA in:  rotation {tot['rotation']}/{n_cells}"
          f"   CA@1.3 {tot['steering']}/{n_cells}   endpoint purity {tot['purity']}/{n_cells}")
    print("  (W = the foundation model beats the no-model expression-PCA baseline on that axis)")
    json.dump(dict(per_tissue=allres, totals=tot, n_cells=n_cells),
              open(os.path.join(RESULTS, "baseline_ALL.json"), "w"), indent=1)
    print(f"\n[done] -> results/baseline_*.json")


if __name__ == "__main__":
    main()
