"""manifold_fit — discover and characterise a cell-state MANIFOLD in C2S activation space.

Generic over the ordering variable: `--kind cyclic` (cell-cycle phase, circadian) or `--kind ordered`
(pseudotime, developmental stage). Given per-cell state vectors H (n_cells, d) and the variable, it answers the
three questions that decide whether Goodfire-style manifold steering is even applicable:

  1. IS THERE A MANIFOLD?  cross-validated decodability of the variable from H.
       cyclic  -> ridge-predict (cos t, sin t) on held-out cells, report circular corr + median |angle error|
       ordered -> ridge-predict t, report Spearman rho
     Null: the same pipeline on SHUFFLED labels (holds H's covariance fixed; never feature-shuffle).

  2. IS IT CURVED?  the crux — manifold steering can only beat linear steering if the structure is nonlinear.
       * ring_score      : circular corr between the angle in the top-2 PC plane and the true phase (cyclic).
       * radius_cv       : std/mean of the radius of the binned knots about their centroid. Low = clean RING
                           (a filled blob or a straight line does not give a low-CV constant radius).
       * chord_deviation : for knots separated by k bins, how far the straight CHORD midpoint sits from the
                           manifold, in units of the local knot spacing. 0 = the path is straight (linear
                           steering == manifold steering); large = genuinely curved, so the chord leaves the
                           data manifold. THIS is the number that licenses manifold steering.
       * intrinsic dim   : participation ratio of the knot cloud.

  3. WHERE IS IT STRONGEST?  all of the above per layer -> pick the layer to steer at.

Outputs the fitted knots (mean activation per bin) to an npz so manifold_steer.py can steer along them.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np


# ---------------------------------------------------------------- helpers
def circ_corr(a, b):
    """Jammalamadaka-SenGupta circular correlation."""
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def ridge_cv(X, Y, folds=5, alpha=1e3, seed=0):
    """Out-of-fold ridge predictions (Y may be multi-column)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    Y = np.asarray(Y); Y2 = Y.reshape(len(Y), -1)
    pred = np.zeros_like(Y2, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y2[tr])
        pred[te] = np.asarray(m.predict(sc.transform(X[te]))).reshape(len(te), -1)
    return pred


def decodability(H, t, kind, n_null=5, seed=0):
    """Q1: is the variable decodable from the activations, vs a label-shuffle null?"""
    rng = np.random.default_rng(seed)
    if kind == "cyclic":
        Y = np.column_stack([np.cos(t), np.sin(t)])
        P = ridge_cv(H, Y, seed=seed)
        that = np.arctan2(P[:, 1], P[:, 0])
        obs = circ_corr(that, t)
        med_err = float(np.rad2deg(np.median(circ_dist(that, t))))
        null = []
        for i in range(n_null):
            tp = rng.permutation(t)
            Pn = ridge_cv(H, np.column_stack([np.cos(tp), np.sin(tp)]), seed=seed)
            null.append(circ_corr(np.arctan2(Pn[:, 1], Pn[:, 0]), tp))
        null = np.array(null)
        return dict(metric="circular_corr", value=obs, median_abs_err_deg=med_err,
                    null_mean=float(null.mean()), null_sd=float(null.std()),
                    z=float((obs - null.mean()) / (null.std() + 1e-9)))
    from scipy.stats import spearmanr
    P = ridge_cv(H, t, seed=seed).ravel()
    obs = float(spearmanr(P, t).statistic)
    null = []
    for i in range(n_null):
        tp = rng.permutation(t)
        null.append(float(spearmanr(ridge_cv(H, tp, seed=seed).ravel(), tp).statistic))
    null = np.array(null)
    return dict(metric="spearman", value=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float((obs - null.mean()) / (null.std() + 1e-9)))


def knots(H, t, kind, n_bins=12):
    """Mean activation per bin of the ordering variable = the discrete manifold M_h."""
    if kind == "cyclic":
        edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        b = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
        centers = (edges[:-1] + edges[1:]) / 2
    else:
        q = np.quantile(t, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
        b = np.clip(np.digitize(t, q) - 1, 0, n_bins - 1)
        centers = np.array([t[b == k].mean() if (b == k).any() else np.nan for k in range(n_bins)])
    K = np.stack([H[b == k].mean(0) if (b == k).sum() >= 3 else np.full(H.shape[1], np.nan)
                  for k in range(n_bins)])
    n = np.array([(b == k).sum() for k in range(n_bins)])
    return K, centers, b, n


def curvature(K, kind):
    """Q2: is the manifold CURVED? chord_deviation is the licence for manifold steering."""
    ok = np.isfinite(K).all(1)
    Kf = K[ok]
    nb = len(Kf)
    C = Kf.mean(0)
    R = np.linalg.norm(Kf - C, axis=1)
    # local spacing between consecutive knots (wrap for cyclic)
    step = np.linalg.norm(np.diff(Kf, axis=0), axis=1)
    if kind == "cyclic":
        step = np.r_[step, np.linalg.norm(Kf[0] - Kf[-1])]
    med_step = float(np.median(step))
    # chord deviation: midpoint of the straight line between knots i,j vs the true mid-arc knot
    devs = {}
    for k in range(2, nb // 2 + 1):
        d = []
        for i in range(nb if kind == "cyclic" else nb - k):
            j = (i + k) % nb
            mid_idx = (i + k // 2) % nb
            if k % 2 != 0:
                continue                                   # need an exact mid knot
            chord_mid = 0.5 * (Kf[i] + Kf[j])
            d.append(np.linalg.norm(chord_mid - Kf[mid_idx]))
        if d:
            devs[k] = float(np.mean(d) / (med_step + 1e-12))
    # intrinsic dimensionality of the knot cloud (participation ratio of eigenvalues)
    Xc = Kf - C
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))
    out = dict(radius_mean=float(R.mean()), radius_cv=float(R.std() / (R.mean() + 1e-12)),
               median_knot_step=med_step, chord_deviation=devs,
               chord_deviation_halfway=devs.get(max(devs) if devs else 0, None),
               participation_ratio=pr, var_pc1=float(ev[0] / ev.sum()), var_pc12=float(ev[:2].sum() / ev.sum()))
    return out


def ring_score(H, t, n_perm=500, seed=0):
    """Cyclic only: does the top-2 PC plane of the activations arrange cells in a RING ordered by phase?"""
    X = H - H.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    P = (U * S)[:, :2]
    ang = np.arctan2(P[:, 1], P[:, 0])
    obs = abs(circ_corr(ang, t))
    rng = np.random.default_rng(seed)
    null = np.array([abs(circ_corr(ang, rng.permutation(t))) for _ in range(n_perm)])
    return dict(ring_circ_corr=obs, null_mean=float(null.mean()),
                z=float((obs - null.mean()) / (null.std() + 1e-12)), p=float((null >= obs).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, help="dir with layer_LL_activations.npy + row_cell_ids.npy")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--kind", choices=["cyclic", "ordered"], default="cyclic")
    ap.add_argument("--var-key", default=None, help="obs key for the ordering variable; omit for cell-cycle phase")
    ap.add_argument("--label-key", default="clusters")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results/manifold_fit.json")
    ap.add_argument("--knots-out", default="results/manifold_knots.npz")
    a = ap.parse_args()

    import anndata, glob
    ad = anndata.read_h5ad(a.h5ad)
    cell_ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    if a.var_key:
        t_all = np.asarray(ad.obs[a.var_key]).astype(float)
        if a.kind == "cyclic":
            t_all = np.mod(t_all, 2 * np.pi)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cc_phase import phase_angle
        t_all, s_sc, g_sc, info = phase_angle(ad)
        print(f"cell-cycle phase from {info['n_cc_genes']} genes; PC12 var {info['pc12_var_ratio']:.3f}", flush=True)
    t = t_all[cell_ids]
    lab = np.asarray(ad.obs[a.label_key]).astype(str)[cell_ids] if a.label_key in ad.obs else None

    layers = a.layers or sorted(int(os.path.basename(p).split("_")[1]) for p in
                                glob.glob(os.path.join(a.states, "layer_*_activations.npy")))
    res, knot_store = {}, {}
    for L in layers:
        H = np.load(os.path.join(a.states, f"layer_{L:02d}_activations.npy")).astype(np.float64)
        H = H[:len(t)] if len(H) > len(t) else H
        rec = {"n_cells": int(len(H)), "d": int(H.shape[1])}
        rec["decodability"] = decodability(H, t, a.kind)
        K, centers, b, nper = knots(H, t, a.kind, a.n_bins)
        rec["curvature"] = curvature(K, a.kind)
        rec["bin_counts"] = nper.tolist()
        if a.kind == "cyclic":
            rec["ring"] = ring_score(H, t)
        knot_store[f"L{L:02d}"] = K
        res[f"L{L:02d}"] = rec
        dec = rec["decodability"]
        extra = f" ring={rec['ring']['ring_circ_corr']:.3f}(z{rec['ring']['z']:+.0f})" if a.kind == "cyclic" else ""
        cd = rec["curvature"]["chord_deviation"]
        cdmax = cd[max(cd)] if cd else float("nan")
        print(f"  L{L:02d} decode {dec['metric']}={dec['value']:+.3f} (z{dec['z']:+.0f})"
              + (f" err={dec.get('median_abs_err_deg'):.0f}deg" if "median_abs_err_deg" in dec else "")
              + extra + f" | chord_dev={cdmax:.2f} radius_cv={rec['curvature']['radius_cv']:.2f} "
              f"PR={rec['curvature']['participation_ratio']:.1f}", flush=True)

    # best layer = strongest decodability among layers whose manifold is curved
    best = max(res.items(), key=lambda kv: kv[1]["decodability"]["value"])
    res["_best_layer"] = best[0]
    print(f"\nBEST decodable layer: {best[0]}  ({best[1]['decodability']['metric']}="
          f"{best[1]['decodability']['value']:+.3f})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez_compressed(a.knots_out, centers=centers, kind=a.kind, n_bins=a.n_bins,
                        **{k: v for k, v in knot_store.items()})
    print(f"[done] -> {a.out} ; knots -> {a.knots_out}", flush=True)


if __name__ == "__main__":
    main()
