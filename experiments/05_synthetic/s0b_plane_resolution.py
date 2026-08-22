"""S0b - local phase resolution measured IN THE FITTED PHASE PLANE.

Why this exists. The pre-registered S0 measured pairwise distance in the ambient 20-PC whitened
space and returned ~0 for every arm, including raw expression, which provably carries phase
(circ-R2 0.929). Diagnosis: phase explains R2 = 0.020-0.026 of ambient pairwise distance globally
and 0.0006-0.0034 inside a 30-degree window. The instrument had no power. That is an instrument
failure, not a null, and S0's numbers must not be read as evidence either way.

This version measures distance in the 2-D plane where the phase coordinate lives - the same space
the knot-gap / metric-stretch claim was made in.

Circularity control: the plane is fitted by ridge on cos(phi), sin(phi) using a TRAIN half of the
cells and every reported number is computed on the held-out half, so the scored cells never
contributed to the plane. The fit is identical in form for every arm.

Positive controls are run first. If the instrument cannot recover a planted stretch, nothing else
is reported.
"""
import os
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s0_local_resolution import load_arms, circ_diff_deg, BT

SEED, N_BINS, N_REPEATS = 0, 12, 200


def fit_plane(X_tr, phi_tr, X_te, alpha=1.0):
    """Ridge cos/sin readout fitted on train, applied to test. Returns test coords in the plane."""
    Y = np.c_[np.cos(np.deg2rad(phi_tr)), np.sin(np.deg2rad(phi_tr))]
    return Ridge(alpha=alpha).fit(X_tr, Y).predict(X_te)


def window_stats(P, phi, lo, width, n_sub, rng, n_repeats=N_REPEATS, shuffle=False):
    """In-plane metric and noise inside one window.

    metric_per_deg : OLS slope of in-plane distance on |dphi| -- how much representational
                     distance the arm spends per degree of phase here (the single-cell analogue
                     of the knot-gap measurement)
    noise          : residual sd about that fit
    dprime_per_deg : metric_per_deg / noise -- resolution
    """
    span = (phi - lo) % 360.0
    sel = np.where(span < width)[0]
    if len(sel) < n_sub:
        return None
    met, noi, dpr = [], [], []
    for _ in range(n_repeats):
        take = rng.choice(sel, size=n_sub, replace=False)
        Pw, pw = P[take], phi[take]
        if shuffle:
            pw = rng.permutation(pw)
        i, j = np.triu_indices(n_sub, k=1)
        dphi = circ_diff_deg(pw[i], pw[j])
        dP = np.linalg.norm(Pw[i] - Pw[j], axis=1)
        if dphi.std() < 1e-9:
            continue
        b = np.polyfit(dphi, dP, 1)
        sd = (dP - np.polyval(b, dphi)).std(ddof=2)
        if sd > 1e-12:
            met.append(b[0]); noi.append(sd); dpr.append(b[0] / sd)
    if not dpr:
        return None
    return {"metric_per_deg": float(np.mean(met)), "noise": float(np.mean(noi)),
            "dprime_per_deg": float(np.mean(dpr)),
            "dprime_sd": float(np.std(dpr)), "n_window": int(len(sel))}


def profile(X, phi, k, rng, n_sub):
    """Split-half plane fit, then the 12-window profile on the held-out half."""
    n = len(X)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    Z = PCA(n_components=min(k, X.shape[1], n - 1), whiten=True,
            random_state=SEED).fit_transform(X.astype(np.float64))
    P = fit_plane(Z[tr], phi[tr], Z[te])
    pt = phi[te]
    width = 360.0 / N_BINS
    edges = np.arange(N_BINS) * width
    real = [window_stats(P, pt, lo, width, n_sub, rng) for lo in edges]
    null = [window_stats(P, pt, lo, width, n_sub, rng, n_repeats=30, shuffle=True)
            for lo in edges]
    return real, null, edges


def synthetic_ring(n, stretch_lo, stretch_hi, stretch_factor, noise, rng, d=50):
    """A ring in d dims. Arc [stretch_lo, stretch_hi) gets stretch_factor more distance per degree.

    Ground truth: dprime_per_deg should be stretch_factor times higher inside that arc.
    """
    phi = rng.uniform(0, 360, n)
    # arc-length coordinate: uniform elsewhere, stretched inside the target arc
    def arc(p):
        base = p.copy()
        inside = (p >= stretch_lo) & (p < stretch_hi)
        extra = (stretch_factor - 1.0) * np.clip(p - stretch_lo, 0, stretch_hi - stretch_lo)
        return base + extra
    s = np.deg2rad(arc(phi))
    X = np.zeros((n, d))
    X[:, 0], X[:, 1] = np.cos(s), np.sin(s)
    X += rng.normal(0, noise, (n, d))          # isotropic noise, incl. the 48 null dims
    return X, phi


def main(k=20):
    rng = np.random.default_rng(SEED)
    out = {"k": k, "n_bins": N_BINS, "n_repeats": N_REPEATS, "controls": {}, "arms": {}}

    # ---- positive / negative controls -------------------------------------------------
    print("CONTROLS (instrument must be flat on uniform, peaked on planted stretch)")
    ok = True
    for label, factor in [("uniform_ring", 1.0), ("planted_stretch_2x", 2.0)]:
        Xs, ps = synthetic_ring(3000, 150.0, 240.0, factor, 0.15, rng)
        real, null, edges = profile(Xs, ps, k, rng, n_sub=70)
        prof = np.array([r["dprime_per_deg"] if r else np.nan for r in real])
        g1s = [i for i, lo in enumerate(edges) if 150.0 <= lo < 240.0]
        contrast = float(np.nanmean(prof[g1s]) - np.nanmean(prof))
        out["controls"][label] = {"profile": [None if np.isnan(v) else float(v) for v in prof],
                                  "contrast_arc_minus_all": contrast}
        print(f"  {label:20s} " + " ".join(f"{v:5.3f}" for v in prof) +
              f"  | arc-vs-all {contrast:+.4f}")
        if label == "uniform_ring" and abs(contrast) > 0.02:
            ok = False
        if label == "planted_stretch_2x" and contrast <= 0.02:
            ok = False
    if not ok:
        print("\nINSTRUMENT FAILED ITS CONTROLS - not reporting real data.")
        json.dump(out, open(f"{BT}/manifolds/synthetic/results/s0b_plane_k{k}.json", "w"), indent=1)
        return
    print("  -> controls passed\n")

    # ---- real arms ---------------------------------------------------------------------
    arms = load_arms()
    groups = [("primary_3000", ["c2s_2b_L21", "expression", "scgpt_L11", "state_L11"]),
              ("secondary_2000", ["maxtoki_L8", "geneformer_L11", "expression"])]
    for label, names in groups:
        common = set(arms[names[0]][2].tolist())
        for nm in names[1:]:
            common &= set(arms[nm][2].tolist())
        common = np.array(sorted(common))
        width = 360.0 / N_BINS
        counts = []
        for nm in names:
            _, phi, ci = arms[nm]
            p = phi[np.isin(ci, common)]
            counts += [int(((p - lo) % 360.0 < width).sum()) for lo in np.arange(N_BINS) * width]
        n_sub = max(25, int(np.min(counts)) // 2)   # //2 because scoring is on the held-out half
        print(f"[{label}] {len(common)} shared cells, matched n = {n_sub} per window (held-out half)")
        for nm in names:
            emb, phi, ci = arms[nm]
            m = np.isin(ci, common)
            order = np.argsort(ci[m])
            real, null, edges = profile(emb[m][order], phi[m][order], k,
                                        np.random.default_rng(rng.integers(1 << 30)), n_sub)
            out["arms"].setdefault(nm, {})[label] = {
                "windows": [{"lo": float(lo), "real": r, "null": nl}
                            for lo, r, nl in zip(edges, real, null)]}
            prof = [r["dprime_per_deg"] if r else float("nan") for r in real]
            nprof = [nl["dprime_per_deg"] if nl else float("nan") for nl in null]
            print(f"  {nm:16s} d'/deg: " + " ".join(f"{v:5.3f}" for v in prof) +
                  f"  | null {np.nanmean(nprof):+.4f}")

    # ---- the pre-registered contrast ----------------------------------------------------
    edges = np.arange(N_BINS) * (360.0 / N_BINS)
    g1s = [i for i, lo in enumerate(edges) if 150.0 <= lo < 240.0]
    for metric in ("dprime_per_deg", "metric_per_deg"):
        e = np.array([w["real"][metric] if w["real"] else np.nan
                      for w in out["arms"]["expression"]["primary_3000"]["windows"]])
        c = np.array([w["real"][metric] if w["real"] else np.nan
                      for w in out["arms"]["c2s_2b_L21"]["primary_3000"]["windows"]])
        d = c - e
        out.setdefault("primary_contrast", {})[metric] = {
            "delta_all": float(np.nanmean(d)), "delta_g1s": float(np.nanmean(d[g1s])),
            "contrast": float(np.nanmean(d[g1s]) - np.nanmean(d)),
            "per_window": [None if np.isnan(v) else float(v) for v in d]}
        print(f"\nPRIMARY [{metric}] C2S - expression: all {np.nanmean(d):+.4f} | "
              f"G1->S {np.nanmean(d[g1s]):+.4f} | contrast {np.nanmean(d[g1s]) - np.nanmean(d):+.4f}")

    p = f"{BT}/manifolds/synthetic/results/s0b_plane_k{k}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
