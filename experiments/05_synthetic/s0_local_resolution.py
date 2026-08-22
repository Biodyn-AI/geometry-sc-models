"""S0 - does the C2S metric stretch at G1->S buy local phase resolution?

Pre-registered in S0_PREREGISTRATION.md. Read that first; the primary contrast is fixed there.

Two resolution metrics per 30-degree window of the cell-cycle loop, per representation:
  slope_to_noise : OLS slope of ||dx|| on |dphi| within the window, divided by residual sd
  dprime         : (mean ||dx|| at |dphi| in [10,20] - mean at |dphi| < 3) / pooled sd

Every window is subsampled to the same n so that uneven cell density around the loop cannot
drive the result, and every arm is reduced to the same number of whitened PCs so distances are
in units of each representation's own spread.
"""
import os
import json, sys
import numpy as np
from sklearn.decomposition import PCA

BT = os.environ.get("GEOMSC_BIOTENSOR", "")   # see docs/DATA.md
C2S = os.environ.get("GEOMSC_C2S", "")        # see docs/DATA.md
SEED, N_BINS, N_REPEATS = 0, 12, 50
NEAR_DEG, FAR_LO, FAR_HI = 3.0, 10.0, 20.0


def circ_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def load_arms():
    """Return {name: (emb, phi_deg, cell_idx)}. phi in degrees, 0-360."""
    arms = {}
    sub = np.load(f"{BT}/data/cellcycle/k562_cc_substrate.npz", allow_pickle=True)
    phi_all = np.rad2deg(sub["phi"]) % 360.0
    idx_all = sub["cell_idx"]

    # C2S-2B L21. row_cell_ids is arange(3000): POSITIONAL rows into the substrate, not global
    # cell ids (verified against the manifest and cc_benchmark_c2s.py, which asserts cell order).
    # Every other arm stores global ids, so these must be mapped through the substrate first.
    act = np.load(f"{C2S}/data/act_k562/layer_21_activations.npy")
    rid = np.load(f"{C2S}/data/act_k562/row_cell_ids.npy")
    assert (rid == np.arange(len(idx_all))).all(), "C2S row convention changed - re-check"
    arms["c2s_2b_L21"] = (act, phi_all, idx_all)

    for name, f in [("expression", "expr_k562"), ("scgpt_L11", "scgptcc_k562"),
                    ("maxtoki_L8", "maxtoki_k562"), ("geneformer_L11", "geneformer_k562"),
                    ("state_L11", "statecc_k562")]:
        d = np.load(f"{BT}/data/cellcycle/{f}.npz", allow_pickle=True)
        arms[name] = (d["emb"], np.rad2deg(d["phi"]) % 360.0, d["cell_idx"])
    return arms


def prep(emb, k, seed=SEED):
    """Whitened PCA to k dims. Whitening puts every arm on its own spread scale."""
    k = min(k, emb.shape[1], emb.shape[0] - 1)
    return PCA(n_components=k, whiten=True, random_state=seed).fit_transform(
        emb.astype(np.float64))


def window_resolution(X, phi, lo, hi, n_sub, rng, n_repeats=N_REPEATS, shuffle=False):
    """Both resolution metrics inside the circular window [lo, hi). None if too few cells."""
    span = (phi - lo) % 360.0
    sel = np.where(span < ((hi - lo) % 360.0 or 360.0))[0]
    if len(sel) < n_sub:
        return None
    slopes, dprimes = [], []
    for _ in range(n_repeats):
        take = rng.choice(sel, size=n_sub, replace=False)
        Xw, pw = X[take], phi[take]
        if shuffle:
            pw = rng.permutation(pw)
        i, j = np.triu_indices(n_sub, k=1)
        dphi = circ_diff_deg(pw[i], pw[j])
        dx = np.linalg.norm(Xw[i] - Xw[j], axis=1)
        if dphi.std() < 1e-9:
            continue
        b = np.polyfit(dphi, dx, 1)
        resid = dx - np.polyval(b, dphi)
        sd = resid.std(ddof=2)
        if sd > 1e-12:
            slopes.append(b[0] / sd)
        near, far = dx[dphi < NEAR_DEG], dx[(dphi >= FAR_LO) & (dphi <= FAR_HI)]
        if len(near) >= 5 and len(far) >= 5:
            pooled = np.sqrt((near.var(ddof=1) + far.var(ddof=1)) / 2.0)
            if pooled > 1e-12:
                dprimes.append((far.mean() - near.mean()) / pooled)
    if not slopes or not dprimes:
        return None
    return {"slope_to_noise": float(np.mean(slopes)),
            "slope_to_noise_sd": float(np.std(slopes)),
            "dprime": float(np.mean(dprimes)),
            "dprime_sd": float(np.std(dprimes)),
            "n_window": int(len(sel))}


def run_group(arms, names, k, rng_master, out, label):
    """Score one set of arms on their own shared cells, with their own matched n."""
    common = set(arms[names[0]][2].tolist())
    for n in names[1:]:
        common &= set(arms[n][2].tolist())
    common = np.array(sorted(common))
    edges = np.arange(N_BINS) * (360.0 / N_BINS)
    width = 360.0 / N_BINS

    counts = []
    for n in names:
        _, phi, ci = arms[n]
        p = phi[np.isin(ci, common)]
        counts += [int(((p - lo) % 360.0 < width).sum()) for lo in edges]
    n_sub = int(np.min(counts))
    print(f"\n[{label}] {len(names)} arms, {len(common)} shared cells; "
          f"window counts {min(counts)}-{max(counts)} -> matched n = {n_sub}")
    if n_sub < 20:
        print(f"[{label}] SKIPPED - sparsest window has only {n_sub} cells")
        return
    out.setdefault("groups", {})[label] = {"n_common_cells": int(len(common)),
                                           "n_sub": n_sub, "arms": names}

    for name in names:
        emb, phi, ci = arms[name]
        m = np.isin(ci, common)
        order = np.argsort(ci[m])
        X = prep(emb[m][order], k)
        p = phi[m][order]
        rng = np.random.default_rng(rng_master.integers(1 << 30))
        real = [window_resolution(X, p, lo, lo + width, n_sub, rng) for lo in edges]
        null = [window_resolution(X, p, lo, lo + width, n_sub, rng, n_repeats=10,
                                  shuffle=True) for lo in edges]
        out["arms"][name] = {
            "group": label, "dim_used": int(X.shape[1]),
            "windows": [{"lo": float(lo), "hi": float(lo + width), "real": r, "null": nl}
                        for lo, r, nl in zip(edges, real, null)]}
        prof = [r["dprime"] if r else float("nan") for r in real]
        nprof = [nl["dprime"] if nl else float("nan") for nl in null]
        print(f"  {name:16s} d' per window: " +
              " ".join(f"{v:5.2f}" for v in prof) +
              f"   | null {np.nanmean(nprof):+.3f}")


def main(k=20):
    rng_master = np.random.default_rng(SEED)
    arms = load_arms()
    out = {"k": k, "n_bins": N_BINS, "n_repeats": N_REPEATS, "arms": {}}
    edges = np.arange(N_BINS) * (360.0 / N_BINS)

    # maxtoki and geneformer were extracted on a 2000-cell subset. Scoring everything on the
    # 2000 would cost the primary contrast a third of its cells for no reason, so the two
    # populations are run as separate groups, each internally matched.
    run_group(arms, ["c2s_2b_L21", "expression", "scgpt_L11", "state_L11"],
              k, rng_master, out, "primary_3000")
    run_group(arms, ["maxtoki_L8", "geneformer_L11", "expression"],
              k, rng_master, out, "secondary_2000")

    # the pre-registered contrast: G1->S window = knot bins 5-8 = 150-240 deg
    g1s = [i for i, lo in enumerate(edges) if 150.0 <= lo < 240.0]
    out["g1s_window_indices"] = g1s
    for metric in ("dprime", "slope_to_noise"):
        e = np.array([w["real"][metric] if w["real"] else np.nan
                      for w in out["arms"]["expression"]["windows"]])
        c = np.array([w["real"][metric] if w["real"] else np.nan
                      for w in out["arms"]["c2s_2b_L21"]["windows"]])
        delta = c - e
        res = {"delta_all_windows_mean": float(np.nanmean(delta)),
               "delta_g1s_mean": float(np.nanmean(delta[g1s])),
               "contrast_g1s_minus_all": float(np.nanmean(delta[g1s]) - np.nanmean(delta)),
               "delta_per_window": [None if np.isnan(v) else float(v) for v in delta]}
        out.setdefault("primary_contrast", {})[metric] = res
        print(f"\nPRIMARY [{metric}] delta(C2S - expression):"
              f"  all windows {res['delta_all_windows_mean']:+.4f}"
              f" | G1->S {res['delta_g1s_mean']:+.4f}"
              f" | contrast {res['contrast_g1s_minus_all']:+.4f}")

    path = f"{BT}/manifolds/synthetic/results/s0_local_resolution_k{k}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
