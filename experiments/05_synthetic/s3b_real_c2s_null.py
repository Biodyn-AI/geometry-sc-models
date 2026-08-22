"""S3b - does the real C2S metric-stretch claim survive a noise floor?

Motivation from S3. On synthetic corpora the shuffle-null floor for knot-gap CV is ~0.14 with a
192-d model embedding at ~450 cells/bin. The C2S claim is model CV 0.318 vs data CV 0.193, and
NEITHER number has a null anywhere in the programme. If the floor is that large, the comparison of
two raw CVs across two representations with different centroid noise may not mean what it is read
to mean. It is the only certified model-specific geometry claim in the corpus, so this matters.

Procedure
  1. Rebuild the 12 knots from the cached C2S-2B L21 activations and the canonical phi, and check
     the rebuild against the stored manifold_knots_cc.npz. If it does not match, stop.
  2. Report gap CV and max/min for C2S and for raw expression on the same cells.
  3. Shuffle phi, rebuild knots, recompute - 500 draws per representation. That is the centroid
     estimation-noise floor for THAT representation.
  4. Report CV excess over the floor and a z.
  5. Do it twice: faithful (bins as they fall, uneven n) and corrected (equal cells per bin).
"""
import os
import json, sys
import numpy as np
from sklearn.decomposition import PCA

BT = os.environ.get("GEOMSC_BIOTENSOR", "")   # see docs/DATA.md
C2S = os.environ.get("GEOMSC_C2S", "")        # see docs/DATA.md
N_BINS, N_NULL = 12, 500


def knots(X, theta, n_bins=N_BINS, n_per_bin=None, rng=None):
    w = 360.0 / n_bins
    out = np.zeros((n_bins, X.shape[1]))
    for i in range(n_bins):
        m = np.where(((theta - i * w) % 360.0) < w)[0]
        if len(m) == 0:
            return None
        if n_per_bin is not None:
            if len(m) < n_per_bin:
                return None
            m = rng.choice(m, n_per_bin, replace=False)
        out[i] = X[m].mean(0)
    return out


def gap_stats(K):
    n = len(K)
    g = np.array([np.linalg.norm(K[(i + 1) % n] - K[i]) for i in range(n)])
    return g, float(g.std() / g.mean()), float(g.max() / g.min()), int(g.argmax())


def analyse(X, theta, label, n_per_bin=None, seed=0):
    rng = np.random.default_rng(seed)
    K = knots(X, theta, n_per_bin=n_per_bin, rng=rng)
    g, cv, mm, am = gap_stats(K)
    ncv, nmm = [], []
    for j in range(N_NULL):
        sh = np.random.default_rng(10_000 + j).permutation(theta)
        Kn = knots(X, sh, n_per_bin=n_per_bin, rng=np.random.default_rng(seed + j))
        if Kn is None:
            continue
        _, c, m, _ = gap_stats(Kn)
        ncv.append(c); nmm.append(m)
    ncv, nmm = np.array(ncv), np.array(nmm)
    z = (cv - ncv.mean()) / ncv.std()
    p = float((ncv >= cv).mean())
    print(f"  {label:24s} CV {cv:.3f}  null {ncv.mean():.3f} +- {ncv.std():.3f}  "
          f"excess {cv - ncv.mean():+.3f}  z {z:+.2f}  p {p:.3f} | "
          f"max/min {mm:.2f} (null {nmm.mean():.2f})  argmax bin {am}")
    return {"cv": cv, "max_over_min": mm, "argmax_bin": am, "gaps": g.tolist(),
            "null_cv_mean": float(ncv.mean()), "null_cv_sd": float(ncv.std()),
            "cv_excess": float(cv - ncv.mean()), "cv_z": float(z), "p_one_sided": p,
            "null_max_over_min_mean": float(nmm.mean())}


def main():
    sub = np.load(f"{BT}/data/cellcycle/k562_cc_substrate.npz", allow_pickle=True)
    phi = np.rad2deg(sub["phi"]) % 360.0
    act = np.load(f"{C2S}/data/act_k562/layer_21_activations.npy")
    rid = np.load(f"{C2S}/data/act_k562/row_cell_ids.npy")
    assert (rid == np.arange(len(phi))).all(), "C2S rows are positional; convention changed"

    # step 1 - does the rebuild match the stored knots?
    stored = np.load(f"{C2S}/results/manifold_knots_cc.npz", allow_pickle=True)
    Kmine = knots(act, phi)
    Kst = stored["L21"]
    cos = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                         for a, b in zip(Kmine, Kst)]))
    rel = float(np.linalg.norm(Kmine - Kst) / np.linalg.norm(Kst))
    print(f"knot rebuild check vs stored L21: mean cosine {cos:.6f}, rel L2 diff {rel:.4f}")
    if cos < 0.99:
        print("REBUILD DOES NOT MATCH THE STORED KNOTS - stopping, the null would not be faithful.")
        print("(stored knots may use a different bin convention or cell subset)")
        return

    expr = np.load(f"{BT}/data/cellcycle/expr_k562.npz", allow_pickle=True)["emb"]
    out = {"n_null": N_NULL, "knot_rebuild_cosine": cos}

    for tag, npb in (("faithful_uneven_bins", None), ("corrected_equal_n", None)):
        if tag == "corrected_equal_n":
            w = 360.0 / N_BINS
            npb = min(int((((phi - i * w) % 360.0) < w).sum()) for i in range(N_BINS))
            print(f"\n[{tag}] equal n per bin = {npb}")
        else:
            print(f"\n[{tag}] bins as they fall")
        out[tag] = {
            "n_per_bin": npb,
            "c2s_2b_L21_raw2304": analyse(act, phi, "C2S-2B L21 (raw 2304-d)", npb),
            "expression_raw": analyse(expr, phi, "raw expression", npb),
            "c2s_2b_L21_pca20": analyse(
                PCA(20, whiten=True, random_state=0).fit_transform(act.astype(np.float64)),
                phi, "C2S-2B L21 (whitened PC20)", npb),
            "expression_pca20": analyse(
                PCA(20, whiten=True, random_state=0).fit_transform(expr.astype(np.float64)),
                phi, "expression (whitened PC20)", npb)}

    p = f"{BT}/manifolds/synthetic/results/s3b_real_c2s_null.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
