"""S3 - what makes a model warp the metric of a manifold it inherits?

Pre-registered question (manifolds/FORMATION.md Part 5). C2S stretches the cell-cycle metric at the
G1->S restriction point (knot-gap CV 0.318 vs the data's 0.193, largest gaps at 150-240 deg where
the data's largest gap is elsewhere). It is the only certified model-specific geometry claim in the
programme, and its cause is unknown. The leading candidate -- that the model spends distance where
its own output changes fastest -- was checked once on real data via output sensitivity and did NOT
line up. S0 separately showed the warp buys no single-cell resolution.

Here we plant a circle with a UNIFORM metric and break exactly one thing at a time:

  uniform    control. Nothing broken. The model should not warp.
  sharp      half the phase genes peak inside the arc, so the emitted gene distribution turns over
             fastest there.                              <- the output-change hypothesis
  occupancy  3x more CELLS inside the arc, gene layout untouched.
                                                          <- the density hypothesis
  noisy      cells inside the arc get extra dispersion, so prediction is hardest there.
                                                          <- the entropy hypothesis

PRE-REGISTERED PREDICTION: `sharp` warps, `occupancy` does not. `noisy` is genuinely open.

Statistic. Bin cells into 12 phase bins, take bin centroids, and measure consecutive centroid gaps
around the loop -- exactly the measurement that produced the C2S claim. Report
  stretch = mean(gap inside arc) / mean(gap outside arc)
for the MODEL and for the raw data on the same cells, and the model-minus-data difference. Only the
difference is a model property; a stretch present in both is inherited.

Centroids are computed on an equal number of cells per bin, so the occupancy arm cannot win by
having better-estimated centroids inside the arc.
"""
import os
import json, sys, time
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import ring_corpus, tokenize, train, cell_embeddings  # noqa: E402

OUT = os.environ.get("GEOMSC_RESULTS",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
N_BINS, ARC = 12, (150.0, 240.0)
N_TRAIN, N_EVAL, STEPS = 20000, 6000, 3000


def lognorm(counts):
    x = counts.astype(np.float64)
    x = x / np.clip(x.sum(1, keepdims=True), 1, None) * 1e4
    return np.log1p(x)


def knot_gaps(X, theta, n_bins=N_BINS, k=20, n_per_bin=None, seed=0):
    """Consecutive centroid gaps around the loop, in whitened-PCA space.

    Equal cells per bin so centroid noise is identical everywhere on the loop.
    """
    rng = np.random.default_rng(seed)
    Z = PCA(n_components=min(k, X.shape[1], X.shape[0] - 1), whiten=True,
            random_state=0).fit_transform(X.astype(np.float64))
    width = 360.0 / n_bins
    members = [np.where(((theta - i * width) % 360.0) < width)[0] for i in range(n_bins)]
    if n_per_bin is None:
        n_per_bin = min(len(m) for m in members)
    if n_per_bin < 10:
        return None
    cents = np.array([Z[rng.choice(m, n_per_bin, replace=False)].mean(0) for m in members])
    gaps = np.array([np.linalg.norm(cents[(i + 1) % n_bins] - cents[i]) for i in range(n_bins)])
    return gaps, n_per_bin


def stretch_index(gaps, arc=ARC, n_bins=N_BINS):
    """mean gap starting inside the arc / mean gap starting outside it."""
    width = 360.0 / n_bins
    starts = np.arange(n_bins) * width
    inside = (starts >= arc[0]) & (starts < arc[1])
    return float(gaps[inside].mean() / gaps[~inside].mean())


def run_arm(arm, seed, quiet=True):
    t0 = time.time()
    counts, theta, meta = ring_corpus(N_TRAIN + N_EVAL, arm=arm, arc=ARC, seed=seed)
    tr_c, tr_t = counts[:N_TRAIN], theta[:N_TRAIN]
    ev_c, ev_t = counts[N_TRAIN:], theta[N_TRAIN:]

    data_tr = tokenize(tr_c, seed=seed)
    model, hist = train(data_tr, meta["n_genes"], steps=STEPS, seed=seed, quiet=quiet)
    data_ev = tokenize(ev_c, seed=seed + 1)
    E = cell_embeddings(model, data_ev)

    # equal cells per bin for both arms of the comparison
    width = 360.0 / N_BINS
    npb = min(len(np.where(((ev_t - i * width) % 360.0) < width)[0]) for i in range(N_BINS))
    Xd = lognorm(ev_c)
    gm, _ = knot_gaps(E, ev_t, n_per_bin=npb, seed=seed)
    gd, _ = knot_gaps(Xd, ev_t, n_per_bin=npb, seed=seed)

    # NULL: shuffle phase, rebuild bins, recompute gaps. This is the centroid-estimation noise
    # floor, and it differs between representations (192-d model vs 1000-d data), so gap CV is
    # NOT comparable across arms without it. 20 draws each.
    def null_cv(X, n_draws=20):
        cvs, strs = [], []
        for j in range(n_draws):
            sh = np.random.default_rng(1000 + j).permutation(ev_t)
            out = knot_gaps(X, sh, n_per_bin=npb, seed=seed + j)
            if out is None:
                continue
            g = out[0]
            cvs.append(g.std() / g.mean()); strs.append(stretch_index(g))
        return float(np.mean(cvs)), float(np.std(cvs)), float(np.mean(strs))

    nm_cv, nm_sd, nm_str = null_cv(E)
    nd_cv, nd_sd, nd_str = null_cv(Xd)

    r = {"arm": arm, "seed": seed, "meta": meta, "val_corr": hist[-1]["val_corr"],
         "n_per_bin": int(npb), "secs": round(time.time() - t0, 1),
         "model": {"gaps": gm.tolist(), "cv": float(gm.std() / gm.mean()),
                   "max_over_min": float(gm.max() / gm.min()),
                   "stretch": stretch_index(gm), "argmax_bin": int(gm.argmax()),
                   "null_cv": nm_cv, "null_cv_sd": nm_sd, "null_stretch": nm_str,
                   "cv_excess": float(gm.std() / gm.mean() - nm_cv),
                   "cv_z": float((gm.std() / gm.mean() - nm_cv) / max(nm_sd, 1e-9))},
         "data": {"gaps": gd.tolist(), "cv": float(gd.std() / gd.mean()),
                  "max_over_min": float(gd.max() / gd.min()),
                  "stretch": stretch_index(gd), "argmax_bin": int(gd.argmax()),
                  "null_cv": nd_cv, "null_cv_sd": nd_sd, "null_stretch": nd_str,
                  "cv_excess": float(gd.std() / gd.mean() - nd_cv),
                  "cv_z": float((gd.std() / gd.mean() - nd_cv) / max(nd_sd, 1e-9))}}
    r["model_minus_data"] = {
        "stretch": r["model"]["stretch"] - r["data"]["stretch"],
        "cv": r["model"]["cv"] - r["data"]["cv"],
        "cv_excess": r["model"]["cv_excess"] - r["data"]["cv_excess"]}
    print(f"  {arm:10s} seed {seed}  val {r['val_corr']:+.3f} n/bin {npb:4d} | "
          f"CV model {r['model']['cv']:.3f} (null {r['model']['null_cv']:.3f}, "
          f"excess {r['model']['cv_excess']:+.3f}, z {r['model']['cv_z']:+.1f})  "
          f"data {r['data']['cv']:.3f} (null {r['data']['null_cv']:.3f}, "
          f"excess {r['data']['cv_excess']:+.3f}) | "
          f"stretch m {r['model']['stretch']:.3f} d {r['data']['stretch']:.3f} "
          f"diff {r['model_minus_data']['stretch']:+.3f}  ({r['secs']:.0f}s)")
    return r


def main(seeds=(0, 1, 2), arms=("uniform", "sharp", "occupancy", "noisy")):
    print(f"S3: {len(arms)} arms x {len(seeds)} seeds, {N_TRAIN} train cells, {STEPS} steps\n")
    rows = []
    for arm in arms:
        for s in seeds:
            rows.append(run_arm(arm, s))
            json.dump(rows, open(f"{OUT}/s3_metric_warp.json", "w"), indent=1)

    print("\n=== SUMMARY: model-minus-data stretch (>0 means the MODEL warps the arc) ===")
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm]
        if not g: continue
        st = np.array([r["model_minus_data"]["stretch"] for r in g])
        ex = np.array([r["model"]["cv_excess"] for r in g])
        dx = np.array([r["data"]["cv_excess"] for r in g])
        am = [r["model"]["argmax_bin"] for r in g]
        print(f"  {arm:10s} stretch diff {st.mean():+.3f}+-{st.std():.3f} | "
              f"CV excess model {ex.mean():+.3f}+-{ex.std():.3f} data {dx.mean():+.3f} | "
              f"model argmax bins {am}  (arc = bins 5,6,7)")
    print(f"\nwrote {OUT}/s3_metric_warp.json")


if __name__ == "__main__":
    main(seeds=tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0, 1, 2))
