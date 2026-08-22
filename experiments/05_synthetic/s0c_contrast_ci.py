"""S0c - a confidence interval on the S0b primary contrast.

S0b gave contrast = +0.0069 (C2S minus expression, G1->S windows minus all windows) against a
planted-2x-stretch reference of +0.0612 from the same instrument. Direction is as predicted, size
is ~9x smaller. This script asks whether it is distinguishable from zero.

Design: repeat the whole S0b pipeline over independent outer splits (a fresh train/test split of
the cells and fresh window subsamples each time), collect the contrast, and report the
distribution. Also run the identical loop with phase shuffled, which is the calibrated zero.
"""
import os
import json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s0_local_resolution import load_arms, BT
from s0b_plane_resolution import profile, synthetic_ring, N_BINS

N_OUTER = 60


def contrast_once(Xa, pa, Xb, pb, k, rng, n_sub, shuffle=False):
    """(arm A minus arm B) G1->S-minus-all contrast for one outer split."""
    if shuffle:
        pa, pb = rng.permutation(pa), rng.permutation(pb)
    ra, _, edges = profile(Xa, pa, k, rng, n_sub)
    rb, _, _ = profile(Xb, pb, k, rng, n_sub)
    a = np.array([r["dprime_per_deg"] if r else np.nan for r in ra])
    b = np.array([r["dprime_per_deg"] if r else np.nan for r in rb])
    d = a - b
    g1s = [i for i, lo in enumerate(edges) if 150.0 <= lo < 240.0]
    if np.all(np.isnan(d[g1s])) or np.all(np.isnan(d)):
        return np.nan
    return float(np.nanmean(d[g1s]) - np.nanmean(d))


def summarise(v, label):
    v = np.asarray([x for x in v if np.isfinite(x)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"  {label:28s} mean {v.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"n={len(v)}  frac>0 {np.mean(v > 0):.2f}")
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(v)), "frac_gt0": float(np.mean(v > 0))}


def main(k=20):
    rng = np.random.default_rng(1)
    arms = load_arms()
    out = {"k": k, "n_outer": N_OUTER}

    names = ["c2s_2b_L21", "expression"]
    common = set(arms[names[0]][2].tolist()) & set(arms[names[1]][2].tolist())
    common = np.array(sorted(common))
    prepped = {}
    for nm in names:
        emb, phi, ci = arms[nm]
        m = np.isin(ci, common)
        o = np.argsort(ci[m])
        prepped[nm] = (emb[m][o], phi[m][o])
    width = 360.0 / N_BINS
    counts = [int(((prepped["expression"][1] - lo) % 360.0 < width).sum())
              for lo in np.arange(N_BINS) * width]
    n_sub = max(25, int(np.min(counts)) // 2)
    print(f"cells {len(common)}, matched n = {n_sub} per window, {N_OUTER} outer splits\n")

    print("REAL")
    real = [contrast_once(*prepped["c2s_2b_L21"], *prepped["expression"], k=k,
                          rng=rng, n_sub=n_sub) for _ in range(N_OUTER)]
    out["real"] = summarise(real, "C2S - expression contrast")

    print("CALIBRATED ZERO (phase shuffled)")
    null = [contrast_once(*prepped["c2s_2b_L21"], *prepped["expression"], k=k,
                          rng=rng, n_sub=n_sub, shuffle=True) for _ in range(N_OUTER)]
    out["shuffled"] = summarise(null, "same, phase shuffled")

    print("REFERENCE EFFECT SIZE (planted stretch, same instrument)")
    for factor in (2.0, 1.5, 1.2):
        vals = []
        for _ in range(max(12, N_OUTER // 4)):
            Xs, ps = synthetic_ring(3000, 150.0, 240.0, factor, 0.15, rng)
            Xu, pu = synthetic_ring(3000, 150.0, 240.0, 1.0, 0.15, rng)
            vals.append(contrast_once(Xs, ps, Xu, pu, k, rng, n_sub))
        out[f"planted_{factor}x"] = summarise(vals, f"planted {factor}x stretch vs uniform")

    p = f"{BT}/manifolds/synthetic/results/s0c_contrast_ci_k{k}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
