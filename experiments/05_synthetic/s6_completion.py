"""S6 - does the model complete a manifold it never observed?

The sharpest operationalisation of "invents vs reparametrises". Delete a contiguous arc of the ring
from training entirely, so no cell in the gap is ever seen. Then ask whether the model's
representation still bridges it.

The real-data version was run once and came back negative: deleting an intermediate population and
steering toward it produced the wrong lineage -- "steering interpolates, it does not generate". But
on real data you get one gap of one size. Here we sweep the gap and get a curve: how wide a hole
can the model bridge?

Three readouts, in increasing strength:

  geometry  do held-out gap cells land at the right phase? (circular decodability of gap cells from
            a readout fitted ONLY on training cells -- the model never saw this region)
  metric    is the representational path across the gap continuous, or does it collapse? Measured as
            the ratio of the gap's chord length to the mean arc length of an equal span of
            observed ring. ~1 = bridged smoothly; >>1 or ~0 = broken.
  baseline  the same two numbers computed on raw expression. If expression bridges the gap just as
            well, the bridging is the DATA's linear structure, not model invention.

The baseline arm is what makes this a real test. A ring in expression space is already linear-ish
locally, so interpolation across a modest gap is expected for free; only an advantage over that
counts as the model supplying something.
"""
import os
import json, os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import ring_corpus, tokenize, train, cell_embeddings  # noqa: E402
from s3_metric_warp import lognorm, OUT  # noqa: E402

N_POOL, STEPS = 30000, 3000
GAPS = (0.0, 30.0, 60.0, 90.0, 120.0)      # degrees of the ring removed from training


def circ_decode(X_tr, t_tr, X_te, t_te, k=20):
    """Fit cos/sin readout on training cells only; score held-out cells. Returns R_diff and error."""
    p = PCA(n_components=min(k, X_tr.shape[1], len(X_tr) - 1), whiten=True, random_state=0)
    Z_tr = p.fit_transform(X_tr.astype(np.float64))
    Z_te = p.transform(X_te.astype(np.float64))
    Y = np.c_[np.cos(np.deg2rad(t_tr)), np.sin(np.deg2rad(t_tr))]
    P = Ridge(alpha=1.0).fit(Z_tr, Y).predict(Z_te)
    pred = np.rad2deg(np.arctan2(P[:, 1], P[:, 0])) % 360.0
    d = np.deg2rad(pred - t_te)
    return {"R_diff": float(np.abs(np.mean(np.exp(1j * d)))),
            "median_err_deg": float(np.median(np.abs(np.angle(np.exp(1j * d))) * 180 / np.pi))}


def bridge_ratio(X, theta, lo, hi, k=20, n_knots=24):
    """Chord across the gap divided by mean arc step of an equal span elsewhere on the ring."""
    Z = PCA(n_components=min(k, X.shape[1], len(X) - 1), whiten=True,
            random_state=0).fit_transform(X.astype(np.float64))
    w = 360.0 / n_knots
    K, ok = [], []
    for i in range(n_knots):
        m = np.where(((theta - i * w) % 360.0) < w)[0]
        K.append(Z[m].mean(0) if len(m) >= 10 else np.full(Z.shape[1], np.nan))
        ok.append(len(m) >= 10)
    K, ok = np.array(K), np.array(ok)
    steps = [np.linalg.norm(K[(i + 1) % n_knots] - K[i])
             for i in range(n_knots) if ok[i] and ok[(i + 1) % n_knots]]
    if not steps:
        return None
    span = max(1, int(round((hi - lo) / w)))
    i_lo = int(lo // w) % n_knots
    i_hi = (i_lo + span) % n_knots
    if not (ok[i_lo] and ok[i_hi]):
        return None
    chord = np.linalg.norm(K[i_hi] - K[i_lo])
    return float(chord / (np.mean(steps) * span))


def run(gap_deg, seed=0):
    t0 = time.time()
    counts, theta, meta = ring_corpus(N_POOL, arm="uniform", seed=seed)
    lo = 150.0
    hi = (lo + gap_deg) % 360.0
    in_gap = (((theta - lo) % 360.0) < gap_deg) if gap_deg > 0 else np.zeros(len(theta), bool)

    tr = np.where(~in_gap)[0][:20000]
    te = np.where(in_gap)[0] if gap_deg > 0 else np.where(~in_gap)[0][20000:22000]
    if len(te) < 200:
        return None

    data_tr = tokenize(counts[tr], seed=seed)
    model, hist = train(data_tr, meta["n_genes"], steps=STEPS, seed=seed, quiet=True)
    E_tr = cell_embeddings(model, data_tr)
    E_te = cell_embeddings(model, tokenize(counts[te], seed=seed + 1))
    D_tr, D_te = lognorm(counts[tr]), lognorm(counts[te])

    allE = np.zeros((len(counts), E_tr.shape[1])); allE[tr] = E_tr; allE[te] = E_te
    keep = np.concatenate([tr, te])

    r = {"gap_deg": gap_deg, "seed": seed, "n_train": len(tr), "n_gap": len(te),
         "val_corr": hist[-1]["val_corr"],
         "model_decode": circ_decode(E_tr, theta[tr], E_te, theta[te]),
         "data_decode": circ_decode(D_tr, theta[tr], D_te, theta[te]),
         "model_bridge": bridge_ratio(allE[keep], theta[keep], lo, lo + max(gap_deg, 30.0)),
         "data_bridge": bridge_ratio(np.vstack([D_tr, D_te]), theta[keep], lo,
                                     lo + max(gap_deg, 30.0)),
         "secs": round(time.time() - t0, 1)}
    md, dd = r["model_decode"], r["data_decode"]
    print(f"  gap {gap_deg:5.0f} deg  n_gap {len(te):5d} | "
          f"model R_diff {md['R_diff']:.3f} err {md['median_err_deg']:4.0f}deg | "
          f"data R_diff {dd['R_diff']:.3f} err {dd['median_err_deg']:4.0f}deg | "
          f"bridge model {r['model_bridge']:.2f} data {r['data_bridge']:.2f}  ({r['secs']:.0f}s)")
    return r


def main(seeds=(0, 1)):
    print("S6: can the model bridge a region of the manifold it never saw?")
    print("    (chance R_diff ~ 0.03; a model that merely interpolates the DATA's linear")
    print("     structure will match the data arm -- only an advantage over it is invention)\n")
    # RESUME: keep anything already on disk so a crash costs nothing
    path = f"{OUT}/s6_completion.json"
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["gap_deg"], r["seed"]) for r in rows}
    if done:
        print(f"resuming; {len(done)} rows already done: {sorted(done)}\n")
    for s in seeds:
        print(f"seed {s}")
        for g in GAPS:
            if (g, s) in done:
                print(f"  gap {g:5.0f} deg  seed {s}  [cached]")
                continue
            r = run(g, s)
            if r:
                rows.append(r)
                json.dump(rows, open(path, "w"), indent=1)

    print("\n=== SUMMARY: model minus data, on held-out gap cells ===")
    for g in GAPS:
        sub = [r for r in rows if r["gap_deg"] == g]
        if not sub:
            continue
        dm = np.mean([r["model_decode"]["R_diff"] - r["data_decode"]["R_diff"] for r in sub])
        mm = np.mean([r["model_decode"]["R_diff"] for r in sub])
        dd = np.mean([r["data_decode"]["R_diff"] for r in sub])
        print(f"  gap {g:5.0f} deg  model {mm:.3f}  data {dd:.3f}  diff {dm:+.3f}")
    print(f"\nwrote {OUT}/s6_completion.json")


if __name__ == "__main__":
    main()
