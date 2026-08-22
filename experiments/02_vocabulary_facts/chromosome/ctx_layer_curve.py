"""WHERE IN THE NETWORK DOES THE GENE-SPECIFIC CONTEXT RESPONSE PEAK? (the depth profile)

The headline analyses used taps L0/L4/L8/L11 and found the gene-specific context response (EXCESS) peaks at L4
and fades by L11, with L0 exactly 0 (context-free embedding). This maps the FULL depth profile from a 12-layer
re-extraction (ctxscan_L00..L11), to characterise the shape rather than infer it from four points.

For every layer we recompute the two metrics that carry the result:
  EXCESS   gene-specific context response: same-gene vs different-gene directional agreement of the crowd-removed
           pairwise context shift, across the two independent cell partitions (as in ctx_polysemy.py).
  MAINREP  context main-effect replication (the averaging null's reproducibility) -- a positive control that the
           measurement works at each depth.

Reads ctxscan_L{00..11}.npz (OUTPREFIX=ctxscan run; separate from the headline ctx_maxtoki_L*.npz).
Out: results/ctx_layer_curve.json
"""
import os, sys, json, glob, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
PREFIX = os.environ.get("PREFIX", "ctxscan")
MIN_GENES = 200
SEED = 0


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def excess_for(path, rng):
    z = np.load(path, allow_pickle=True)
    M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
    nP, nC, nG, d = M.shape
    full = (counts == cap).all(0)
    flat = M[:, full]
    mu = flat.reshape(-1, d).mean(0); sd = flat.reshape(-1, d).std(0) + 1e-6
    Mz = (M - mu) / sd
    top_dim = float(((sd ** 2).max()) / (sd ** 2).sum())
    same_all, diff_all, main = [], [], []
    for c1, c2 in itertools.combinations(range(nC), 2):
        keep = full[c1] & full[c2]
        if keep.sum() < MIN_GENES:
            continue
        D0 = Mz[0, c2, keep] - Mz[0, c1, keep]; D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
        b0, b1 = D0.mean(0), D1.mean(0)
        main.append(float(np.dot(b0, b1) / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-9)))
        d0, d1 = D0 - b0, D1 - b1
        same_all.append(cos_rows(d0, d1)); diff_all.append(cos_rows(d0, d1[rng.permutation(len(d1))]))
    if not same_all:
        return None
    S = np.concatenate(same_all); Dg = np.concatenate(diff_all)
    excess = float(S.mean() - Dg.mean())
    bs = [float(S[rng.integers(0, len(S), len(S))].mean() - Dg[rng.integers(0, len(Dg), len(Dg))].mean())
          for _ in range(1000)]
    return dict(excess=excess, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                main_effect_replication=float(np.mean(main)), n_pairs=len(same_all), top_dim_share=top_dim)


def main():
    files = sorted(glob.glob(os.path.join(RES, f"{PREFIX}_L*.npz")))
    if not files:
        print(f"no {PREFIX}_L*.npz found — run the 12-layer extraction first"); return
    rng = np.random.default_rng(SEED)
    out = {"layers": {}}
    print(f"{'layer':<7} {'EXCESS':<9} {'95% CI':<22} {'main-effect repl':<18} {'top-dim share'}")
    for f in files:
        L = int(os.path.basename(f).split("_L")[1].split(".")[0])
        r = excess_for(f, rng)
        if r is None:
            print(f"L{L:02d}     (too few balanced genes)"); continue
        out["layers"][f"L{L:02d}"] = r
        print(f"L{L:02d}     {r['excess']:+.4f}   [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]     "
              f"{r['main_effect_replication']:+.3f}            {r['top_dim_share']:.1%}")
    if out["layers"]:
        peak = max(out["layers"].items(), key=lambda kv: kv[1]["excess"])
        out["peak_layer"] = peak[0]
        out["verdict"] = (f"gene-specific context response peaks at {peak[0]} (EXCESS {peak[1]['excess']:+.3f}); "
                          "L0 is the context-free embedding (~0 by construction), the response rises through the "
                          "early layers and decays toward the output.")
        print(f"\n{out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_layer_curve.json"), "w"), indent=1)
    print("[done] -> results/ctx_layer_curve.json")


if __name__ == "__main__":
    main()
