"""ctx_polysemy_c2s — HEADLINE #1: does a gene respond to cell context GENE-SPECIFICALLY and reproducibly?
Port of ctx_polysemy.py's EXCESS metric.

For each context pair (c1,c2) with >=MIN_SHARED balanced genes:
  D_p = Mz[p,c2] - Mz[p,c1]            # per-gene contextual shift, partition p in {0,1} (disjoint cells)
  b_p = D_p.mean(0)                    # context main effect (the 'crowd' moves together) -> mainrep=cos(b0,b1)
  d_p = D_p - b_p                      # crowd-removed = GENE-SPECIFIC shift
  same = cos_rows(d0, d1)              # does gene g's own shift agree across the two disjoint halves?
  diff = cos_rows(d0, d1[perm])        # does it agree with a DIFFERENT gene's shift?
  EXCESS = mean(same) - mean(diff)     # gene-specific agreement in excess of crowd agreement
Bootstrap CI over pooled per-gene values. Verdict POSITIVE iff CI-lo > 0. L0 embedding tap should give ~0.

Out: results/ctx_polysemy_c2s.json
"""
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_lib_c2s as L


def excess_for_tap(tap, bases, min_shared=200, n_boot=2000, seed=0):
    d = L.load(tap, bases)
    full = L.balanced(d["counts"], d["cap"])
    Mz, top_dim = L.zscore_dims(d["M"], full)
    nC = len(d["contexts"])
    rng = np.random.default_rng(seed)
    same_all, diff_all, mainreps = [], [], []
    pairs = 0
    for c1 in range(nC):
        for c2 in range(c1 + 1, nC):
            keep = full[c1] & full[c2]
            if keep.sum() < min_shared:
                continue
            D0 = Mz[0, c2, keep] - Mz[0, c1, keep]
            D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
            b0, b1 = D0.mean(0), D1.mean(0)
            mainreps.append(float(b0 @ b1 / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-12)))
            dp0, dp1 = D0 - b0, D1 - b1
            same = L.cos_rows(dp0, dp1)
            perm = rng.permutation(keep.sum())
            diff = L.cos_rows(dp0, dp1[perm])
            same_all.append(same); diff_all.append(diff); pairs += 1
    if pairs == 0:
        return dict(tap=tap, error="no context pair reached min_shared", top_dim_share=top_dim)
    same = np.concatenate(same_all); diff = np.concatenate(diff_all)
    excess = float(same.mean() - diff.mean())
    ns, nd = len(same), len(diff)
    boot = np.array([same[rng.integers(0, ns, ns)].mean() - diff[rng.integers(0, nd, nd)].mean()
                     for _ in range(n_boot)])
    return dict(tap=tap, excess=excess, ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)), mainrep=float(np.mean(mainreps)),
                same_mean=float(same.mean()), diff_mean=float(diff.mean()),
                n_pairs=pairs, n_gene_obs=int(ns), top_dim_share=round(top_dim, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="ctx_bases")
    ap.add_argument("--taps", type=int, nargs="+", default=None)
    ap.add_argument("--min-shared", type=int, default=200)
    ap.add_argument("--out", default="results/ctx_polysemy_c2s.json")
    a = ap.parse_args()
    import glob
    taps = a.taps or sorted(int(os.path.basename(p).split("_L")[1].split(".")[0]) for p in glob.glob(os.path.join(a.bases, "ctx_c2s_L*.npz")))
    res = {}
    for t in taps:
        r = excess_for_tap(t, a.bases, a.min_shared)
        res[f"L{t:02d}"] = r
        if "excess" in r:
            print(f"  L{t:02d}: EXCESS={r['excess']:+.3f} CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"mainrep={r['mainrep']:.3f} pairs={r['n_pairs']} top_dim={r['top_dim_share']}", flush=True)
        else:
            print(f"  L{t:02d}: {r.get('error')}", flush=True)
    best = max((v for v in res.values() if "excess" in v), key=lambda v: v["excess"], default=None)
    verdict = "no balanced data"
    if best:
        verdict = "POSITIVE (gene-specific + reproducible)" if best["ci_lo"] > 0 else "null (CI touches 0)"
        print(f"\nVERDICT: {verdict} | peak EXCESS {best['excess']:+.3f} at L{best['tap']:02d}", flush=True)
    res["_verdict"] = verdict
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
