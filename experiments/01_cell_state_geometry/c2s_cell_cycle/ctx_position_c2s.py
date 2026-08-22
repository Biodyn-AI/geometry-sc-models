"""ctx_position_c2s — HEADLINE deflation #1: is the gene-specific context response just token-RANK encoding?
Port of ctx_position_confound.py (the control the C2S pilot could not run — needs stored token rank).

Same per-(context-pair) crowd-removed shifts d0,d1 as EXCESS, plus rank = mean token position per (ctx,gene):
  1. rho = Spearman(shift-magnitude, |Δrank|)   -> is a bigger shift just a bigger rank change?
  2. median-split on |Δrank|: EXCESS in rank-STABLE vs rank-MOVING genes (should be similar if not position).
  3. OLS-residualise d0,d1 on A=[1, rank_c1, rank_c2, |Δrank|] -> r0,r1; recompute EXCESS -> excess_residualised.
Verdict SURVIVES iff excess_residualised > 0.05.  Out: results/ctx_position_c2s.json
"""
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_lib_c2s as L


def for_tap(tap, bases, min_shared=200, seed=0):
    d = L.load(tap, bases)
    full = L.balanced(d["counts"], d["cap"])
    Mz, _ = L.zscore_dims(d["M"], full)
    rank = L.rank_mean(d["rank_tok"])                    # [nC, nG]
    nC = len(d["contexts"]); rng = np.random.default_rng(seed)
    S_lo, D_lo, S_hi, D_hi, S_res, D_res, mags, dranks = ([] for _ in range(8))
    for c1 in range(nC):
        for c2 in range(c1 + 1, nC):
            keep = full[c1] & full[c2]
            if keep.sum() < min_shared:
                continue
            D0 = Mz[0, c2, keep] - Mz[0, c1, keep]; D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
            d0 = D0 - D0.mean(0); d1 = D1 - D1.mean(0)
            dr = np.abs(rank[c2, keep] - rank[c1, keep])
            mag = 0.5 * (np.linalg.norm(d0, axis=1) + np.linalg.norm(d1, axis=1))
            mags.append(mag); dranks.append(dr)
            perm = rng.permutation(keep.sum())
            med = np.median(dr); lo, hi = dr <= med, dr > med
            S_lo.append(L.cos_rows(d0[lo], d1[lo])); D_lo.append(L.cos_rows(d0[lo], d1[perm][lo]))
            S_hi.append(L.cos_rows(d0[hi], d1[hi])); D_hi.append(L.cos_rows(d0[hi], d1[perm][hi]))
            A = np.column_stack([np.ones(keep.sum()), rank[c1, keep], rank[c2, keep], dr])
            proj = lambda V: V - A @ np.linalg.lstsq(A, V, rcond=None)[0]
            r0, r1 = proj(d0), proj(d1)
            S_res.append(L.cos_rows(r0, r1)); D_res.append(L.cos_rows(r0, r1[perm]))
    if not S_res:
        return dict(tap=tap, error="no pair reached min_shared")
    ex = lambda S, D: float(np.concatenate(S).mean() - np.concatenate(D).mean())
    rho = float(spearmanr(np.concatenate(mags), np.concatenate(dranks)).statistic)
    return dict(tap=tap, rho_mag_vs_rank=rho, excess_rank_stable=ex(S_lo, D_lo),
                excess_rank_moving=ex(S_hi, D_hi), excess_residualised=ex(S_res, D_res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="ctx_bases")
    ap.add_argument("--taps", type=int, nargs="+", default=None)
    ap.add_argument("--min-shared", type=int, default=200)
    ap.add_argument("--out", default="results/ctx_position_c2s.json")
    a = ap.parse_args()
    import glob
    taps = a.taps or sorted(int(os.path.basename(p).split("_L")[1].split(".")[0]) for p in glob.glob(os.path.join(a.bases, "ctx_c2s_L*.npz")))
    res = {}
    for t in taps:
        r = for_tap(t, a.bases, a.min_shared)
        res[f"L{t:02d}"] = r
        if "excess_residualised" in r:
            print(f"  L{t:02d}: EXCESS_resid={r['excess_residualised']:+.3f} "
                  f"(stable {r['excess_rank_stable']:+.3f} / moving {r['excess_rank_moving']:+.3f}) "
                  f"rho(mag,Δrank)={r['rho_mag_vs_rank']:+.3f}", flush=True)
    valid = [v for v in res.values() if "excess_residualised" in v]
    best = max(valid, key=lambda v: v["excess_residualised"]) if valid else None
    verdict = ("SURVIVES rank control" if best and best["excess_residualised"] > 0.05
               else "COLLAPSES under rank control" if best else "no data")
    if best:
        print(f"\nVERDICT: {verdict} | best EXCESS_resid {best['excess_residualised']:+.3f} at L{best['tap']:02d}", flush=True)
    res["_verdict"] = verdict
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
