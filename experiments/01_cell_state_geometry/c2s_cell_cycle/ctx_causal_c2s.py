"""ctx_causal_c2s — HEADLINE #4: is the functional-context axis CAUSALLY USED? Port of ctx_causal.py.
Runs on the pod (needs the model). Reuses steer_c2s.propagation_test.

Build the functional axis u = mean a(nuclear genes) - mean a(surface genes) from the ctx cube at the peak-EXCESS
layer (in that layer's hidden space). Steer at that decoder layer: push a random gene-level half of a cell's
nuclear/surface genes toward +u / -u (dose = fraction x residual norm), read the nuclear-minus-surface next-gene
logit swing at the disjoint other half. Clean-causal gate: signed swing grows with dose, norm-matched random ~0.

Readout note: uses the first-subword-of-gene-name proxy (steer_c2s.pole_swing), an approximation of the route's
full teacher-forced gene-name log-prob scorer. Flagged in the write-up.
Out: results/ctx_causal_c2s.json
"""
import os, sys, json, argparse, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_lib_c2s as L
from ctx_functional_axes_c2s import AXES, G2G_PATH, pole_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, help="peak-EXCESS ctx/steer layer (block index)")
    ap.add_argument("--axis", default="nuclear_vs_surface")
    ap.add_argument("--bases", default="ctx_bases")
    ap.add_argument("--panel", default="data/ts_panel_celltype.h5ad")
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-cells", type=int, default=40)
    ap.add_argument("--alphas", default="0,0.25,0.5,1.0")
    ap.add_argument("--out", default="results/ctx_causal_c2s.json")
    a = ap.parse_args()
    alphas = tuple(float(x) for x in a.alphas.split(","))

    import anndata
    from steer_c2s import C2SSteerer, propagation_test
    g2g = pickle.load(open(G2G_PATH, "rb"))
    d = L.load(a.layer, a.bases)
    full = L.balanced(d["counts"], d["cap"])
    Mz, _ = L.zscore_dims(d["M"], full)
    A = L.a_space(Mz, full)
    Ag, Bg = AXES[a.axis]
    ia, ib = pole_rows(d["genes"], g2g, Ag, Bg)
    ia = ia[np.isfinite(A[ia]).all(1)]; ib = ib[np.isfinite(A[ib]).all(1)]
    # direction in the layer's hidden space: use the RAW (un-z-scored) main effect so it matches residual space
    Araw = np.array([np.nanmean(d["M"][:, np.where(full[:, g])[0], g, :].reshape(-1, d["M"].shape[-1]), 0)
                     if full[:, g].any() else np.full(d["M"].shape[-1], np.nan) for g in range(len(d["genes"]))])
    u = np.nanmean(Araw[ia], 0) - np.nanmean(Araw[ib], 0)
    nuc = [d["genes"][i] for i in ia]; surf = [d["genes"][i] for i in ib]
    print(f"axis {a.axis} @ L{a.layer}: nuclear={len(nuc)} surface={len(surf)} genes", flush=True)

    adata = anndata.read_h5ad(a.panel)
    st = C2SSteerer(a.model)
    rng = np.random.default_rng(0); rand = rng.standard_normal(st.hidden)
    rows = propagation_test(st, adata, u, a.layer, nuc, surf, source_genes=nuc + surf,
                            n_cells=a.n_cells, alphas=alphas, random_dir=rand, min_present=6)
    amax = max(alphas)
    signed = np.array([r[f"signed_{amax}"] for r in rows if f"signed_{amax}" in r], float)
    randv = np.array([r[f"rand_{amax}"] for r in rows if f"rand_{amax}" in r], float)
    sm = float(np.nanmean(signed)) if signed.size else None
    # sign test: signed swing > random swing across cells
    k = int(np.nansum(signed > randv)) if signed.size and randv.size else 0
    n = int(min(signed.size, randv.size))
    from math import comb
    sign_p = float(sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)) if n else 1.0
    summary = dict(axis=a.axis, layer=a.layer, n_cells=len(rows), n_nuclear=len(nuc), n_surface=len(surf),
                   signed_mean=sm, signed_frac_pos=float(np.nanmean(signed > 0)) if signed.size else None,
                   rand_mean=float(np.nanmean(randv)) if randv.size else None, sign_k=k, sign_n=n, sign_p=sign_p,
                   dose=[{"frac": x, "signed": float(np.nanmean([r.get(f"signed_{x}", np.nan) for r in rows])),
                          "rand": float(np.nanmean([r.get(f"rand_{x}", np.nan) for r in rows]))}
                         for x in alphas if x > 0])
    gate = (sm is not None and sm > 0 and sign_p < 0.05 and summary["rand_mean"] is not None
            and abs(sm) > 2 * abs(summary["rand_mean"]))
    summary["clean_causal"] = bool(gate)
    print(f"signed@{amax}={sm} frac>0={summary['signed_frac_pos']} rand={summary['rand_mean']} "
          f"sign_p={sign_p:.4f} -> clean_causal={gate}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
