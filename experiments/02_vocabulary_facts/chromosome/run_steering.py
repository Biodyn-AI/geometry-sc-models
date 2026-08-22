"""Gate-3 CAUSAL steering on C2S — runs on the pod (needs the model).

For each antipodal survivor, build the steering axis from the CTX BASIS at layer L:
    u = centroid(pole-A marker genes) - centroid(pole-B marker genes)      # hidden-space vector at layer L
Then the split-half propagation test (steer_c2s.propagation_test): push a random half of a cell's axis-marker
positions toward +u / -u after block L, read the pole-A-minus-pole-B next-token logit swing at the OTHER half.
A causal channel: signed swing grows with dose, a norm-matched random direction stays flat.

Verdict per hypothesis: signed swing at max dose > 0 and monotone, random ~0, in a majority of cells.
Out: results/steering.json
Usage: python run_steering.py --layer 8 --panel data/ts_panel.h5ad --hypotheses antipodal_gata1_spi1,...
"""
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gene_sets as S


def ctx_centroid_dir(M, syms, poleA, poleB):
    pos = {s: i for i, s in enumerate(syms)}
    a = [pos[g.upper()] for g in poleA if g.upper() in pos]
    b = [pos[g.upper()] for g in poleB if g.upper() in pos]
    if len(a) < 3 or len(b) < 3:
        return None, len(a), len(b)
    return M[a].mean(0) - M[b].mean(0), len(a), len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, help="ctx/steer layer (block index)")
    ap.add_argument("--panel", default="data/ts_panel.h5ad")
    ap.add_argument("--bases", default="bases")
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--hypotheses", default="", help="comma-sep antipodal names; default all antipodal")
    ap.add_argument("--n-cells", type=int, default=30)
    ap.add_argument("--alphas", default="0,0.25,0.5,1.0", help="dose as FRACTION of residual norm")
    ap.add_argument("--out", default="results/steering.json")
    args = ap.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(","))

    import anndata
    from steer_c2s import C2SSteerer, propagation_test
    z = np.load(os.path.join(args.bases, f"c2s_ctx_L{args.layer:02d}.npz"), allow_pickle=True)
    M, syms = z["M"].astype(np.float64), np.char.upper(z["symbols"].astype(str))
    adata = anndata.read_h5ad(args.panel)
    st = C2SSteerer(args.model)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(st.hidden)

    # steerable specs: the cell-cycle G1/S<->G2/M axis (the best-covered decodable direction) + antipodal axes
    specs = []
    cc = S.H["cellcycle_circle"]
    deg = np.rad2deg(np.asarray(cc["coord"], float)) % 360
    ccA = [g for g, d in zip(cc["genes"], deg) if d < 90]         # G1/S arc
    ccB = [g for g, d in zip(cc["genes"], deg) if d >= 90]        # G2/M arc
    specs.append(("cellcycle_g1s_vs_g2m", ccA, ccB, cc["genes"]))
    for name in (args.hypotheses.split(",") if args.hypotheses else list(S.H)):
        h = S.H.get(name, {})
        if h.get("kind") == "antipodal":
            pA = [g for g, s in zip(h["axis_genes"], h["axis_sign"]) if s > 0]
            pB = [g for g, s in zip(h["axis_genes"], h["axis_sign"]) if s < 0]
            specs.append((name, pA, pB, h["axis_genes"]))

    print(f"steering layer {args.layer} | {len(specs)} axes | {args.n_cells} cells | frac-doses {alphas}", flush=True)
    res = {}
    for name, poleA, poleB, source in specs:
        u, na, nb = ctx_centroid_dir(M, syms, poleA, poleB)
        if u is None:
            print(f"[{name}] SKIP — poles undercovered in ctx (A={na} B={nb})", flush=True)
            res[name] = dict(skipped=f"A={na} B={nb}"); json.dump(res, open(args.out, "w"), indent=1); continue
        rows = propagation_test(st, adata, u, args.layer, poleA, poleB, source_genes=source,
                                n_cells=args.n_cells, alphas=alphas, random_dir=rand, min_present=4)
        amax = max(alphas)
        signed = np.array([r[f"signed_{amax}"] for r in rows if f"signed_{amax}" in r], float)
        randv = np.array([r[f"rand_{amax}"] for r in rows if f"rand_{amax}" in r], float)
        sm = float(np.nanmean(signed)) if signed.size else None
        fp = float(np.nanmean(signed > 0)) if signed.size else None
        rm = float(np.nanmean(randv)) if randv.size else None
        summary = dict(n_cells=len(rows), n_poleA=na, n_poleB=nb, signed_mean=sm, signed_frac_pos=fp,
                       rand_mean=rm,
                       dose=[{"frac": a,
                              "signed": (float(np.nanmean([r.get(f"signed_{a}", np.nan) for r in rows])) if rows else None),
                              "rand": (float(np.nanmean([r.get(f"rand_{a}", np.nan) for r in rows])) if rows else None)}
                             for a in alphas if a > 0])
        res[name] = dict(summary=summary, rows=rows)
        smt = f"{sm:+.3f}" if sm is not None else "n/a(0 cells)"
        print(f"[{name}] cells={len(rows)} signed@{amax}={smt} "
              f"(frac>0 {fp if fp is None else round(fp,2)}) rand {rm if rm is None else round(rm,3)}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)                # persist per axis
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
