"""HOW FAR DOES THE SHALLOW BASELINE GO? -- width-matched, and with a bigger corpus.

coocc_matched.py established, on 14,769 matched genes with both nulls collapsing to chance:
    raw fetal-gut profile 0.044 | LSA-256 of the SAME cells 0.555 | MaxToki-217M 0.438 | MaxToki-1B 0.798
So the shallow factorisation already beats the 217M. Two confounds decide whether it also catches the 1B.

  WIDTH. The 1B table is 2304-dim, LSA is 256-dim, and this project has already been bitten once by reading a
  width advantage as a capability advantage (model_scale.py). The established fair control is ISOTROPIC RANDOM
  PROJECTION to a common width -- NOT PCA truncation, which is biased because it discards exactly the
  low-variance directions the coordinate lives in. Project both model tables to 256 and 64 dims, several seeds.

  CORPUS SIZE. Distributional embeddings improve with corpus size, and the LSA above saw only 8,000 of the
  62,849 fetal-gut cells. If LSA at full corpus keeps climbing, the model's remaining lead is a data-budget
  artifact rather than a representational one. Built by streaming the h5ad so the dense matrix is never formed.

Out: results/coocc_fair.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from table_grid import load as load_mt
from shallow_coocc_baseline import build_binary, lsa, probe, TOPK

SEED = 0
WIDTHS = [64, 256]
PROJ_SEEDS = [0, 1, 2]
CORPUS = [8000, 25000, 62849]


def stream_binary(path, want, topk=TOPK, n_cells=None, rng=np.random.default_rng(SEED)):
    """Binary top-K cell x gene matrix streamed from a CSR h5ad. `want` = symbols to keep, in order."""
    with h5py.File(path, "r") as f:
        X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
        fnm = f["var"]["feature_name"]
        syms = np.char.upper(G._dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]].astype(str))
        col = {s: i for i, s in enumerate(want)}
        keepcol = np.full(g, -1, np.int64)
        for j, s in enumerate(syms):
            if s in col:
                keepcol[j] = col[s]
        sel = np.arange(n) if (n_cells is None or n_cells >= n) else np.sort(rng.choice(n, n_cells, replace=False))
        indptr = X["indptr"][:]; data = X["data"]; idx = X["indices"]
        rows, cols = [], []
        for blk in range(0, len(sel), 2000):
            for c, r in enumerate(sel[blk:blk + 2000], start=blk):
                s0, e0 = int(indptr[r]), int(indptr[r + 1])
                if e0 <= s0:
                    continue
                ii = idx[s0:e0]; vv = data[s0:e0]
                m = keepcol[ii] >= 0
                ii, vv = keepcol[ii[m]], vv[m]
                if len(ii) > topk:
                    p = np.argpartition(-vv, topk)[:topk]; ii = ii[p]
                rows.append(np.full(len(ii), c, np.int32)); cols.append(ii.astype(np.int32))
        r = np.concatenate(rows); k = np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, k)), shape=(len(sel), len(want)))


def main():
    C = coords()
    tabs = {nm: load_mt(w, "output") for nm, w in [("MaxToki-217M", "217M"), ("MaxToki-1B", "1B")]}
    Pd, sd = G.basis("coexpr_devel")
    keep = [i for i, s in enumerate(sd) if s in C.index and C.chromosome[s] in AUTOSOMES]
    Pd = np.asarray(Pd[keep], np.float32); sd = np.array(sd)[keep]

    common = sorted(set.intersection(*[set(s) for _, s in tabs.values()], set(sd))
                    & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    y = np.array([C.chromosome[q] for q in common])
    st = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])
    print(f"{len(common)} matched genes | chance {1/22:.3f}\n")
    res = {"n_genes": len(common), "chance": 1 / 22, "width_matched": {}, "corpus": {}}

    def sub(M, syms):
        pi = {q: i for i, q in enumerate(syms)}
        return M[[pi[q] for q in common]]

    # ---------- width-matched ----------
    print("=== WIDTH-MATCHED (isotropic random projection; mean +- sd over 3 seeds) ===", flush=True)
    print(f"{'representation':<34} {'width':<7} {'random':<16} {'group'}")
    print("-" * 72)
    B8 = build_binary(Pd)
    for D in WIDTHS:
        E = sub(lsa(B8, dims=D), sd)
        r = probe(E, y, groups)
        res["width_matched"][f"LSA fetal gut @{D}"] = r
        print(f"{'LSA fetal gut (8k cells)':<34} {D:<7} {r['random']:<16.3f} {r['group']:.3f}", flush=True)
        del E; gc.collect()
        for nm, (M, s) in tabs.items():
            X0 = sub(M.astype(np.float32), s)
            rs, gs = [], []
            for ps in PROJ_SEEDS:
                rg = np.random.default_rng(ps)
                R = rg.normal(0, 1.0 / np.sqrt(D), size=(X0.shape[1], D)).astype(np.float32)
                p = probe(X0 @ R, y, groups); rs.append(p["random"]); gs.append(p["group"])
            res["width_matched"][f"{nm} @{D}"] = {"random": float(np.mean(rs)), "random_sd": float(np.std(rs)),
                                                  "group": float(np.mean(gs)), "group_sd": float(np.std(gs))}
            print(f"{nm:<34} {D:<7} {np.mean(rs):.3f} +- {np.std(rs):.3f}    "
                  f"{np.mean(gs):.3f} +- {np.std(gs):.3f}", flush=True)
            del X0; gc.collect()
        json.dump(res, open(os.path.join(HERE, "results", "coocc_fair.json"), "w"), indent=1)
    del B8; gc.collect()

    # ---------- corpus size ----------
    print("\n=== CORPUS SIZE (LSA-256, fetal gut, streamed) ===", flush=True)
    for n in CORPUS:
        try:
            B = stream_binary(G.FETAL_GUT, list(sd), n_cells=n)
            E = sub(lsa(B, dims=256), sd)
            r = probe(E, y, groups)
            res["corpus"][str(n)] = r
            print(f"  {n:>6} cells   random {r['random']:.3f}  group {r['group']:.3f}"
                  f"   ({B.nnz/1e6:.0f}M nonzeros)", flush=True)
            del B, E; gc.collect()
        except MemoryError:
            print(f"  {n:>6} cells   skipped (memory)", flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "coocc_fair.json"), "w"), indent=1)

    print("\n=== VERDICT ===")
    for D in WIDTHS:
        l = res["width_matched"].get(f"LSA fetal gut @{D}", {}).get("group", 0)
        for nm in tabs:
            m = res["width_matched"].get(f"{nm} @{D}", {}).get("group", 0)
            print(f"  @{D:<5} LSA {l:.3f} vs {nm} {m:.3f} -> "
                  f"{'MODEL leads' if m > l else 'SHALLOW BASELINE leads'}")
    if res["corpus"]:
        ks = sorted(res["corpus"], key=int)
        print(f"  corpus trend: " + " -> ".join(f"{k}:{res['corpus'][k]['group']:.3f}" for k in ks))
    print("\n[done] -> results/coocc_fair.json")


if __name__ == "__main__":
    main()
