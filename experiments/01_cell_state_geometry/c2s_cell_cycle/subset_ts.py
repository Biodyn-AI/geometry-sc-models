"""Build a compact TS panel h5ad on the SAME seed-0 cells as c2s_gm_lib.build_coexpr, so the coexpr
baseline and the C2S ctx basis describe identical cells. Sparse -> small enough to upload to the pod.
var_names = gene symbols (feature_name), which cell_sentences.anndata_to_ranked_genes needs.
Out: <out>/ts_panel.h5ad
"""
import os, sys, argparse
import numpy as np
import h5py
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2s_gm_lib as G


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tissue", type=int, default=G.N_CELLS_PER_TISSUE)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import anndata

    rng = np.random.default_rng(0)          # MUST match build_coexpr's rng sequence
    blocks, obs_tissue, syms0 = [], [], None
    for fn in G.TS_FILES:
        with h5py.File(os.path.join(G.TS_RAW, fn), "r") as f:
            X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
            fnm = f["var"]["feature_name"]
            syms = _dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]]
            if syms0 is None:
                syms0 = syms
            assert np.array_equal(syms, syms0), f"{fn}: gene space differs"
            sel = np.sort(rng.choice(n, min(args.n_per_tissue, n), replace=False))
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            rows_data, rows_idx, rows_ptr = [], [], [0]
            for r in sel:
                s, e = int(indptr[r]), int(indptr[r + 1])
                rows_data.append(data[s:e]); rows_idx.append(idx[s:e])
                rows_ptr.append(rows_ptr[-1] + (e - s))
            block = sp.csr_matrix((np.concatenate(rows_data) if rows_data else np.zeros(0),
                                   np.concatenate(rows_idx) if rows_idx else np.zeros(0, int),
                                   np.array(rows_ptr)), shape=(len(sel), g))
            blocks.append(block)
            obs_tissue += [fn.replace("tabula_sapiens_", "").replace(".h5ad", "")] * len(sel)
            print(f"[{fn}] sampled {len(sel)} cells", flush=True)
    Xall = sp.vstack(blocks).tocsr()
    import pandas as pd
    ad = anndata.AnnData(X=Xall,
                         obs=pd.DataFrame({"tissue": obs_tissue}),
                         var=pd.DataFrame(index=np.char.upper(syms0.astype(str))))
    ad.var_names_make_unique()
    out = os.path.join(args.out, "ts_panel.h5ad")
    ad.write_h5ad(out)
    print(f"-> {out}  shape={ad.shape}  size={os.path.getsize(out)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
