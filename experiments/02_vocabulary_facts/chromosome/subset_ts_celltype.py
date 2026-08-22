"""Build a CELL-TYPE-STRATIFIED TS panel for Thread B (the NOTE's 12 cell types x 1000 cells).
Each cell keeps obs.cell_type (short label); var_names = gene symbols. Sparse -> uploadable.
Out: <out>/ts_panel_celltype.h5ad
"""
import os, sys, argparse
import numpy as np, h5py, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2s_gm_lib as G

# (cell_type string in obs, source file key, short label)
TARGETS = [
    ("B cell", "immune", "Bcell"),
    ("CD4-positive, alpha-beta T cell", "immune", "CD4T"),
    ("CD8-positive, alpha-beta T cell", "immune", "CD8T"),
    ("neutrophil", "immune", "neutrophil"),
    ("macrophage", "lung", "macrophage"),
    ("classical monocyte", "lung", "cMono"),
    ("intermediate monocyte", "lung", "iMono"),
    ("kidney epithelial cell", "kidney", "kidneyEpi"),
    ("capillary endothelial cell", "lung", "capEndo"),
    ("basal cell", "lung", "basal"),
    ("pulmonary alveolar type 1 cell", "lung", "AT1"),
    ("pulmonary alveolar type 2 cell", "lung", "AT2"),
]
FILES = {"immune": "tabula_sapiens_immune_subset_20000.h5ad",
         "lung": "tabula_sapiens_lung.h5ad", "kidney": "tabula_sapiens_kidney.h5ad"}


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import anndata, pandas as pd
    rng = np.random.default_rng(0)
    blocks, labels, syms0 = [], [], None
    for ct, fkey, short in TARGETS:
        with h5py.File(os.path.join(G.TS_RAW, FILES[fkey]), "r") as f:
            X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
            fnm = f["var"]["feature_name"]
            syms = dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]]
            if syms0 is None:
                syms0 = syms
            assert np.array_equal(syms, syms0), f"{fkey}: gene space differs"
            cobs = f["obs"]["cell_type"]
            cats = dec(cobs["categories"][:]).astype(str); codes = cobs["codes"][:]
            if ct not in set(cats):
                print(f"  !! {ct} not in {fkey}; skip"); continue
            ci = np.where(codes == list(cats).index(ct))[0]
            sel = np.sort(rng.choice(ci, min(args.n_per_type, len(ci)), replace=False))
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            rd, ri, rp = [], [], [0]
            for r in sel:
                s, e = int(indptr[r]), int(indptr[r + 1])
                rd.append(data[s:e]); ri.append(idx[s:e]); rp.append(rp[-1] + (e - s))
            blocks.append(sp.csr_matrix((np.concatenate(rd) if rd else np.zeros(0),
                                         np.concatenate(ri) if ri else np.zeros(0, int),
                                         np.array(rp)), shape=(len(sel), g)))
            labels += [short] * len(sel)
            print(f"  {short:<11} ({ct[:32]:<32} @ {fkey}): {len(sel)} cells", flush=True)
    Xall = sp.vstack(blocks).tocsr()
    ad = anndata.AnnData(X=Xall, obs=pd.DataFrame({"cell_type": labels}),
                         var=pd.DataFrame(index=np.char.upper(syms0.astype(str))))
    ad.var_names_make_unique()
    out = os.path.join(args.out, "ts_panel_celltype.h5ad")
    ad.write_h5ad(out)
    print(f"-> {out}  shape={ad.shape}  {len(set(labels))} cell types  size={os.path.getsize(out)/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
