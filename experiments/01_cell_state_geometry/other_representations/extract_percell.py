"""Generic per-cell UCE (100M, layer 2) residual-stream extractor for the geometry tests
(cell-cycle on K562, cell-type on Tabula Sapiens). Produces per-cell mean-pooled gene-token residual
+ CLS residual, keyed by original row index, plus any requested obs columns.

This is the cross-model-comparable substrate: the same mean-pool-over-gene-tokens convention used for
scGPT/Geneformer/STATE/MaxToki. The downstream curvature analyses read the target (cell-cycle phase,
cell type) from the source h5ad by cell_idx.

Output npz: emb (N,1280), cls (N,1280), cell_idx (N,), + one array per CARRY_OBS column.

Run (.venv_state):
  UCE_H5AD=<path> UCE_OUT=<path.npz> [UCE_GENECOL=auto|var_names|<col>] [CARRY_OBS=col1,col2]
  [UCE_NCELLS=N] [UCE_DEVICE=mps|cpu] python extract_percell.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from uce_loader import (  # noqa: E402
    load_uce_model, load_tok_aux, build_gene_maps, sample_cell_sentences,
    forward_residual, gene_position_mask,
)

H5AD = os.environ["UCE_H5AD"]
OUT = os.environ["UCE_OUT"]
GENECOL = os.environ.get("UCE_GENECOL", "auto")
XLAYER = os.environ.get("UCE_XLAYER", "")   # e.g. "decontXcounts" to use raw counts from a layer
CARRY_OBS = [c for c in os.environ.get("CARRY_OBS", "").split(",") if c]
HOOK_LAYER = int(os.environ.get("UCE_LAYER", "2"))
BATCH = int(os.environ.get("UCE_BATCH", "16"))
DEVICE = os.environ.get("UCE_DEVICE", "mps")
N_CELLS = int(os.environ.get("UCE_NCELLS", str(10**9)))


def gene_symbols(adata):
    if GENECOL == "var_names":
        return np.array([str(s).upper() for s in adata.var_names])
    if GENECOL != "auto" and GENECOL in adata.var.columns:
        return np.array([str(s).upper() for s in adata.var[GENECOL].values])
    vn = np.array([str(v) for v in adata.var_names])
    if np.mean([v.isdigit() for v in vn]) > 0.5 and "index" in adata.var.columns:
        return np.array([str(s).upper() for s in adata.var["index"].values])
    return np.array([s.upper() for s in vn])


def main():
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, info = load_uce_model(device=dev, dtype=torch.float32)
    g2i, off, cdf = load_tok_aux()
    vocab = set(g2i.keys())
    chrom_ok = set(cdf[cdf.species == "human"]["gene_symbol"].astype(str).str.upper())
    print(f"[load] UCE-100M on {dev}; hook layer {HOOK_LAYER}", flush=True)

    adata = ad.read_h5ad(H5AD)
    n0 = min(N_CELLS, adata.n_obs)
    adata = adata[:n0].copy()
    syms = gene_symbols(adata)
    keep, seen = [], set()
    for j, g in enumerate(syms):
        if g in vocab and g in chrom_ok and g not in seen:
            keep.append(j); seen.add(g)
    keep = np.array(keep, dtype=int)
    print(f"[genes] {len(keep)}/{adata.n_vars} map to UCE vocab", flush=True)

    X = adata.layers[XLAYER] if XLAYER and XLAYER in adata.layers else adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    X = X[:, keep]
    kept_syms = syms[keep]
    pe_row_idxs, chroms, starts = build_gene_maps(kept_syms, g2i, off, cdf)
    carried = {c: adata.obs[c].astype(str).values.copy() for c in CARRY_OBS if c in adata.obs.columns}
    n = adata.n_obs

    emb = np.zeros((n, info["d_model"]), np.float32)
    cls = np.zeros((n, info["d_model"]), np.float32)
    xuce = np.zeros((n, info["d_model"]), np.float32)   # UCE's true final output (X_uce)
    filled = np.zeros(n, bool)
    if os.path.exists(OUT):
        try:
            z = np.load(OUT, allow_pickle=True)
            if z["emb"].shape[1] == info["d_model"] and "xuce" in z.files:
                for e, cl, xu, ci in zip(z["emb"], z["cls"], z["xuce"], z["cell_idx"]):
                    if int(ci) < n:
                        emb[int(ci)] = e; cls[int(ci)] = cl; xuce[int(ci)] = xu; filled[int(ci)] = True
                print(f"[resume] pre-filled {int(filled.sum())}", flush=True)
        except Exception as ex:
            print(f"[resume] ignored ({ex})", flush=True)

    def flush():
        keepm = filled
        if keepm.sum() == 0:
            return
        idx = np.nonzero(keepm)[0]
        d = dict(emb=emb[keepm], cls=cls[keepm], xuce=xuce[keepm], cell_idx=idx)
        for c, v in carried.items():
            d[c] = v[keepm]
        np.savez(OUT, **d)

    todo = np.nonzero(~filled)[0]
    for bstart in range(0, len(todo), BATCH):
        rows = todo[bstart:bstart + BATCH]
        counts = torch.from_numpy(X[rows].toarray()).float()
        rng = np.random.default_rng(int(rows[0]))
        bs, msk, longest = sample_cell_sentences(counts, pe_row_idxs, chroms, starts, rng)
        bs = bs[:, :longest]; msk = msk[:, :longest]
        resid, tok_idx, cls_emb = forward_residual(model, bs, msk, dev, hook_layer=HOOK_LAYER)
        resid_bf = resid.permute(1, 0, 2).float().cpu()
        gmask = gene_position_mask(tok_idx).unsqueeze(-1).float()
        pooled = (resid_bf * gmask).sum(1) / gmask.sum(1).clamp(min=1)
        cls_resid = resid_bf[:, 0, :]
        xuce_b = cls_emb.float().cpu().numpy()
        for k, ci in enumerate(rows):
            emb[ci] = pooled[k].numpy(); cls[ci] = cls_resid[k].numpy()
            xuce[ci] = xuce_b[k]; filled[ci] = True
        if (bstart // BATCH) % 5 == 0:
            print(f"  cells={int(filled.sum())}/{n}", flush=True)
        if (bstart // BATCH) % 25 == 0 and bstart > 0:
            flush()
    flush()
    print(f"saved {OUT}  kept={int(filled.sum())}/{n}", flush=True)


if __name__ == "__main__":
    main()
