"""Extract per-cell UCE (100M, layer 2 of 4) residual-stream embeddings on a Setty-schema
differentiation trajectory, for the cross-model branch-point curvature comparison.

Mirrors route_branchpoint/extract_state.py in output schema: a mean-pooled per-cell residual over the
gene-token positions (the cross-model convention used for scGPT/Geneformer/STATE/MaxToki), plus the CLS
cell embedding (UCE's native X_uce) for a robustness check.

Input  : a Setty-schema h5ad (raw counts in .X, gene symbols in var_names or var['index'],
         obs['palantir_pseudotime'], obs['clusters']).
Output : data/branchpoint/uce_<suffix>.npz  with
         emb (N,1280) f32   : mean-pooled layer-2 residual over gene tokens
         cls (N,1280) f32   : layer-2 residual at the CLS position
         pseudotime (N,)    : palantir_pseudotime
         clusters (N,) str  : cluster labels
         cell_idx (N,)      : original row indices

Run (.venv_state):
  UCE_SUFFIX=setty    BP_H5AD=".../data/hematopoiesis/setty19_cd34_bm.h5ad"        python extract_branchpoint.py
  UCE_SUFFIX=lung     BP_H5AD=".../data/pancreas/lung_airway_setty_schema.h5ad"   python extract_branchpoint.py
  UCE_SUFFIX=gut      BP_H5AD=".../data/pancreas/gut_setty_schema.h5ad"           python extract_branchpoint.py
  UCE_SUFFIX=pancreas BP_H5AD=".../data/pancreas/pancreas_setty_schema.h5ad"      python extract_branchpoint.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
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

DATA = f"{_DATA}"
SUFFIX = os.environ.get("UCE_SUFFIX", "setty")
DEFAULT_H5AD = {
    "setty": f"{DATA}/hematopoiesis/setty19_cd34_bm.h5ad",
    "lung": f"{DATA}/pancreas/lung_airway_setty_schema.h5ad",
    "gut": f"{DATA}/pancreas/gut_setty_schema.h5ad",
    "pancreas": f"{DATA}/pancreas/pancreas_setty_schema.h5ad",
}
H5AD = os.environ.get("BP_H5AD", DEFAULT_H5AD.get(SUFFIX, DEFAULT_H5AD["setty"]))
OUT = os.environ.get("BP_OUT", f"{DATA}/branchpoint/uce_{SUFFIX}.npz")
HOOK_LAYER = int(os.environ.get("UCE_LAYER", "2"))
BATCH = int(os.environ.get("UCE_BATCH", "16"))
DEVICE = os.environ.get("UCE_DEVICE", "mps")
N_CELLS = int(os.environ.get("UCE_NCELLS", "10**9".replace("10**9", str(10**9))))


def gene_symbols(adata):
    """Setty stores symbols in var_names; the setty-schema tissues store them in var['index']."""
    vn = np.array([str(v) for v in adata.var_names])
    frac_digit = np.mean([v.isdigit() for v in vn])
    if frac_digit > 0.5 and "index" in adata.var.columns:
        return np.array([str(s).upper() for s in adata.var["index"].values])
    return np.array([s.upper() for s in vn])


def main():
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, info = load_uce_model(device=dev, dtype=torch.float32)
    g2i, off, cdf = load_tok_aux()
    vocab = set(g2i.keys())
    chrom_ok = set(cdf[cdf.species == "human"]["gene_symbol"].astype(str).str.upper())
    print(f"[load] UCE-100M d={info['d_model']} L={info['nlayers']} on {dev}; hook layer {HOOK_LAYER}", flush=True)

    adata = ad.read_h5ad(H5AD)
    n0 = min(N_CELLS, adata.n_obs)
    adata = adata[:n0].copy()
    syms = gene_symbols(adata)

    # keep genes UCE can tokenize (in the .pt vocab AND the human chrom CSV), first occurrence only
    keep, seen = [], set()
    for j, g in enumerate(syms):
        if g in vocab and g in chrom_ok and g not in seen:
            keep.append(j); seen.add(g)
    keep = np.array(keep, dtype=int)
    kept_syms = syms[keep]
    print(f"[genes] {len(keep)}/{adata.n_vars} genes map to UCE vocab", flush=True)

    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    X = X[:, keep]
    pt = adata.obs["palantir_pseudotime"].astype(float).values.copy()
    cats = adata.obs["clusters"].astype(str).values.copy()
    n = adata.n_obs

    pe_row_idxs, chroms, starts = build_gene_maps(kept_syms, g2i, off, cdf)

    emb = np.zeros((n, info["d_model"]), np.float32)
    cls = np.zeros((n, info["d_model"]), np.float32)
    xuce = np.zeros((n, info["d_model"]), np.float32)   # UCE's true final output (X_uce)
    filled = np.zeros(n, bool)

    # resume
    if os.path.exists(OUT):
        try:
            z = np.load(OUT, allow_pickle=True)
            if z["emb"].shape[1] == info["d_model"] and "xuce" in z.files:
                for e, cl, xu, ci in zip(z["emb"], z["cls"], z["xuce"], z["cell_idx"]):
                    if int(ci) < n:
                        emb[int(ci)] = e; cls[int(ci)] = cl; xuce[int(ci)] = xu; filled[int(ci)] = True
                print(f"[resume] pre-filled {int(filled.sum())} cells", flush=True)
        except Exception as ex:
            print(f"[resume] ignored existing ({ex})", flush=True)

    def flush():
        keepm = filled
        if keepm.sum() == 0:
            return
        idx = np.nonzero(keepm)[0]
        np.savez(OUT, emb=emb[keepm], cls=cls[keepm], xuce=xuce[keepm], pseudotime=pt[keepm],
                 clusters=cats[keepm].astype(str), cell_idx=idx)

    todo = np.nonzero(~filled)[0]
    for bstart in range(0, len(todo), BATCH):
        rows = todo[bstart:bstart + BATCH]
        counts = torch.from_numpy(X[rows].toarray()).float()
        # deterministic per-cell rng seeded by original row index (resume-safe, reproducible)
        rng = np.random.default_rng(int(rows[0]))
        bs, msk, longest = sample_cell_sentences(counts, pe_row_idxs, chroms, starts, rng)
        bs = bs[:, :longest]; msk = msk[:, :longest]
        resid, tok_idx, cls_emb = forward_residual(model, bs, msk, dev, hook_layer=HOOK_LAYER)
        resid_bf = resid.permute(1, 0, 2).float().cpu()      # (batch, seq, d)
        gmask = gene_position_mask(tok_idx).unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (resid_bf * gmask).sum(1) / gmask.sum(1).clamp(min=1)
        cls_resid = resid_bf[:, 0, :]
        xuce_b = cls_emb.float().cpu().numpy()               # (batch, d) = UCE final X_uce
        for k, ci in enumerate(rows):
            emb[ci] = pooled[k].numpy(); cls[ci] = cls_resid[k].numpy()
            xuce[ci] = xuce_b[k]; filled[ci] = True
        done = int(filled.sum())
        if (bstart // BATCH) % 5 == 0:
            print(f"  batch {bstart//BATCH}: cells={done}/{n}", flush=True)
        if (bstart // BATCH) % 20 == 0 and bstart > 0:
            flush()
    flush()
    print(f"saved {OUT}  kept={int(filled.sum())}/{n}  "
          f"emb norm mean={np.linalg.norm(emb[filled], axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
