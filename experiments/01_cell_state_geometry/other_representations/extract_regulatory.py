"""Extract per-gene-token UCE (100M, layer 2) residual-stream activations on the Replogle K562 subset,
for the Route B bilinear-vs-linear regulatory-logic comparison. Mirrors route_state/extract.py's storage
layout so route_state/train_compare.py's machinery applies unchanged.

UCE samples 1024 gene tokens per cell (expression-weighted, with replacement). We keep the residual at
each UNIQUE sampled gene position per cell (first occurrence), mirroring the "one row per expressed gene
per cell" convention of the other models' extractors.

Output (data/uce_acts/uce_L2/):
  layer_02_activations.npy  (N_tok, 1280) float16
  layer_02_gene_ids.npy     (N_tok,) int32   — local index into gene_vocab (-> symbol)
  layer_02_cell_ids.npy     (N_tok,) int32   — subset cell row
  gene_vocab.json           {local_idx: symbol}
  cell_perturbation.npy     (n_cells,) str   — obs['gene'] per cell (perturbation label)
  extraction_metadata.json

Run (.venv_state):
  UCE_NCELLS=1200 UCE_DEVICE=mps python extract_regulatory.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, warnings
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
    forward_residual, gene_position_mask, CHROM_TOKEN_OFFSET,
)
import config as C  # noqa: E402

DATA = f"{_DATA}"
SUBSET = os.path.join(DATA, "state_activations", "replogle_k562_subset.h5ad")
N_CELLS = int(os.environ.get("UCE_NCELLS", "1200"))
BATCH = int(os.environ.get("UCE_BATCH", "16"))
DEVICE = os.environ.get("UCE_DEVICE", "mps")
LAYER = C.LAYER


def main():
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, info = load_uce_model(device=dev, dtype=torch.float32)
    g2i, off, cdf = load_tok_aux()
    vocab = set(g2i.keys())
    chrom_ok = set(cdf[cdf.species == "human"]["gene_symbol"].astype(str).str.upper())
    print(f"[load] UCE-100M on {dev}; hooking layer {LAYER}", flush=True)

    adata = ad.read_h5ad(SUBSET)
    adata = adata[:N_CELLS].copy()
    perturb = adata.obs["gene"].astype(str).values.copy() if "gene" in adata.obs else np.array([""] * adata.n_obs)
    syms = np.array([str(s).upper() for s in adata.var_names])
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
    n_cells = adata.n_obs

    pe_row_idxs, chroms, starts = build_gene_maps(kept_syms, g2i, off, cdf)
    # invert token-index -> local gene index (for gene_ids)
    tok_to_local = {int(t): i for i, t in enumerate(pe_row_idxs.numpy())}

    outdir = Path(C.ACT_DIR); outdir.mkdir(parents=True, exist_ok=True)
    PAD = 1024
    max_rows = n_cells * PAD
    acts = np.lib.format.open_memmap(outdir / f"layer_{LAYER:02d}_activations.npy", mode="w+",
                                     dtype=np.float16, shape=(max_rows, info["d_model"]))
    gene_ids = np.empty(max_rows, np.int32)
    cell_ids = np.empty(max_rows, np.int32)
    n = 0
    for bstart in range(0, n_cells, BATCH):
        rows = np.arange(bstart, min(bstart + BATCH, n_cells))
        counts = torch.from_numpy(X[rows].toarray()).float()
        rng = np.random.default_rng(int(rows[0]))
        bs, msk, longest = sample_cell_sentences(counts, pe_row_idxs, chroms, starts, rng)
        bs = bs[:, :longest]; msk = msk[:, :longest]
        resid, tok_idx, _ = forward_residual(model, bs, msk, dev, hook_layer=LAYER)
        resid_bf = resid.permute(1, 0, 2).float().cpu().numpy()      # (B, seq, d)
        gmask = gene_position_mask(tok_idx).numpy()                  # (B, seq)
        toks = tok_idx.numpy()
        for k, ci in enumerate(rows):
            pos = np.nonzero(gmask[k])[0]
            seen_g = set()
            for p in pos:
                loc = tok_to_local.get(int(toks[k, p]))
                if loc is None or loc in seen_g:
                    continue                                        # unique genes only
                seen_g.add(loc)
                acts[n] = resid_bf[k, p].astype(np.float16)
                gene_ids[n] = loc
                cell_ids[n] = int(ci)
                n += 1
        if (bstart // BATCH) % 10 == 0:
            print(f"  cells={rows[-1]+1}/{n_cells} tokens={n}", flush=True)

    acts.flush()
    final = np.lib.format.open_memmap(outdir / f"layer_{LAYER:02d}_activations.npy", mode="r+",
                                      dtype=np.float16, shape=(max_rows, info["d_model"]))
    trimmed = np.lib.format.open_memmap(outdir / f"layer_{LAYER:02d}_activations_trim.npy", mode="w+",
                                        dtype=np.float16, shape=(n, info["d_model"]))
    CH = 200000
    for s in range(0, n, CH):
        trimmed[s:min(s + CH, n)] = final[s:min(s + CH, n)]
    trimmed.flush()
    os.replace(outdir / f"layer_{LAYER:02d}_activations_trim.npy",
               outdir / f"layer_{LAYER:02d}_activations.npy")
    np.save(outdir / f"layer_{LAYER:02d}_gene_ids.npy", gene_ids[:n])
    np.save(outdir / f"layer_{LAYER:02d}_cell_ids.npy", cell_ids[:n])
    np.save(outdir / "cell_perturbation.npy", perturb[:n_cells])
    used = np.unique(gene_ids[:n])
    json.dump({int(u): kept_syms[int(u)] for u in used}, open(outdir / "gene_vocab.json", "w"))
    json.dump(dict(layer=LAYER, d_model=info["d_model"], n_tokens=int(n), n_cells=int(n_cells),
                   n_unique_genes=int(len(used)), source="replogle_k562_subset",
                   mean_tokens_per_cell=float(n / max(n_cells, 1))),
              open(outdir / "extraction_metadata.json", "w"), indent=2)
    print(f"[done] {n} tokens, {n_cells} cells, {len(used)} genes -> {outdir}")


if __name__ == "__main__":
    main()
