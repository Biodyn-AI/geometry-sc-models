"""Extract STATE-SE per-cell L11 residual for Tabula Sapiens immune/kidney/lung cells (cell-type geometry).

Mirrors route_state/extract.py's faithful pipeline (safetensors SE model injected into STATE's own
Inference + create_dataloader, forward hook on transformer_encoder.layers[11]) but:
  - input = Tabula Sapiens processed h5ad (raw counts from the `decontXcounts` layer; var = gene symbols),
  - stratified cell-type subsample per tissue (cheap; CPU/MPS courteous),
  - mean-pools the per-gene residual to ONE per-cell 2048-vector on the fly (no big memmap),
  - stores per-cell embedding + cell_type + tissue for the downstream curvature test.

Run:  ../../.venv_state/bin/python ts_extract.py [n_per_tissue] [smoke]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os, sys
from pathlib import Path
import anndata as ad
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "route_state"))
from state_loader import load_state_se, load_protein_embeds  # noqa: E402

TS_DIR = f"{_DATA}/biodyn-work/subproject_09_causal_mediation_circuit_map/implementation/results/preprocessed"
TISSUES = {"immune": "tabula_sapiens_immune_subset_20000_processed.h5ad",
           "kidney": "tabula_sapiens_kidney_processed.h5ad",
           "lung": "tabula_sapiens_lung_processed.h5ad"}
CACHE = f"{_DATA}/state_geometry_cache"
LAYER = 11
PAD = 2048
DEVICE = os.environ.get("STATE_DEVICE", "mps")
BATCH = int(os.environ.get("STATE_BATCH", "8"))


def subsample(adata, n, seed=0, top_types=8, cap=70, min_per_type=30):
    """Concentrate the subsample on the tissue's most abundant cell types (>= cap cells each where possible),
    so each retained type has enough cells for stable 5-fold CV in the curvature test."""
    rng = np.random.default_rng(seed)
    ct = adata.obs["cell_type"].astype(str).values
    u, cnt = np.unique(ct, return_counts=True)
    order = np.argsort(-cnt)
    keep_types = [t for t in u[order] if cnt[list(u).index(t)] >= min_per_type][:top_types]
    picked = []
    for t in keep_types:
        gi = np.where(ct == t)[0]
        picked.append(rng.choice(gi, min(cap, len(gi)), replace=False))
        if sum(len(p) for p in picked) >= n:
            break
    sel = np.sort(np.concatenate(picked))[:n]
    return sel


def extract_tissue(tissue, fname, n, model, cfg, pe_dict, dev):
    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    a = ad.read_h5ad(os.path.join(TS_DIR, fname))
    sel = subsample(a, n)
    a = a[sel].copy()
    cell_type = a.obs["cell_type"].astype(str).values.copy()
    # STATE needs raw counts: use decontXcounts layer if present
    if "decontXcounts" in a.layers:
        a.X = a.layers["decontXcounts"]
    a.layers.clear()
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)
    a = inferer._convert_to_csr(a)
    gene_col = inferer._auto_detect_gene_column(a)
    cfg.model.batch_size = BATCH
    dl = create_dataloader(cfg, adata=a, adata_name=tissue, shape_dict=None,
                           data_dir=CACHE, shuffle=False, protein_embeds=pe_dict,
                           precision=None, gene_column=gene_col)
    captured = {}
    h = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", o.detach()))
    n_cells = a.n_obs
    emb = np.zeros((n_cells, 2048), np.float32)
    got = np.zeros(n_cells, bool)
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            model._compute_embedding_for_batch(batch)
            res = captured["x"].float().cpu().numpy()          # (B, R, d)
            counts = batch[7].cpu().numpy() if batch[7] is not None else None
            idxs = batch[3].cpu().numpy()                      # (B,) local cell rows
            B, T = batch[0].shape
            for i in range(B):
                valid = np.zeros(T, bool); valid[1:T] = True   # skip CLS
                if counts is not None:
                    valid &= (counts[i, :T] > 0)
                pos = np.nonzero(valid)[0]
                emb[int(idxs[i])] = res[i, pos].mean(0)
                got[int(idxs[i])] = True
            if bi % 20 == 0:
                print(f"    [{tissue}] batch {bi} cells~{got.sum()}/{n_cells}", flush=True)
    h.remove()
    assert got.all(), f"{(~got).sum()} cells missed"
    return emb, cell_type


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    if smoke:
        n_per = 40
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, cfg, genes, info = load_state_se(device=dev, dtype=torch.float32)
    pe_dict, gene_list, _ = load_protein_embeds()
    print(f"[load] SE on {dev}; hooking L{LAYER}; n_per_tissue={n_per}", flush=True)
    tissues = ["immune"] if smoke else list(TISSUES)
    out = os.path.join(CACHE, "state_L11_ts_percell" + ("_smoke" if smoke else "") + ".npz")
    embs, cts, tis = [], [], []
    done = set()
    if os.path.exists(out):                       # resume: reuse tissues already extracted
        try:
            z = np.load(out, allow_pickle=True)
            for t in np.unique(z["tissue"]):
                m = z["tissue"] == t
                embs.append(z["E"][m]); cts.append(z["cell_type"][m]); tis.append(z["tissue"][m])
                done.add(str(t))
            print(f"[resume] loaded tissues {sorted(done)} from {out}", flush=True)
        except Exception as ex:
            print(f"[resume] could not load ({ex}); starting fresh", flush=True)
    for t in tissues:
        if t in done:
            print(f"  [{t}] already done, skipping", flush=True); continue
        e, c = extract_tissue(t, TISSUES[t], n_per, model, cfg, pe_dict, dev)
        embs.append(e); cts.append(c); tis.append(np.full(len(c), t))
        print(f"  [{t}] -> {e.shape}", flush=True)
        # checkpoint after each tissue
        np.savez(out, E=np.concatenate(embs), cell_type=np.concatenate(cts), tissue=np.concatenate(tis))
        print(f"  [ckpt] wrote {sum(len(x) for x in cts)} cells ({len(tis)} tissues) -> {out}", flush=True)
    E = np.concatenate(embs); cell_type = np.concatenate(cts); tissue = np.concatenate(tis)
    np.savez(out, E=E, cell_type=cell_type, tissue=tissue)
    print(f"[done] {E.shape} cells, {len(set(cell_type))} types, {len(set(tissue))} tissues -> {out}")


if __name__ == "__main__":
    main()
