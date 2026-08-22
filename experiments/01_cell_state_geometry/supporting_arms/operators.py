"""Stage-1 operator library for the manifold-discovery port (handoff §2).

Each operator maps a cell -> a per-cell feature vector on the bio-sae RESIDUAL STREAM (d_model=512 scGPT /
1152 Geneformer). Per-cell embedding = mean-pool over the cell's tokens (same recipe as
route_geometry/geometry_test.py::per_cell_embeddings).

Branches
  raw_mean      : mean-pooled L11 residual (d_model)
  svd50         : top-50 PCA of raw_mean
  linear_sae    : trained linear TopK atlas latents at L11 (Tier-A baseline)   [reuse cached]
  bilinear      : trained atomic bilinear latents at L11 (Tier-A winner)       [reuse cached]
  linear_drift  : residual drift map f_drift = [d_early - d_mid ; d_mid - d_late] on pooled layer blocks
  bilinear_drift: bilinear latents pooled per block (scGPT: AEs at L02/L05/L08/L11), same difference map

Heavy layer streams are cached OUTSIDE the repo (common.DATA). Cheap to re-score afterwards.
Run:  ../.venv/bin/python operators.py [scgpt|geneformer]   (builds + caches all internal operators)
"""
import os
import sys
import numpy as np
import torch

import common as K

sys.path.insert(0, K.ROUTE_B)
sys.path.insert(0, os.path.join(K.ROUTE_B, "baseline"))
import config as C                       # route_b/config.py
from bilinear_ae import BilinearAE       # route_b/bilinear_ae.py
import sae_baseline as SB                # route_b/baseline/sae_baseline.py

BATCH = 8192


def percell_layer_mean(model_base, layer, n_cells):
    """Per-cell mean-pooled residual at a given layer. Cached npy in common.DATA."""
    cache = os.path.join(K.DATA, f"{model_base}_percell_L{layer:02d}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    tgt = C.TARGETS[f"{model_base}_L{layer:02d}"]
    d_model = tgt["d_model"]
    acts = np.load(tgt["acts"], mmap_mode="r")
    cell_ids = np.load(tgt["cell_ids"]).astype(np.int64)
    n_pos = acts.shape[0]
    ssum = np.zeros((n_cells, d_model), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.float64)
    for s in range(0, n_pos, BATCH):
        e = min(s + BATCH, n_pos)
        xb = np.asarray(acts[s:e], dtype=np.float64)
        cid = cell_ids[s:e]
        np.add.at(ssum, cid, xb)
        np.add.at(cnt, cid, 1.0)
    out = (ssum / np.clip(cnt, 1, None)[:, None]).astype(np.float32)
    np.save(cache, out)
    print(f"    cached per-cell mean {model_base} L{layer:02d} {out.shape}", flush=True)
    return out


def block_means(model_base, blocks, n_cells):
    """dict{early,mid,late} of per-cell block-averaged residuals."""
    out = {}
    for name, layers in blocks.items():
        acc = np.zeros((n_cells, C.TARGETS[f"{model_base}_L{layers[0]:02d}"]["d_model"]), dtype=np.float64)
        for L in layers:
            acc += percell_layer_mean(model_base, L, n_cells)
        out[name] = (acc / len(layers)).astype(np.float32)
    return out


def linear_drift(model_base, blocks, n_cells):
    b = block_means(model_base, blocks, n_cells)
    return np.concatenate([b["early"] - b["mid"], b["mid"] - b["late"]], axis=1)


def _encode_bilinear_layer(model_base, layer, n_cells):
    """Per-cell mean-pooled ATOMIC bilinear latents at `layer` (streams that layer's residual once)."""
    cache = os.path.join(K.DATA, f"{model_base}_bilat_L{layer:02d}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    run = f"{model_base}_L{layer:02d}_atomic_muon"
    ck = torch.load(os.path.join(K.ROUTE_B, "runs", run, "latest.pt"), map_location="cpu", weights_only=False)
    bil = BilinearAE.from_config(ck["config"]).to(K.DEV).eval()
    bil.load_state_dict(ck["model"])
    mean = ck["mean"].float() if torch.is_tensor(ck["mean"]) else torch.tensor(ck["mean"]).float()
    tgt = C.TARGETS[f"{model_base}_L{layer:02d}"]
    nf = tgt["n_features"]
    acts = np.load(tgt["acts"], mmap_mode="r")
    cell_ids = np.load(tgt["cell_ids"]).astype(np.int64)
    ssum = torch.zeros(n_cells, nf, dtype=torch.float64)
    cnt = np.zeros(n_cells, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, acts.shape[0], BATCH):
            e = min(s + BATCH, acts.shape[0])
            xb = torch.from_numpy(np.asarray(acts[s:e], dtype=np.float32)) - mean
            h = bil.encode(xb.to(K.DEV)).double().cpu()
            ssum.index_add_(0, torch.from_numpy(cell_ids[s:e]), h)
            np.add.at(cnt, cell_ids[s:e], 1.0)
    out = (ssum / torch.tensor(cnt).clamp_min(1)[:, None]).float().numpy()
    np.save(cache, out)
    print(f"    cached bilinear latents {model_base} L{layer:02d} {out.shape}", flush=True)
    return out


def bilinear_drift_scgpt(n_cells):
    """scGPT bilinear drift: block reps from trained AEs at L02(early)/L05,L08(mid)/L11(late)."""
    early = _encode_bilinear_layer("scgpt", 2, n_cells)
    mid = 0.5 * (_encode_bilinear_layer("scgpt", 5, n_cells) + _encode_bilinear_layer("scgpt", 8, n_cells))
    late = _encode_bilinear_layer("scgpt", 11, n_cells)
    return np.concatenate([early - mid, mid - late], axis=1)


def cached_latents(model_key):
    """linear_sae + bilinear (atomic) per-cell latents from route_geometry/results/emb_cache."""
    d = np.load(os.path.join(K.ROUTE_GEO, "results", "emb_cache", f"{model_key}.npz"), allow_pickle=True)
    return dict(linear_sae=d["linear_sae"], bilinear=d["atomic"])


def build_internal(model_key, with_drift=True):
    """Return dict{operator_name: (n_cells, dim) float32} for the internal panel."""
    model_base = "scgpt" if model_key.startswith("scgpt") else "geneformer"
    blocks = K.SCGPT_BLOCKS if model_base == "scgpt" else K.GENEFORMER_BLOCKS
    lat = cached_latents(model_key)
    n_cells = lat["linear_sae"].shape[0]
    ops = dict(linear_sae=lat["linear_sae"], bilinear=lat["bilinear"])
    raw = percell_layer_mean(model_base, 11, n_cells)
    ops["raw_mean"] = raw
    rc = raw - raw.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(rc, full_matrices=False)
    ops["svd50"] = (rc @ Vt[:50].T).astype(np.float32)
    if with_drift:
        ops["linear_drift"] = linear_drift(model_base, blocks, n_cells).astype(np.float32)
        if model_base == "scgpt":
            ops["bilinear_drift"] = bilinear_drift_scgpt(n_cells).astype(np.float32)
    return ops


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "scgpt"
    key = "scgpt_L11" if which == "scgpt" else "geneformer_L11"
    print(f"building operators for {key} ...", flush=True)
    ops = build_internal(key, with_drift=True)
    for name, X in ops.items():
        print(f"  {name:15s} {X.shape}", flush=True)
    print("done.")
