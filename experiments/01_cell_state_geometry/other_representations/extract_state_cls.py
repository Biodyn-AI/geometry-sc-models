"""Extract STATE SE-600M's CLS cell token on Setty, at the cross-model tap layer and the final layer.

WHY. See extract_scgpt_cls.py. route_branchpoint/extract_state.py already knows position 0 is CLS
(`valid[1:T] = True  # skip CLS`) but mean-pools the gene positions and discards it. This saves it.

Three readouts per cell from ONE forward pass:
  cls        : CLS at layer TAP (default 11) -- matched depth to the cross-model `emb` convention
  cls_final  : CLS at the last transformer layer -- STATE's shipped cell embedding depth
  emb        : mean-pool over expressed gene positions at TAP (excludes CLS) -- cross-model default

Out: data/celltoken/state_setty.npz
Run: ../../.venv_state/bin/python extract_state_cls.py [n_cells]
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
import torch

HERE = Path(__file__).resolve().parent
ROUTE_STATE = HERE.parent / "route_state"
sys.path.insert(0, str(ROUTE_STATE))
from state_loader import load_state_se, load_protein_embeds  # noqa: E402

ROOT = f"{_DATA}"
SETTY = os.environ.get("BP_H5AD", os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data/celltoken/state_setty.npz"))
TAP = int(os.environ.get("STATE_LAYER", "11"))
BATCH = int(os.environ.get("STATE_BATCH", "8"))
DEVICE = os.environ.get("STATE_DEVICE", "mps")


def main(n_cells):
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, cfg, genes, info = load_state_se(device=dev, dtype=torch.float32)
    pe_dict, gene_list, _ = load_protein_embeds()
    n_layers = len(model.transformer_encoder.layers)
    FINAL = n_layers - 1
    print(f"[load] STATE SE d={info['d_model']} L={n_layers} on {dev}; tap={TAP} final={FINAL}", flush=True)

    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)

    adata = ad.read_h5ad(SETTY)
    n0 = min(n_cells, adata.n_obs)
    rows = np.arange(n0)
    adata = adata[:n0].copy()
    n = adata.n_obs
    cats = adata.obs["clusters"].astype(str).values.copy()
    pt = adata.obs["palantir_pseudotime"].astype(float).values.copy()
    print(f"[data] N={n}", flush=True)

    adata = inferer._convert_to_csr(adata)
    gene_col = inferer._auto_detect_gene_column(adata)
    cfg.model.batch_size = BATCH
    dl = create_dataloader(cfg, adata=adata, adata_name="setty", shape_dict=None,
                           data_dir=str(Path(SETTY).parent), shuffle=False,
                           protein_embeds=pe_dict, precision=None, gene_column=gene_col)

    cap = {}
    h1 = model.transformer_encoder.layers[TAP].register_forward_hook(
        lambda m, i, o: cap.__setitem__("tap", o.detach()))
    h2 = model.transformer_encoder.layers[FINAL].register_forward_hook(
        lambda m, i, o: cap.__setitem__("fin", o.detach()))

    d = info["d_model"]
    cls_t = np.zeros((n, d), np.float32)
    cls_f = np.zeros((n, d), np.float32)
    emb_t = np.zeros((n, d), np.float32)
    filled = np.zeros(n, bool)

    def flush():
        if filled.sum() == 0:
            return
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.savez(OUT, cls=cls_t[filled], cls_final=cls_f[filled], emb=emb_t[filled],
                 pseudotime=pt[filled], clusters=cats[filled].astype(str),
                 cell_idx=np.asarray(rows)[filled])

    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            cap.clear()
            model._compute_embedding_for_batch(batch)
            rt = cap["tap"].float().cpu().numpy()               # (B,R,d)
            rf = cap["fin"].float().cpu().numpy()
            bs = batch[0].cpu().numpy()
            counts = batch[7].cpu().numpy() if batch[7] is not None else None
            idxs = batch[3].cpu().numpy()
            B, T = bs.shape
            for i in range(B):
                ci = int(idxs[i])
                if ci >= n or filled[ci]:
                    continue
                valid = np.zeros(T, bool); valid[1:T] = True     # skip CLS
                if counts is not None:
                    valid &= (counts[i, :T] > 0)
                pos = np.nonzero(valid)[0]
                if len(pos):
                    cls_t[ci] = rt[i, 0]                         # CLS at tap
                    cls_f[ci] = rf[i, 0]                         # CLS at final
                    emb_t[ci] = rt[i, pos].mean(0)               # genes at tap
                    filled[ci] = True
            seen += B
            if bi % 10 == 0:
                print(f"  batch {bi}: cells={seen}/{n} filled={int(filled.sum())}", flush=True)
            if bi % 25 == 0 and bi > 0:
                flush()
    h1.remove(); h2.remove()
    flush()
    print(f"saved {OUT} kept={int(filled.sum())}/{n} "
          f"cls={np.linalg.norm(cls_t[filled],axis=1).mean():.2f} "
          f"emb={np.linalg.norm(emb_t[filled],axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
