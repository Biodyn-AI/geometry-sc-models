"""Stage-3 external holdout extractor (handoff §2 Stage-3, §4 step 5; pipeline §6 Phase 7).

Runs STRICT non-overlap external cells through frozen scGPT whole-human (mirrors
repos/bio-sae/scgpt_src/01_extract_activations.py) and produces COMPACT per-cell operator features
(not raw token dumps). Two panels:
  external_ts : held-out Tabula Sapiens immune+kidney+lung, excluding every internal cell_idx (3-tissue,
                same ruler structure as internal). The decisive external holdout.
  krasnow     : Krasnow lung Smart-seq2 -> expected-to-fail cross-platform control.

Output npz (outside repo, common.DATA): per-cell lin_feat, bil_feat, raw11, block means, bilinear-drift
block latents, cell_type, tissue. Zero internal obs-id overlap (immune/kidney/lung excluded by cell_idx).

Run (schedule when machine idle):  ../.venv/bin/python extract_external.py [external_ts|krasnow] [n_per]
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch

import common as K
sys.path.insert(0, K.ROUTE_B); sys.path.insert(0, os.path.join(K.ROUTE_B, "baseline"))
import config as C
from bilinear_ae import BilinearAE
import sae_baseline as SB

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ = 512, 12, 8, 1200
DRIFT_LAYERS = sorted(set(sum(K.SCGPT_BLOCKS.values(), [])))          # [1..11]
BIL_LAYERS = [2, 5, 8, 11]


def load_cat(h5group, col):
    c = h5group[col]
    if isinstance(c, h5py.Group):
        cats, codes = c["categories"][:], c["codes"][:]
        if cats.dtype.kind in ("O", "S"):
            cats = np.array([x.decode() if isinstance(x, bytes) else x for x in cats])
        return cats[codes]
    d = c[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in d]) if d.dtype.kind in ("O", "S") else d


def gene_names_of(path):
    with h5py.File(path) as f:
        return load_cat(f["var"], "feature_name")


def sparse_row(fX, i, ncols):
    ip = fX["indptr"]; a, b = int(ip[i]), int(ip[i + 1])
    row = np.zeros(ncols, np.float32)
    row[fX["indices"][a:b]] = fX["data"][a:b]
    return row


def tokenize(expr, gene_names, vocab, pad_id):
    nz = np.where(expr > 0)[0]
    ids, val = [], []
    for idx in nz:
        g = gene_names[idx]
        if g in vocab:
            ids.append(vocab[g]); val.append(expr[idx])
    if not ids:
        return None
    ids = np.array(ids, np.int64); val = np.array(val, np.float32)
    o = np.argsort(-val)[:MAX_SEQ]
    ids, val = ids[o], val[o]
    n = len(ids)
    gi = np.pad(ids, (0, MAX_SEQ - n), constant_values=pad_id)
    gv = np.pad(val, (0, MAX_SEQ - n), constant_values=-2)
    mask = np.zeros(MAX_SEQ, bool); mask[n:] = True
    return gi, gv, mask, n


def internal_excludes():
    cd = json.load(open(K.TS_META))["cell_data"]
    ex = {}
    for c in cd:
        ex.setdefault(c["tissue"], set()).add(int(c["cell_idx"]))
    return ex


def panel_cells(panel, n_per, seed=0):
    """Return list of (path, row_idx, tissue). Strict non-overlap for TS tissues."""
    rng = np.random.default_rng(seed)
    ex = internal_excludes()
    out = []
    if panel == "external_ts":
        specs = [("immune", os.path.join(K.RAW, "tabula_sapiens_immune_subset_20000.h5ad")),
                 ("kidney", os.path.join(K.RAW, "tabula_sapiens_kidney.h5ad")),
                 ("lung", os.path.join(K.RAW, "tabula_sapiens_lung.h5ad"))]
        for tis, path in specs:
            with h5py.File(path) as f:
                n = f["X"]["indptr"].shape[0] - 1
            avail = np.array([i for i in range(n) if i not in ex.get(tis, set())])
            sel = rng.choice(avail, size=min(n_per, len(avail)), replace=False)
            out += [(path, int(i), tis) for i in sel]
    else:  # krasnow
        path = os.path.join(K.RAW, "krasnow_lung_smartsq2.h5ad")
        with h5py.File(path) as f:
            n = f["X"]["indptr"].shape[0] - 1
        sel = rng.choice(n, size=min(n_per, n), replace=False)
        out = [(path, int(i), "lung") for i in sel]
    return out


def build_model(vocab):
    sys.path.insert(0, K.SCGPT_REPO)
    import scgpt  # noqa
    from scgpt.model.model import TransformerModel
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL, nlayers=N_LAYERS,
                         vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=-2,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False, do_dab=False,
                         use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(K.SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main(panel, n_per):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"panel={panel} n_per={n_per} device={dev}", flush=True)
    vocab = json.load(open(K.SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    model = build_model(vocab).to(dev)

    # frozen operator dictionaries
    sae, sae_cfg = SB.load_sae(str(C.TARGETS["scgpt_L11"]["linear_sae_dir"]), device="cpu")
    lin_mean = torch.tensor(sae_cfg["activation_mean"]) if sae_cfg.get("activation_mean") is not None else torch.zeros(D_MODEL)
    bils = {}
    for L in BIL_LAYERS:
        ck = torch.load(os.path.join(K.ROUTE_B, "runs", f"scgpt_L{L:02d}_atomic_muon", "latest.pt"),
                        map_location="cpu", weights_only=False)
        b = BilinearAE.from_config(ck["config"]).eval(); b.load_state_dict(ck["model"])
        bils[L] = (b, ck["mean"].float() if torch.is_tensor(ck["mean"]) else torch.tensor(ck["mean"]).float())

    outs = {i: [] for i in DRIFT_LAYERS}      # per-layer raw means placeholder (we store blocks)
    rec = dict(lin=[], bil=[], raw11=[], be=[], bm=[], bl=[],
               bd_e=[], bd_m=[], bd_l=[], cell_type=[], tissue=[])

    layer_out = {}
    hooks = []
    for i, lyr in enumerate(model.transformer_encoder.layers):
        hooks.append(lyr.register_forward_hook(lambda mod, inp, out, i=i: layer_out.__setitem__(i, out.detach())))

    cells = panel_cells(panel, n_per)
    # cache gene-name maps per file
    gmap = {}
    t0 = __import__("time").time()
    kept = 0
    for ci, (path, row, tis) in enumerate(cells):
        if path not in gmap:
            gmap[path] = gene_names_of(path)
        gn = gmap[path]
        with h5py.File(path) as f:
            expr = sparse_row(f["X"], row, len(gn))
            ct = load_cat(f["obs"], "cell_type")[row]
        tk = tokenize(expr, gn, vocab, pad_id)
        if tk is None:
            continue
        gi, gv, mask, n = tk
        with torch.no_grad():
            layer_out.clear()
            model._encode(src=torch.tensor(gi)[None].to(dev), values=torch.tensor(gv)[None].to(dev),
                          src_key_padding_mask=torch.tensor(mask)[None].to(dev))
            h = {L: layer_out[L][0, :n].float().cpu() for L in range(N_LAYERS)}   # (n, 512) per layer
        blk = {name: torch.stack([h[L] for L in lyrs]).mean(0).mean(0) for name, lyrs in K.SCGPT_BLOCKS.items()}
        raw11 = h[11].mean(0)
        lin = SB.sae_features(sae, (h[11] - lin_mean).numpy(), device="cpu", activation_mean=np.zeros(D_MODEL, np.float32)).mean(0)
        with torch.no_grad():
            bil = bils[11][0].encode(h[11] - bils[11][1]).mean(0).numpy()
            bd = {L: bils[L][0].encode(h[L] - bils[L][1]).mean(0).numpy() for L in BIL_LAYERS}
        rec["lin"].append(lin); rec["bil"].append(bil); rec["raw11"].append(raw11.numpy())
        rec["be"].append(blk["early"].numpy()); rec["bm"].append(blk["mid"].numpy()); rec["bl"].append(blk["late"].numpy())
        rec["bd_e"].append(bd[2]); rec["bd_m"].append(0.5 * (bd[5] + bd[8])); rec["bd_l"].append(bd[11])
        rec["cell_type"].append(str(ct)); rec["tissue"].append(tis)
        kept += 1
        if kept % 100 == 0:
            print(f"  {kept}/{len(cells)} cells | {kept/(__import__('time').time()-t0):.1f} c/s", flush=True)
    for hk in hooks:
        hk.remove()
    outp = os.path.join(K.DATA, f"external_{panel}.npz")
    np.savez(outp,
             lin_feat=np.array(rec["lin"], np.float32), bil_feat=np.array(rec["bil"], np.float32),
             raw11=np.array(rec["raw11"], np.float32),
             blk_early=np.array(rec["be"], np.float32), blk_mid=np.array(rec["bm"], np.float32),
             blk_late=np.array(rec["bl"], np.float32),
             bd_early=np.array(rec["bd_e"], np.float32), bd_mid=np.array(rec["bd_m"], np.float32),
             bd_late=np.array(rec["bd_l"], np.float32),
             cell_type=np.array(rec["cell_type"]), tissue=np.array(rec["tissue"]))
    print(f"saved {outp}  kept={kept}  tissues={dict(zip(*np.unique(rec['tissue'], return_counts=True)))}")


if __name__ == "__main__":
    panel = sys.argv[1] if len(sys.argv) > 1 else "external_ts"
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else (700 if panel == "external_ts" else 1500)
    main(panel, n_per)
