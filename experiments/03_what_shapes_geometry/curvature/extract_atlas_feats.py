"""Route Q — per-cell LINEAR-ATLAS feature vectors, computed the domain-correct way (prereg §5.5).

Why this file exists. The linear TopK SAE atlas was trained on per-GENE-TOKEN L11 activations
(reconstruction VE 0.953). The branch-point representation x is the MEAN-POOLED per-cell L11 residual,
on which the atlas reconstructs at only VE 0.508 (Setty). Feeding the cell embedding straight into the
atlas encoder therefore cripples it -- and a crippled atlas would *understate* R^2_SAE and so *inflate*
our novelty claim. That bias points toward our own headline, so we must not accept it.

The fix: encode each cell's gene tokens with the atlas (in-distribution, where VE=0.95), then mean-pool
the FEATURE activations over the cell's tokens:

        h_bar(cell) = mean_t  TopK(W_enc (x_t - mu))          <- in-distribution
   vs.  h_hat(cell) =         TopK(W_enc (x_bar - mu))        <- out-of-distribution (the naive way)

h_bar is "what the linear atlas says about this cell", and is the fair baseline for asking whether the
atlas already spans Q's interaction directions.

Free correctness proof: the mean-pooled token activations must reproduce the cached `emb` bit-for-bit.

Run (bio_mech_interp env -- .venv lacks anndata/scgpt):
  ~/anaconda3/envs/bio_mech_interp/bin/python extract_atlas_feats.py
Out: results/atlas_feats_scgpt_setty.npz  (h_bar, h_hat, emb_check, cell_idx)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_REPO = os.path.join(MI, "external", "scGPT")
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
SAE_DIR = (f"{_DATA}/biodyn-work/"
           "subproject_42_sparse_autoencoder_biological_map/experiments/scgpt_atlas/"
           "sae_models/layer11_x4_k32")
BIO_SAE_SRC = f"{_DATA}/biomi_automation/repos/bio-sae/src"
SETTY = f"{_DATA}/hematopoiesis/setty19_cd34_bm.h5ad"
CACHED = f"{_DATA}/branchpoint/scgpt_setty.npz"
OUT = os.path.join(RES, "atlas_feats_scgpt_setty.npz")

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11


def load_setty():
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        shape = f["X"].attrs["shape"]
    return gn, int(shape[0]), int(shape[1])


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


def build_model(vocab):
    sys.path.insert(0, SCGPT_REPO)
    from scgpt.model.model import TransformerModel
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL,
                         nlayers=N_LAYERS, vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=-2,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False,
                         do_dab=False, use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main():
    sys.path.insert(0, BIO_SAE_SRC)
    from sae_model import TopKSAE

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    vocab = json.load(open(SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    gn, N, G = load_setty()

    cached = np.load(CACHED)
    keep_idx = cached["cell_idx"]
    emb_cached = cached["emb"]

    sae = TopKSAE.load(os.path.join(SAE_DIR, "sae_final.pt"), device=dev).eval()
    mu = torch.from_numpy(np.load(os.path.join(SAE_DIR, "activation_mean.npy")).astype(np.float32)).to(dev)
    n_feat = sae.n_features
    print(f"scGPT+atlas | N={N} cells | SAE n_feat={n_feat} k={sae.k} | dev={dev}", flush=True)

    model = build_model(vocab).to(dev)
    model.transformer_encoder.enable_nested_tensor = False
    layer_out = {}
    hk = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda mod, inp, out: layer_out.__setitem__("h", out.detach()))

    M = len(keep_idx)
    h_bar = np.zeros((M, n_feat), np.float32)
    h_hat = np.zeros((M, n_feat), np.float32)
    emb_check = np.zeros((M, D_MODEL), np.float32)

    t0 = time.time()
    with h5py.File(SETTY, "r") as f:
        fX = f["X"]
        for j, i in enumerate(keep_idx):
            expr = sparse_row(fX, int(i), G)
            tk = tokenize(expr, gn, vocab, pad_id)
            gi, gv, mask, n = tk
            with torch.no_grad():
                layer_out.clear()
                model._encode(src=torch.tensor(gi)[None].to(dev),
                              values=torch.tensor(gv)[None].to(dev),
                              src_key_padding_mask=torch.tensor(mask)[None].to(dev))
                toks = layer_out["h"][0, :n].float()             # (n_tok, 512) in-distribution
                emb = toks.mean(0)
                emb_check[j] = emb.cpu().numpy()
                hb, _ = sae.encode(toks - mu)                    # per-token atlas features
                h_bar[j] = hb.float().mean(0).cpu().numpy()      # <- domain-correct pooling
                hh, _ = sae.encode((emb - mu)[None])             # naive: encode the cell embedding
                h_hat[j] = hh[0].float().cpu().numpy()
            if (j + 1) % 250 == 0:
                print(f"  {j+1}/{M} | {(j+1)/(time.time()-t0):.1f} c/s", flush=True)
    hk.remove()

    r = np.corrcoef(emb_check.ravel(), emb_cached.ravel())[0, 1]
    relerr = np.abs(emb_check - emb_cached).max() / (np.abs(emb_cached).max() + 1e-12)
    print(f"[correctness] mean-pooled acts vs cached emb: r={r:.6f}  max relerr={relerr:.2e}", flush=True)

    nz_bar = (h_bar > 0).sum(1).mean()
    nz_hat = (h_hat > 0).sum(1).mean()
    print(f"[feats] mean nonzero: h_bar={nz_bar:.1f}/{n_feat}  h_hat={nz_hat:.1f}/{n_feat}", flush=True)

    np.savez(OUT, h_bar=h_bar, h_hat=h_hat, emb_check=emb_check, cell_idx=keep_idx,
             corr_emb=r, relerr_emb=relerr)
    print(f"saved {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
