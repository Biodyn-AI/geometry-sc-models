"""Re-extract scGPT Setty cell embeddings under scGPT's OWN pretraining input convention (51-bin values).

WHY. `route_branchpoint/extract_scgpt.py` fed RAW COUNTS into scGPT's value encoder (inherited from bio-sae's
L11 atlas convention). But the whole-human checkpoint's args.json says `input_style: "binned"`, `n_bins: 51`
-- the model was pretrained on quantile-BINNED values. Feeding raw counts puts the cell embedding
off-distribution. That is fine for every result so far (they are relative comparisons within one convention),
but it BREAKS scGPT's pretrained MVC head: native_decode.py stage 1 found the head reads raw-count embeddings
only weakly (within-cell Spearman +0.09, marker r +0.10), so it cannot be used as a circularity-free readout.

This extractor changes exactly ONE thing -- values are quantile-binned to 1..50 before the value encoder --
so that the cell embedding lands in the distribution the MVC head was trained against. Everything else (layer
11, mean-pool over non-pad gene tokens, MAX_SEQ=1200, descending-expression sort) is identical to
extract_scgpt.py, so the two are directly comparable.

Note: scGPT's preprocessing is CP10k -> log1p -> quantile-bin. Quantile binning is invariant to both (a per-cell
scalar multiply and a monotone map preserve within-cell quantiles), so binning the raw counts directly is
equivalent and we skip the normalization.

Binning logic: route_a/scgpt/scgpt_io.py:86 bin_row (verbatim from external/scGPT preprocess.py).

Out: data/branchpoint/scgptbin_setty.npz   (emb, pseudotime, clusters, cell_idx -- same schema as scgpt_setty.npz)
Run: ../../.venv/bin/python extract_scgpt_binned.py [n_cells]
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

MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_REPO = os.path.join(MI, "external", "scGPT")
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
ROOT = f"{_DATA}"
# BP_H5AD / BP_OUT override the dataset (same convention as route_branchpoint/extract_scgpt.py), so the same
# corrected extractor serves setty / pancreas / lung / gut.
SETTY = os.environ.get("BP_H5AD", os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data/branchpoint/scgptbin_setty.npz"))

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11
N_BINS = 51            # args.json: n_bins=51, input_style="binned"
PAD_VALUE = -2


def load_setty():
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in f["obs"]["__categories"]["clusters"][:]])
        clusters = cats[f["obs"]["clusters"][:]]
        pt = f["obs"]["palantir_pseudotime"][:].astype(np.float64)
        shape = f["X"].attrs["shape"]
    return gn, clusters, pt, int(shape[0]), int(shape[1])


def sparse_row(fX, i, ncols):
    ip = fX["indptr"]; a, b = int(ip[i]), int(ip[i + 1])
    row = np.zeros(ncols, np.float32)
    row[fX["indices"][a:b]] = fX["data"][a:b]
    return row


def _digitize(x, bins, rng):
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    return np.ceil(rng.random(len(x)) * (right - left) + left).astype(np.int64)


def bin_row(expr_nonzero, n_bins, rng):
    """scGPT quantile binning -> 1..n_bins-1 (0 reserved for zeros). scgpt_io.py:86."""
    bins = np.quantile(expr_nonzero, np.linspace(0, 1, n_bins - 1))
    return _digitize(expr_nonzero, bins, rng)


def tokenize(expr, gene_names, vocab, pad_id, rng):
    """Identical to extract_scgpt.py::tokenize EXCEPT values are 51-bin quantile-binned (the pretraining regime)."""
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
    val = bin_row(val, N_BINS, rng).astype(np.float32)          # <-- the ONLY change vs extract_scgpt.py
    n = len(ids)
    gi = np.pad(ids, (0, MAX_SEQ - n), constant_values=pad_id)
    gv = np.pad(val, (0, MAX_SEQ - n), constant_values=PAD_VALUE)
    mask = np.zeros(MAX_SEQ, bool); mask[n:] = True
    return gi, gv, mask, n


def build_model(vocab):
    sys.path.insert(0, SCGPT_REPO)
    import scgpt  # noqa
    from scgpt.model.model import TransformerModel
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL, nlayers=N_LAYERS,
                         vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=PAD_VALUE,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False, do_dab=False,
                         use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main(n_cells):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    vocab = json.load(open(SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    gn, clusters, pt, N, G = load_setty()
    N = min(n_cells, N)
    rng = np.random.default_rng(0)
    print(f"scGPT BINNED (n_bins={N_BINS}) Setty extraction | N={N} | dev={dev}", flush=True)
    model = build_model(vocab).to(dev)

    layer_out = {}
    hk = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda mod, inp, out: layer_out.__setitem__("h", out.detach()))

    embs = np.zeros((N, D_MODEL), np.float32)
    keep = np.zeros(N, bool)
    t0 = time.time()
    with h5py.File(SETTY, "r") as f:
        fX = f["X"]
        for i in range(N):
            tk = tokenize(sparse_row(fX, i, G), gn, vocab, pad_id, rng)
            if tk is None:
                continue
            gi, gv, mask, n = tk
            with torch.no_grad():
                layer_out.clear()
                model._encode(src=torch.tensor(gi)[None].to(dev),
                              values=torch.tensor(gv)[None].to(dev),
                              src_key_padding_mask=torch.tensor(mask)[None].to(dev))
                embs[i] = layer_out["h"][0, :n].float().mean(0).cpu().numpy()
            keep[i] = True
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N} | {(i+1)/(time.time()-t0):.1f} c/s", flush=True)
    hk.remove()
    embs, pt2, clu2 = embs[keep], pt[:N][keep], clusters[:N][keep]
    np.savez(OUT, emb=embs, pseudotime=pt2, clusters=clu2.astype(str), cell_idx=np.where(keep)[0])
    print(f"saved {OUT} | kept={keep.sum()}/{N} | emb norm mean={np.linalg.norm(embs,axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
