"""Extract scGPT's NATIVE CELL TOKEN (<cls>) on Setty, under scGPT's own binned input convention.

WHY. route_uce's cell-token experiments compared UCE's shipped cell embedding (X_uce) only against
UCE's OWN gene-mean-pool -- no other model's cell token was ever extracted (their npz files carry
`emb` only). So "UCE's embedding is a dramatically lower-dimensional manifold" is a within-UCE
statement, not a cross-model one. This route extracts the analogous native cell token from the other
four models so the comparison can actually be made.

scGPT's cell embedding is the <cls> token (vocab id 60695) prepended to the gene sequence with
value 0, read out at the FINAL transformer layer (L11 of 12) -- cf. scGPT's
cell_emb_style="cls" -> _get_cell_emb_from_layer -> layer_output[:, 0, :].
route_branchpoint/extract_scgpt.py never prepends <cls> at all; it hooks L11 and mean-pools genes.

INPUT CONVENTION. Values are 51-bin quantile-binned (args.json: input_style="binned", n_bins=51),
i.e. this follows route_steering/extract_scgpt_binned.py, NOT the raw-counts extractor. The
raw-counts cache (data/branchpoint/scgpt_setty.npz) is off-distribution: its emb collapses onto one
axis (PC1=81%, PR=1.5), which is a bug artifact, not scGPT's geometry. Binned gives PR=9.2, PC1=22%.
Binning logic verbatim from route_steering/extract_scgpt_binned.py::bin_row.

Saves BOTH readouts from ONE forward pass, so they are directly comparable:
  cls (N,512) : <cls> token at L11  -- scGPT's native cell embedding
  emb (N,512) : mean-pool over gene tokens at L11 (excludes <cls>) -- the cross-model default
NOTE emb here will differ slightly from data/branchpoint/scgptbin_setty.npz's emb, because the
presence of <cls> in the sequence changes attention. That is intended: both readouts come from the
same pass, so the cls-vs-meanpool contrast is clean.

Out: data/celltoken/scgptbin_setty.npz
Run: ../../.venv/bin/python extract_scgpt_cls.py [n_cells]
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
SETTY = os.environ.get("BP_H5AD", os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data/celltoken/scgptbin_setty.npz"))

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11
N_BINS = 51
PAD_VALUE = -2
CLS_VALUE = 0.0          # scGPT tokenize_and_pad_batch(append_cls=True) prepends <cls> with value 0


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
    bins = np.quantile(expr_nonzero, np.linspace(0, 1, n_bins - 1))
    return _digitize(expr_nonzero, bins, rng)


def tokenize(expr, gene_names, vocab, pad_id, cls_id, rng):
    """51-bin quantile-binned values, descending sort, with <cls> PREPENDED (value 0)."""
    nz = np.where(expr > 0)[0]
    ids, val = [], []
    for idx in nz:
        g = gene_names[idx]
        if g in vocab:
            ids.append(vocab[g]); val.append(expr[idx])
    if not ids:
        return None
    ids = np.array(ids, np.int64); val = np.array(val, np.float32)
    o = np.argsort(-val)[:MAX_SEQ - 1]                     # -1 leaves room for <cls>
    ids, val = ids[o], val[o]
    val = bin_row(val, N_BINS, rng).astype(np.float32)
    ids = np.concatenate([[cls_id], ids])                  # <cls> at position 0
    val = np.concatenate([[CLS_VALUE], val]).astype(np.float32)
    n = len(ids)                                           # includes <cls>
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
                         use_batch_labels=False, cell_emb_style="cls", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main(n_cells):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    vocab = json.load(open(SCGPT_VOCAB))
    pad_id, cls_id = vocab["<pad>"], vocab["<cls>"]
    gn, clusters, pt, N, G = load_setty()
    N = min(n_cells, N)
    rng = np.random.default_rng(0)
    print(f"scGPT CLS extraction (binned) | N={N} | cls_id={cls_id} | dev={dev}", flush=True)
    model = build_model(vocab).to(dev)

    layer_out = {}
    hooks = [model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda mod, inp, out: layer_out.__setitem__("h", out.detach()))]

    cls_e = np.zeros((N, D_MODEL), np.float32)
    emb_e = np.zeros((N, D_MODEL), np.float32)
    keep = np.zeros(N, bool)
    t0 = time.time()
    with h5py.File(SETTY, "r") as f:
        fX = f["X"]
        for i in range(N):
            tk = tokenize(sparse_row(fX, i, G), gn, vocab, pad_id, cls_id, rng)
            if tk is None:
                continue
            gi, gv, mask, n = tk
            with torch.no_grad():
                layer_out.clear()
                model._encode(src=torch.tensor(gi)[None].to(dev),
                              values=torch.tensor(gv)[None].to(dev),
                              src_key_padding_mask=torch.tensor(mask)[None].to(dev))
                h = layer_out["h"][0].float()
                cls_e[i] = h[0].cpu().numpy()               # <cls> = native cell embedding
                emb_e[i] = h[1:n].mean(0).cpu().numpy()     # genes only, excludes <cls>
            keep[i] = True
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{N} | {(i+1)/(time.time()-t0):.1f} c/s", flush=True)
    for hk in hooks:
        hk.remove()
    idx = np.where(keep)[0]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, cls=cls_e[keep], emb=emb_e[keep], pseudotime=pt[:N][keep],
             clusters=clusters[:N][keep].astype(str), cell_idx=idx)
    print(f"saved {OUT} kept={keep.sum()}/{N} "
          f"cls_norm={np.linalg.norm(cls_e[keep],axis=1).mean():.2f} "
          f"emb_norm={np.linalg.norm(emb_e[keep],axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
