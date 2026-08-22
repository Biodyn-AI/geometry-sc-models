"""scGPT L11 cell embeddings for the K562 cell-cycle substrate, under scGPT's OWN binned input convention.

scGPT has never been run on K562 anywhere in this program (CROSS_MODEL_GEOMETRY_RESULTS.md:62 records it as
"- (no cycling data)"), yet it is the ONLY model with a pretrained cell-embedding -> expression head (the MVC
decoder), and therefore the only one that can give a circularity-free gene readout of a steered path. So this
extraction is the single new forward pass the whole route needs.

Convention: 51-bin quantile binning, exactly as route_steering/extract_scgpt_binned.py. That extractor exists
because feeding RAW counts to the value encoder (route_branchpoint/extract_scgpt.py) puts the cell embedding
off-distribution and BREAKS the pretrained MVC head -- native_decode.py's gate went from within-cell Spearman
+0.085 (raw, FAILS) to +0.260 (binned, PASSES). We inherit the corrected convention from the start; cc_decode.py
re-runs the same honesty gate on these embeddings before trusting any readout.

Everything else (layer 11, mean-pool over non-pad gene tokens, MAX_SEQ=1200, descending-expression sort) is
identical to extract_scgpt_binned.py, so these embeddings are directly comparable to the Setty ones.

Out: data/cellcycle/scgptcc_k562.npz  (emb, phi, s_score, g2m_score, cc_phase, cell_idx)
Run: ../../.venv/bin/python extract_scgpt_cc.py [n_cells]
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

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import build_substrate, REPLOGLE, DATA, _load_categorical  # noqa: E402

MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_REPO = os.path.join(MI, "external", "scGPT")
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
OUT = os.path.join(DATA, "scgptcc_k562.npz")

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11
N_BINS, PAD_VALUE = 51, -2


def _digitize(x, bins, rng):
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    return np.ceil(rng.random(len(x)) * (right - left) + left).astype(np.int64)


def bin_row(expr_nonzero, n_bins, rng):
    """scGPT quantile binning -> 1..n_bins-1 (0 reserved for zeros). route_a/scgpt/scgpt_io.py:86."""
    bins = np.quantile(expr_nonzero, np.linspace(0, 1, n_bins - 1))
    return _digitize(expr_nonzero, bins, rng)


def tokenize(expr, gene_names, vocab, pad_id, rng):
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
    val = bin_row(val, N_BINS, rng).astype(np.float32)
    n = len(ids)
    gi = np.pad(ids, (0, MAX_SEQ - n), constant_values=pad_id)
    gv = np.pad(val, (0, MAX_SEQ - n), constant_values=PAD_VALUE)
    mask = np.zeros(MAX_SEQ, bool); mask[n:] = True
    return gi, gv, mask, n


def _import_transformer_model():
    """Load scGPT's TransformerModel WITHOUT executing scgpt/__init__.py.

    The package __init__ imports the gene tokenizer (-> torchtext) and utils (-> anndata); neither is installed
    in either project venv, and neither is needed here. scgpt/model/model.py itself imports only torch, numpy,
    tqdm and its two siblings (.dsbn, .grad_reverse). So we register the two parent packages as namespace stubs
    with a __path__ and let the normal import machinery resolve the relative imports from there. No installs,
    no changes to a venv shared with other sessions.
    """
    import importlib.util, types
    root = os.path.join(SCGPT_REPO, "scgpt")
    for name, p in (("scgpt", root), ("scgpt.model", os.path.join(root, "model"))):
        if name not in sys.modules:
            pkg = types.ModuleType(name); pkg.__path__ = [p]; sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location("scgpt.model.model", os.path.join(root, "model", "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scgpt.model.model"] = mod
    spec.loader.exec_module(mod)
    return mod.TransformerModel


def build_model(vocab):
    TransformerModel = _import_transformer_model()
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL, nlayers=N_LAYERS,
                         vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=PAD_VALUE,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False, do_dab=False,
                         use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main(n_cells=None):
    sub = build_substrate()
    sel = sub["cell_idx"].astype(int)
    if n_cells:
        sel = sel[:n_cells]
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    vocab = json.load(open(SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    rng = np.random.default_rng(0)

    with h5py.File(REPLOGLE, "r") as f:
        genes = _load_categorical(f["var"], "gene_name_index").astype(str)
    in_vocab = sum(g in vocab for g in genes)
    print(f"scGPT BINNED (n_bins={N_BINS}) K562 cell-cycle extraction | N={len(sel)} | dev={dev}")
    print(f"  genes {len(genes)} | in scGPT vocab: {in_vocab}", flush=True)

    model = build_model(vocab).to(dev)
    layer_out = {}
    hk = model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda mod, inp, out: layer_out.__setitem__("h", out.detach()))

    embs = np.zeros((len(sel), D_MODEL), np.float32)
    keep = np.zeros(len(sel), bool)
    t0 = time.time()
    with h5py.File(REPLOGLE, "r") as f:
        X = f["X"]
        for i, r in enumerate(sel):
            tk = tokenize(np.asarray(X[int(r), :], dtype=np.float32), genes, vocab, pad_id, rng)
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
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(sel)} | {(i+1)/(time.time()-t0):.1f} c/s", flush=True)
    hk.remove()

    np.savez(OUT, emb=embs[keep], phi=sub["phi"][:len(sel)][keep],
             s_score=sub["s_score"][:len(sel)][keep], g2m_score=sub["g2m_score"][:len(sel)][keep],
             cc_phase=sub["cc_phase"][:len(sel)][keep].astype(str), cell_idx=sel[keep],
             total_counts=sub["total_counts"][:len(sel)][keep])
    print(f"saved {OUT} | kept={keep.sum()}/{len(sel)} | "
          f"emb norm mean={np.linalg.norm(embs[keep], axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
