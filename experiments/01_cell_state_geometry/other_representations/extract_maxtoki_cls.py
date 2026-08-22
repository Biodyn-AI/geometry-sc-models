"""Extract MaxToki-217M's cell-summary token on Setty, at the cross-model tap depth and the final layer.

WHY. See extract_scgpt_cls.py. route_maxtoki/extract_maxtoki.py masks out <bos>/<eos>
(`gene_mask = cell.gene_positions >= 0`) and mean-pools the gene tokens. This saves the cell token.

CAVEAT -- MaxToki has NO designed cell embedding. It is a causal LlamaForCausalLM: unlike scGPT's
<cls>, Geneformer's [CLS], STATE's CLS or UCE's decoder output, nothing in MaxToki was trained to be
a cell representation. Under causal attention the LAST token (<eos>) is the only position that has
seen the whole cell, so it is the closest available analog -- but it is an analog, not a shipped
embedding, and should be read as the weakest member of the comparison.

Three readouts per cell from ONE forward pass (tokenization: [<bos>, ranked genes..., <eos>]):
  cls        : <eos> at hidden_states[8] -- matched depth to the cross-model `emb` convention
  cls_final  : <eos> at the last hidden state
  emb        : mean-pool over gene positions at hidden_states[8] -- cross-model default

Out: data/celltoken/maxtoki_setty.npz
Run: ../../../maxtoki/.venv/bin/python extract_maxtoki_cls.py [n_cells]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch

MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MAXTOKI_SETUP)
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402

PROJ = f"{_DATA}"
SETTY = os.environ.get("BP_H5AD", f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad")
OUT = os.environ.get("BP_OUT", f"{PROJ}/data/celltoken/maxtoki_setty.npz")
BIOM = f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data"
NAME_ID_PKL = f"{BIOM}/gene_name_id_dict_gc104M.pkl"

LAYER_HS = 8
MAX_LEN = 2048


def load_setty():
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        cats = np.array([x.decode() if isinstance(x, bytes) else x
                         for x in f["obs"]["__categories"]["clusters"][:]])
        clusters = cats[f["obs"]["clusters"][:]]
        pt = f["obs"]["palantir_pseudotime"][:].astype(np.float64)
        shape = f["X"].attrs["shape"]
    return gn, clusters, pt, int(shape[0]), int(shape[1])


def sparse_row(fX, i, ncols):
    ip = fX["indptr"]; a, b = int(ip[i]), int(ip[i + 1])
    row = np.zeros(ncols, np.float32)
    row[fX["indices"][a:b]] = fX["data"][a:b]
    return row


def main(n_cells):
    dev = os.environ.get("MT_DEVICE", "cpu")
    xt = MaxTokiAttentionExtractor(device=dev)
    gn, clusters, pt, N, G = load_setty()
    N = min(n_cells, N)
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))
    var_ensembl = [name_id.get(s) for s in gn]
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    var_idx, token_ids, medians = tok.make_var_mapping(var_ensembl)
    n_hs = xt.n_layers + 1
    print(f"MaxToki CLS | N={N} | {len(var_idx)}/{G} genes mapped | hs[{LAYER_HS}] + hs[{n_hs-1}] "
          f"| d={xt.hidden_size} | dev={dev}", flush=True)

    D = xt.hidden_size
    cls_t = np.zeros((N, D), np.float32)
    cls_f = np.zeros((N, D), np.float32)
    emb_t = np.zeros((N, D), np.float32)
    keep = np.zeros(N, bool)
    t0 = time.time()
    with h5py.File(SETTY, "r") as f:
        fX = f["X"]
        for i in range(N):
            raw = sparse_row(fX, i, G)
            s = raw.sum()
            if s <= 0:
                continue
            en = np.log1p(raw / s * 1e4)
            cell = tok.tokenize_cell(en, var_idx, token_ids, medians, max_len=MAX_LEN)
            if cell is None or cell.n_genes_in_cell == 0:
                continue
            with torch.no_grad():
                _, hidden = xt.forward_with_hidden_states(torch.from_numpy(cell.token_ids[None, :]))
            ht = hidden[LAYER_HS][0].float()
            hf = hidden[n_hs - 1][0].float()
            gmask = cell.gene_positions >= 0
            cls_t[i] = ht[-1].numpy()                     # <eos> at tap
            cls_f[i] = hf[-1].numpy()                     # <eos> at final
            emb_t[i] = ht[gmask].mean(0).numpy()          # genes at tap
            keep[i] = True
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{N} | {(i+1)/(time.time()-t0):.2f} c/s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, cls=cls_t[keep], cls_final=cls_f[keep], emb=emb_t[keep],
             pseudotime=pt[:N][keep], clusters=clusters[:N][keep].astype(str),
             cell_idx=np.where(keep)[0])
    print(f"saved {OUT} kept={keep.sum()}/{N} "
          f"cls={np.linalg.norm(cls_t[keep],axis=1).mean():.2f} "
          f"emb={np.linalg.norm(emb_t[keep],axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
