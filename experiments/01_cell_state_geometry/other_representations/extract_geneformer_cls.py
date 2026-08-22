"""Extract Geneformer's [CLS] cell token on Setty, at BOTH the cross-model tap layer and the final layer.

WHY. See extract_scgpt_cls.py. route_branchpoint/extract_geneformer.py already places [CLS] at
position 0 (tokenize -> [2, ...ranked genes..., 3]) but mean-pools over gene positions and throws the
[CLS] away. This saves it.

Geneformer V2-316M: 18 layers (0..17), hidden 1152. Two readouts per cell:
  cls        : [CLS] at layer 11 -- matched depth to the cross-model `emb` convention
  cls_final  : [CLS] at layer 17 -- the final layer, Geneformer V2's shipped cell embedding
  emb        : mean-pool over gene positions at layer 11 (excludes [CLS]/[SEP]) -- cross-model default
All three come from ONE forward pass.

Preprocessing identical to route_branchpoint/extract_geneformer.py: log1p(counts/rowsum*1e4),
rank by (norm / gene_median) descending, [CLS]..genes..[SEP].

Out: data/celltoken/geneformer_setty.npz
Run: ~/anaconda3/envs/bio_mech_interp/bin/python extract_geneformer_cls.py [n_cells]
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

TOK_DIR = f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data"
ROOT = f"{_DATA}"
SETTY = os.environ.get("BP_H5AD", os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad"))
OUT = os.environ.get("BP_OUT", os.path.join(ROOT, "data/celltoken/geneformer_setty.npz"))
MODEL_NAME, MODEL_SUBFOLDER = "ctheodoris/Geneformer", "Geneformer-V2-316M"
HIDDEN_DIM, MAX_SEQ_LEN = 1152, 2048
TAP, FINAL = 11, 17
CLS_ID, SEP_ID = 2, 3


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


def tokenize(expr_norm, var_idx, token_ids, medians):
    e = expr_norm[var_idx]
    nz = e > 0
    if nz.sum() == 0:
        return None
    e_nz, t_nz, m_nz = e[nz], token_ids[nz], medians[nz]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.nan_to_num(e_nz / m_nz, nan=0.0, posinf=0.0)
    order = np.argsort(-norm)
    ranked = t_nz[order][:MAX_SEQ_LEN - 2]
    return np.concatenate([[CLS_ID], ranked, [SEP_ID]]).astype(np.int64)


def main(n_cells):
    from transformers import BertForMaskedLM
    dev = torch.device(os.environ.get("GF_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    token_dict = pickle.load(open(os.path.join(TOK_DIR, "token_dictionary_gc104M.pkl"), "rb"))
    median_dict = pickle.load(open(os.path.join(TOK_DIR, "gene_median_dictionary_gc104M.pkl"), "rb"))
    name_id = pickle.load(open(os.path.join(TOK_DIR, "gene_name_id_dict_gc104M.pkl"), "rb"))

    gn, clusters, pt, N, G = load_setty()
    rows = np.arange(min(n_cells, N))
    var_idx, tok_ids, meds = [], [], []
    for i in range(G):
        ens = name_id.get(gn[i])
        if ens and ens in token_dict:
            var_idx.append(i); tok_ids.append(token_dict[ens]); meds.append(median_dict.get(ens, 1.0))
    var_idx, tok_ids, meds = np.array(var_idx), np.array(tok_ids), np.array(meds)
    print(f"Geneformer CLS | M={len(rows)} | {len(var_idx)}/{G} genes mapped | dev={dev}", flush=True)

    model = BertForMaskedLM.from_pretrained(MODEL_NAME, subfolder=MODEL_SUBFOLDER,
                                            output_hidden_states=False, output_attentions=False,
                                            attn_implementation=os.environ.get("GF_ATTN", "sdpa")).to(dev).eval()
    cap = {}
    model.bert.encoder.layer[TAP].register_forward_hook(lambda m, i, o: cap.__setitem__("tap", o[0].detach()))
    model.bert.encoder.layer[FINAL].register_forward_hook(lambda m, i, o: cap.__setitem__("fin", o[0].detach()))

    cls_t = np.zeros((len(rows), HIDDEN_DIM), np.float32)
    cls_f = np.zeros((len(rows), HIDDEN_DIM), np.float32)
    emb_t = np.zeros((len(rows), HIDDEN_DIM), np.float32)
    keep = np.zeros(len(rows), bool)

    # Checkpoint/resume. This Mac has hard-reset under parallel MPS load before, and a full pass is
    # ~1h at the contended rate, so never hold more than CKPT cells of unsaved work.
    CKPT = int(os.environ.get("CKPT", "250"))
    row_to_local = {int(r): j for j, r in enumerate(rows)}
    if os.path.exists(OUT):
        try:
            z = np.load(OUT, allow_pickle=True)
            if {"cls", "cls_final", "emb"} <= set(z.files) and z["cls"].shape[1] == HIDDEN_DIM:
                for a, b, c, ci in zip(z["cls"], z["cls_final"], z["emb"], z["cell_idx"]):
                    loc = row_to_local.get(int(ci))
                    if loc is not None:
                        cls_t[loc], cls_f[loc], emb_t[loc], keep[loc] = a, b, c, True
                print(f"[resume] pre-filled {int(keep.sum())} cells from {OUT}", flush=True)
        except Exception as ex:
            print(f"[resume] ignored existing ({ex})", flush=True)

    def flush():
        if keep.sum() == 0:
            return
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        tmp = OUT + ".tmp"
        np.savez(tmp, cls=cls_t[keep], cls_final=cls_f[keep], emb=emb_t[keep],
                 pseudotime=pt[rows][keep], clusters=clusters[rows][keep].astype(str),
                 cell_idx=rows[keep])
        os.replace(tmp + ".npz", OUT)

    t0 = time.time()
    with h5py.File(SETTY, "r") as f:
        fX = f["X"]
        for j, r in enumerate(rows):
            if keep[j]:
                continue
            raw = sparse_row(fX, int(r), G)
            s = raw.sum()
            if s <= 0:
                continue
            en = np.log1p(raw / s * 1e4)
            tk = tokenize(en, var_idx, tok_ids, meds)
            if tk is None:
                continue
            n = len(tk)
            with torch.no_grad():
                cap.clear()
                model(input_ids=torch.tensor(tk)[None].to(dev),
                      attention_mask=torch.ones(1, n, dtype=torch.long, device=dev))
                ht = cap["tap"][0].float(); hf = cap["fin"][0].float()
                cls_t[j] = ht[0].cpu().numpy()
                cls_f[j] = hf[0].cpu().numpy()
                emb_t[j] = ht[1:n - 1].mean(0).cpu().numpy()     # genes only
            keep[j] = True
            if (j + 1) % 200 == 0:
                print(f"  {j+1}/{len(rows)} | {(j+1)/(time.time()-t0):.2f} c/s", flush=True)
            if (j + 1) % CKPT == 0:
                flush()
                print(f"  [ckpt] {int(keep.sum())} cells written", flush=True)
    flush()
    print(f"saved {OUT} kept={keep.sum()}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
