"""Dump MaxToki in-silico TF-activation embedding shifts (cross-model replication of perturb_invert.py).

MaxToki is RANK-encoded (Geneformer-style): a cell is the list of its genes sorted by
normalized-expression / gene-median, descending -- there are no expression VALUES in the input. So the
in-silico activation of TF g is not "set its value high" (scGPT) but **move g to RANK 1**, inserting it at the
front if it is not expressed. That is exactly Theodoris et al.'s in-silico activation, and it is the faithful
analogue of what perturb_invert.py does to scGPT.

Only DUMPS embeddings; all statistics are done uniformly across models by score_shifts.py (which runs in the
biotensor .venv). This keeps the cross-model comparison honest -- one scorer, three models.

Note: the INDIRECT score (pool excluding the perturbed token) is not computed here. In scGPT it was numerically
indistinguishable from the direct score (0.707/0.707, 0.701/0.694, ...) because the perturbed token is ~1/1200
of the mean-pool; the same 1/2048 argument holds here. See PERTURB_INVERT_RESULTS.md.

Out: data/branchpoint/shifts_maxtoki_setty.npz  {base, pert, tfs, src_rows, tr_rows}
Run with the project venv; see docs/REPRODUCING.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, time, pickle
import numpy as np
import h5py
import torch

MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MAXTOKI_SETUP)
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402

PROJ = f"{_DATA}"
SETTY = f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad"
EMB = f"{PROJ}/data/maxtoki_acts/maxtoki_setty.npz"
OUT = f"{PROJ}/data/branchpoint/shifts_maxtoki_setty.npz"
NAME_ID_PKL = (f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/"
               "crispri_validation/data/gene_name_id_dict_gc104M.pkl")
TF_DB = (f"{_DATA}/biodyn-work/network_inference/data/"
         "dorothea_trrust_union_immune.tsv")

LAYER_HS, MAX_LEN, SEED, BATCH = 8, 2048, 0, 8


def load_setty():
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        G = int(f["X"].attrs["shape"][1])
    return gn, G


def sparse_rows(rows, G):
    out = np.zeros((len(rows), G), np.float32)
    with h5py.File(SETTY, "r") as f:
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        for r, i in enumerate(rows):
            a, b = int(ip[i]), int(ip[i + 1])
            out[r, ind[a:b]] = dat[a:b]
    return out


def main(n_src=16):
    genes, G = load_setty()
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))
    var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in genes])
    # symbol -> position within the kept (mapped) gene set
    sym_at = {genes[v]: k for k, v in enumerate(var_idx)}

    tfs_db = sorted({l.split("\t")[0] for l in open(TF_DB).read().splitlines()[1:] if l.strip()})
    tfs = [t for t in tfs_db if t in sym_at]
    print(f"TF universe: {len(tfs_db)} in DB -> {len(tfs)} mapped into MaxToki's vocab", flush=True)

    # ---- source cells: SAME construction as perturb_invert.py (held-out progenitors), SEED=0
    z = np.load(EMB, allow_pickle=True)
    y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int)
    ok = np.isfinite(y)
    y, ci = y[ok], ci[ok]
    yz = (y - y.mean()) / (y.std() + 1e-9)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = pool[rng.choice(len(pool), min(n_src, len(pool)), replace=False)]
    print(f"source progenitors: {len(src)} (of {len(pool)} held-out)", flush=True)

    xt = MaxTokiAttentionExtractor(dtype=torch.float32)
    print(f"[load] MaxToki d={xt.hidden_size} L={xt.n_layers} on {xt.device}; tap hs[{LAYER_HS}]", flush=True)

    # ---- baseline rank sequences for the source cells
    expr = sparse_rows(ci[src], G)
    base_seq = []
    for i in range(len(src)):
        rs = expr[i].sum() or 1.0
        en = np.log1p(expr[i] / rs * 1e4)[var_idx]                 # normalized, kept-gene order
        nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        order = np.argsort(-norm[nz])
        ranked = np.nonzero(nz)[0][order][: MAX_LEN - 2]           # positions in the kept-gene set, rank order
        base_seq.append(ranked)

    def embed(seqs):
        """seqs: list of gene-position arrays (rank order) -> (B, d) mean-pooled hs[LAYER_HS] over gene tokens."""
        out = np.zeros((len(seqs), xt.hidden_size), np.float32)
        for a in range(0, len(seqs), BATCH):
            ch = seqs[a:a + BATCH]
            L = max(len(s) for s in ch) + 2
            ids = np.full((len(ch), L), tok.EOS, np.int64)         # pad with EOS; masked out below
            am = np.zeros((len(ch), L), np.int64)
            for j, s in enumerate(ch):
                seq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]])
                ids[j, :len(seq)] = seq; am[j, :len(seq)] = 1
            with torch.no_grad():
                _, hid = xt.forward_with_hidden_states(torch.from_numpy(ids), torch.from_numpy(am))
            h = hid[LAYER_HS].float().cpu().numpy()                # (B, L, d)
            for j, s in enumerate(ch):
                out[a + j] = h[j, 1:1 + len(s)].mean(0)            # gene tokens only (drop BOS/EOS)
        return out

    base = embed(base_seq)
    zc = z["emb"][ok][src]
    print(f"[sanity] re-embed vs cached maxtoki_setty: r={np.corrcoef(base.ravel(), zc.ravel())[0,1]:.4f}",
          flush=True)

    # ---- CHECKPOINTED screen. A first attempt was killed at 550/601 TFs by a background-task lifetime cap
    # and lost an hour of GPU. Save every 25 TFs and resume from the last checkpoint.
    pert = np.zeros((len(tfs), len(src), xt.hidden_size), np.float32)
    start = 0
    if os.path.exists(OUT):
        try:
            ck = np.load(OUT, allow_pickle=True)
            if "progress" in ck.files and ck["pert"].shape == pert.shape and list(ck["tfs"]) == tfs:
                pert = ck["pert"].copy(); start = int(ck["progress"])
                print(f"[resume] continuing from TF {start}/{len(tfs)}", flush=True)
        except Exception as e:
            print(f"[resume] ignored existing file ({e})", flush=True)

    def save(prog):
        tmp = OUT + ".tmp"
        np.savez(tmp, base=base, pert=pert, tfs=np.array(tfs), src_rows=src, tr_rows=tr,
                 progress=np.array(prog))
        os.replace(tmp + ".npz", OUT)

    t0 = time.time()
    for k in range(start, len(tfs)):
        p = sym_at[tfs[k]]
        seqs = []
        for s in base_seq:
            rest = s[s != p]                                       # drop it if already ranked
            seqs.append(np.concatenate([[p], rest])[: MAX_LEN - 2])  # ACTIVATION: force to rank 1
        pert[k] = embed(seqs)
        if (k + 1) % 25 == 0:
            save(k + 1)
            el = time.time() - t0
            print(f"  {k+1}/{len(tfs)} TFs | {el/60:.1f} min | "
                  f"eta {(len(tfs)-k-1)*el/(k+1-start)/60:.1f} min", flush=True)

    save(len(tfs))
    print(f"saved {OUT} | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
