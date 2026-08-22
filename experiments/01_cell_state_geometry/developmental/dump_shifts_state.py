"""Dump STATE SE-600M in-silico TF-activation embedding shifts (cross-model replication of perturb_invert.py).

STATE-SE consumes RAW COUNTS (+ RDA) with ESM2 protein embeddings as the gene vocabulary, through its own
dataloader. So the in-silico activation of TF g is "set g's count to the cell's max count" -- the counts-space
analogue of scGPT's force-to-top-bin and MaxToki's force-to-rank-1.

All perturbed cells (n_tf x n_src) plus the baselines are packed into ONE AnnData and pushed through STATE's
standard inference path in a single pass -- far cheaper than rebuilding a dataloader per TF.

Only DUMPS embeddings; score_shifts.py does the statistics uniformly for all three models.

NOTE this is STATE-**SE** (the embedding model, scGPT/Geneformer's direct analogue), NOT STATE-**ST** (the
perturbation-TRAINED transition model). SE is the apples-to-apples cross-model replication. ST is the model that
could actually overturn the negative, and it is a different, harder experiment (cell SETS, trained on K562
CRISPRi, so out-of-distribution on Setty CD34+) -- see PERTURB_INVERT_RESULTS.md.

Out: data/branchpoint/shifts_state_setty.npz  {base, pert, tfs, src_rows, tr_rows}
Run (.venv_state):  ../../.venv_state/bin/python dump_shifts_state.py [n_src=16]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "route_state"))
from state_loader import load_state_se, load_protein_embeds  # noqa: E402

PROJ = f"{_DATA}"
SETTY = f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad"
EMB = f"{PROJ}/data/branchpoint/state_setty.npz"
OUT = f"{PROJ}/data/branchpoint/shifts_state_setty.npz"
TF_DB = (f"{_DATA}/biodyn-work/network_inference/data/"
         "dorothea_trrust_union_immune.tsv")
LAYER, SEED = int(os.environ.get("STATE_LAYER", "11")), 0
BATCH = int(os.environ.get("STATE_BATCH", "8"))


def main(n_src=16):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg, _, info = load_state_se(device=dev, dtype=torch.float32)
    pe_dict, _, _ = load_protein_embeds()
    d = info["d_model"]
    print(f"[load] STATE SE d={d} L={info['nlayers']} on {dev}; hooking layer {LAYER}", flush=True)

    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)

    # ---- source cells: SAME construction as perturb_invert.py (held-out progenitors), SEED=0
    z = np.load(EMB, allow_pickle=True)
    y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int)
    ok = np.isfinite(y); y, ci = y[ok], ci[ok]
    yz = (y - y.mean()) / (y.std() + 1e-9)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = pool[rng.choice(len(pool), min(n_src, len(pool)), replace=False)]

    full = ad.read_h5ad(SETTY)
    sub = full[ci[src]].copy()                                   # the n_src progenitors
    genes = np.array(sub.var_names)
    sym_at = {g: i for i, g in enumerate(genes)}
    tfs_db = sorted({l.split("\t")[0] for l in open(TF_DB).read().splitlines()[1:] if l.strip()})
    tfs = [t for t in tfs_db if t in sym_at and t in pe_dict]    # must be in STATE's protein-embed vocab
    print(f"TF universe: {len(tfs_db)} in DB -> {len(tfs)} in STATE's vocab & Setty panel", flush=True)
    print(f"source progenitors: {len(src)} (of {len(pool)} held-out)", flush=True)

    D0 = np.asarray(sub.X.todense() if sp.issparse(sub.X) else sub.X, dtype=np.float32)   # (n_src, G)
    rowmax = D0.max(1)                                           # per-cell max count = the "forced high" level

    cap = {}
    model.transformer_encoder.layers[LAYER].register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", o.detach()))
    cfg.model.batch_size = BATCH

    def embed_blocks(mats):
        """mats: list of (n_src, G) dense count matrices -> (len(mats), n_src, d) embeddings, one pass."""
        big = ad.AnnData(X=sp.vstack([sp.csr_matrix(M) for M in mats], format="csr"), var=sub.var.copy(),
                         obs=sub.obs.iloc[np.tile(np.arange(len(src)), len(mats))].reset_index(drop=True))
        big.obs_names = [str(i) for i in range(big.n_obs)]
        big = inferer._convert_to_csr(big)
        dl = create_dataloader(cfg, adata=big, adata_name="setty_pert", shape_dict=None,
                              data_dir=str(Path(SETTY).parent), shuffle=False, protein_embeds=pe_dict,
                              precision=None, gene_column=inferer._auto_detect_gene_column(big))
        E = np.zeros((big.n_obs, d), np.float32)
        with torch.no_grad():
            for batch in dl:
                model._compute_embedding_for_batch(batch)
                res = cap["x"].float().cpu().numpy()
                counts = batch[7].cpu().numpy() if batch[7] is not None else None
                idxs = batch[3].cpu().numpy()
                T = batch[0].shape[1]
                for i in range(len(idxs)):
                    valid = np.zeros(T, bool); valid[1:T] = True     # skip CLS
                    if counts is not None:
                        valid &= (counts[i, :T] > 0)
                    pos = np.nonzero(valid)[0]
                    if len(pos):
                        E[int(idxs[i])] = res[i, pos].mean(0)
        return E.reshape(len(mats), len(src), d)

    base = embed_blocks([D0])[0]
    zc = z["emb"][ok][src]
    print(f"[sanity] re-embed vs cached state_setty: r={np.corrcoef(base.ravel(), zc.ravel())[0,1]:.4f}",
          flush=True)

    # ---- CHECKPOINTED, in TF chunks. (A MaxToki run was killed at 92% by a background lifetime cap and lost
    # an hour of GPU; never run a long screen without resumable checkpoints.)
    CHUNK = 40
    pert = np.zeros((len(tfs), len(src), d), np.float32)
    start = 0
    if os.path.exists(OUT):
        try:
            ck = np.load(OUT, allow_pickle=True)
            if "progress" in ck.files and ck["pert"].shape == pert.shape and list(ck["tfs"]) == tfs:
                pert = ck["pert"].copy(); start = int(ck["progress"])
                print(f"[resume] continuing from TF {start}/{len(tfs)}", flush=True)
        except Exception as e:
            print(f"[resume] ignored existing ({e})", flush=True)

    def save(prog):
        np.savez(OUT + ".tmp", base=base, pert=pert, tfs=np.array(tfs), src_rows=src, tr_rows=tr,
                 progress=np.array(prog))
        os.replace(OUT + ".tmp.npz", OUT)

    t0 = time.time()
    for a in range(start, len(tfs), CHUNK):
        b = min(a + CHUNK, len(tfs))
        mats = []
        for tf in tfs[a:b]:
            Dk = D0.copy(); Dk[:, sym_at[tf]] = rowmax
            mats.append(Dk)
        pert[a:b] = embed_blocks(mats)
        save(b)
        el = time.time() - t0
        print(f"  {b}/{len(tfs)} TFs | {el/60:.1f} min | eta {(len(tfs)-b)*el/(b-start)/60:.1f} min", flush=True)

    save(len(tfs))
    print(f"saved {OUT} | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
