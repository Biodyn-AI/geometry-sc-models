"""EXTRACT STATE-SE (Arc SE-600M) per-gene CONTEXTUAL representations, stratified by cell type.

Cross-model sibling of ctx_extract_maxtoki.py / ctx_extract_scgpt.py: produces results/ctx_state_L*.npz in the
IDENTICAL schema so the shared downstream analysis (gene x context interaction, functional-axis projection) runs
unchanged across scGPT / MaxToki / STATE-SE.

THE CRUX -- gene identity per sequence position.
STATE feeds each cell as a "sentence" of gene tokens. The dataloader collator
(state.emb.data.loader.VCIDatasetSentenceCollator) returns a tuple whose element [0] is `batch_sentences`
(B, pad_length=2048):
  * position 0 is the CLS/cell token (overwritten inside _compute_embedding_for_batch) -- we SKIP it,
  * positions 1.. hold GLOBAL protein-embedding indices: the collator ranks the cell's expressed genes by
    expression (sample_cell_sentences), then remaps local -> global via `ds_emb_idxs`, and global indices are
    positions in `GENES = list(protein_embeds.keys())` (the ESM2 protein-embedding gene list, gene SYMBOLS).
  * element [7] is per-position count weight (100*expression_weight); >0 marks a genuinely EXPRESSED gene
    (unexpressed padding genes are sampled with count 0). element [3] is the cell's adata row index.
So the gene at (cell i, position p>=1, count>0) is  GENES[ batch_sentences[i, p] ]  -- a gene symbol, which we
map to Ensembl via the Geneformer name_id dict. The residual stream at that position (output of
transformer_encoder.layers[L], shape (B, 2048, d=2048)) is that gene's contextual representation.

Discipline (mirrors the siblings): contexts = cell types with >= MIN_CELLS; two independent cell PARTITIONS
(split-half); an occurrence CAP per (gene, context, partition) so token counts are balanced (kills the
heteroscedasticity/anisotropy confound); panel = genes reaching FLOOR in >= 2 contexts, capped to MAX_GENES,
mappable to Ensembl.

Two passes: pass 1 collates ONLY (CPU) to pick contexts + gene panel + cell partitions; pass 2 runs the STATE
forward and accumulates per-gene hidden states (panel genes only) under the cap.

CAVEAT (STATE architecture): STATE-SE uses FROZEN ESM2 protein embeddings as the *input* gene tokens (there is
no learned per-gene input embedding). The contextual rep we extract is the transformer's LAYER-L residual for
that gene's position -- it is contextualised by the whole cell, but the gene's *identity signal* enters only
through a fixed protein embedding plus its expression rank. This is exactly why the correctness check below
matters: if the mapping were wrong the reps would be random and the nuclear/surface axis would sit at AUC ~0.5.

Out: results/ctx_state_L{tap:02d}.npz  -- M[part, ctx, gene, 2048] float16, counts[part, ctx, gene] int32,
     genes (Ensembl), contexts (cell-type strings), cap.
Run: ../../.venv_state/bin/python -u ctx_extract_state.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, pickle, collections, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import anndata as ad
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "route_state"))
from state_loader import load_state_se, load_protein_embeds  # noqa: E402

# ---- data (same TS tissues as ts_extract.py) --------------------------------------------------------------
TS_DIR = (f"{_DATA}/biodyn-work/subproject_09_causal_mediation_circuit_map/"
          "implementation/results/preprocessed")
TISSUES = {"immune": "tabula_sapiens_immune_subset_20000_processed.h5ad",
           "kidney": "tabula_sapiens_kidney_processed.h5ad",
           "lung":   "tabula_sapiens_lung_processed.h5ad"}
CACHE = f"{_DATA}/state_geometry_cache"
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
GENE2GO = (f"{_DATA}/perturb/"
           "gene2go_all.pkl")

# ---- settings (match the siblings) ------------------------------------------------------------------------
N_PER      = int(os.environ.get("NPER", 600))       # cells subsampled per tissue
MIN_CELLS  = int(os.environ.get("MINCELLS", 100))   # context = cell type with >= this many cells (pooled)
CAP        = int(os.environ.get("CAP", 20))         # max occurrences per (gene, ctx, partition)
FLOOR      = int(os.environ.get("FLOOR", 12))       # panel: gene must reach this in >= 2 contexts
MAX_GENES  = int(os.environ.get("MAXGENES", 5000))
TAPS       = [int(x) for x in os.environ.get("TAPS", "4,8,11").split(",")]
DEVICE     = os.environ.get("STATE_DEVICE", "mps")
BATCH      = int(os.environ.get("STATE_BATCH", "8"))
NPART, SEED = 2, 0
# subsample: concentrate on abundant types so pooled contexts clear MIN_CELLS
SUB_CAP, SUB_MINTYPE, SUB_TOPTYPES = 150, 40, 10


def subsample(adata, n, seed=SEED):
    """Greedy top-cell-type subsample (mirrors ts_extract.subsample; wider cap so contexts clear MIN_CELLS)."""
    rng = np.random.default_rng(seed)
    ct = adata.obs["cell_type"].astype(str).values
    u, cnt = np.unique(ct, return_counts=True)
    order = np.argsort(-cnt)
    keep_types = [t for t in u[order] if cnt[list(u).index(t)] >= SUB_MINTYPE][:SUB_TOPTYPES]
    picked = []
    for t in keep_types:
        gi = np.where(ct == t)[0]
        picked.append(rng.choice(gi, min(SUB_CAP, len(gi)), replace=False))
        if sum(len(p) for p in picked) >= n:
            break
    sel = np.sort(np.concatenate(picked))[:n]
    return sel


def setup_tissue(tissue, fname, model, cfg, pe_dict):
    """Load + subsample the tissue and build STATE's own dataloader (verbatim ts_extract plumbing).
    Returns (dataloader, cell_type_array_aligned_to_row_idx, n_obs)."""
    from state.emb.inference import Inference
    from state.emb.data import create_dataloader
    a = ad.read_h5ad(os.path.join(TS_DIR, fname))
    sel = subsample(a, N_PER)
    a = a[sel].copy()
    cell_type = a.obs["cell_type"].astype(str).values.copy()
    if "decontXcounts" in a.layers:                 # STATE needs raw counts
        a.X = a.layers["decontXcounts"]
    a.layers.clear()
    inferer = Inference(cfg=cfg)
    inferer.init_from_model(model, protein_embeds=pe_dict)
    a = inferer._convert_to_csr(a)
    gene_col = inferer._auto_detect_gene_column(a)
    cfg.model.batch_size = BATCH
    dl = create_dataloader(cfg, adata=a, adata_name=tissue, shape_dict=None, data_dir=CACHE,
                           shuffle=False, protein_embeds=pe_dict, precision=None, gene_column=gene_col)
    return dl, cell_type, a.n_obs


def expressed_positions(batch, i, T):
    """positions (>=1, count>0) of genuinely expressed genes for cell i in the batch (skip CLS)."""
    valid = np.zeros(T, bool); valid[1:T] = True
    counts = batch[7]
    if counts is not None:
        valid &= (counts[i, :T].cpu().numpy() > 0)
    return np.nonzero(valid)[0]


def main():
    dev = DEVICE if (DEVICE != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, cfg, _genes, info = load_state_se(device=dev, dtype=torch.float32)
    pe_dict, GENES, _ = load_protein_embeds()          # GENES = list(protein_embeds.keys()) = gene symbols
    GENES = np.array(GENES)
    nG = len(GENES)
    print(f"[load] STATE-SE on {dev}; d={info['d_model']} nlayers={info['nlayers']} genes={nG}; "
          f"taps={TAPS} NPER={N_PER} CAP={CAP} FLOOR={FLOOR}", flush=True)

    # symbol -> Ensembl (Geneformer name_id; uppercase, first wins)
    sym2ens = {}
    for s, e in pickle.load(open(NAME_ID, "rb")).items():
        sym2ens.setdefault(str(s).upper(), e)

    # ---- PASS 1: collate only -> occurrences per (cell_type, gene) + per-cell context/partition -----------
    print("[pass 1] collating (no forward) to choose contexts + gene panel", flush=True)
    type_of = {}            # tissue -> np.array(cell_type per row idx)
    n_of = {}               # tissue -> n_obs
    # first learn the full cell-type universe (from subsample) to size the occurrence matrix
    for t, f in TISSUES.items():
        a = ad.read_h5ad(os.path.join(TS_DIR, f))
        sel = subsample(a, N_PER)
        type_of[t] = a.obs["cell_type"].astype(str).values[sel].copy()
        n_of[t] = len(sel)
        del a
    all_types = sorted(set(np.concatenate([type_of[t] for t in TISSUES])))
    tidx = {c: i for i, c in enumerate(all_types)}
    occ = np.zeros((len(all_types), nG), np.int64)         # occurrence per (cell_type, gene global idx)
    total_by_type = collections.Counter()
    for t in type_of:
        for c in type_of[t]:
            total_by_type[c] += 1

    for t, f in TISSUES.items():
        dl, cell_type, n_obs = setup_tissue(t, f, model, cfg, pe_dict)
        seen = 0
        for batch in dl:
            bs = batch[0].cpu().numpy()
            idxs = batch[3].cpu().numpy()
            B, T = batch[0].shape
            for i in range(B):
                pos = expressed_positions(batch, i, T)
                toks = bs[i, pos]
                toks = toks[(toks >= 0) & (toks < nG)]
                ti = tidx[cell_type[int(idxs[i])]]
                np.add.at(occ[ti], toks, 1)
                seen += 1
            del batch
        print(f"    [{t}] collated {seen}/{n_obs} cells", flush=True)
        del dl

    # contexts = pooled cell types with >= MIN_CELLS
    contexts = sorted([c for c in total_by_type if total_by_type[c] >= MIN_CELLS],
                      key=lambda c: -total_by_type[c])
    cidx = {c: i for i, c in enumerate(contexts)}
    print(f"[pass 1] {len(contexts)} contexts (>= {MIN_CELLS} cells): "
          + ", ".join(f"{c[:26]}({total_by_type[c]})" for c in contexts), flush=True)

    # panel: gene reaches FLOOR in >= 2 contexts AND maps to Ensembl; sort by total ctx count; cap
    ctx_rows = [tidx[c] for c in contexts]
    occ_ctx = occ[ctx_rows]                                # (n_ctx, nG)
    reach = (occ_ctx >= FLOOR).sum(0)                      # in how many contexts each gene clears FLOOR
    cand = np.where(reach >= 2)[0]
    cand = [g for g in cand if str(GENES[g]).upper() in sym2ens]
    cand.sort(key=lambda g: -occ_ctx[:, g].sum())
    panel = np.array(cand[:MAX_GENES], np.int64)          # gene GLOBAL indices
    panel_syms = np.array([str(GENES[g]).upper() for g in panel])
    panel_ens = np.array([sym2ens[s] for s in panel_syms])
    g2panel = np.full(nG, -1, np.int64); g2panel[panel] = np.arange(len(panel))
    print(f"[pass 1] panel = {len(panel)} genes (reach {FLOOR} in >=2 contexts, mappable to Ensembl)", flush=True)

    # per-cell context idx + partition (split each context's cells into 2 halves)
    rng = np.random.default_rng(SEED)
    cell_ctx = {}; cell_part = {}
    # build a global cell registry per (tissue,row) with its context; then split per-context
    reg = []  # (tissue, row)
    reg_ctx = []
    for t in TISSUES:
        cc = np.array([cidx.get(c, -1) for c in type_of[t]], np.int64)
        cell_ctx[t] = cc
        cell_part[t] = np.full(n_of[t], -1, np.int64)
        for r in range(n_of[t]):
            if cc[r] >= 0:
                reg.append((t, r)); reg_ctx.append(int(cc[r]))
    reg_ctx = np.array(reg_ctx)
    for c in range(len(contexts)):
        members = np.where(reg_ctx == c)[0]; rng.shuffle(members)
        half = len(members) // 2
        for k, m in enumerate(members):
            t, r = reg[m]
            cell_part[t][r] = 0 if k < half else 1

    # ---- PASS 2: forward pass, accumulate per-gene hidden states under the cap ----------------------------
    d = info["d_model"]
    acc = {L: np.zeros((NPART, len(contexts), len(panel), d), np.float32) for L in TAPS}
    cnts = np.zeros((NPART, len(contexts), len(panel)), np.int32)
    print(f"[pass 2] forward on {dev}; taps {TAPS}; accumulator "
          f"{sum(a.nbytes for a in acc.values())/2**30:.2f} GB", flush=True)

    hooks = []
    captured = {}
    for L in TAPS:
        hooks.append(model.transformer_encoder.layers[L].register_forward_hook(
            (lambda LL: (lambda m, inp, out: captured.__setitem__(LL, out.detach())))(L)))

    with torch.no_grad():
        for t, f in TISSUES.items():
            dl, cell_type, n_obs = setup_tissue(t, f, model, cfg, pe_dict)
            cc = cell_ctx[t]; pp = cell_part[t]
            done = 0
            for bi, batch in enumerate(dl):
                model._compute_embedding_for_batch(batch)
                bs = batch[0].cpu().numpy()
                idxs = batch[3].cpu().numpy()
                B, T = batch[0].shape
                res = {L: captured[L].float().cpu().numpy() for L in TAPS}   # (B, R, d) each
                for i in range(B):
                    row = int(idxs[i])
                    ci = int(cc[row]); part = int(pp[row])
                    if ci < 0 or part < 0:
                        continue
                    pos = expressed_positions(batch, i, T)
                    toks = bs[i, pos]
                    ok = (toks >= 0) & (toks < nG)
                    pos = pos[ok]; gi = g2panel[toks[ok]]
                    inpanel = gi >= 0
                    if not inpanel.any():
                        continue
                    pos = pos[inpanel]; gi = gi[inpanel]                 # each gene distinct within a cell
                    allowed = cnts[part, ci, gi] < CAP
                    if not allowed.any():
                        continue
                    gi = gi[allowed]; pos = pos[allowed]
                    for L in TAPS:
                        np.add.at(acc[L][part, ci], gi, res[L][i, pos])
                    cnts[part, ci, gi] += 1
                done += B
                if bi % 10 == 0:
                    frac = float((cnts >= CAP).mean())
                    print(f"    [{t}] batch {bi} cells~{done}/{n_obs} | {frac:.1%} (gene,ctx,part) at cap",
                          flush=True)
                del batch, res
            print(f"    [{t}] done {done} cells", flush=True)
            del dl
    for h in hooks:
        h.remove()

    # ---- save (identical schema to ctx_maxtoki / ctx_scgpt) ----------------------------------------------
    os.makedirs(HERE / "results", exist_ok=True)
    for L in TAPS:
        M = acc[L] / np.maximum(cnts[..., None], 1)
        out = HERE / "results" / f"ctx_state_L{L:02d}.npz"
        np.savez_compressed(out, M=M.astype(np.float16), counts=cnts, genes=panel_ens,
                            contexts=np.array(contexts), cap=CAP)
        # sanity: no all-zero gene slice among covered (part,ctx,gene)
        covered = cnts > 0
        zero_slices = int((covered & (np.abs(M).sum(-1) == 0)).sum())
        print(f"  wrote {out}  M{M.shape}  covered={int(covered.sum())}  zero-cov-slices={zero_slices}",
              flush=True)
    frac_cap = float((cnts >= CAP).mean())
    print(f"[pass 2] {frac_cap:.1%} of (gene,ctx,part) cells reached the {CAP}-occurrence cap", flush=True)

    # ---- MANDATORY correctness check: nuclear vs surface functional axis ---------------------------------
    validate(panel_syms, contexts)


# =========================== correctness check =============================================================
def _auc(labels, scores):
    """ROC-AUC via Mann-Whitney rank statistic. labels in {0,1}."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def validate(panel_syms, contexts, tap=8):
    print(f"\n[validate] nuclear-vs-surface axis on L{tap}", flush=True)
    npz = HERE / "results" / f"ctx_state_L{tap:02d}.npz"
    if not npz.exists():
        alt = [L for L in TAPS if (HERE / "results" / f"ctx_state_L{L:02d}.npz").exists()]
        if not alt:
            print("[validate] no output file to validate"); return
        tap = alt[0]; npz = HERE / "results" / f"ctx_state_L{tap:02d}.npz"
        print(f"[validate] L8 missing; using L{tap}", flush=True)
    z = np.load(npz, allow_pickle=True)
    M = z["M"].astype(np.float32); counts = z["counts"]           # (P,C,G,d), (P,C,G)
    P, C, G, d = M.shape

    # combine partitions: count-weighted mean -> rep[ctx, gene, d]; valid where total count > 0
    tot = counts.sum(0)                                            # (C,G)
    rep = (M[0] * counts[0][..., None] + M[1] * counts[1][..., None]) / np.maximum(tot[..., None], 1)
    valid = tot > 0                                                # (C,G)

    # z-score each of the d dims over the count-balanced (context,gene) entries
    ent = rep[valid]                                              # (n_entries, d)
    mu = ent.mean(0); sd = ent.std(0) + 1e-8
    repz = (rep - mu) / sd

    # a(g) = per-gene mean over contexts of the z-scored rep (contexts where the gene is valid)
    a = np.zeros((G, d), np.float32); ok = np.zeros(G, bool)
    for g in range(G):
        cv = np.where(valid[:, g])[0]
        if len(cv) == 0:
            continue
        a[g] = repz[cv, g].mean(0); ok[g] = True

    # GO poles
    g2g = pickle.load(open(GENE2GO, "rb"))
    NUC = {"GO:0005634", "GO:0000785", "GO:0003677"}
    SUR = {"GO:0005886", "GO:0005576", "GO:0005615"}
    def has(sym, S):
        gs = g2g.get(sym) or g2g.get(sym.upper())
        return bool(gs) and len(gs & S) > 0
    lab = np.full(G, -1)
    for g in range(G):
        if not ok[g]:
            continue
        s = panel_syms[g]
        n, u = has(s, NUC), has(s, SUR)
        if n and not u:
            lab[g] = 1
        elif u and not n:
            lab[g] = 0
    pole = np.where(lab >= 0)[0]
    y = lab[pole]; X = a[pole]
    n_nuc = int((y == 1).sum()); n_sur = int((y == 0).sum())
    print(f"[validate] pole genes: {n_nuc} nuclear + {n_sur} surface = {len(pole)}", flush=True)
    if n_nuc < 5 or n_sur < 5:
        print("[validate] too few pole genes to score"); return

    # 5-fold CV: train axis u on train poles, score held-out by <a,u>; pool held-out predictions
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pole))
    folds = np.array_split(perm, 5)
    ys, ss = [], []
    per_fold = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        u = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        u = u / (np.linalg.norm(u) + 1e-12)
        sc = X[te] @ u
        ys.append(y[te]); ss.append(sc)
        per_fold.append(_auc(y[te], sc))
    auc = _auc(np.concatenate(ys), np.concatenate(ss))
    print(f"[validate] nuclear-vs-surface axis ROC-AUC = {auc:.3f}  (pooled 5-fold; per-fold "
          f"{np.nanmean(per_fold):.3f}+-{np.nanstd(per_fold):.3f})", flush=True)
    verdict = "PASS (>0.60): gene->position mapping is correct" if auc > 0.60 else \
              "FAIL (<=0.60): mapping likely WRONG -- debug before trusting"
    print(f"[validate] {verdict}", flush=True)


if __name__ == "__main__":
    main()
