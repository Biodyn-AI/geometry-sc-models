"""ctx_extract_c2s — the Thread-B data cube for C2S-Scale (port of ctx_extract_maxtoki.py).

Builds M[partition, context, gene, dim] = per-(cell-type, gene) mean residual, over a cell-type-stratified
panel, with the controls the polysemy PILOT lacked baked in at extraction:
  * OCCURRENCE CAP (CAP=50) per (gene,ctx,partition), shuffle-then-cap -> a random subsample -> kills
    heteroscedasticity.
  * CELL-LEVEL SPLIT-HALF: part = cell_index % 2 (disjoint cells; per-partition means).
  * TOKEN RANK stored (rank_tok = mean abs token position of the gene's subword span; rank_ord = gene ordinal
    in the cell sentence) so the position/rank confound is residualisable downstream.
  * STOPWORD / high-frequency-gene filter at panel build (MT-*, MALAT1, NEAT1, RACK1, RPL*/RPS*, EEF1A1, ACTB,
    B2M, TMSB4X, + document-frequency > DF_MAX) -> these sit at rank~0 in nearly every cell sentence.
  * PAIRWISE panel: gene must reach FLOOR=25 occurrences in >=2 contexts (not the all-context intersection
    that leaves only ~65 housekeeping genes), top MAX_GENES by cross-context count.

Out: <out>/ctx_c2s_L{tap:02d}.npz  {M float16 [2,nC,nG,H], counts int32 [2,nC,nG], rank_tok f32, rank_ord f32,
     genes(symbols), contexts, cap}
"""
from __future__ import annotations
import os, sys, re, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, attribute_tokens_to_genes, build_cell_sentence

STOP_RE = re.compile(r"^(MT-|RPL|RPS|MRPL|MRPS)")
STOP_SET = {"MALAT1", "NEAT1", "RACK1", "EEF1A1", "EEF2", "ACTB", "ACTG1", "B2M", "TMSB4X", "TMSB10",
            "FTL", "FTH1", "TPT1", "GAPDH"}


def is_stopword(sym):
    return bool(STOP_RE.match(sym)) or sym in STOP_SET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    p.add_argument("--h5ad", required=True, help="cell-type-stratified panel; obs[celltype-key], var=symbols")
    p.add_argument("--celltype-key", default="cell_type")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 4, 6, 9, 13, 17])
    p.add_argument("--max-genes", type=int, default=512, help="genes per cell sentence")
    p.add_argument("--cap", type=int, default=50)
    p.add_argument("--floor", type=int, default=25)
    p.add_argument("--panel-genes", type=int, default=6000)
    p.add_argument("--df-max", type=float, default=0.5, help="drop genes in top-K of > this fraction of cells")
    p.add_argument("--organism", default="human")
    p.add_argument("--no-prompt", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompt = not a.no_prompt
    rng = np.random.default_rng(a.seed)

    adata = anndata.read_h5ad(a.h5ad)
    ct = np.asarray(adata.obs[a.celltype_key]).astype(str)
    contexts = sorted(set(ct))
    ctx_idx = {c: i for i, c in enumerate(contexts)}
    nC = len(contexts)
    print(f"panel {adata.shape} | {nC} contexts: {contexts}", flush=True)

    # ---- PASS 1: rank genes per cell, count occurrences per context, document frequency ----
    print("PASS 1: ranking + panel selection", flush=True)
    cell_genes = [None] * adata.n_obs
    occ = np.zeros((nC, adata.n_vars), dtype=np.int64)          # occurrences per (context, gene col)
    dfc = np.zeros(adata.n_vars, dtype=np.int64)                # cells where gene is in top-K
    sym2col = {s.upper(): i for i, s in enumerate(adata.var_names)}
    var_up = np.char.upper(np.asarray(adata.var_names).astype(str))
    for ci in range(adata.n_obs):
        genes = anndata_to_ranked_genes(adata, ci, max_genes=a.max_genes)
        cell_genes[ci] = genes if genes else None
        if not genes:
            continue
        cols = [sym2col.get(g.upper()) for g in genes]
        cidx = ctx_idx[ct[ci]]
        for c in cols:
            if c is not None:
                occ[cidx, c] += 1
                dfc[c] += 1
        if (ci + 1) % 2000 == 0:
            print(f"  {ci+1}/{adata.n_obs}", flush=True)
    df = dfc / max(1, sum(g is not None for g in cell_genes))
    reach = (occ >= a.floor).sum(0)                            # #contexts reaching FLOOR
    keep_mask = (reach >= 2) & (df <= a.df_max)
    for i, s in enumerate(var_up):
        if keep_mask[i] and is_stopword(s):
            keep_mask[i] = False
    cand = np.where(keep_mask)[0]
    cand = cand[np.argsort(-occ[:, cand].sum(0))][:a.panel_genes]
    panel_cols = np.sort(cand)
    panel_syms = var_up[panel_cols]
    col2gi = {int(c): i for i, c in enumerate(panel_cols)}
    nG = len(panel_cols)
    print(f"  panel: {nG} genes (reach>=2 & df<={a.df_max} & non-stopword; top {a.panel_genes})", flush=True)

    # ---- load model ----
    dt = getattr(torch, a.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    if not tok.is_fast:
        raise RuntimeError("fast tokenizer required")
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    H = model.config.hidden_size
    taps = a.layers
    acc = {L: np.zeros((2, nC, nG, H), dtype=np.float64) for L in taps}
    cnts = np.zeros((2, nC, nG), dtype=np.int32)
    rsum_tok = np.zeros((2, nC, nG), dtype=np.float64)
    rsum_ord = np.zeros((2, nC, nG), dtype=np.float64)

    # ---- PASS 2: forward + accumulate (shuffle cell order so the cap draws a random subsample) ----
    order = np.arange(adata.n_obs); rng.shuffle(order)
    print(f"PASS 2: forward + accumulate | taps {taps} | H {H}", flush=True)
    done = 0
    for ci in order:
        genes = cell_genes[ci]
        if genes is None:
            continue
        part = int(ci) % 2
        cidx = ctx_idx[ct[ci]]
        # which ranked genes are panel genes (and not yet capped in this cell's partition/context)?
        gi_of_rank = {}
        for r, g in enumerate(genes):
            c = sym2col.get(g.upper())
            if c is not None and int(c) in col2gi:
                gi = col2gi[int(c)]
                if cnts[part, cidx, gi] < a.cap:
                    gi_of_rank[r] = gi
        if not gi_of_rank:
            continue
        cs = build_cell_sentence(genes, organism=a.organism, prompt=prompt)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_len)
        tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_hidden_states=True)
        hid = {L: out.hidden_states[L + 1][0].float().cpu().numpy() for L in taps}   # (seq,H)
        for r, gi in gi_of_rank.items():
            mask = tok2gene == r
            if not mask.any():
                continue
            toks = np.where(mask)[0]
            for L in taps:
                acc[L][part, cidx, gi] += hid[L][toks].mean(0)
            cnts[part, cidx, gi] += 1
            rsum_tok[part, cidx, gi] += float(toks.mean())
            rsum_ord[part, cidx, gi] += float(r)
        done += 1
        if done % 500 == 0:
            full = int(((cnts == a.cap).all(0)).sum())
            print(f"  {done} cells | balanced(gene,ctx) at cap: {full}", flush=True)

    # ---- finalise ----
    denom = np.maximum(cnts, 1)[..., None]
    rank_tok = np.where(cnts > 0, rsum_tok / np.maximum(cnts, 1), np.nan).astype(np.float32)
    rank_ord = np.where(cnts > 0, rsum_ord / np.maximum(cnts, 1), np.nan).astype(np.float32)
    for L in taps:
        M = (acc[L] / denom).astype(np.float16)
        out = os.path.join(a.out, f"ctx_c2s_L{L:02d}.npz")
        np.savez_compressed(out, M=M, counts=cnts, rank_tok=rank_tok, rank_ord=rank_ord,
                            genes=panel_syms, contexts=np.array(contexts), cap=a.cap)
        print(f"  L{L:02d} -> {out}  M={M.shape}", flush=True)
    bal = int(((cnts == a.cap).all(0)).sum())
    json.dump({"model": a.model, "layers": taps, "nC": nC, "nG": int(nG), "cap": a.cap, "floor": a.floor,
               "contexts": contexts, "balanced_gene_ctx_at_cap": bal, "H": int(H)},
              open(os.path.join(a.out, "ctx_manifest.json"), "w"), indent=2)
    print(f"done. balanced (gene,ctx) cells at cap in BOTH halves: {bal}", flush=True)


if __name__ == "__main__":
    main()
