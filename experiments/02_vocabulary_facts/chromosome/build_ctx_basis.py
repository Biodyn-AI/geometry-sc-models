"""build_ctx_basis — the C2S MODEL BASIS for route_genemanifold Thread-A.

The direct analog of gm_lib.build_ctx: per-gene MEAN of the C2S residual stream at that gene's token
position, layer L, over a diverse Tabula Sapiens panel. C2S-Scale has no learned gene table, so this
context-aware activation basis is the model's gene representation.

Streams per-gene sums (never stores per-occurrence rows), so memory is O(n_genes x n_layers x d_model),
not O(n_tokens x d_model). Reuses the proven tokenize->attribute pattern of extract_activations.py.

Discipline carried from the route:
  * min-cnt floor (>=12 tokens/gene) so a gene's mean is a stable estimate (gm_lib.build_ctx).
  * occurrence CAP per gene (default 200) equalises estimation noise between ubiquitous housekeeping
    genes and rare genes, removing the heteroscedasticity that manufactures spurious geometry
    (NOTE_gene_context_representation: cap at 50/context; here we pool the panel so cap higher).

Out: <out>/c2s_ctx_L{L:02d}.npz  with M (n_genes,d_model) float32, symbols, counts.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from cell_sentences import anndata_to_ranked_genes, attribute_tokens_to_genes, build_cell_sentence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    p.add_argument("--h5ad", required=True, help="TS panel h5ad; var_names = gene symbols")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 13, 17, 21, 25],
                   help="block indices; hidden_states[L+1] captured")
    p.add_argument("--max-cells", type=int, default=100000)
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--min-cnt", type=int, default=12, help="drop genes seen in <min_cnt tokens")
    p.add_argument("--occ-cap", type=int, default=200, help="stop accumulating a gene past this many "
                   "occurrences (equalises estimation noise; 0 = no cap)")
    p.add_argument("--organism", default="human")
    p.add_argument("--no-prompt", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompt = not args.no_prompt
    cap = args.occ_cap if args.occ_cap and args.occ_cap > 0 else None

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise RuntimeError("fast tokenizer required for offset_mapping")
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    model = model.eval()
    d = model.config.hidden_size
    nL = len(args.layers)

    adata = anndata.read_h5ad(args.h5ad)
    n_cells = min(args.max_cells, adata.n_obs)
    print(f"panel {adata.shape} | {n_cells} cells | layers {args.layers} | d_model {d} | "
          f"min_cnt {args.min_cnt} | occ_cap {cap}", flush=True)

    sums = {}   # symbol -> (nL, d) float64 running sum
    cnts = {}   # symbol -> int
    done = 0
    for ci in range(n_cells):
        genes = anndata_to_ranked_genes(adata, ci, max_genes=args.max_genes)
        if not genes:
            continue
        # skip genes already capped, to save the forward's attribution work? no — position matters,
        # so we forward the full sentence and just don't ACCUMULATE capped genes.
        cs = build_cell_sentence(genes, organism=args.organism, prompt=prompt)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()},
                        output_hidden_states=True)
        hid = np.stack([out.hidden_states[L + 1][0].float().cpu().numpy() for L in args.layers])  # (nL,seq,d)
        for g in range(len(genes)):
            sym = genes[g].upper()
            if cap is not None and cnts.get(sym, 0) >= cap:
                continue
            mask = tok2gene == g
            if not mask.any():
                continue
            vec = hid[:, mask, :].mean(axis=1)          # (nL, d) mean over the gene's subword tokens
            if sym not in sums:
                sums[sym] = vec.astype(np.float64)
                cnts[sym] = 1
            else:
                sums[sym] += vec
                cnts[sym] += 1
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{n_cells} cells | {len(sums)} genes seen", flush=True)

    keep = sorted(s for s in sums if cnts[s] >= args.min_cnt)
    syms = np.array(keep)
    counts = np.array([cnts[s] for s in keep])
    print(f"genes kept (>= {args.min_cnt} tokens): {len(keep)} / {len(sums)} seen", flush=True)
    for li, L in enumerate(args.layers):
        M = np.stack([sums[s][li] / cnts[s] for s in keep]).astype(np.float32)
        out = os.path.join(args.out, f"c2s_ctx_L{L:02d}.npz")
        np.savez(out, M=M, symbols=syms, counts=counts)
        print(f"  L{L:02d} -> {out}  M={M.shape}", flush=True)
    json.dump({"model": args.model, "layers": args.layers, "n_cells": done,
               "n_genes": len(keep), "min_cnt": args.min_cnt, "occ_cap": cap,
               "max_genes": args.max_genes, "prompt": prompt}, open(os.path.join(args.out, "ctx_manifest.json"), "w"), indent=2)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
