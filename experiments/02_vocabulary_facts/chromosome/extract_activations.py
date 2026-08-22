"""
Phase 0 adapter for C2S-Scale: residual-stream activation extraction with subword -> gene
attribution. Memory-safe two-pass design that scales to many layers x millions of positions.

Two units of analysis (see README "unit-of-analysis fork"):
  * gene-token : one row per gene occurrence, mean-pooled over that gene's subword tokens
                 (atlas-native; drives Phases 1-5/7/12). Scaffold tokens dropped.
  * cell-summary : one row per cell, the residual stream at the last real token
                 (decoder-native; drives Phases 6/8/11).

Residual stream comes from HF `output_hidden_states=True`: hidden_states[L+1] is the
post-block residual stream of transformer block L (hidden_states[0] is the embedding).

Two passes so nothing is held in RAM:
  PASS 1 (no model): tokenize + attribute every cell, count total rows, build row_gene_names /
    row_cell_ids. Cheap (tokenizer only).
  PASS 2 (model): re-tokenize + forward each cell, write mean-pooled gene rows straight into
    preallocated per-layer memmaps (dtype --store-dtype, default float16). Row order is identical
    to pass 1 because tokenization/attribution are deterministic.

Outputs per run dir:
  layer_{L:02d}_activations.npy   (n_rows, d_model)  float16 (or float32)
  row_gene_names.json  row_cell_ids.npy  manifest.json   (shared across layers)

Usage:
  python extract_activations.py --model vandijklab/C2S-Scale-Gemma-2-2B \
      --h5ad data/cells.h5ad --layers 0 3 6 9 13 17 21 25 --max-cells 1500 --max-genes 512 \
      --unit gene-token --store-dtype float16 --out runs/2b_atlas
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cell_sentences import (
    SCAFFOLD,
    anndata_to_ranked_genes,
    attribute_tokens_to_genes,
    build_cell_sentence,
)


def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Phase-0 activation extraction (two-pass)")
    p.add_argument("--model", required=True)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="transformer block indices (0-based); hidden_states[L+1] is captured")
    p.add_argument("--max-cells", type=int, default=1500)
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--unit", choices=["gene-token", "cell-summary"], default="gene-token")
    p.add_argument("--organism", default="human")
    p.add_argument("--no-prompt", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="model compute dtype")
    p.add_argument("--store-dtype", default="float16", choices=["float16", "float32"],
                   help="on-disk activation dtype (float16 halves disk; loaders cast to float32). "
                        "USE float32 for models with massive late-layer activations (e.g. Gemma-2 "
                        "27B) — fp16 max is 65504 and overflows to inf -> NaN SAEs. fp16 values are "
                        "clipped to +-60000 as a safety net, but that distorts the massive dims.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    import anndata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32
    prompt = not args.no_prompt

    def _cast(x):  # clip to fp16 range so massive activations don't overflow to inf -> NaN SAEs
        if store_dtype == np.float16:
            x = np.clip(x, -60000.0, 60000.0)
        return x.astype(store_dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset_mapping (gene attribution).")
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    adata = anndata.read_h5ad(args.h5ad)
    n_cells = min(args.max_cells, adata.n_obs)

    # ---- PASS 1: count rows + build row metadata (tokenizer only, no model) ----
    print(f"PASS 1: counting rows over {n_cells} cells...")
    cell_genes = []          # ranked gene list per cell (None if empty) — reused in pass 2
    row_gene_names, row_cell_ids = [], []
    for ci in range(n_cells):
        genes = anndata_to_ranked_genes(adata, ci, max_genes=args.max_genes)
        cell_genes.append(genes if genes else None)
        if not genes:
            continue
        if args.unit == "cell-summary":
            row_gene_names.append("<CELL>")
            row_cell_ids.append(ci)
            continue
        cs = build_cell_sentence(genes, organism=args.organism, prompt=prompt)
        enc = tok(cs.text, return_offsets_mapping=True, truncation=True, max_length=max_len)
        tok2gene = attribute_tokens_to_genes(enc["offset_mapping"], cs.gene_spans)
        for g in range(len(genes)):
            if (tok2gene == g).any():
                row_gene_names.append(genes[g])
                row_cell_ids.append(ci)
        if (ci + 1) % 200 == 0:
            print(f"  counted {ci + 1}/{n_cells} | rows so far {len(row_gene_names):,}")
    n_rows = len(row_gene_names)
    if n_rows == 0:
        raise SystemExit("no rows produced (check data / max_genes)")

    # infer d_model from a single forward-free load? we need the model; load it now.
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    model = model.eval()
    d_model = model.config.hidden_size
    print(f"  total rows: {n_rows:,} | d_model {d_model} | store {store_dtype.__name__} | "
          f"disk ≈ {n_rows * d_model * store_dtype().itemsize * len(args.layers) / 1e9:.1f} GB")

    # ---- preallocate per-layer memmaps ----
    mmaps = {}
    for L in args.layers:
        path = os.path.join(args.out, f"layer_{L:02d}_activations.npy")
        mmaps[L] = np.lib.format.open_memmap(path, mode="w+", dtype=store_dtype,
                                             shape=(n_rows, d_model))

    # ---- PASS 2: forward + write rows straight to disk ----
    print("PASS 2: forward + write...")
    off = 0
    for ci in range(n_cells):
        genes = cell_genes[ci]
        if genes is None:
            continue
        cs = build_cell_sentence(genes, organism=args.organism, prompt=prompt)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(model.device) for k, v in enc.items()},
                        output_hidden_states=True)
        hidden = [out.hidden_states[L + 1][0].float().cpu().numpy() for L in args.layers]  # (seq,d)

        if args.unit == "cell-summary":
            for li, L in enumerate(args.layers):
                mmaps[L][off] = _cast(hidden[li][-1])
            off += 1
        else:
            for g in range(len(genes)):
                mask = tok2gene == g
                if not mask.any():
                    continue
                for li, L in enumerate(args.layers):
                    mmaps[L][off] = _cast(hidden[li][mask].mean(axis=0))
                off += 1
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{n_cells} cells | {off:,}/{n_rows:,} rows")

    for L in args.layers:
        mmaps[L].flush()
        del mmaps[L]
    assert off == n_rows, f"row count mismatch: wrote {off}, expected {n_rows}"

    np.save(os.path.join(args.out, "row_cell_ids.npy"), np.asarray(row_cell_ids, dtype=np.int64))
    with open(os.path.join(args.out, "row_gene_names.json"), "w") as f:
        json.dump(row_gene_names, f)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"model": args.model, "unit": args.unit, "layers": args.layers,
                   "max_genes": args.max_genes, "prompt": prompt, "store_dtype": args.store_dtype,
                   "n_cells": int(sum(g is not None for g in cell_genes)),
                   "n_rows": n_rows, "d_model": int(d_model)}, f, indent=2)
    print(f"done: {n_rows:,} rows x {len(args.layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
