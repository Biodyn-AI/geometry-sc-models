"""steer_c2s — Gate-3 causal steering harness for C2S-Scale (Gemma-2), ported from route_genemanifold/steer_lib.py.

THE ONE VALIDITY RULE (verbatim from STEERING_TOOL.md): a steering result is evidence ONLY if the readout is
causally downstream of, and INDEPENDENT from, the intervention. So the primary readout is the MODEL'S OWN
next-token logits — never a probe fitted on the steering directions. Every run carries: a norm-matched RANDOM
direction (~0 effect), and a DOSE-RESPONSE over alpha (monotone, not a step).

C2S SPECIFICS.
  * We steer at a DECODER-LAYER site (residual after block L), not "embed": Gemma-2 scales the embedding by
    sqrt(hidden) right after embed_tokens, which would rescale an embed-site edit. Layer-site steering also
    matches where the ctx basis (and hence the direction) is measured.
  * "next-gene logit" readout: C2S emits SUBWORD tokens, and a gene name is several subwords. We read the mean
    next-token logit over the FIRST subword of each readout gene's name (leading space, as in a cell sentence).
  * Split-half propagation: push a RANDOM half of a cell's gene positions toward pole A (+u) or B (-u); read
    the pole-A-minus-pole-B logit swing at the OTHER half — disjoint positions, so the effect must propagate
    through attention. Direction u = ctx-basis centroid(pole-A genes) - centroid(pole-B genes) at layer L.
"""
from __future__ import annotations
import os, sys, json
from contextlib import contextmanager
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, attribute_tokens_to_genes, build_cell_sentence


class C2SSteerer:
    def __init__(self, model_dir="vandijklab/C2S-Scale-Gemma-2-2B", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dt = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, device_map="auto").eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, device_map="auto").eval()
        self.device = next(self.model.parameters()).device
        self.hidden = self.model.config.hidden_size
        self.max_len = min(getattr(self.tok, "model_max_length", 8192) or 8192, 8192)
        self._first_tok = {}

    def first_subtoken(self, sym):
        """Token id of the first subword of ' <SYM>' (leading space = mid-sentence gene start)."""
        if sym not in self._first_tok:
            ids = self.tok(" " + sym, add_special_tokens=False)["input_ids"]
            self._first_tok[sym] = ids[0] if ids else None
        return self._first_tok[sym]

    def encode(self, genes, organism="human", prompt=True):
        cs = build_cell_sentence(genes, organism=organism, prompt=prompt)
        enc = self.tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                       truncation=True, max_length=self.max_len)
        tok2gene = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        return enc, tok2gene, cs

    @contextmanager
    def steering(self, vec, alpha, pos_mask, layer):
        """Add alpha * unit(vec) to the residual after decoder block `layer` at positions in pos_mask (seq,bool)."""
        torch = self.torch
        v = np.asarray(vec, dtype=np.float64).ravel(); v = v / (np.linalg.norm(v) + 1e-12)
        d = torch.tensor(v, dtype=torch.float32, device=self.device)
        m = torch.tensor(np.asarray(pos_mask, bool), device=self.device)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            add = (alpha * d).to(h.dtype)
            h = h + m.view(1, -1, 1).to(h.dtype) * add
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        handle = self.model.model.layers[int(layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def logits(self, enc):
        torch = self.torch
        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        return out.logits[0].float().cpu().numpy()          # (seq, vocab)

    def logits_and_resid_norm(self, enc, layer, pos_mask):
        """One forward: next-token logits AND the mean L2 norm of the residual after block `layer` over the
        pushed positions. The norm sets the steering scale (Gemma-2 residuals are large, so a dose must be a
        FRACTION of the residual norm — the route's '× residual norm' convention)."""
        torch = self.torch
        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True)
        lg = out.logits[0].float().cpu().numpy()
        h = out.hidden_states[layer + 1][0].float().cpu().numpy()      # (seq, d)
        m = np.asarray(pos_mask, bool)
        scale = float(np.linalg.norm(h[m], axis=1).mean()) if m.any() else 1.0
        return lg, scale

    def read_positions(self, tok2gene, gene_idx_set):
        """Last token index of each gene in gene_idx_set (the position whose NEXT-token logits we read)."""
        pos = {}
        for t, g in enumerate(tok2gene):
            if g in gene_idx_set:
                pos[g] = t                                   # keep last token of the gene
        return list(pos.values())

    def pole_swing(self, logits_row, poleA_toks, poleB_toks):
        a = np.asarray([t for t in poleA_toks if t is not None], int)
        b = np.asarray([t for t in poleB_toks if t is not None], int)
        a, b = a[a < logits_row.shape[-1]], b[b < logits_row.shape[-1]]
        la = float(logits_row[a].mean()) if len(a) else np.nan
        lb = float(logits_row[b].mean()) if len(b) else np.nan
        return la - lb                                       # pole-A minus pole-B mass


def propagation_test(st, adata, direction, layer, poleA_genes, poleB_genes, source_genes,
                     n_cells=30, alphas=(0.0, 0.25, 0.5, 1.0), seed=0, max_genes=256, random_dir=None,
                     min_present=4):
    """Split-half propagation: in each cell, push a RANDOM half of the gene positions toward +/- direction at
    layer L; read the pole-A-minus-pole-B next-token logit swing at the OTHER half. Signed (+u vs -u), with a
    dose sweep and a norm-matched random control. direction/random_dir live in layer-L hidden space."""
    rng = np.random.default_rng(seed)
    pA = [st.first_subtoken(g.upper()) for g in poleA_genes]
    pB = [st.first_subtoken(g.upper()) for g in poleB_genes]
    src = set(g.upper() for g in source_genes)
    rows = []
    used = 0
    for ci in range(adata.n_obs):
        if used >= n_cells:
            break
        genes = anndata_to_ranked_genes(adata, ci, max_genes=max_genes)
        if len(genes) < 20:
            continue
        # positions that are source-axis genes present in this cell
        present = [i for i, g in enumerate(genes) if g.upper() in src]
        if len(present) < min_present:
            continue
        enc, tok2gene, _ = st.encode(genes)
        seq = enc["input_ids"].shape[1]
        # random half of the PRESENT axis-genes are the push set; read at the complementary half
        rng.shuffle(present)
        half = len(present) // 2
        push_g, read_g = set(present[:half]), set(present[half:])
        push_mask = np.zeros(seq, bool)
        for t, g in enumerate(tok2gene):
            if g in push_g:
                push_mask[t] = True
        read_pos = st.read_positions(tok2gene, read_g)
        if not push_mask.any() or not read_pos:
            continue
        base, scale = st.logits_and_resid_norm(enc, layer, push_mask)   # dose is FRACTION x residual norm
        b_swing = np.mean([st.pole_swing(base[p], pA, pB) for p in read_pos])
        cell = dict(cell=int(ci), n_push=int(push_mask.sum()), n_read=len(read_pos),
                    base=float(b_swing), resid_norm=float(scale))
        for a in alphas:                                                # a = fraction of residual norm
            if a == 0:
                cell["pos_0.0"] = 0.0; cell["neg_0.0"] = 0.0; cell["rand_0.0"] = 0.0
                continue
            aa = a * scale
            with st.steering(direction, aa, push_mask, layer):
                lp = st.logits(enc)
            with st.steering(direction, -aa, push_mask, layer):
                ln = st.logits(enc)
            dp = np.mean([st.pole_swing(lp[p], pA, pB) for p in read_pos]) - b_swing
            dn = np.mean([st.pole_swing(ln[p], pA, pB) for p in read_pos]) - b_swing
            cell[f"pos_{a}"] = float(dp); cell[f"neg_{a}"] = float(dn); cell[f"signed_{a}"] = float(dp - dn)
            if random_dir is not None:
                with st.steering(random_dir, aa, push_mask, layer):
                    lr = st.logits(enc)
                cell[f"rand_{a}"] = float(np.mean([st.pole_swing(lr[p], pA, pB) for p in read_pos]) - b_swing)
        rows.append(cell); used += 1
    return rows
