"""
Cell-sentence construction and subword -> gene attribution for C2S-Scale.

The pipeline's Phase 0 assumes 1 token = 1 gene. C2S-Scale breaks that: a cell is a
"cell sentence" of space-separated gene *names* run through the standard Gemma-2
tokenizer, so a gene symbol spans several subword tokens and the input also carries an
instruction scaffold. We build the cell sentence ourselves, so we know every gene's
exact character span, and use the tokenizer's `offset_mapping` to attribute each token
position back to its gene (or to the scaffold).

No model or HF download is needed to *build* sentences or to test the attribution logic
(offsets are the only tokenizer-dependent input) — see test_span_attribution.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

SCAFFOLD = -1  # gene id for tokens that belong to the prompt/special tokens, not a gene

# C2S-Scale cell-type-prediction prompt (from the model card). Kept configurable because
# the atlas may prefer a bare cell sentence over a task prompt; VERIFY exact template on
# the real tokenizer/model before large runs.
DEFAULT_PROMPT_TEMPLATE = (
    "The following is a list of {n_genes} gene names ordered by descending expression "
    "level in a {organism} cell. "
)


@dataclass
class CellSentence:
    text: str
    genes: List[str]                       # gene symbols, in sentence order
    gene_spans: List[Tuple[int, int]]      # (char_start, char_end) of each gene in `text`

    def gene_span_array(self) -> np.ndarray:
        """(n_genes, 2) int array of character spans, for vectorized attribution."""
        return np.asarray(self.gene_spans, dtype=np.int64).reshape(-1, 2)


def build_cell_sentence(
    genes: Sequence[str],
    organism: str = "human",
    prompt: bool = True,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> CellSentence:
    """Build a cell sentence and record each gene's character span.

    Args:
        genes: gene symbols already ranked by descending expression (highest first).
        organism: filled into the prompt template.
        prompt: if True, prepend the instruction scaffold (its span is not attributed
            to any gene); if False, the text is the bare space-separated gene list.
        prompt_template: format string with {n_genes} and {organism}.
    """
    prefix = prompt_template.format(n_genes=len(genes), organism=organism) if prompt else ""
    parts = []
    spans: List[Tuple[int, int]] = []
    cursor = len(prefix)
    for i, g in enumerate(genes):
        if i > 0:
            cursor += 1  # the joining space
        start = cursor
        end = start + len(g)
        spans.append((start, end))
        parts.append(g)
        cursor = end
    text = prefix + " ".join(parts)
    # sanity: recorded spans must slice back to the gene symbols
    for g, (s, e) in zip(genes, spans):
        assert text[s:e] == g, f"span mismatch for {g!r}: got {text[s:e]!r}"
    return CellSentence(text=text, genes=list(genes), gene_spans=spans)


def attribute_tokens_to_genes(
    offsets: Sequence[Tuple[int, int]],
    gene_spans: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Map each token to a gene index (or SCAFFOLD) by maximum character overlap.

    Args:
        offsets: per-token (char_start, char_end) from a fast tokenizer with
            return_offsets_mapping=True. Special tokens are typically (0, 0).
        gene_spans: (char_start, char_end) per gene, from build_cell_sentence.

    Returns:
        (n_tokens,) int array: gene index in [0, n_genes) or SCAFFOLD (-1).

    A token is assigned to the gene it overlaps most; zero-overlap tokens (special
    tokens, punctuation, prompt scaffold) map to SCAFFOLD. Overlap (not midpoint) is
    used so leading-space subwords like "▁GENE" attribute correctly.
    """
    gs = np.asarray(gene_spans, dtype=np.int64).reshape(-1, 2)
    starts, ends = gs[:, 0], gs[:, 1]
    out = np.full(len(offsets), SCAFFOLD, dtype=np.int64)
    for t, (ts, te) in enumerate(offsets):
        if te <= ts:  # empty / special token
            continue
        overlap = np.minimum(te, ends) - np.maximum(ts, starts)
        best = int(np.argmax(overlap))
        if overlap[best] > 0:
            out[t] = best
    return out


def anndata_to_ranked_genes(
    adata,
    cell_index: int,
    max_genes: int = 1024,
    layer: str | None = None,
    var_names=None,
) -> List[str]:
    """Rank one cell's genes by descending expression and return their symbols.

    Args:
        adata: AnnData (cells x genes). Uses `adata.X` unless `layer` is given.
        cell_index: row to convert.
        max_genes: cap on genes per cell (token-budget control — C2S-Scale ctx is 8192
            tokens and each gene is several subwords, so cap well below the raw gene count).
        layer: optional adata.layers key to read counts from instead of X.
        var_names: optional override for gene symbols (defaults to adata.var_names).
    """
    import scipy.sparse as sp

    x = adata.layers[layer] if layer is not None else adata.X
    row = x[cell_index]
    row = row.toarray().ravel() if sp.issparse(row) else np.asarray(row).ravel()
    names = np.asarray(var_names if var_names is not None else adata.var_names)

    nz = np.nonzero(row > 0)[0]
    order = nz[np.argsort(-row[nz], kind="stable")]  # descending expression, stable ties
    return [str(names[i]) for i in order[:max_genes]]
