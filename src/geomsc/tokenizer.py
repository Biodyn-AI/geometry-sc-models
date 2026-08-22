"""scGPT's published input encoding: 51 quantile value bins with expression-sorted truncation.

This is a standalone reimplementation of the input scheme described in scGPT (Cui et al., Nature
Methods 21:1470-1480, 2024) and used by that model's reference implementation. It is included here so
that the synthetic experiments in ``experiments/05_synthetic`` run without any dependency on a
private codebase, and so that anyone can check exactly what the "the model's own tokenised input"
rows of Table 1 in the paper were given.

Three steps, in this order:

1. **Select.** Keep the genes a cell actually expresses (count > 0) that are in the vocabulary.
2. **Truncate by expression.** Sort genes by descending expression and keep the top ``max_seq_len``.
   Sorting is used for truncation only; the model has no positional encoding over gene order in the
   synthetic setting, so the order itself carries no information there.
3. **Quantile-bin the values.** Map the surviving expression values onto ``1..n_bins-1`` using the
   cell's own quantiles, with 0 reserved for "not expressed". Ties are broken by a random draw
   between the left and right digitisation, which is what makes the binning reproducible only under
   a fixed seed. See the note in :func:`bin_row`.

Reserved value tokens, following the same convention as the reference implementation:

===================  =====================================================
value                meaning
===================  =====================================================
``0 .. n_bins-1``    quantile bin of an expressed gene
``n_bins``           MASK, the token the masked-value objective predicts
``n_bins + 1``       PAD, positions past the end of a short cell
===================  =====================================================

Padded gene positions carry ``PAD_VALUE`` in the float array so that a masked loss can ignore them
without a separate length tensor.
"""

from __future__ import annotations

import numpy as np

__all__ = ["N_BINS", "MAXLEN", "PAD_VALUE", "bin_row", "tokenize_csr_row", "batch_of"]

N_BINS = 51
"""Number of quantile bins. 51 is the scGPT default and is what every run in the paper used."""

MAXLEN = 512
"""Default truncation length. The synthetic runs override this to 128; see the note there."""

PAD_VALUE = -2.0
"""Sentinel written into the float value array at padded positions."""


def _digitize(x: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Digitise with random tie-breaking between the left- and right-inclusive bin edges.

    ``np.digitize`` has to choose a side for values that fall exactly on a bin edge. Single-cell
    counts are small integers, so edge ties are common rather than rare, and always choosing one
    side puts a visible spike in the bin histogram. Drawing uniformly between the two answers
    spreads tied values evenly across the bins they are tied between.

    This is why the tokenisation is a function of the random state as well as the data. Supplement
    A.3 records a case where an unseeded global generator here made a model's cached embeddings
    irreproducible run to run.
    """
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    rands = rng.random(len(x))
    return np.ceil(rands * (right - left) + left).astype(np.int64)


def bin_row(expr_nonzero: np.ndarray, n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """Quantile-bin one cell's non-zero expression into ``1 .. n_bins-1``.

    Bin edges are the cell's *own* ``n_bins - 1`` quantiles, so the encoding is per-cell and carries
    no information about absolute library size. Bin 0 is reserved for genes that are not expressed
    and therefore never appears in the output of this function.

    Parameters
    ----------
    expr_nonzero
        The cell's non-zero expression values, already restricted to vocabulary genes.
    n_bins
        Number of bins, normally :data:`N_BINS`.
    rng
        Generator used for tie-breaking. Pass a seeded generator for reproducibility.
    """
    bins = np.quantile(expr_nonzero, np.linspace(0, 1, n_bins - 1))
    return _digitize(expr_nonzero, bins, rng)


def tokenize_csr_row(
    col_idx,
    values,
    col2vocab,
    rng: np.random.Generator,
    max_seq_len: int = MAXLEN,
    n_bins: int = N_BINS,
    pad_token: int | None = None,
):
    """Tokenise one cell given its sparse representation.

    Parameters
    ----------
    col_idx, values
        The cell's non-zero column indices and their expression values, i.e. one row of a CSR
        matrix. ``values`` may be raw counts or normalised expression; only the ordering and the
        within-cell quantiles are used.
    col2vocab
        Array mapping gene column index to vocabulary id, with ``-1`` for genes outside the
        vocabulary. Those genes are dropped.
    rng
        Generator for bin tie-breaking.
    max_seq_len
        Truncation length. Genes are ranked by descending expression and the tail is dropped, so a
        cell expressing more genes than this loses its lowest-expressed ones.
    n_bins
        Number of quantile bins.
    pad_token
        Vocabulary id written at padded gene positions. Required in practice; the default of
        ``None`` will raise inside ``np.pad`` rather than silently writing a real gene id.

    Returns
    -------
    ``(gene_ids, gene_values, n_genes)`` with both arrays of length ``max_seq_len``, or ``None`` if
    the cell expresses no vocabulary gene at all. ``n_genes`` is the number of real positions before
    padding starts.
    """
    vids, expr = [], []
    for c, v in zip(col_idx, values):
        vid = col2vocab[c]
        if vid >= 0 and v > 0:
            vids.append(vid)
            expr.append(v)
    if not vids:
        return None

    vids = np.asarray(vids, dtype=np.int64)
    expr = np.asarray(expr, dtype=np.float32)

    order = np.argsort(-expr)              # descending expression; used for truncation only
    vids, expr = vids[order], expr[order]
    if len(vids) > max_seq_len:
        vids, expr = vids[:max_seq_len], expr[:max_seq_len]

    vals = bin_row(expr, n_bins, rng).astype(np.float32)
    n = len(vids)
    pad = max_seq_len - n
    gid = np.pad(vids, (0, pad), constant_values=pad_token).astype(np.int32)
    val = np.pad(vals, (0, pad), constant_values=PAD_VALUE).astype(np.float32)
    return gid, val, n


def batch_of(data, idx, dev, rng, mask_ratio: float, n_bins: int = N_BINS):
    """Assemble one training batch and apply the masked-value objective.

    Parameters
    ----------
    data
        Dict with tensors ``gid`` ``[N, L]``, ``val`` ``[N, L]`` and ``n`` ``[N]``, as produced by
        tokenising a corpus with :func:`tokenize_csr_row`.
    idx
        Indices of the cells in this batch.
    dev
        Torch device.
    rng
        Torch generator on ``dev``, for the mask draw.
    mask_ratio
        Fraction of real (non-padded) positions to mask. The paper's runs use 0.40.
    n_bins
        Number of quantile bins; ``n_bins`` is the MASK token and ``n_bins + 1`` the PAD token.

    Returns
    -------
    ``(gid, vin, pad, mask, val)`` where ``vin`` is the masked input the model sees, ``mask`` marks
    the positions the loss is taken over, and ``val`` holds the unmasked targets. Loss must be
    averaged over ``mask`` only; see the note in ``experiments/05_synthetic/synth_lib.py`` about
    the Apple-Silicon fault that made boolean indexing here unsafe.
    """
    import torch  # local import so the tokeniser is usable without torch installed

    gid = data["gid"][idx].long().to(dev)
    val = data["val"][idx].to(dev)
    n = data["n"][idx].to(dev)

    L = gid.shape[1]
    ar = torch.arange(L, device=dev)[None, :]
    pad = ar >= n[:, None]

    vbin = val.clamp(0, n_bins - 1).long()
    vbin = torch.where(pad, torch.full_like(vbin, n_bins + 1), vbin)      # PAD token

    r = torch.rand(gid.shape, device=dev, generator=rng)
    mask = (~pad) & (r < mask_ratio)
    vin = torch.where(mask, torch.full_like(vbin, n_bins), vbin)          # MASK token
    return gid, vin, pad, mask, val
