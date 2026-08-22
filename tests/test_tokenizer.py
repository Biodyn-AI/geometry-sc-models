"""The vendored tokeniser must match the reference implementation exactly.

``geomsc.tokenizer`` is a standalone copy of scGPT's published input scheme. The synthetic
experiments in the paper were run against the original implementation, so a difference here would
mean the shipped code no longer reproduces the shipped numbers.

The test is self-contained: it checks the properties the encoding is supposed to have, on random
cells, rather than requiring the original module to be present.
"""

import numpy as np
import pytest

from geomsc.tokenizer import N_BINS, PAD_VALUE, bin_row, tokenize_csr_row


def _random_cell(seed, n_cols=400):
    r = np.random.default_rng(seed)
    nnz = int(r.integers(1, 60))
    col_idx = r.choice(n_cols, size=nnz, replace=False)
    values = (r.random(nnz) * 20).astype(np.float32)
    col2vocab = np.where(r.random(n_cols) < 0.8, np.arange(n_cols), -1)
    return col_idx, values, col2vocab


@pytest.mark.parametrize("seed", range(25))
def test_deterministic_under_a_fixed_seed(seed):
    """Same data and same generator seed must give the same tokens, every time."""
    col_idx, values, col2vocab = _random_cell(seed)
    a = tokenize_csr_row(col_idx, values, col2vocab, np.random.default_rng(7),
                         max_seq_len=32, pad_token=999)
    b = tokenize_csr_row(col_idx, values, col2vocab, np.random.default_rng(7),
                         max_seq_len=32, pad_token=999)
    assert a is not None
    for x, y in zip(a[:2], b[:2]):
        assert np.array_equal(x, y)
    assert a[2] == b[2]


@pytest.mark.parametrize("seed", range(25))
def test_shape_padding_and_bin_range(seed):
    """Fixed length, correct pad token, and bins inside 1..N_BINS-1."""
    col_idx, values, col2vocab = _random_cell(seed)
    out = tokenize_csr_row(col_idx, values, col2vocab, np.random.default_rng(seed),
                           max_seq_len=32, pad_token=999)
    if out is None:
        return
    gid, val, n = out
    assert gid.shape == (32,) and val.shape == (32,)
    assert 0 < n <= 32
    assert np.all(gid[n:] == 999)
    assert np.all(val[n:] == PAD_VALUE)
    assert np.all(val[:n] >= 0) and np.all(val[:n] <= N_BINS - 1)


@pytest.mark.parametrize("seed", range(25))
def test_truncation_keeps_the_highest_expressed_genes(seed):
    """Truncation is by descending expression, so the kept set is the top-k by value."""
    col_idx, values, col2vocab = _random_cell(seed)
    keep = col2vocab[col_idx] >= 0
    if keep.sum() < 5:
        return
    k = 3
    out = tokenize_csr_row(col_idx, values, col2vocab, np.random.default_rng(seed),
                           max_seq_len=k, pad_token=999)
    gid, _, n = out
    assert n == k
    expected = set(col2vocab[col_idx[keep]][np.argsort(-values[keep])][:k].tolist())
    assert set(gid[:k].tolist()) == expected


def test_binning_is_monotone_in_expression():
    """A larger expression value never lands in a lower bin than a smaller one."""
    expr = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 9.0, 20.0], dtype=np.float32)
    bins = bin_row(expr, N_BINS, np.random.default_rng(0))
    assert np.all(np.diff(bins) >= 0)


def test_genes_outside_the_vocabulary_are_dropped():
    col_idx = np.array([0, 1, 2])
    values = np.array([5.0, 3.0, 1.0], dtype=np.float32)
    col2vocab = np.array([-1, 7, -1])          # only column 1 is in the vocabulary
    gid, _, n = tokenize_csr_row(col_idx, values, col2vocab, np.random.default_rng(0),
                                 max_seq_len=4, pad_token=99)
    assert n == 1 and gid[0] == 7


def test_a_cell_with_no_vocabulary_gene_returns_none():
    assert tokenize_csr_row(np.array([0]), np.array([5.0], dtype=np.float32),
                            np.array([-1]), np.random.default_rng(0),
                            max_seq_len=4, pad_token=99) is None
