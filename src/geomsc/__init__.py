"""Shared machinery for *The geometry of single-cell foundation models*.

This package holds the pieces that more than one experiment needs. Everything specific to a single
result lives beside that result under ``experiments/``.

Modules
-------
``tokenizer``
    scGPT's published input encoding (51 quantile bins, expression-sorted truncation), vendored so
    the synthetic experiments run with no external dependency. Verified byte-identical to the
    reference implementation by ``tests/test_tokenizer.py``.
"""

__version__ = "1.0.0"
