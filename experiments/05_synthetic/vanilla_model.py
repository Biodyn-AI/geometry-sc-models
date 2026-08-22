"""A plain, fully-standard masked-value transformer for the synthetic experiments.

Standard multi-head softmax attention, GELU feed-forward, pre-norm residual blocks. Nothing here is
novel and nothing needs citing beyond the original transformer. It is deliberately ordinary for two
reasons. The synthetic programme needs many small corpora rather than one large model, so a small
standard model is the right instrument. And a claim such as "the embedding table is a map rather
than an index" should not be attributable to an unusual attention or feed-forward variant; on a
plain transformer it is a statement about transformers.

The input encoding is scGPT's published scheme (gene embedding plus binned-value embedding, 51
quantile bins, expression-sorted truncation), implemented in ``geomsc.tokenizer``.

Size used throughout the paper: d = 192, 4 layers, 4 heads, feed-forward 384, about 1.4M parameters.
Supplement D.1 explains that choice.
"""
import math
import torch
import torch.nn as nn

N_BINS = 51


class Block(nn.Module):
    """Pre-norm transformer block: standard softmax MHA + GELU MLP."""

    def __init__(self, d, nh, dff, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x, pad):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class VanillaSCT(nn.Module):
    """Standard transformer over (gene id, binned value) tokens, masked-value objective.

    This model has one configuration. There is no architecture switch, and no experiment in the
    paper varies the architecture within this file; Supplement D.1 explains the choice of size.
    model is always the standard GELU variant.
    """

    def __init__(self, V, d=192, nl=4, nh=4, dff=384, n_bins=N_BINS):
        super().__init__()
        self.d, self.nl, self.nh, self.dff, self.n_bins = d, nl, nh, dff, n_bins
        self.gene_emb = nn.Embedding(V, d)
        self.val_emb = nn.Embedding(n_bins + 2, d)      # bins 0..n_bins-1, + mask, + pad
        self.blocks = nn.ModuleList([Block(d, nh, dff) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.dec = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, gid, vbin, pad):
        x = self.gene_emb(gid) + self.val_emb(vbin)
        for b in self.blocks:
            x = b(x, pad)
        return self.norm(x)

    def forward(self, gid, vbin, pad):
        return self.dec(self.encode(gid, vbin, pad)).squeeze(-1)
