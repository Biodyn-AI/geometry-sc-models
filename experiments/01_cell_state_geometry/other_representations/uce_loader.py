"""Load UCE (Universal Cell Embedding, Rosen et al. Nature 2026) — the 100M / 4-layer variant
from the HuggingFace mirror `minwoosun/uce-100m`, and expose its transformer residual stream.

UCE is architecturally a sibling of STATE-SE: a cell is a *set of gene tokens*, each gene embedded by
its ESM2 protein embedding (token_dim=5120), fed through an nn.TransformerEncoder that emits a CLS cell
embedding (`X_uce`). This module reproduces snap-stanford/UCE's own tokenizer
(`eval_data.sample_cell_sentences` + `data_proc/data_utils.adata_path_to_prot_chrom_starts`) and the
`evaluate.run_eval` forward path exactly — human-only, self-contained (no accelerate / figshare).

Model class: `repos/UCE/model.py:TransformerModel` (pinned commit in repos/UCE/PINNED_COMMIT.txt).
The 100M checkpoint carries the full 145469x5120 `pe_embedding.weight` token table, so no external
`all_tokens.torch` is needed.

Residual stream = output of `model.transformer_encoder.layers[L]` : (seq_len, batch, d_model=1280).
Position 0 is the CLS token; gene tokens have index in [4, 143574); chrom delimiters and pad are excluded.

Verified on CPU and MPS (Apple Silicon), Python 3.12.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- pinned locations -------------------------------------------------------------------------
REPO = f"{_DATA}/biomi_automation/repos/UCE"
_HF_HUB = _os.path.join(_os.environ.get(
    "HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface")), "hub")
MISC = _os.path.join(
    _HF_HUB, "models--minwoosun--uce-misc", "snapshots",
    "bffb91084e4476698984e7e01f6170ce291f4074")
CKPT = _os.path.join(
    _HF_HUB, "models--minwoosun--uce-100m", "snapshots",
    "25b78197adc2fbd56fd6a141f2066b755ddba19c", "pytorch_model.bin")
HUMAN_PT = f"{MISC}/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
CHROM_CSV = f"{MISC}/species_chrom.csv"
OFFSETS = f"{MISC}/species_offsets.pkl"

# ---- tokenization constants (eval_single_anndata.py defaults) ---------------------------------
PAD_LENGTH = 1536
SAMPLE_SIZE = 1024
PAD_IDX = 0
CHROM_LEFT_IDX = 1
CHROM_RIGHT_IDX = 2
CLS_IDX = 3
CHROM_TOKEN_OFFSET = 143574
SPECIES = "human"

# model dims (config.json of minwoosun/uce-100m)
TOKEN_DIM = 5120
D_MODEL = 1280
NHEAD = 20
D_HID = 5120
NLAYERS = 4
OUTPUT_DIM = 1280
DROPOUT = 0.05
PE_ROWS = 145469


def load_uce_model(device: str = "cpu", dtype=torch.float32):
    """Construct TransformerModel and load the 100M checkpoint (strict=True; pe_embedding is in the ckpt)."""
    sys.path.insert(0, REPO)
    from model import TransformerModel  # noqa: E402

    m = TransformerModel(token_dim=TOKEN_DIM, d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID,
                         nlayers=NLAYERS, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    # pe_embedding must exist (145469 x 5120) before the strict load; the checkpoint fills it.
    m.pe_embedding = nn.Embedding.from_pretrained(torch.zeros(PE_ROWS, TOKEN_DIM))
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval().to(device=device, dtype=dtype)
    info = dict(missing=list(missing), unexpected=list(unexpected),
                d_model=D_MODEL, nlayers=NLAYERS)
    return m, info


def load_tok_aux():
    """Return (gene2idx, offset, chrom_df). gene2idx maps UPPERCASE human symbol -> row in the human .pt.
    chrom_df has a GLOBAL 'spec_chrom' categorical (built on the full CSV so codes are species-global)."""
    hum = torch.load(HUMAN_PT)                       # OrderedDict {gene_symbol: 5120-vector}
    spec_pe_genes = [k.upper() for k in hum.keys()]
    gene2idx = {g: i for i, g in enumerate(spec_pe_genes)}
    with open(OFFSETS, "rb") as f:
        offset = pickle.load(f)[SPECIES]             # 13466
    df = pd.read_csv(CHROM_CSV)
    df["spec_chrom"] = pd.Categorical(df["species"] + "_" + df["chromosome"])  # global codes
    return gene2idx, offset, df


def uce_vocab(gene2idx):
    """Set of UPPERCASE human symbols UCE can tokenize."""
    return set(gene2idx.keys())


def build_gene_maps(gene_symbols, gene2idx, offset, chrom_df):
    """Given a list of UPPERCASE symbols (all must be in gene2idx AND the human chrom CSV), return
    (pe_row_idxs [n], chroms [n] global codes, starts [n]) — the per-gene static maps."""
    vn = [g.upper() for g in gene_symbols]
    pe_row_idxs = torch.tensor([gene2idx[g] + offset for g in vn]).long()
    spec_chrom = chrom_df[chrom_df.species == SPECIES].set_index("gene_symbol")
    # keep first occurrence if the CSV has duplicate symbols
    spec_chrom = spec_chrom[~spec_chrom.index.duplicated(keep="first")]
    gene_chrom = spec_chrom.loc[vn]
    chroms = gene_chrom["spec_chrom"].cat.codes.values.astype(np.int64)
    starts = gene_chrom["start"].values.astype(np.float64)
    return pe_row_idxs, chroms, starts


def sample_cell_sentences(counts, pe_row_idxs, chroms, starts, rng: np.random.Generator):
    """Replicate eval_data.sample_cell_sentences for a batch of cells. `counts` is (ncell, n_genes)
    raw counts over the SAME gene axis as pe_row_idxs/chroms/starts. `rng` gives reproducibility
    (UCE's own path is unseeded). Returns (cell_sentences long (ncell,PAD), mask (ncell,PAD), longest)."""
    ncell = counts.shape[0]
    cell_sentences = torch.zeros((ncell, PAD_LENGTH))
    mask = torch.zeros((ncell, PAD_LENGTH))
    longest = 0
    for c in range(ncell):
        w = torch.log1p(counts[c]).numpy().astype(np.float64)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        w = w / w.sum()
        choice = rng.choice(np.arange(len(w)), size=SAMPLE_SIZE, p=w, replace=True)
        chosen_chrom = chroms[choice]
        order = np.argsort(chosen_chrom, kind="stable")
        choice = choice[order]
        new_chrom = chroms[choice]
        chosen_starts = starts[choice]
        ordered = np.full((PAD_LENGTH,), CLS_IDX)      # pos 0 = CLS
        i = 1
        uq = np.unique(new_chrom)
        rng.shuffle(uq)
        for chrom in uq:
            ordered[i] = int(chrom) + CHROM_TOKEN_OFFSET
            i += 1                                       # chrom-open
            loc = np.where(new_chrom == chrom)[0]
            sbs = np.argsort(chosen_starts[loc], kind="stable")
            to_add = choice[loc[sbs]]
            ordered[i:i + len(to_add)] = pe_row_idxs[to_add].numpy()
            i += len(to_add)                             # genes
            ordered[i] = CHROM_RIGHT_IDX
            i += 1                                       # chrom-close
        longest = max(longest, i)
        mask[c, :] = torch.concat((torch.ones(i), torch.zeros(PAD_LENGTH - i)))
        ordered[i:] = PAD_IDX
        cell_sentences[c, :] = torch.from_numpy(ordered)
    return cell_sentences.long(), mask, longest


@torch.no_grad()
def forward_residual(model, batch_sentences, mask, device, hook_layer=2):
    """Run the UCE forward path (evaluate.run_eval inner loop) and capture the residual stream after
    transformer_encoder.layers[hook_layer]. Returns (resid (seq,batch,d), token_idx (batch,seq),
    cls_embedding (batch,d)). Positions with token_idx in [4, CHROM_TOKEN_OFFSET) are gene tokens."""
    captured = {}
    h = model.transformer_encoder.layers[hook_layer].register_forward_hook(
        lambda mod, i, o: captured.__setitem__("resid", o.detach()))
    bs = batch_sentences.to(device)
    msk = mask.to(device)
    src = bs.permute(1, 0)                              # (seq, batch)
    src = model.pe_embedding(src.long())               # (seq, batch, 5120)
    src = nn.functional.normalize(src, dim=2)
    gene_output, cls_embedding = model.forward(src, mask=msk)
    h.remove()
    return captured["resid"], batch_sentences, cls_embedding


def gene_position_mask(token_idx):
    """Boolean (batch, seq): True at gene-token positions (exclude CLS / chrom-delim / pad)."""
    return (token_idx >= 4) & (token_idx < CHROM_TOKEN_OFFSET)


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, info = load_uce_model(device=dev)
    g2i, off, cdf = load_tok_aux()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded UCE-100M on {dev}: d_model={info['d_model']} nlayers={info['nlayers']} "
          f"params={n/1e6:.1f}M  missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    print(f"human offset={off}  vocab={len(g2i)} genes")
    print("sample vocab:", list(g2i)[:8])
