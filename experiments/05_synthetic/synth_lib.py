"""Shared machinery for the synthetic-corpus experiments (S3, S4, S6).

Corpora are generated from a latent structure we choose, then pushed through scGPT's published input
encoding (``geomsc.tokenizer``: 51 quantile bins, expression-sorted truncation) into a plain standard
transformer (``vanilla_model.VanillaSCT``: softmax multi-head attention, GELU feed-forward,
pre-norm). Across arms of an experiment, only the data changes.

The point of the design is that the ground truth is planted rather than inferred. We choose the
latent structure, break exactly one property of it, and ask what the model's copy of that structure
does. That is what lets Section 5 of the paper state a mechanism instead of a correlation.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

# scGPT's published input scheme, vendored in this repository so the synthetic experiments run
# with no external dependency. ``tests/test_tokenizer.py`` checks it against the reference
# implementation on 200 random cells.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomsc.tokenizer import tokenize_csr_row, batch_of, N_BINS  # noqa: E402
from vanilla_model import VanillaSCT  # noqa: E402

MAXLEN = 128          # synthetic cells are small; 512 would be mostly padding
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------------------------------ corpus generation

def ring_corpus(n_cells, n_genes=1000, n_phase=300, arm="uniform", arc=(150.0, 240.0),
                kappa=4.0, amp=6.0, lib=3000, seed=0):
    """Cells on a phase circle. Returns (counts uint16 [n_cells, n_genes], theta_deg, meta).

    arm:
      uniform   - phase-gene preferred phases spread evenly; cells spread evenly
      sharp     - HALF the phase genes have their preferred phase inside `arc`, so the emitted
                  gene distribution turns over fastest there (output-change hypothesis)
      occupancy - genes even, but 3x more CELLS sampled inside `arc` (density hypothesis)
      noisy     - genes even, cells even, but cells inside `arc` get extra dispersion, so the
                  next-token distribution is harder to predict there (entropy hypothesis)
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.deg2rad(arc[0]), np.deg2rad(arc[1])

    # cell phases
    if arm == "occupancy":
        n_in = int(n_cells * 0.5)
        th = np.concatenate([rng.uniform(lo, hi, n_in),
                             rng.uniform(0, 2 * np.pi, n_cells - n_in)])
        rng.shuffle(th)
    else:
        th = rng.uniform(0, 2 * np.pi, n_cells)

    # phase-gene preferred phases
    if arm == "sharp":
        k = n_phase // 2
        mu = np.concatenate([rng.uniform(lo, hi, k), rng.uniform(0, 2 * np.pi, n_phase - k)])
    else:
        mu = rng.uniform(0, 2 * np.pi, n_phase)

    base = rng.gamma(2.0, 1.0, n_genes) + 0.1          # background rate per gene
    counts = np.zeros((n_cells, n_genes), dtype=np.uint16)
    inside = ((th - lo) % (2 * np.pi)) < ((hi - lo) % (2 * np.pi))

    for i in range(n_cells):
        rate = base.copy()
        rate[:n_phase] *= 1.0 + amp * np.exp(kappa * (np.cos(th[i] - mu) - 1.0))
        if arm == "noisy" and inside[i]:
            rate = rate * rng.gamma(1.0, 1.0, n_genes)  # extra dispersion inside the arc only
        p = rate / rate.sum()
        counts[i] = rng.multinomial(lib, p).astype(np.uint16)

    return counts, np.rad2deg(th) % 360.0, {
        "arm": arm, "arc": list(arc), "n_genes": n_genes, "n_phase": n_phase,
        "kappa": kappa, "amp": amp, "lib": lib, "seed": seed,
        "frac_cells_in_arc": float(inside.mean()),
        "frac_phase_genes_in_arc": float(
            (((mu - lo) % (2 * np.pi)) < ((hi - lo) % (2 * np.pi))).mean())}


def group_corpus(n_cells, n_genes=1000, n_groups=20, consistency=0.6, n_programs=40,
                 genes_per_program=60, lib=3000, seed=0):
    """Cells whose active gene sets are enriched for arbitrary GROUPS of genes.

    Group membership is a fixed property of the vocabulary that appears in NO single cell -- the
    synthetic analogue of a gene's chromosome. `consistency` is the probability that a program
    draws its genes from within one group rather than at random; it is the knob that sets how
    strongly group membership shows up in corpus-wide co-occurrence.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, n_genes)                 # the ground-truth label

    programs = []
    for _ in range(n_programs):
        if rng.random() < consistency:
            g = rng.integers(0, n_groups)
            pool = np.where(group == g)[0]
            if len(pool) < genes_per_program:
                pool = np.concatenate([pool, rng.choice(n_genes, genes_per_program, replace=False)])
        else:
            pool = np.arange(n_genes)
        programs.append(rng.choice(pool, size=min(genes_per_program, len(pool)), replace=False))

    base = rng.gamma(2.0, 1.0, n_genes) + 0.1
    counts = np.zeros((n_cells, n_genes), dtype=np.uint16)
    for i in range(n_cells):
        rate = base.copy()
        for p in rng.choice(len(programs), size=3, replace=False):
            rate[programs[p]] *= 8.0
        counts[i] = rng.multinomial(lib, rate / rate.sum()).astype(np.uint16)

    return counts, group, {"n_groups": n_groups, "consistency": consistency,
                           "n_programs": n_programs, "genes_per_program": genes_per_program,
                           "n_genes": n_genes, "lib": lib, "seed": seed}


# ------------------------------------------------------------------ tokenize + train

def tokenize(counts, seed=0, maxlen=MAXLEN):
    """Project tokenizer, one row at a time. Vocab id == gene column; pad id == n_genes."""
    rng = np.random.default_rng(seed)
    n, g = counts.shape
    col2vocab = np.arange(g)
    gid = np.zeros((n, maxlen), dtype=np.int32)
    val = np.zeros((n, maxlen), dtype=np.float32)
    ln = np.zeros(n, dtype=np.int32)
    for i in range(n):
        nz = np.nonzero(counts[i])[0]
        out = tokenize_csr_row(nz, counts[i][nz].astype(np.float32), col2vocab, rng,
                               max_seq_len=maxlen, pad_token=g)
        if out is None:
            continue
        gid[i], val[i], ln[i] = out
    return {"gid": torch.from_numpy(gid), "val": torch.from_numpy(val),
            "n": torch.from_numpy(ln)}


def train(data, V, d=192, nl=4, nh=4, dff=384, steps=3000, bs=64,
          lr=3e-4, mask_ratio=0.4, seed=0, log_every=500, quiet=False,
          freeze_gene_emb=False):
    """Masked-value objective, same as the real runs. Returns (model, history)."""
    torch.manual_seed(seed)
    model = VanillaSCT(V + 1, d=d, nl=nl, nh=nh, dff=dff).to(DEV)
    if freeze_gene_emb:
        model.gene_emb.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n = data["gid"].shape[0]
    n_val = min(1000, n // 10)
    tr_idx, va_idx = np.arange(n_val, n), np.arange(n_val)
    rng = np.random.default_rng(seed)
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.from_numpy(rng.choice(tr_idx, bs, replace=False))
        gid, vin, pad, mask, val = batch_of(data, idx, DEV, gen, mask_ratio)
        pred = model(gid, vin, pad)
        mf = mask.float()
        loss = (((pred - val) ** 2) * mf).sum() / mf.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vi = torch.from_numpy(va_idx)
                gid, vin, pad, mask, val = batch_of(data, vi, DEV, gen, mask_ratio)
                mf = mask.detach().cpu().numpy().astype(bool)
                p = model(gid, vin, pad).float().cpu().numpy()[mf]
                t = val.float().cpu().numpy()[mf]
                vc = float(np.corrcoef(p, t)[0, 1]) if len(p) > 10 and t.std() > 0 else float("nan")
            hist.append({"step": step, "train_mse": float(loss.item()), "val_corr": vc})
            if not quiet:
                print(f"    step {step:5d}  mse {loss.item():7.3f}  val_corr {vc:+.4f}"
                      f"  {time.time() - t0:5.0f}s")
    return model, hist


@torch.no_grad()
def cell_embeddings(model, data, layer=None, bs=256):
    """Mean-pool over non-pad gene tokens -- the same construct every cell-level result uses."""
    model.eval()
    n = data["gid"].shape[0]
    outs = []
    gen = torch.Generator(device=DEV).manual_seed(0)
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n))
        gid, vin, pad, _, _ = batch_of(data, idx, DEV, gen, 0.0)
        x = model.gene_emb(gid) + model.val_emb(vin)
        nl = len(model.blocks) if layer is None else layer
        for b in model.blocks[:nl]:
            x = b(x, pad)
        m = (~pad).float().unsqueeze(-1)
        outs.append(((x * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(outs)


def gene_table(model):
    """The learned input gene-embedding table W_E, minus the pad row."""
    return model.gene_emb.weight.detach().float().cpu().numpy()[:-1]
