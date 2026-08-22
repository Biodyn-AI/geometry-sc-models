"""ctx_lib_c2s — shared machinery for Thread-B analyses on the C2S ctx cube.

Cube per tap (ctx_c2s_L{tap}.npz): M[2, nC, nG, H] per-(partition,context,gene) mean residual; counts[2,nC,nG];
rank_tok/rank_ord[2,nC,nG]; genes(symbols); contexts; cap.

Discipline carried from route_genemanifold Thread B:
  * balanced mask = (counts==cap) in BOTH partitions -> equal token count per estimate (heteroscedasticity).
  * per-DIM z-score (Timkey) over balanced entries -> tame Gemma-2 massive-activation dims (report top_dim_share).
  * everything cross-partition (0 vs 1 are disjoint cells) so reproducibility is real.
"""
import os
import numpy as np


def load(tap, bases="ctx_bases"):
    z = np.load(os.path.join(bases, f"ctx_c2s_L{tap:02d}.npz"), allow_pickle=True)
    return dict(M=z["M"].astype(np.float32), counts=z["counts"].astype(np.int32),
                rank_tok=z["rank_tok"].astype(np.float32), rank_ord=z["rank_ord"].astype(np.float32),
                genes=np.char.upper(z["genes"].astype(str)), contexts=z["contexts"].astype(str),
                cap=int(z["cap"]))


def balanced(counts, cap):
    """[nC, nG] bool: (gene,context) reached the cap in BOTH partitions."""
    return (counts == cap).all(0)


def zscore_dims(M, full):
    """Per-dim z-score using mean/std over all balanced (context,gene) entries across both partitions."""
    flat = M[:, full].reshape(-1, M.shape[-1])          # (2*n_full, H)
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    return (M - mu) / sd, float((flat.var(0).max()) / (flat.var(0).sum() + 1e-12))  # Mz, top_dim_share


def cos_rows(A, B):
    num = (A * B).sum(1)
    return num / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def a_space(Mz, full):
    """[nG, H] gene main effect a(g) = mean over partitions and balanced contexts of Mz. NaN where none."""
    nC, nG = full.shape; H = Mz.shape[-1]
    A = np.full((nG, H), np.nan, dtype=np.float64)
    for g in range(nG):
        cs = np.where(full[:, g])[0]
        if len(cs):
            A[g] = Mz[:, cs, g, :].reshape(-1, H).mean(0)
    return A


def rank_mean(rank_tok):
    """[nC, nG] mean token-rank over partitions (the RoPE-visible position confound var)."""
    return np.nanmean(rank_tok, axis=0)


def axis_from_poles(A, gene_idx, poleA_rows, poleB_rows):
    """Unit axis in a(g)-space: mean(A[poleA]) - mean(A[poleB]). Rows are indices into A (genes)."""
    a = np.nanmean(A[poleA_rows], axis=0); b = np.nanmean(A[poleB_rows], axis=0)
    u = a - b
    return u / (np.linalg.norm(u) + 1e-9)


def along_axis_interaction(Mz, full, vec, rank=None):
    """Projection q[p,c,g]=<Mz,vec> masked to balanced; optional rank-residualise on [1,rank,rank^2] per
    partition; then double-centre (subtract context-mean + gene-mean, add grand mean) -> gene x context
    interaction I[p]. Returns I (2, nC, nG) with NaN off-balanced."""
    q = np.tensordot(Mz, vec, axes=([3], [0]))          # (2, nC, nG)
    qm = np.where(full[None], q, np.nan)
    if rank is not None:
        for p in range(2):
            fin = np.isfinite(qm[p]) & np.isfinite(rank)
            r = rank[fin]; y = qm[p][fin]
            Amat = np.column_stack([np.ones_like(r), r, r * r])
            coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
            res = y - Amat @ coef
            tmp = np.full_like(qm[p], np.nan); tmp[fin] = res; qm[p] = tmp
    cm = np.nanmean(qm, axis=1, keepdims=True)          # context-mean (over genes? no: over contexts axis=1)
    gm = np.nanmean(qm, axis=2, keepdims=True)          # gene-mean
    grand = np.nanmean(qm, axis=(1, 2), keepdims=True)
    return qm - cm - gm + grand


def axis_power(I):
    """Cross-partition covariance of the interaction = reproducibility of along-axis gene x context modulation."""
    sel = np.isfinite(I[0]) & np.isfinite(I[1])
    return float(np.nanmean(I[0][sel] * I[1][sel])) if sel.any() else 0.0
