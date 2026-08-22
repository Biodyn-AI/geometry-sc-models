"""Build an 'expression-space' cell-cycle dataset: emb = raw log1p gene expression, no foundation model at all.

Tests the developmental claim -- "the curved manifold is IN the expression data; the model only preserves it"
-- for the cell cycle. cc_geometry.py / cc_steering.py call prep(X, 20) themselves (PCA-20 + whiten), so we
just hand them the 3000 x 6546 log1p expression matrix in place of a model embedding, with the SAME phi. If the
fixed-stalls / local-winds pattern replicates here, the loop and its navigability are properties of the DATA.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import REPLOGLE, DATA, _load_categorical, to_lognorm, build_substrate

sub = build_substrate()
sel = sub["cell_idx"].astype(int)
with h5py.File(REPLOGLE, "r") as f:
    X = np.stack([np.asarray(f["X"][int(i), :], dtype=np.float32) for i in sel])
Xlog = to_lognorm(X).astype(np.float32)
out = os.path.join(DATA, "expr_k562.npz")
np.savez(out, emb=Xlog, phi=sub["phi"], s_score=sub["s_score"], g2m_score=sub["g2m_score"],
         cc_phase=sub["cc_phase"].astype(str), cell_idx=sel)
print(f"[expr] {Xlog.shape} log1p expression (no model) | -> {out}")
