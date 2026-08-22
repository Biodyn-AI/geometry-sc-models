"""Shared paths + helpers for Tier-B manifold-geometry build (route_manifold).

READ-ONLY w.r.t. route_b / route_geometry. Heavy external caches go OUTSIDE the repo under
projects/biotensor/data/route_manifold/ (gitignored). CPU-only, capped threads (co-tenant with STATE-ST).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBASE = os.path.abspath(os.path.join(HERE, ".."))
ROUTE_B = os.path.join(CODEBASE, "route_b")
ROUTE_GEO = os.path.join(CODEBASE, "route_geometry")
DATA = os.path.abspath(os.path.join(CODEBASE, "..", "data", "route_manifold"))   # outside repo
RESULTS = os.path.join(HERE, "results")
os.makedirs(DATA, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

for p in (ROUTE_B, os.path.join(ROUTE_B, "baseline"), ROUTE_GEO):
    if p not in sys.path:
        sys.path.insert(0, p)

# Co-tenant friendliness: CPU, capped threads.
DEV = "cpu"
N_THREADS = max(1, min(4, (os.cpu_count() or 4) - 2))
torch.set_num_threads(N_THREADS)

# ---- external drive roots ----
EXP = (f"{_DATA}/biodyn-work/"
       "subproject_42_sparse_autoencoder_biological_map/experiments")
MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
RAW = os.path.join(MI, "data", "raw")
SCGPT_REPO = os.path.join(MI, "external", "scGPT")
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
TS_META = os.path.join(EXP, "phase3_multitissue", "ts_activations", "extraction_metadata.json")

# ---- layer-block boundaries for the drift operator (handoff §2) ----
# scGPT 12 layers (0-11); Geneformer 18 layers (0-17).
SCGPT_BLOCKS = dict(early=[1, 2, 3, 4], mid=[5, 6, 7, 8], late=[9, 10, 11])
# Geneformer full blocks would be L0-5/L6-11/L12-17. To bound shared-drive reads (18x18.7GB) we use a
# representative-layer subset per block (documented in MODEL_NOTES.md); still a faithful early/mid/late drift.
GENEFORMER_BLOCKS = dict(early=[3], mid=[9], late=[15])   # 1 rep layer/block: bound shared-drive reads


def load_ruler(model_key):
    """model_key in {'scgpt_L11','geneformer_L11'} -> ruler dict."""
    if model_key.startswith("scgpt"):
        r = np.load(os.path.join(ROUTE_GEO, "results", "scgpt_L11_ruler.npz"), allow_pickle=True)
        return dict(kind="celltype", cell_type=r["cell_type"], tissue=r["tissue"])
    r = np.load(os.path.join(ROUTE_GEO, "results", "geneformer_L11_ruler.npz"), allow_pickle=True)
    return dict(kind="cellcycle", s_score=r["s_score"].astype(float),
                g2m_score=r["g2m_score"].astype(float), cc_phase=r["cc_phase"])


def zscore(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
