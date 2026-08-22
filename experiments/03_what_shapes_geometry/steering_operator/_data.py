"""Minimal, self-contained data loaders for the manifold_steer benchmarks (h5py + numpy + scipy only).

Kept separate from the package core so `manifold_steer.py` has ZERO data dependencies. These read the project's
cached files directly; nothing here imports route_steering or any foundation model.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import numpy as np
import h5py
from scipy.sparse import csr_matrix

PROJ = f"{_DATA}"
SETTY_H5AD = f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad"
SETTY_NPZ = f"{PROJ}/data/branchpoint/scgptbin_setty.npz"          # only for pseudotime/cell_idx/clusters
LARRY_H5AD = f"{PROJ}/data/larry/LARRY_sp500_ranking1_adata_preprocessed.h5ad"


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def load_setty():
    """Returns dict: counts (log1p CP10k, cells x genes), pseudotime, clusters, genes. Model-free counts."""
    z = np.load(SETTY_NPZ, allow_pickle=True)
    y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int); clu = z["clusters"]
    ok = np.isfinite(y); y, ci, clu = y[ok], ci[ok], clu[ok]
    with h5py.File(SETTY_H5AD, "r") as f:
        genes = _dec(f["var"]["index"][:])
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        E = np.zeros((len(ci), len(genes)), np.float32)
        for r, i in enumerate(ci):
            s, e = int(ip[i]), int(ip[i + 1])
            E[r, ind[s:e]] = dat[s:e]
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    E = np.log1p(E / tot * 1e4)
    return dict(counts=E, pseudotime=y, clusters=clu, genes=genes)


def load_larry():
    """Returns dict: Xpca (cells x 40, the counts manifold), state (labels), time (2/4/6), clone (per cell)."""
    with h5py.File(LARRY_H5AD, "r") as f:
        cats = _dec(f['obs/__categories/state_info'][:])
        si = f['obs/state_info'][:]; ti = f['obs/time_info'][:]
        Xpca = f['obsm/X_pca'][:].astype(np.float64)
        ind = f['obsm/X_clone/indices'][:]; iptr = f['obsm/X_clone/indptr'][:]
        Xc = csr_matrix((np.ones_like(ind, np.int8), ind, iptr), shape=(len(si), int(ind.max()) + 1))
        clone = np.asarray(Xc.argmax(1)).ravel()
    tcats = ['2', '4', '6']
    return dict(Xpca=Xpca, state=np.array([cats[c] for c in si]),
                time=np.array([int(tcats[t]) for t in ti]), clone=clone)

