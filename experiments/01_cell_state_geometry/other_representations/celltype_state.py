"""STATE-SE cell-type geometry/curvature test (the Tabula Sapiens test scGPT got).

Loads per-cell L11 residual embeddings extracted through STATE-SE (ts_extract.py) for TS immune/kidney/lung
cells, then:
  1. k-NN cell-type recovery (balanced accuracy) vs shuffled-label null  (geometry_test.py style),
  2. curvature = kNN - linear classification accuracy in a matched 20D PCA space, reusing route_lineage's
     curvature() READ-ONLY (linear logistic vs quad/bilinear vs kNN),
  3. MANDATORY synthetic linearly-embedded control: labels that are an argmax(linear) function of the PCA
     features by construction -> curvature must be ~0 (route_lineage.linear_control discipline).

Run:  ../../.venv_state/bin/python celltype_state.py [smoke]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_lineage"))
from lineage_manifold import curvature, cv_acc, pca_reduce, chance  # READ-ONLY reuse  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                                    # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score                  # noqa: E402
from sklearn.preprocessing import StandardScaler                                      # noqa: E402
from sklearn.metrics import balanced_accuracy_score                                   # noqa: E402

CACHE = f"{_DATA}/state_geometry_cache"
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)


def knn_recovery(E, labels, k=15, seed=0):
    y = np.asarray(labels)
    keep = np.array([np.sum(y == v) >= 5 for v in y])
    Xz = StandardScaler().fit_transform(np.nan_to_num(E.astype(np.float64)))[keep]
    yk = y[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    knn = KNeighborsClassifier(k)
    acc = cross_val_score(knn, Xz, yk, cv=cv, scoring="balanced_accuracy").mean()
    null = cross_val_score(knn, Xz, np.random.default_rng(seed).permutation(yk), cv=cv,
                           scoring="balanced_accuracy").mean()
    return dict(bal_acc=float(acc), null=float(null), margin=float(acc - null), n=int(keep.sum()))


def synthetic_linear_control(X, y, d=20, seed=0):
    """Labels that are argmax(linear) of the PCA features -> linearly separable BY CONSTRUCTION.
    Same curvature() ladder must then show curvature_acc ~ 0 (no nonlinear structure exists)."""
    k = len(np.unique(y))
    Xr = StandardScaler().fit_transform(pca_reduce(X, d))
    rng = np.random.default_rng(seed)
    for _ in range(30):
        W = rng.standard_normal((Xr.shape[1], k))
        ysyn = np.argmax(Xr @ W, axis=1)
        _, c = np.unique(ysyn, return_counts=True)
        if len(c) == k and c.min() >= 15:
            break
    return curvature(X, ysyn, d=d)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    npz = os.path.join(CACHE, "state_L11_ts_percell" + ("_smoke" if smoke else "") + ".npz")
    z = np.load(npz, allow_pickle=True)
    E, cell_type, tissue = z["E"], z["cell_type"], z["tissue"]
    print(f"[load] {E.shape} cells, {len(set(cell_type))} types, {len(set(tissue))} tissues")

    out = {"model": "state_se_L11", "test": "cell_type_curvature",
           "n_cells": int(len(E)), "n_types": int(len(set(cell_type))),
           "tissues": sorted(set(tissue.tolist()))}

    # 1. recovery
    ct = knn_recovery(E, cell_type)
    ti = knn_recovery(E, tissue)
    out["celltype_knn"] = ct
    out["tissue_knn"] = ti
    print(f"  celltype kNN bal_acc={ct['bal_acc']:.3f} (null {ct['null']:.3f}, +{ct['margin']:.3f}) n={ct['n']}")
    print(f"  tissue   kNN bal_acc={ti['bal_acc']:.3f} (null {ti['null']:.3f}, +{ti['margin']:.3f})")

    # 2. curvature (drop ultra-rare types for stable CV; encode labels)
    y = np.asarray(cell_type)
    keep = np.array([np.sum(y == v) >= 25 for v in y])
    Xk = E[keep].astype(np.float64)
    yk = np.unique(y[keep], return_inverse=True)[1]
    cur = curvature(Xk, yk, d=20)
    out["curvature"] = cur
    print(f"  CURVATURE (20D) chance={cur['chance']:.3f} | lin={cur['linear_acc']:.3f} "
          f"quad={cur['quad_acc']:.3f} knn={cur['knn_acc']:.3f} | curv(knn-lin)={cur['curvature_acc']:+.3f} "
          f"quad_gain={cur['quad_gain_acc']:+.3f}")

    # 3. mandatory synthetic linear control
    syn = synthetic_linear_control(Xk, yk, d=20)
    out["synthetic_linear_control"] = syn
    print(f"  SYN linear control    | lin={syn['linear_acc']:.3f} knn={syn['knn_acc']:.3f} "
          f"curv={syn['curvature_acc']:+.3f}  (must be ~0)")

    out["verdict"] = dict(
        celltype_recovered=bool(ct["margin"] > 0.1),
        curvature_acc=cur["curvature_acc"],
        synthetic_control_curvature=syn["curvature_acc"],
        curved=bool(cur["curvature_acc"] > max(0.03, syn["curvature_acc"] + 0.03)),
    )
    tag = "_smoke" if smoke else ""
    json.dump(out, open(os.path.join(RESULTS, f"state_se_celltype_curvature{tag}.json"), "w"), indent=1)
    print(f"\n[done] curved={out['verdict']['curved']} -> results/state_se_celltype_curvature{tag}.json")


if __name__ == "__main__":
    main()
