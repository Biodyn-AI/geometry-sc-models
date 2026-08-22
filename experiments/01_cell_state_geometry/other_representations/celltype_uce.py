"""UCE cell-type geometry/curvature test (the Tabula Sapiens test scGPT/STATE got).

Per-cell mean-pooled layer-2 UCE residual (extract_percell.py on a TS tissue) as the embedding; then
  1. kNN cell-type recovery (balanced accuracy) vs shuffled-label null,
  2. curvature = kNN - linear classification accuracy in matched 20D PCA (route_lineage.curvature READ-ONLY),
  3. mandatory synthetic linearly-embedded control (labels = argmax(linear) of PCA features).
Identical machinery to route_state_geometry/celltype_state.py, on UCE embeddings.

Run:  ../../.venv_state/bin/python celltype_uce.py  [tissue=immune]
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
sys.path.insert(0, os.path.join(HERE, "..", "route_state_geometry"))
from lineage_manifold import curvature, pca_reduce  # READ-ONLY reuse  # noqa: E402
from celltype_state import knn_recovery, synthetic_linear_control  # READ-ONLY reuse  # noqa: E402

DATA = f"{_DATA}"
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)


def main():
    tissue = sys.argv[1] if len(sys.argv) > 1 else "immune"
    npz = os.path.join(DATA, "celltype", f"uce_ts_{tissue}.npz")
    z = np.load(npz, allow_pickle=True)
    E = z["emb"].astype(np.float64)
    cell_type = z["cell_type"] if "cell_type" in z.files else z["cell_ontology_class"]
    print(f"[load] UCE per-cell {E.shape}, {len(set(cell_type))} types (tissue={tissue})")

    out = {"model": "uce_100m_L2", "test": "cell_type_curvature", "tissue": tissue,
           "n_cells": int(len(E)), "n_types": int(len(set(cell_type)))}

    ct = knn_recovery(E, cell_type)
    out["celltype_knn"] = ct
    print(f"  celltype kNN bal_acc={ct['bal_acc']:.3f} (null {ct['null']:.3f}, +{ct['margin']:.3f}) n={ct['n']}")

    y = np.asarray(cell_type)
    keep = np.array([np.sum(y == v) >= 25 for v in y])
    Xk = E[keep].astype(np.float64)
    yk = np.unique(y[keep], return_inverse=True)[1]
    cur = curvature(Xk, yk, d=20)
    out["curvature"] = cur
    print(f"  CURVATURE(20D) chance={cur['chance']:.3f} | lin={cur['linear_acc']:.3f} "
          f"quad={cur['quad_acc']:.3f} knn={cur['knn_acc']:.3f} | curv={cur['curvature_acc']:+.3f}")

    syn = synthetic_linear_control(Xk, yk, d=20)
    out["synthetic_linear_control"] = syn
    print(f"  SYN linear control | lin={syn['linear_acc']:.3f} knn={syn['knn_acc']:.3f} "
          f"curv={syn['curvature_acc']:+.3f}  (must be ~0)")

    out["verdict"] = dict(
        celltype_recovered=bool(ct["margin"] > 0.1),
        curvature_acc=cur["curvature_acc"],
        synthetic_control_curvature=syn["curvature_acc"],
        curved=bool(cur["curvature_acc"] > max(0.03, syn["curvature_acc"] + 0.03)),
    )
    json.dump(out, open(os.path.join(RESULTS, f"uce_celltype_curvature_{tissue}.json"), "w"), indent=1)
    print(f"\n[done] curved={out['verdict']['curved']} -> results/uce_celltype_curvature_{tissue}.json")


if __name__ == "__main__":
    main()
