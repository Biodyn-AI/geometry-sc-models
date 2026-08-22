"""UCE cell-cycle geometry/curvature test (the K562 test every other model got).

Per-cell mean-pooled layer-2 UCE residual (from extract_percell.py on replogle_k562_subset) as the
embedding; Tirosh S/G2M cell-cycle phase from the SAME cells' expression as the target angle; linear vs
bilinear vs kNN decode of phase in a matched 20D PCA space (curvature = kNN - linear), with the mandatory
synthetic linearly-embedded control. Analysis methods reused READ-ONLY from route_state_geometry/common.py
(which in turn reuses route_topology/cellcycle_loop.py) — identical machinery to the STATE cell-cycle test.

Run:  ../../.venv_state/bin/python cellcycle_uce.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os, sys
import numpy as np
import anndata as ad
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_state_geometry"))
import common as K  # noqa: E402

DATA = f"{_DATA}"
NPZ = os.path.join(DATA, "cellcycle", "uce_k562.npz")
SUBSET = os.path.join(DATA, "state_activations", "replogle_k562_subset.h5ad")
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)


def main():
    z = np.load(NPZ, allow_pickle=True)
    E = z["emb"].astype(np.float64)
    cell_idx = z["cell_idx"].astype(int)
    print(f"[load] UCE per-cell {E.shape} ({len(cell_idx)} K562 cells)")

    a = ad.read_h5ad(SUBSET, backed="r")
    Xsub = a.X[cell_idx]
    X = np.asarray(Xsub.todense()) if sp.issparse(Xsub) else np.asarray(Xsub)
    var = a.var_names.values
    pert = a.obs["gene"].astype(str).values[cell_idx] if "gene" in a.obs else None

    phi_all, nmark = K.phase_from_expression(X, var)
    print(f"[phase] {len(phi_all)} cells, {nmark} S+G2M markers, "
          f"spread={np.std(np.angle(np.exp(1j*phi_all))):.2f} rad")

    out = {"model": "uce_100m_L2", "test": "cell_cycle_curvature",
           "n_cells": int(len(E)), "n_markers": int(nmark), "reps": {}}

    # trim to a multiple of 5 (reused decode_phase averages equal-length folds)
    n5 = len(E) // 5 * 5
    rep = K.curvature_report(E[:n5], phi_all[:n5])
    out["reps"]["raw_mean_all"] = rep
    print(f"  raw_mean(all {n5}) | lin(full)={rep['linear_full_circ_r2']:+.3f} bil={rep['bilinear_circ_r2']:+.3f}"
          f" | 20D lin={rep['linear_20d']:+.3f} knn={rep['knn_20d']:+.3f} curv={rep['curvature']:+.3f}"
          f" | align rho={rep['loop_align_rho']:.3f}")

    if pert is not None:
        ctrl = (pert == "non-targeting")
        cc = np.where(ctrl)[0]; cc = cc[: len(cc) // 5 * 5]
        if len(cc) >= 100:
            phi_c, _ = K.phase_from_expression(X[cc], var)
            repc = K.curvature_report(E[cc], phi_c)
            out["reps"]["raw_mean_ctrl"] = repc
            print(f"  raw_mean(ctrl {len(cc)}) | 20D lin={repc['linear_20d']:+.3f} knn={repc['knn_20d']:+.3f}"
                  f" curv={repc['curvature']:+.3f}")

    target = max(0.05, rep["linear_20d"])
    E_syn = K.synthetic_linear_control(phi_all[:n5], d=E.shape[1], noise_target_r2=target)
    reps_syn = K.curvature_report(E_syn, phi_all[:n5])
    out["reps"]["synthetic_linear_control"] = reps_syn
    print(f"  SYN linear-embedded | 20D lin={reps_syn['linear_20d']:+.3f} knn={reps_syn['knn_20d']:+.3f}"
          f" curv={reps_syn['curvature']:+.3f}  (must be ~0)")

    out["verdict"] = dict(
        curvature_all=rep["curvature"],
        synthetic_control_curvature=reps_syn["curvature"],
        curved=bool(rep["curvature"] > max(0.02, reps_syn["curvature"] + 0.02)),
    )
    json.dump(out, open(os.path.join(RESULTS, "uce_cellcycle_curvature.json"), "w"), indent=1)
    print(f"\n[done] curved={out['verdict']['curved']} -> results/uce_cellcycle_curvature.json")


if __name__ == "__main__":
    main()
