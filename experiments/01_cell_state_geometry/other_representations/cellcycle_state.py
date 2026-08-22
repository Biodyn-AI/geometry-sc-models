"""STATE-SE cell-cycle geometry/curvature test (the K562 test every other model got).

Per-cell mean-pooled L11 residual (from the cached Route-B STATE activations) as the embedding; Tirosh
S/G2M cell-cycle phase from the SAME cells' expression as the target angle; linear vs bilinear vs kNN
decode of phase in a matched 20D PCA space (curvature = kNN - linear). MANDATORY synthetic linearly-embedded
control. All analysis methods reused READ-ONLY from route_topology/cellcycle_loop.py (via common.py).

Run:  ../.venv_state/bin/python cellcycle_state.py    (CPU)
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os
import numpy as np
import anndata as ad
import common as K

DATA = f"{_DATA}"
ACT = os.path.join(DATA, "state_activations", "state_L11")
SUBSET = os.path.join(DATA, "state_activations", "replogle_k562_subset.h5ad")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def per_cell_mean(chunk=400000):
    """Stream the cached per-gene L11 residual tokens -> per-cell mean-pooled embedding (n_cells, 2048)."""
    cid = np.load(os.path.join(ACT, "layer_11_cell_ids.npy")).astype(np.int64)
    acts = np.load(os.path.join(ACT, "layer_11_activations.npy"), mmap_mode="r")   # (n_tok, 2048) f16
    n_cells = int(cid.max()) + 1
    d = acts.shape[1]
    s = np.zeros((n_cells, d), np.float64); c = np.zeros(n_cells, np.float64)
    for a in range(0, len(cid), chunk):
        b = min(a + chunk, len(cid))
        np.add.at(s, cid[a:b], np.asarray(acts[a:b], dtype=np.float64))
        np.add.at(c, cid[a:b], 1.0)
    return (s / np.clip(c, 1, None)[:, None]).astype(np.float32), n_cells


def main():
    cache = os.path.join(K.CACHE, "state_L11_percell_raw.npz")
    if os.path.exists(cache):
        E = np.load(cache)["raw_mean"]
        print(f"[cache] loaded per-cell raw_mean {E.shape}")
    else:
        E, n = per_cell_mean()
        np.savez(cache, raw_mean=E)
        print(f"[extract] per-cell mean-pooled L11 residual {E.shape} ({n} cells) -> cached")

    a = ad.read_h5ad(SUBSET, backed="r")
    X = np.asarray(a.X[: len(E)].todense()) if hasattr(a.X, "todense") else np.asarray(a.X[: len(E)])
    var = a.var_names.values
    pert = np.load(os.path.join(ACT, "cell_perturbation.npy"), allow_pickle=True)
    ctrl = (pert == "non-targeting")

    phi_all, nmark = K.phase_from_expression(X, var)
    print(f"[phase] {len(phi_all)} cells, {nmark} S+G2M markers, "
          f"circular spread={np.std(np.angle(np.exp(1j*phi_all))):.2f} rad")

    out = {"model": "state_se_L11", "test": "cell_cycle_curvature",
           "n_cells": int(len(E)), "n_ctrl": int(ctrl.sum()), "n_markers": int(nmark), "reps": {}}

    # --- primary: all 1200 cells ---
    rep = K.curvature_report(E, phi_all)
    out["reps"]["raw_mean_all"] = rep
    print(f"  raw_mean(all 1200)   | lin(full)={rep['linear_full_circ_r2']:+.3f} bil={rep['bilinear_circ_r2']:+.3f}"
          f" | 20D lin={rep['linear_20d']:+.3f} knn={rep['knn_20d']:+.3f} curv={rep['curvature']:+.3f}"
          f" | align rho={rep['loop_align_rho']:.3f}")

    # --- robustness: control cells only (matches other models' control-cell phase) ---
    # reused decode_phase does np.mean over per-fold arrays -> needs n % 5 == 0 (folds equal length)
    cc = np.where(ctrl)[0]; cc = cc[: len(cc) // 5 * 5]
    phi_c, _ = K.phase_from_expression(X[cc], var)
    repc = K.curvature_report(E[cc], phi_c)
    out["reps"]["raw_mean_ctrl"] = repc
    print(f"  raw_mean(ctrl {int(ctrl.sum())})   | lin(full)={repc['linear_full_circ_r2']:+.3f}"
          f" bil={repc['bilinear_circ_r2']:+.3f} | 20D lin={repc['linear_20d']:+.3f} knn={repc['knn_20d']:+.3f}"
          f" curv={repc['curvature']:+.3f}")

    # --- MANDATORY synthetic linearly-embedded control (matched linear-decodability) ---
    target = max(0.05, rep["linear_20d"])
    E_syn = K.synthetic_linear_control(phi_all, d=E.shape[1], noise_target_r2=target)
    reps_syn = K.curvature_report(E_syn, phi_all)
    out["reps"]["synthetic_linear_control"] = reps_syn
    print(f"  SYN linear-embedded  | 20D lin={reps_syn['linear_20d']:+.3f} knn={reps_syn['knn_20d']:+.3f}"
          f" curv={reps_syn['curvature']:+.3f}  (must be ~0)")

    out["verdict"] = dict(
        curvature_all=rep["curvature"], curvature_ctrl=repc["curvature"],
        synthetic_control_curvature=reps_syn["curvature"],
        curved=bool(rep["curvature"] > max(0.02, reps_syn["curvature"] + 0.02)),
    )
    json.dump(out, open(os.path.join(RESULTS, "state_se_cellcycle_curvature.json"), "w"), indent=1)
    print(f"\n[done] curved={out['verdict']['curved']}  -> results/state_se_cellcycle_curvature.json")


if __name__ == "__main__":
    main()
