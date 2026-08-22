"""Cross-model CELL-TOKEN manifold comparison on Setty blood -- the test route_uce could not run.

route_uce/manifold_celltoken.py compared UCE's shipped cell embedding (X_uce) against UCE's OWN
gene-mean-pool and layer-2 CLS, and found it far more compact (PR 27.7 -> 8.6, d90 >200 -> 40) while
*gaining* pseudotime decodability. That is a within-UCE statement: no other model's cell token had
ever been extracted (their npz files carry `emb` only). This script runs the same battery across all
five models, so "is UCE's cell manifold special?" can actually be answered.

Reps per model:
  emb        gene-mean-pool at the model's cross-model tap layer  (the existing comparison default)
  cls        the model's cell token at the SAME tap layer          (matched depth)
  cls_final  the model's cell token at its FINAL layer             (its shipped cell embedding)
For UCE, `xuce` (CLS -> decoder -> L2-normalize) IS the shipped embedding and plays the cls_final role.

Fairness controls:
  * All models scored on the SAME cells (intersection of cell_idx across models).
  * Identical statistics to route_uce (global_shape / decode_pseudotime / tangent_rotation reused
    read-only from route_branchpoint + route_uce).
  * scGPT uses the BINNED input convention. The raw-counts cache is off-distribution and collapses
    onto one axis (PR 1.5, PC1 81%) -- a bug artifact, not scGPT's geometry.

Out: results/celltoken_crossmodel.json
Run: ../../.venv/bin/python compare_celltoken.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_branchpoint"))
sys.path.insert(0, os.path.join(HERE, "..", "route_uce"))
from branchpoint_geometry import decode_pseudotime, synth_linear_trajectory, twonn_id, pca_reduce  # noqa
from manifold_celltoken import global_shape, tangent_rotation  # noqa

ROOT = f"{_DATA}"
CT = f"{ROOT}/data/celltoken"
OUT = os.path.join(HERE, "results", "celltoken_crossmodel.json")

# model -> {rep: (npz_path, key)}
MODELS = {
    "scgpt": {"emb": (f"{CT}/scgptbin_setty.npz", "emb"),
              "cls": (f"{CT}/scgptbin_setty.npz", "cls")},          # 12 layers: tap 11 == final
    "geneformer": {"emb": (f"{CT}/geneformer_setty.npz", "emb"),
                   "cls": (f"{CT}/geneformer_setty.npz", "cls"),
                   "cls_final": (f"{CT}/geneformer_setty.npz", "cls_final")},
    "state": {"emb": (f"{CT}/state_setty.npz", "emb"),
              "cls": (f"{CT}/state_setty.npz", "cls"),
              "cls_final": (f"{CT}/state_setty.npz", "cls_final")},
    "maxtoki": {"emb": (f"{CT}/maxtoki_setty.npz", "emb"),
                "cls": (f"{CT}/maxtoki_setty.npz", "cls"),
                "cls_final": (f"{CT}/maxtoki_setty.npz", "cls_final")},
    "uce": {"emb": (f"{ROOT}/data/branchpoint/uce_setty.npz", "emb"),
            "cls": (f"{ROOT}/data/branchpoint/uce_setty.npz", "cls"),
            "cls_final": (f"{ROOT}/data/xuce/uce_setty.npz", "xuce")},
}


def common_cells():
    """Intersection of cell_idx across every model file that exists."""
    sets, present = [], []
    for m, reps in MODELS.items():
        p = list(reps.values())[0][0]
        if not os.path.exists(p):
            print(f"[missing] {m}: {p}")
            continue
        z = np.load(p, allow_pickle=True)
        sets.append(set(int(i) for i in z["cell_idx"]))
        present.append(m)
    if not sets:
        raise SystemExit("no model files found")
    common = set.intersection(*sets)
    per_model = {m: len(s) for m, s in zip(present, sets)}
    print(f"[cells] per-model: {per_model} | common = {len(common)}")
    # Every model file here is written incrementally (checkpoint/resume), so a crashed or still-running
    # extraction leaves a VALID but PARTIAL npz. Intersecting silently drags the whole comparison down
    # to that model's cell count -- a weak table that still looks finished. Refuse instead.
    # Absolute floor, not a fraction of the largest: by design scGPT/UCE hold 5780 cells and the slower
    # models 3000, so "fraction of largest" would sit ~110 cells under the expected value and false-abort.
    floor = int(os.environ.get("CT_MIN_CELLS", "2000"))
    if len(common) < floor and not os.environ.get("CT_ALLOW_PARTIAL"):
        stragglers = {m: n for m, n in per_model.items() if n <= len(common)}
        raise SystemExit(
            f"[abort] common cells ({len(common)}) < floor ({floor}). "
            f"Likely a partial/in-flight extraction: {stragglers}. "
            f"Re-run once extraction completes, or set CT_ALLOW_PARTIAL=1 / CT_MIN_CELLS to override.")
    print(f"[cells] models present: {present} | common cells = {len(common)}")
    return np.array(sorted(common)), present


def load(model, rep, cells):
    p, key = MODELS[model][rep]
    if not os.path.exists(p):
        return None, None
    z = np.load(p, allow_pickle=True)
    if key not in z.files:
        return None, None
    idx = np.array([int(i) for i in z["cell_idx"]])
    pos = {c: i for i, c in enumerate(idx)}
    sel = np.array([pos[c] for c in cells if c in pos])
    if len(sel) < len(cells):
        return None, None
    return z[key][sel].astype(np.float64), z["pseudotime"][sel].astype(float)


def main():
    cells, present = common_cells()
    res = {}
    for m in present:
        for rep in MODELS[m]:
            X, y = load(m, rep, cells)
            if X is None:
                print(f"[skip] {m}/{rep}")
                continue
            ok = np.isfinite(y)
            X, y = X[ok], y[ok]
            rec = dict(n=int(len(X)), dim=int(X.shape[1]), **global_shape(X))
            dp = decode_pseudotime(X, y, 20)
            sc = synth_linear_trajectory(X, y, 20)
            rec.update(linear_r2=dp["linear_r2"], knn_r2=dp["knn_r2"], curvature=dp["curvature"],
                       bilinear_gain=dp["bilinear_gain"], synth_curvature=sc["curvature"],
                       tangent_rotation_deg=tangent_rotation(X, y))
            res[f"{m}/{rep}"] = rec
            print(f"  [{m:11s} {rep:9s}] dim={rec['dim']:5d} PR={rec['participation_ratio']:6.1f} "
                  f"d90={rec['pca_dim_90']:4d} TwoNN={rec['twonn_id']:5.1f} | "
                  f"linR2={rec['linear_r2']:+.3f} curv={rec['curvature']:+.3f} "
                  f"rot={rec['tangent_rotation_deg'] or float('nan'):.0f}deg", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"dataset": "setty", "n_common_cells": int(len(cells)), "results": res},
              open(OUT, "w"), indent=1)
    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
