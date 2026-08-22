"""External-holdout scoring (handoff §2 Stage-3, §4 step 6; pipeline §6 Phase 7 — decisive gate).

Load the frozen LET heads (trained on INTERNAL only) and re-score trustworthiness + geodesic<->ruler rho on
the external panels with ZERO retraining. Writes results/external_gates_{panel}.json.

Run:  ../.venv/bin/python score_external.py [external_ts|krasnow]
"""
import json
import os
import pickle
import sys
import numpy as np

import common as K
from gates import geodesic_ruler_rho, trust

HEAD_DIR = os.path.join(K.DATA, "let_heads")


def internal_svd_basis():
    """Frozen SVD-50 basis from the INTERNAL scGPT L11 per-cell raw means."""
    raw = np.load(os.path.join(K.DATA, "scgpt_percell_L11.npy"))
    mu = raw.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(raw - mu, full_matrices=False)
    return mu, Vt[:50].T


def external_operators(npz):
    d = np.load(npz, allow_pickle=True)
    mu, V50 = internal_svd_basis()
    ops = dict(
        linear_sae=d["lin_feat"], bilinear=d["bil_feat"], raw_mean=d["raw11"],
        svd50=((d["raw11"] - mu) @ V50).astype(np.float32),
        linear_drift=np.concatenate([d["blk_early"] - d["blk_mid"], d["blk_mid"] - d["blk_late"]], 1),
        bilinear_drift=np.concatenate([d["bd_early"] - d["bd_mid"], d["bd_mid"] - d["bd_late"]], 1),
    )
    ruler = dict(kind="celltype", cell_type=d["cell_type"], tissue=d["tissue"])
    return ops, ruler


def main(panel):
    npz = os.path.join(K.DATA, f"external_{panel}.npz")
    ops, ruler = external_operators(npz)
    n = len(ruler["cell_type"])
    out = {"panel": panel, "n_cells": int(n),
           "tissues": {k: int(v) for k, v in zip(*np.unique(ruler["tissue"], return_counts=True))},
           "n_cell_types": int(len(np.unique(ruler["cell_type"]))), "operators": {}}
    order = ["linear_drift", "linear_sae", "bilinear", "bilinear_drift", "svd50", "raw_mean"]
    for name in order:
        hp = os.path.join(HEAD_DIR, f"{name}.pkl")
        if not os.path.exists(hp) or name not in ops:
            continue
        head = pickle.load(open(hp, "rb"))
        X = ops[name]
        if X.shape[1] != head.W_.shape[1]:
            print(f"  skip {name}: dim {X.shape[1]} != head {head.W_.shape[1]}"); continue
        z = head.transform(X)
        tw = trust(X, z)
        rho = geodesic_ruler_rho(z, ruler, seed=0)
        out["operators"][name] = dict(dim=int(X.shape[1]), trustworthiness=float(tw),
                                      geodesic_ruler_rho=float(rho), pass_trust=bool(tw >= 0.80))
        print(f"  {name:14s} dim={X.shape[1]:5d} tw={tw:.3f} ext_geo_rho={rho:+.3f}"
              f" {'PASS' if tw>=0.80 else 'fail-trust'}", flush=True)
    with open(os.path.join(K.RESULTS, f"external_gates_{panel}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/external_gates_{panel}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "external_ts")
