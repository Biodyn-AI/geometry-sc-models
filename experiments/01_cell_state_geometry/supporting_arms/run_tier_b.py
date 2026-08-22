"""Tier-B internal sweep: operators x LET x gate stack (handoff §4 step 6).

For each operator branch: fit the LET adaptor on the INTERNAL panel (multi-seed), then score the four gates:
  trustworthiness, geodesic<->ruler rho, blocked-permutation null, matched-null branch.
Saves frozen LET heads (for the external holdout) + writes results/internal_gates_{model}.json and a table.

Run:  ../.venv/bin/python run_tier_b.py [scgpt|geneformer]
"""
import json
import os
import pickle
import sys
import numpy as np

import common as K
from operators import build_internal
from let import LETHead
from gates import geodesic_ruler_rho, blocked_permutation, trust, matched_null_ruler

SEEDS = [0, 1, 2, 3, 4]
HEAD_DIR = os.path.join(K.DATA, "let_heads")
os.makedirs(HEAD_DIR, exist_ok=True)


def score_operator(name, X, ruler, save_head=True):
    rhos, tws = [], []
    heads = []
    for sd in SEEDS:
        h = LETHead(d_latent=10, epochs=350, seed=sd).fit(X, ruler)
        z = h.transform(X)
        tws.append(trust(X, z))
        rhos.append(geodesic_ruler_rho(z, ruler, seed=sd))
        heads.append(h)
    best = int(np.argmax(rhos))                      # freeze the best-seed head for external transfer
    if save_head:
        with open(os.path.join(HEAD_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(heads[best], f)
    z_best = heads[best].transform(X)
    bp = blocked_permutation(z_best, ruler, n_perm=200, seed=0)
    # matched-null branch: refit LET to a nonsensical target; must FAIL
    mn = matched_null_ruler(ruler)
    zmn = LETHead(d_latent=10, epochs=350, seed=0).fit(X, mn).transform(X)
    mn_rho = geodesic_ruler_rho(zmn, mn, seed=0)
    return dict(dim=int(X.shape[1]),
                trustworthiness=float(np.mean(tws)), trustworthiness_sd=float(np.std(tws)),
                geodesic_ruler_rho=float(np.mean(rhos)), geodesic_ruler_rho_sd=float(np.std(rhos)),
                blocked_perm=bp, matched_null_rho=float(mn_rho),
                pass_trust=bool(np.mean(tws) >= 0.80),
                pass_null=bool(bp["p"] <= 0.01))


def main(model_key):
    ruler = K.load_ruler(model_key)
    ops = build_internal(model_key, with_drift=True)
    out = {"model": model_key, "ruler": ruler["kind"], "n_cells": int(len(next(iter(ops.values())))),
           "seeds": SEEDS, "operators": {}}
    order = ["linear_drift", "linear_sae", "bilinear", "bilinear_drift", "svd50", "raw_mean"]
    for name in order:
        if name not in ops:
            continue
        print(f"\n== {model_key} :: {name} ==", flush=True)
        res = score_operator(name, ops[name], ruler)
        out["operators"][name] = res
        bp = res["blocked_perm"]
        print(f"  dim={res['dim']:5d} tw={res['trustworthiness']:.3f} geo_rho={res['geodesic_ruler_rho']:+.3f}"
              f"+-{res['geodesic_ruler_rho_sd']:.3f} | null_margin={bp['margin']:+.3f} p={bp['p']:.4f}"
              f" | matched_null_rho={res['matched_null_rho']:+.3f}"
              f" | {'PASS' if res['pass_trust'] and res['pass_null'] else 'partial'}", flush=True)
    with open(os.path.join(K.RESULTS, f"internal_gates_{model_key}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote results/internal_gates_{model_key}.json")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "scgpt"
    main("scgpt_L11" if which == "scgpt" else "geneformer_L11")
