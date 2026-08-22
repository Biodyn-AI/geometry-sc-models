"""GATE 1 — is the tangent rotation curvature (maturation), or is it just branching?

`theta_far` compares the local pseudotime-ascent direction in pseudotime quintile 1 vs quintile 5. Quintile 5
fans across FOUR terminal branches while quintile 1 is one progenitor blob; the calibrated null is linear-in-X
by construction (true tangent constant everywhere) so it cannot express branch fan-out -> fan-out is scored as
rotation. This restricts the rotation measurement to a SINGLE non-branching lineage (progenitors -> one
terminal fate) and re-runs the exact tangent_rotation statistic + calibrated null.

  - within-branch excess survives  => the ascent direction rotates as a cell MATURES along one path (real curvature)
  - within-branch excess collapses => the rotation was the tree fanning out (branching), not maturation curvature

We compute, per model: full-data theta_far/excess (reproduces tangent_diagnostic), then per-lineage
theta_far/excess against the SAME calibrated null. Lineage = shared progenitors + one terminal's clusters.

Run:  ../../.venv/bin/python gate1_within_branch_rotation.py [model ...]
Out:  results/gate1_within_branch_rotation.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import load, prep, zscore, unit, RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from tangent_diagnostic import (tangent_rotation, null_rotation_stats, cv_linear_r2, sig,  # noqa: E402
                                Ridge)

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MODELS = ["scgptbin", "geneformer", "state", "maxtoki"]

PROGENITORS = ["HSC_1", "HSC_2", "Precursors"]
LINEAGES = {"ery": ["Ery_1", "Ery_2"], "myeloid": ["Mono_1", "Mono_2"],
            "lymphoid": ["CLP"], "mega": ["Mega"]}
MIN_CELLS = 250          # need enough cells for 5 quintiles x local probe
MIN_TERMINAL = 40        # skip a lineage with too few terminal cells (Mega ~ borderline)


def rotation_block(Xz, y, nbins=5):
    """theta_far + calibrated-null excess/z for one (Xz, y). Mirrors tangent_diagnostic.run's H1 core."""
    yz = zscore(y)
    lin_r2 = cv_linear_r2(Xz, yz)
    rot, _, w_glob = tangent_rotation(Xz, yz, nbins=nbins)
    th_n, rh_n, _ = null_rotation_stats(Xz, y, w_glob, lin_r2, nbins=nbins, n_seeds=20)
    ts = sig(rot["theta_far"], th_n)
    rs = sig(rot["monotonicity"], rh_n)
    return dict(n=int(len(y)), lin_r2=float(lin_r2), theta_far=float(rot["theta_far"]),
                monotonicity=float(rot["monotonicity"]), split_half=float(rot["split_half_mean"]),
                null_theta_mean=float(th_n.mean()), null_theta_sd=float(th_n.std()),
                excess=float(rot["theta_far"] - th_n.mean()), z_theta=float(ts["z"]), p_theta=float(ts["p_value"]),
                null_rho_mean=float(rh_n.mean()), z_rho=float(rs["z"]), p_rho=float(rs["p_value"]))


def run_model(model):
    X, y, ci, clu = load(model)
    Xz = prep(X)
    full = rotation_block(Xz, y)
    print(f"\n===== {model}  (n={len(y)}) =====")
    print(f"  FULL       theta={full['theta_far']:5.1f}  null={full['null_theta_mean']:4.1f}"
          f"  excess={full['excess']:+5.1f}  z={full['z_theta']:+5.1f}  p={full['p_theta']:.3f}"
          f" | rho={full['monotonicity']:+.2f} (z={full['z_rho']:+.1f})")
    branches = {}
    for fate, terms in LINEAGES.items():
        keep = np.isin(clu, PROGENITORS + terms)
        n_term = int(np.isin(clu, terms).sum())
        if keep.sum() < MIN_CELLS or n_term < MIN_TERMINAL:
            print(f"  {fate:<9}  [skip] cells={int(keep.sum())} terminal={n_term}")
            continue
        # rebuild the whitened space on the lineage subset ONLY (its own local manifold)
        Xb = prep(X[keep]); yb = y[keep]
        b = rotation_block(Xb, yb)
        b["n_terminal"] = n_term
        branches[fate] = b
        print(f"  {fate:<9}  theta={b['theta_far']:5.1f}  null={b['null_theta_mean']:4.1f}"
              f"  excess={b['excess']:+5.1f}  z={b['z_theta']:+5.1f}  p={b['p_theta']:.3f}"
              f" | rho={b['monotonicity']:+.2f} (z={b['z_rho']:+.1f})  [n={b['n']}, term={n_term}]")
    within = [b["excess"] for b in branches.values()]
    within_z = [b["z_theta"] for b in branches.values()]
    rec = dict(model=model, full=full, branches=branches,
               within_excess_mean=float(np.mean(within)) if within else None,
               within_z_mean=float(np.mean(within_z)) if within_z else None,
               n_branches_excess_pos=int(sum(e > 0 for e in within)),
               n_branches_z_ge2=int(sum(z >= 2 for z in within_z)),
               retention=float(np.mean(within) / full["excess"]) if within and full["excess"] else None)
    if within:
        print(f"  -> within-branch excess mean {np.mean(within):+.1f} vs full {full['excess']:+.1f}"
              f"  (retention {rec['retention']:.0%})  |  {rec['n_branches_z_ge2']}/{len(branches)} branches z>=2")
    return rec


def main():
    models = sys.argv[1:] or MODELS
    out = {}
    for m in models:
        try:
            out[m] = run_model(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[skip] {m}: {e}")
    json.dump(out, open(os.path.join(RESULTS, "gate1_within_branch_rotation.json"), "w"), indent=1)
    print("\n================ GATE 1 VERDICT ================")
    print(f"  {'model':<12} {'full excess':>12} {'within mean':>12} {'retention':>10} {'branches z>=2':>14}")
    for m, r in out.items():
        ret = f"{r['retention']:.0%}" if r["retention"] is not None else "n/a"
        wm = f"{r['within_excess_mean']:+.1f}" if r["within_excess_mean"] is not None else "n/a"
        nb = len(r["branches"])
        print(f"  {m:<12} {r['full']['excess']:>+12.1f} {wm:>12} {ret:>10} {r['n_branches_z_ge2']:>10}/{nb}")
    print("  If within-branch excess ~ full excess and z>=2 in most branches, rotation is maturation curvature.")
    print("  If it collapses toward 0, the rotation was branch fan-out.")
    print("[done] -> results/gate1_within_branch_rotation.json")


if __name__ == "__main__":
    main()

