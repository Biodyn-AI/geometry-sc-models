"""ring_analysis — (1) single-cell RING vs FILLED DISK, (2) does phase survive deleting the cell-cycle genes?

Answers the two things the original manifold write-up asserted without testing:

RING vs DISK. Project every CELL (not the bin centroids — averaging ~250 cells makes any cloud look ring-like)
onto the 2-plane spanned by the fitted (cos, sin) phase decoders, and look at the radial distribution.
  * a RING  -> radius concentrated away from 0; few cells near the centre
  * a DISK  -> density uniform in area, i.e. p(r) ∝ r, so a sizeable fraction of cells sit near the centre
Statistics: radius CV; fraction of cells inside 0.5x the median radius (ring ≈ 0, uniform disk ≈ 0.25);
and the same two numbers for a matched synthetic ring and synthetic disk for calibration.

COMPUTE vs RELAY. Compare phase decodability from states extracted on NORMAL sentences against states from
sentences with all 87 cell-cycle genes DELETED. Phase is defined from those genes' expression, so if the model
is merely reading them, decodability must collapse; if it survives, the model infers cycle state from the rest
of the transcriptome.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle
from manifold_fit import decodability, ridge_cv, circ_corr, circ_dist


def radial_profile(H, t, seed=0):
    """Project cells onto the fitted (cos,sin) phase plane and characterise the radial distribution."""
    Y = np.column_stack([np.cos(t), np.sin(t)])
    P = ridge_cv(H, Y, seed=seed)                       # out-of-fold -> honest projection
    P = P - P.mean(0)
    r = np.linalg.norm(P, axis=1)
    med = np.median(r)
    frac_centre = float((r < 0.5 * med).mean())
    rng = np.random.default_rng(seed)
    n = len(r)
    # calibration: matched synthetic ring (constant radius + gaussian jitter) and uniform disk
    ang = rng.uniform(0, 2 * np.pi, n)
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) * (1 + 0.15 * rng.standard_normal(n))[:, None]
    rr = np.linalg.norm(ring, axis=1)
    dr = np.sqrt(rng.uniform(0, 1, n))                  # p(r) ∝ r  == uniform in area
    return dict(radius_cv=float(r.std() / (r.mean() + 1e-12)),
                frac_inside_half_median=frac_centre,
                calib_ring_radius_cv=float(rr.std() / rr.mean()),
                calib_ring_frac_centre=float((rr < 0.5 * np.median(rr)).mean()),
                calib_disk_radius_cv=float(dr.std() / dr.mean()),
                calib_disk_frac_centre=float((dr < 0.5 * np.median(dr)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", required=True, help="states dir, normal sentences")
    ap.add_argument("--nocc", required=True, help="states dir, cell-cycle genes deleted")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", default="results/ring_analysis.json")
    a = ap.parse_args()
    import anndata, glob
    ad = anndata.read_h5ad(a.h5ad)
    theta, _, _, info = phase_angle(ad)
    res = {"phase_info": info, "layers": {}}
    layers = sorted(int(os.path.basename(p).split("_")[1])
                    for p in glob.glob(os.path.join(a.norm, "layer_*_activations.npy")))
    for L in layers:
        rec = {}
        for tag, d in [("normal", a.norm), ("nocc", a.nocc)]:
            f = os.path.join(d, f"layer_{L:02d}_activations.npy")
            if not os.path.exists(f):
                continue
            H = np.load(f).astype(np.float64)
            ids = np.load(os.path.join(d, "row_cell_ids.npy"))
            t = theta[ids][:len(H)]
            H = H[:len(t)]
            dec = decodability(H, t, "cyclic")
            rec[tag] = dict(circ_corr=dec["value"], median_err_deg=dec["median_abs_err_deg"], z=dec["z"])
            if tag == "normal":
                rec["radial"] = radial_profile(H, t)
        if "normal" in rec and "nocc" in rec:
            rec["retention"] = float(rec["nocc"]["circ_corr"] / (rec["normal"]["circ_corr"] + 1e-12))
        res["layers"][f"L{L:02d}"] = rec
        n, c = rec.get("normal", {}), rec.get("nocc", {})
        print(f"  L{L:02d}: normal circ={n.get('circ_corr', float('nan')):+.3f} ({n.get('median_abs_err_deg', n.get('median_err_deg', float('nan'))):.0f}deg)"
              f" | cc-genes-DELETED circ={c.get('circ_corr', float('nan')):+.3f} ({c.get('median_err_deg', float('nan')):.0f}deg)"
              f" | retention {rec.get('retention', float('nan')):.2f}", flush=True)
        if "radial" in rec:
            r = rec["radial"]
            print(f"        radial: cv={r['radius_cv']:.2f} frac_centre={r['frac_inside_half_median']:.3f}"
                  f"  [ring calib cv={r['calib_ring_radius_cv']:.2f}/{r['calib_ring_frac_centre']:.3f}"
                  f" | disk calib cv={r['calib_disk_radius_cv']:.2f}/{r['calib_disk_frac_centre']:.3f}]", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
