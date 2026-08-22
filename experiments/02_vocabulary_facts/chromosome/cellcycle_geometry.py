"""IS THE GENE-LEVEL CELL-CYCLE GEOMETRY A CIRCLE, OR AN AXIS? — the diagnostic that explains the steering
negative (Ihor, 2026-07-18).

WHY. `cellcycle_steer.py` found NO phase tracking at any dose (best circ_corr +0.18, random ~0). A negative is
worthless until you show the instrument could have detected a positive, and until you know WHICH link failed.
Three checks, in the order that makes the negative interpretable:

  D1  READOUT VALIDITY (planted signal, no model). Give the readout genes logits peaked at a known phase T and
      ask whether `induced_phase` recovers T. If it cannot, the negative is instrument blindness.
  D2  DOES THE PHASE PLANE ORGANISE HELD-OUT GENES? Build the (u,v) plane from a direction-half of the marker
      genes and check whether the held-out half lands at its annotated angle.
  D3  CIRCLE OR AXIS? — the decisive one. Take the four WAVE group CENTROIDS directly (no Fourier construction,
      no split) and look at their 2-D layout. A circle puts G1S/S/G2/G2M at ~0/90/180/270 deg in cyclic order.
      Two antipodal clusters mean the "cycle" is really one bipolar contrast.

RESULT (see results/cellcycle_geometry.json): D1 passes exactly; D2's circ_corr looks high (+0.85) but is
MISLEADING because D3 shows why — G1S and S collapse onto the same point (5.8 deg apart, not 90), G2/G2M sit
opposite them, and every cross-cluster cosine is -0.5..-0.6. So the gene-level cell-cycle geometry is a
REPLICATION-vs-MITOSIS AXIS, not a traversable circle. You cannot rotate around a cycle whose four stations are
really two. That explains the steering negative AND section 7's persistent-homology null (no irreducible loop
because there is no loop) while remaining consistent with phase being highly DECODABLE (0.929) -- decodability
of a bipolar contrast does not imply a traversable cyclic coordinate.

SCOPE (important). This is the GENE-TOKEN level -- do the marker GENES form a circle by their peak phase, which
is what route_genemanifold section 7 asked. It is NOT the CELL-level loop (route_cellcycle: per-cell phase from
Tirosh marker z-scores -> PCA(2) -> atan2, where CELLS go round a loop). Those are different objects and this
result says nothing about the cell-level one.

Run: ../../.venv/bin/python -u cellcycle_geometry.py
Out: results/cellcycle_geometry.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "route_cellcycle"))
import gm_lib as G
from cc_common import WAVE, WAVE_ORDER, circ_mean, circ_corr, wrap

PHASE = {k: i * (2 * np.pi / len(WAVE_ORDER)) for i, k in enumerate(WAVE_ORDER)}
BASES = ["maxtoki_lmhead", "maxtoki_we", "scgpt_we"]
SEED = 0


def main():
    out = {}

    # ---- D1: planted-rotation readout check (no model involved)
    print("D1 -- READOUT VALIDITY: plant logits peaked at phase T, does induced_phase recover T?")
    phi_r = np.array([PHASE[k] for k in WAVE_ORDER for _ in range(5)])
    d1 = {}
    for T in (0, 90, 180, 270):
        t = np.radians(T)
        lg = 3.0 * np.cos(phi_r - t)
        w = np.exp(lg - lg.max()); w /= w.sum()
        psi = np.arctan2((w * np.sin(phi_r)).sum(), (w * np.cos(phi_r)).sum())
        err = abs(np.degrees(wrap(psi - t)))
        d1[T] = dict(recovered_deg=float(np.degrees(psi)), err_deg=float(err), ok=bool(err < 25))
        print(f"   planted {T:>3}deg -> recovered {np.degrees(psi):>+7.1f}deg  err {err:>4.1f}  "
              f"{'OK' if err < 25 else 'FAIL'}")
    out["D1_readout"] = d1
    print(f"   -> readout is {'VALID (not blind)' if all(v['ok'] for v in d1.values()) else 'BLIND'}\n")

    for basis in BASES:
        M, syms = G.basis(basis)
        pos = {s: i for i, s in enumerate(syms)}
        rng = np.random.default_rng(SEED)
        rec = {}

        # ---- D2: does the (u,v) plane place HELD-OUT genes at their annotated angle?
        dt, dp, rt, rp = [], [], [], []
        for k in WAVE_ORDER:
            g = [x for x in WAVE[k] if x in pos]
            rng.shuffle(g); h = len(g) // 2
            dt += g[:h]; dp += [PHASE[k]] * h
            rt += g[h:]; rp += [PHASE[k]] * (len(g) - h)
        Ed = np.stack([M[pos[g]] for g in dt]); Er = np.stack([M[pos[g]] for g in rt])
        mu = Ed.mean(0); Ec = Ed - mu; phi = np.array(dp)
        u = (np.cos(phi)[:, None] * Ec).mean(0); v = (np.sin(phi)[:, None] * Ec).mean(0)
        uh = u / np.linalg.norm(u); v = v - (v @ uh) * uh; vh = v / np.linalg.norm(v)
        ang = np.arctan2((Er - mu) @ vh, (Er - mu) @ uh)
        rec["D2_plane_circ_corr"] = float(circ_corr(ang, np.array(rp)))
        rec["D2_recovered_deg"] = {k: float(np.degrees(circ_mean(
            ang[[i for i, x in enumerate(rp) if np.isclose(x, PHASE[k])]]))) for k in WAVE_ORDER}

        # ---- D3: circle or axis? the four group centroids, directly
        C = {}
        for k in WAVE_ORDER:
            g = [x for x in WAVE[k] if x in pos]
            if len(g) >= 4:
                C[k] = M[[pos[x] for x in g]].mean(0)
        ks = [k for k in WAVE_ORDER if k in C]
        X = np.stack([C[k] for k in ks]); X = X - X.mean(0)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        P = (U * s)[:, :2]
        a = np.degrees(np.arctan2(P[:, 1], P[:, 0])); a = (a - a[0]) % 360
        ev = (s ** 2 / (s ** 2).sum())
        rec["D3_angles_deg"] = {k: float(x) for k, x in zip(ks, a)}
        rec["D3_cos"] = {f"{ks[i]}-{ks[j]}": float(Xn[i] @ Xn[j])
                         for i in range(len(ks)) for j in range(i + 1, len(ks))}
        rec["D3_var"] = dict(pc1=float(ev[0]), pc2=float(ev[1]))
        dd = lambda p, q: float(np.linalg.norm(C[p] - C[q]))
        rec["D3_dist"] = {"G1S-S": dd("G1S", "S"), "G2-G2M": dd("G2", "G2M"),
                          "S-G2": dd("S", "G2"), "G1S-G2M": dd("G1S", "G2M")}
        within = (rec["D3_dist"]["G1S-S"] + rec["D3_dist"]["G2-G2M"]) / 2
        between = (rec["D3_dist"]["S-G2"] + rec["D3_dist"]["G1S-G2M"]) / 2
        # THE DECISIVE STATISTIC IS THE MINIMUM ANGULAR GAP, NOT THE DISTANCE RATIO.
        # A perfect circle (4 points at 0/90/180/270, radius r) has adjacent dist r*sqrt2 and opposite 2r, i.e.
        # within/between = 0.707 -- so "within < between" is TRUE FOR A CIRCLE TOO and cannot discriminate.
        # What separates them is the spacing: a circle puts consecutive groups ~90 deg apart; a collapsed pair
        # (two phases sharing one point) drives the minimum gap toward 0.
        srt = np.sort(a)
        gaps = np.concatenate([np.diff(srt), [360.0 - (srt[-1] - srt[0])]])
        rec["D3_min_gap_deg"] = float(gaps.min())
        rec["D3_within_between_ratio"] = float(within / between)
        rec["verdict"] = ("AXIS (a phase pair is collapsed)" if gaps.min() < 45
                          else "circle-like (all four phases resolved)")
        print(f"=== {basis} ===")
        print(f"  D2 held-out phase recovery circ_corr = {rec['D2_plane_circ_corr']:+.3f}   "
              f"(recovered: " + ", ".join(f"{k}={v:+.0f}deg" for k, v in rec['D2_recovered_deg'].items()) + ")")
        print(f"  D3 2-D angles rel. G1S: " + "  ".join(f"{k}={v:.1f}deg" for k, v in rec["D3_angles_deg"].items())
              + f"   (a circle would be 0/90/180/270)")
        print(f"  D3 var PC1={ev[0]:.2f} PC2={ev[1]:.2f};  MIN ANGULAR GAP = {gaps.min():.1f}deg "
              f"(circle needs ~90);  within/between {within/between:.2f} (0.71 for a perfect circle)")
        print(f"  -> {rec['verdict']}\n")
        out[basis] = rec

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "cellcycle_geometry.json"), "w"), indent=1)
    print("[done] -> results/cellcycle_geometry.json")


if __name__ == "__main__":
    main()
