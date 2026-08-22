"""SPECIFICITY vs DETECTION -- the confound-controlled rerun.

WHAT WENT WRONG IN specificity_mechanism.py. It split genes by cell-type specificity (Tau) matched on set size
and chromosome composition, and found LOW-specificity genes carry the chromosome signal far more strongly
(k=12: 0.585 vs 0.231) -- the OPPOSITE of the hypothesis that genomically-clustered cell-type-specific programs
manufacture it. But Tau and detection rate are confounded by construction: a gene confined to one cell type is
by definition detected in few cells. Measured: LOW-spec genes are detected 9.8x more often than HIGH-spec
(mean 0.263 vs 0.027; HIGH-spec median detection is 0.9% of cells), and corr(Tau, detection) = -0.658. A gene
observed in 0.9% of cells contributes almost nothing to a co-occurrence matrix, so the earlier contrast may
say only "genes you observe more are decoded better" -- which is not a mechanism.

THE FIX. Stratify on detection rate and compare Tau WITHIN strata, so the two sets have matched detection
distributions by construction. Strata = 5 detection quintiles x 22 chromosomes; within each cell take equal
numbers of the highest- and lowest-Tau genes. This holds detection AND chromosome composition fixed and varies
only specificity. Reported alongside: the realised detection distributions, so the match is auditable rather
than asserted.

INTERPRETATION EITHER WAY.
  HIGH still loses at matched detection  -> the carrier is BROADLY-EXPRESSED genes being co-modulated, i.e. a
      regional/domain effect on constitutive genes, NOT cell-type on/off programs. The original hypothesis dies
      properly rather than by confound.
  The gap closes at matched detection    -> the earlier result was the detection confound, and specificity is
      not the relevant axis at all.
  HIGH now wins                          -> the original cell-type-program hypothesis is rescued.

Out: results/specificity_matched.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from shallow_coocc_baseline import lsa
from purity_decomposition import decode, build_from_rows, N_TOTAL, DIMS
from specificity_mechanism import tau_specificity, FETAL

SEED = 0
N_QUINT = 5


def detection_matched_split(tau, det, y_chr, ok, n_quint=N_QUINT):
    """HIGH vs LOW Tau, matched on detection quintile AND chromosome."""
    hi, lo = [], []
    valid = ok & (det > 0)
    edges = np.quantile(det[valid], np.linspace(0, 1, n_quint + 1)[1:-1])
    dq = np.digitize(det, edges)
    for c in sorted(set(y_chr)):
        for q in range(n_quint):
            idx = np.nonzero((y_chr == c) & valid & (dq == q))[0]
            if len(idx) < 8:
                continue
            order = idx[np.argsort(tau[idx])]
            k = len(order) // 3
            if k == 0:
                continue
            lo.extend(order[:k].tolist()); hi.extend(order[-k:].tolist())
    return np.array(sorted(hi)), np.array(sorted(lo))


def main():
    C = coords()
    _, sd0 = G.basis("coexpr_devel")
    sd = [s for s in sd0 if s in C.index and C.chromosome[s] in AUTOSOMES]
    y_chr = np.array([C.chromosome[s] for s in sd])
    st = C.loc[list(sd), "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y_chr, st)])

    with h5py.File(FETAL, "r") as f:
        ct = f["obs"]["cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ct["categories"][:]]
        codes = ct["codes"][:]

    print("[tau] scoring cell-type specificity ...", flush=True)
    tau, ok, used = tau_specificity(FETAL, sd, codes, cats)

    rng = np.random.default_rng(SEED)
    usable = [i for i, c in enumerate(cats) if (codes == i).sum() >= N_TOTAL // 12]
    pick = list(rng.choice(usable, size=min(12, len(usable)), replace=False))
    sel = np.concatenate([rng.choice(np.nonzero(codes == t)[0],
                                     size=min(N_TOTAL // len(pick), (codes == t).sum()), replace=False)
                          for t in pick])
    B_all = build_from_rows(FETAL, sd, sel)
    det = np.asarray((B_all > 0).mean(0)).ravel()

    hi, lo = detection_matched_split(tau, det, y_chr, ok)
    n = min(len(hi), len(lo))
    hi, lo = hi[:n], lo[:n]
    print(f"\n[matched sets] {n} genes each")
    print(f"  Tau        HIGH {tau[hi].mean():.3f}  LOW {tau[lo].mean():.3f}   (the axis being varied)")
    print(f"  detection  HIGH {det[hi].mean():.4f}  LOW {det[lo].mean():.4f}   "
          f"(ratio {det[lo].mean()/max(det[hi].mean(),1e-9):.2f}x -- was 9.8x before matching)")
    print(f"  median det HIGH {np.median(det[hi]):.4f}  LOW {np.median(det[lo]):.4f}", flush=True)

    res = {"n_per_set": int(n), "tau_hi": float(tau[hi].mean()), "tau_lo": float(tau[lo].mean()),
           "det_hi": float(det[hi].mean()), "det_lo": float(det[lo].mean()),
           "det_ratio_after_matching": float(det[lo].mean() / max(det[hi].mean(), 1e-9)),
           "chance": 1 / 22, "cell_types_used": used, "runs": {}}

    print(f"\n{'gene set':<12} {'n genes':<9} {'chromosome':<13} {'mean detection'}")
    print("-" * 52)
    for gname, gi in [("HIGH-spec", hi), ("LOW-spec", lo)]:
        sub = B_all[:, gi]
        E = lsa(sub, dims=min(DIMS, sub.shape[1] - 1))
        a = decode(E, y_chr[gi], groups[gi])
        res["runs"][gname] = {"chromosome": a, "n_genes": int(len(gi)), "detection": float(det[gi].mean())}
        print(f"{gname:<12} {len(gi):<9} {a:<13.3f} {det[gi].mean():.4f}", flush=True)
        del E, sub; gc.collect()

    json.dump(res, open(os.path.join(HERE, "results", "specificity_matched.json"), "w"), indent=1)
    dh, dl = res["runs"]["HIGH-spec"]["chromosome"], res["runs"]["LOW-spec"]["chromosome"]
    print("\n=== VERDICT (detection-matched) ===")
    print(f"  HIGH-spec {dh:.3f} vs LOW-spec {dl:.3f}   gap {dl-dh:+.3f}   (unmatched gap was +0.354)")
    if abs(dl - dh) < 0.06:
        print("  -> the earlier contrast was THE DETECTION CONFOUND. Specificity is not the relevant axis.")
    elif dl > dh:
        print("  -> BROADLY-EXPRESSED genes genuinely carry it, at matched detection: a regional/domain effect")
        print("     on constitutive genes, NOT cell-type on/off programs. Original hypothesis properly dead.")
    else:
        print("  -> cell-type-specific genes carry it once detection is controlled: hypothesis rescued.")
    print("\n[done] -> results/specificity_matched.json")


if __name__ == "__main__":
    main()
