"""WHICH GENES MANUFACTURE THE CHROMOSOME SIGNAL? -- testing the cell-type-program mechanism.

WHERE THIS COMES FROM. purity_decomposition.py established that cell-type diversity SPECIFICALLY raises
chromosome decodability (0.284 at k=1 -> 0.524 at k=12 cell types, fixed N=6,000 cells, while a matched-difficulty
ESM2-cluster control stayed flat at 0.077 -> 0.103). Two facts there refuse the simple "diversity" reading:
a single BROAD cell type already gets most of the way (fibroblast alone 0.433) while a single NARROW one gets
almost nothing (erythroblast 0.093), and pooling two narrow purified panels barely helps (0.064/0.078 -> 0.097).
A depth artifact is excluded: lung airway has 96.7% gene coverage and 1,997 genes/cell against fetal gut's 100%
and 1,930, and still scores 0.078 against 0.720.

THE HYPOTHESIS THAT FITS ALL OF IT. The signal is carried by CELL-TYPE-SPECIFIC PROGRAMS THAT ARE GENOMICALLY
CLUSTERED. Genes that switch on and off together between cell states, and that happen to sit near one another,
co-vary across a mixed population and so make same-chromosome genes look co-expressed. Ubiquitous housekeeping
genes, being always on, contribute nothing regardless of how many cell types are present. That predicts the
effect is a property of a GENE CLASS, not of the population per se -- which is directly testable.

THE TEST. Score every gene for cell-type specificity by the Tau index over fetal-gut cell types
(tau = sum_i (1 - x_i/x_max) / (n-1); 0 = uniformly expressed, 1 = confined to one cell type). Split into a
HIGH-specificity and a LOW-specificity set, MATCHED on set size and on chromosome composition so neither the
class-balance nor the number of genes can drive the comparison. Build the co-occurrence factorisation SEPARATELY
within each gene set -- so each set must manufacture its own structure -- and decode chromosome, at low and high
cell-type diversity.

  PREDICTED IF THE HYPOTHESIS HOLDS: high-specificity genes carry a large diversity effect; low-specificity genes
  show little decodability and little diversity effect.
  KILLED IF: both sets behave alike, or the low-specificity set carries it -- then the mechanism is something
  else (gene density, chromatin domain, replication timing) and the cell-type-program story is wrong.

Matched-difficulty control retained throughout: the same ESM2 k-means cluster target, which must stay flat.

Out: results/specificity_mechanism.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from shallow_coocc_baseline import lsa
from coocc_strongest import symbols_of
from purity_decomposition import decode, decode_random_split, build_from_rows, N_TOTAL, DIMS
from sklearn.cluster import KMeans

SEED = 0
DATA = f"{_DATA}"
FETAL = f"{DATA}/pancreas/fetal_gut.h5ad"
BROAD = "fibroblast"          # the k=1 type that already reached 0.433
NARROW = "erythroblast"       # the k=1 type that reached 0.093


def tau_specificity(path, want, codes, cats, min_cells=200):
    """Per-gene Tau index of detection rate across cell types. 0 = ubiquitous, 1 = one-cell-type-specific."""
    use = [i for i in range(len(cats)) if (codes == i).sum() >= min_cells]
    prof = np.zeros((len(want), len(use)), np.float32)
    for j, t in enumerate(use):
        rows = np.nonzero(codes == t)[0]
        if len(rows) > 1200:
            rows = np.random.default_rng(SEED).choice(rows, 1200, replace=False)
        B = build_from_rows(path, want, rows)
        prof[:, j] = np.asarray((B > 0).mean(0)).ravel()
        del B; gc.collect()
    mx = prof.max(1)
    ok = mx > 0
    tau = np.zeros(len(want), np.float32)
    tau[ok] = ((1.0 - prof[ok] / mx[ok, None]).sum(1)) / (len(use) - 1)
    return tau, ok, [cats[t] for t in use]


def matched_split(tau, ok, y_chr, rng):
    """HIGH vs LOW specificity sets, matched on size AND per-chromosome composition."""
    hi, lo = [], []
    for c in sorted(set(y_chr)):
        idx = np.nonzero((y_chr == c) & ok)[0]
        if len(idx) < 6:
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

    Me, se = G.basis("esm2"); pe = {s: i for i, s in enumerate(se)}
    have = np.array([s in pe for s in sd])
    Xe = np.zeros((len(sd), Me.shape[1]), np.float32)
    Xe[have] = Me[[pe[s] for s in np.array(sd)[have]]]
    y_esm = KMeans(n_clusters=22, n_init=4, random_state=SEED).fit_predict(Xe).astype(str)
    del Me, Xe; gc.collect()

    with h5py.File(FETAL, "r") as f:
        ct = f["obs"]["cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ct["categories"][:]]
        codes = ct["codes"][:]

    print("[tau] scoring cell-type specificity ...", flush=True)
    tau, ok, used = tau_specificity(FETAL, sd, codes, cats)
    hi, lo = matched_split(tau, ok, y_chr, np.random.default_rng(SEED))
    print(f"  {len(used)} cell types | HIGH-specificity {len(hi)} genes (mean tau {tau[hi].mean():.3f}) | "
          f"LOW {len(lo)} genes (mean tau {tau[lo].mean():.3f})", flush=True)
    assert len(hi) == len(lo), "gene sets must be size-matched"

    res = {"n_hi": int(len(hi)), "n_lo": int(len(lo)), "tau_hi": float(tau[hi].mean()),
           "tau_lo": float(tau[lo].mean()), "chance": 1 / 22, "cell_types_used": used, "runs": {}}
    rng = np.random.default_rng(SEED)
    usable = [i for i, c in enumerate(cats) if (codes == i).sum() >= N_TOTAL // 12]

    conditions = []
    for nm, t in [(f"k=1 broad ({BROAD})", BROAD), (f"k=1 narrow ({NARROW})", NARROW)]:
        if t in cats:
            idx = np.nonzero(codes == cats.index(t))[0]
            conditions.append((nm, rng.choice(idx, size=min(N_TOTAL, len(idx)), replace=False)))
    pick = list(rng.choice(usable, size=min(12, len(usable)), replace=False))
    sel = np.concatenate([rng.choice(np.nonzero(codes == t)[0],
                                     size=min(N_TOTAL // len(pick), (codes == t).sum()), replace=False)
                          for t in pick])
    conditions.append(("k=12 diverse", sel))

    print(f"\n{'condition':<26} {'gene set':<10} {'n cells':<9} {'chromosome':<13} {'ESM2 control'}")
    print("-" * 78)
    for cname, rows in conditions:
        B_all = build_from_rows(FETAL, sd, rows)
        for gname, gi in [("HIGH-spec", hi), ("LOW-spec", lo)]:
            sub = B_all[:, gi]
            keep = np.asarray((sub > 0).sum(0)).ravel() > 0
            E = lsa(sub, dims=min(DIMS, int(keep.sum()) - 1))
            a_chr = decode(E, y_chr[gi], groups[gi])
            hm = have[gi]
            a_esm = decode_random_split(E[hm], y_esm[gi][hm]) if hm.sum() > 500 else float("nan")
            res["runs"][f"{cname} | {gname}"] = {"chromosome": a_chr, "esm2_control": a_esm,
                                                 "n_cells": int(len(rows)), "n_genes": int(len(gi))}
            print(f"{cname:<26} {gname:<10} {len(rows):<9} {a_chr:<13.3f} {a_esm:.3f}", flush=True)
            del E, sub; gc.collect()
        del B_all; gc.collect()
        json.dump(res, open(os.path.join(HERE, "results", "specificity_mechanism.json"), "w"), indent=1)

    print("\n=== VERDICT ===")
    r = res["runs"]
    def g(c, s): return r.get(f"{c} | {s}", {}).get("chromosome", float("nan"))
    dhi = g("k=12 diverse", "HIGH-spec") - g(f"k=1 narrow ({NARROW})", "HIGH-spec")
    dlo = g("k=12 diverse", "LOW-spec") - g(f"k=1 narrow ({NARROW})", "LOW-spec")
    print(f"  diversity effect (narrow -> k=12): HIGH-spec {dhi:+.3f} | LOW-spec {dlo:+.3f}")
    print(f"  at k=12: HIGH-spec {g('k=12 diverse','HIGH-spec'):.3f} vs LOW-spec {g('k=12 diverse','LOW-spec'):.3f}")
    if dhi > 0.10 and dhi > 2 * max(dlo, 0.0):
        print("  -> CONFIRMED: genomically-clustered CELL-TYPE-SPECIFIC programs manufacture the chromosome signal.")
    elif abs(dhi - dlo) < 0.05:
        print("  -> REFUTED: both gene classes behave alike; the mechanism is not cell-type-program driven.")
    else:
        print("  -> PARTIAL: report both; do not claim the mechanism cleanly.")
    print("\n[done] -> results/specificity_mechanism.json")


if __name__ == "__main__":
    main()
