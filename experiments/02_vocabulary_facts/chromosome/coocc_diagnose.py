"""WHY DOES THE SHALLOW CO-OCCURRENCE BASELINE SPLIT THE TWO NORMAL PANELS 0.551 vs 0.081?

shallow_coocc_baseline.py found that an LSA embedding of the fetal-gut panel decodes chromosome at 0.551
(10-Mb group split) -- better than MaxToki's 0.368 -- while the SAME construction on Tabula Sapiens gives 0.081.
A real biological mechanism (chromosome-wide co-regulation) should not be 7x stronger in one normal human tissue
than in another. One of the two panels is lying. Four candidate explanations, one test each:

  A  PREPROCESSING. gm_lib.build_coexpr always re-normalises TS from raw counts, but build_coexpr_devel keeps
     fetal gut AS-IS when it is already normalised (the `D.max() < 20` branch). If the fetal-gut matrix arrived
     denoised/imputed/batch-corrected, correlation structure was injected before we ever saw it.
  B  COLUMN ADJACENCY. If the fetal-gut var table is in genomic order AND any smoothing was applied along it,
     genes adjacent IN THE FILE become correlated, which reads out as genomic position. Diagnostic: compare
     correlation for pairs adjacent-in-file vs same-chromosome-distant vs different-chromosome. A technical
     smoothing artifact shows up as adjacency >> same-chromosome; real domain co-regulation does not.
  C  FREQUENCY, NON-LINEARLY. The 1-D linear detection-rate probe was at chance, but chromosomes differ in gene
     density and a MONOTONE-but-non-linear function of detection rate could still carry chromosome. The top LSA
     component of a PPMI-weighted binary matrix is essentially frequency, so this is very live. Diagnostic:
     probe chromosome from 20 quantile bins of detection rate + mean + variance.
  D  CONCENTRATION. Balanced accuracy over 22 classes can still be driven by a handful of chromosomes. If one or
     two chromosomes carry it, that points to something local (a CNV, an imprinted domain), not a genome-wide code.

Out: results/coocc_diagnose.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from shallow_coocc_baseline import build_binary, lsa, probe, TOPK, DIMS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

SEED = 0
PANELS = [("coexpr_devel", "fetal gut"), ("coexpr", "Tabula Sapiens")]


def raw_provenance():
    """A: which normalisation branch did each panel take? Look at the ORIGINAL h5ad, not the cache."""
    out = {}
    for path, nm in [(G.FETAL_GUT, "fetal gut"),
                     (os.path.join(G.TS_RAW, G.TS_FILES[0]), "Tabula Sapiens")]:
        try:
            with h5py.File(path, "r") as f:
                d = f["X"]["data"][:200000]
                out[nm] = {"file": os.path.basename(path), "max": float(d.max()), "min": float(d.min()),
                           "integer_valued": bool(np.allclose(d, np.round(d))),
                           "branch_taken": "kept AS-IS (already normalised)"
                           if (d.max() < 20.0 and not np.allclose(d, np.round(d))) else "re-normalised log1p-CP10k"}
        except Exception as e:
            out[nm] = {"error": repr(e)[:80]}
    return out


def adjacency(P, sy, C, n=4000, rng=np.random.default_rng(SEED)):
    """B: correlation by relationship. Column order in P is the file's var order."""
    Z = P - P.mean(1, keepdims=True)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    ch = np.array([C.chromosome[s] for s in sy]); st = C.loc[list(sy), "start"].values.astype(float)

    def mean_corr(pairs):
        if len(pairs) == 0:
            return float("nan")
        a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        return float(np.mean(np.abs(np.einsum("ij,ij->i", Z[a], Z[b]))))

    ng = len(sy)
    adj = [(i, i + 1) for i in rng.choice(ng - 1, min(n, ng - 1), replace=False)]
    i1, i2 = rng.integers(0, ng, n * 8), rng.integers(0, ng, n * 8)
    ok = i1 != i2
    i1, i2 = i1[ok], i2[ok]
    same = ch[i1] == ch[i2]; dist = np.abs(st[i1] - st[i2])
    near = list(zip(i1[same & (dist < 1e7)], i2[same & (dist < 1e7)]))[:n]
    far = list(zip(i1[same & (dist > 5e7)], i2[same & (dist > 5e7)]))[:n]
    diff = list(zip(i1[~same], i2[~same]))[:n]
    return {"adjacent_in_file": mean_corr(adj), "same_chr_<10Mb": mean_corr(near),
            "same_chr_>50Mb": mean_corr(far), "different_chr": mean_corr(diff),
            "n_pairs": {"adjacent": len(adj), "near": len(near), "far": len(far), "diff": len(diff)}}


def frequency_nonlinear(P, y, groups):
    """C: chromosome from detection rate / mean / variance, expanded into quantile bins."""
    det = (P > 0).mean(1); mu = P.mean(1); sd = P.std(1)
    F = np.column_stack([det, mu, sd])
    q = np.column_stack([np.digitize(c, np.quantile(c, np.linspace(0, 1, 21)[1:-1])) for c in (det, mu, sd)])
    oh = np.zeros((len(y), 63), np.float32)
    for j in range(3):
        oh[np.arange(len(y)), j * 21 + q[:, j]] = 1.0
    return probe(np.hstack([F, oh]).astype(np.float32), y, groups)


def per_chrom(X, y, groups):
    """D: which chromosomes carry the signal?"""
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, groups=groups):
        sc = StandardScaler().fit(X[tr])
        pred[te] = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1).fit(
            sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    labs = sorted(set(y))
    r = recall_score(y, pred.astype(str), labels=labs, average=None, zero_division=0)
    return {c: round(float(v), 3) for c, v in zip(labs, r)}


def main():
    C = coords()
    res = {"A_provenance": raw_provenance()}
    print("=== A: PREPROCESSING PROVENANCE ===")
    for k, v in res["A_provenance"].items():
        print(f"  {k:<16} {v}")

    for panel, label in PANELS:
        print(f"\n=== {label} ===", flush=True)
        P, syms = G.basis(panel)
        keep = [i for i, s in enumerate(syms) if s in C.index and C.chromosome[s] in AUTOSOMES]
        P = np.asarray(P[keep], dtype=np.float32); sy = np.array(syms)[keep]
        y = np.array([C.chromosome[s] for s in sy])
        st = C.loc[list(sy), "start"].values.astype(float)
        groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])

        d = {}
        d["B_adjacency"] = adjacency(P, sy, C)
        print("  B mean |corr|:  adjacent-in-file %.4f | same-chr <10Mb %.4f | same-chr >50Mb %.4f | diff-chr %.4f"
              % (d["B_adjacency"]["adjacent_in_file"], d["B_adjacency"]["same_chr_<10Mb"],
                 d["B_adjacency"]["same_chr_>50Mb"], d["B_adjacency"]["different_chr"]), flush=True)

        d["C_frequency_nonlinear"] = frequency_nonlinear(P, y, groups)
        print(f"  C frequency (binned, 66-D): random {d['C_frequency_nonlinear']['random']:.3f}"
              f"  group {d['C_frequency_nonlinear']['group']:.3f}", flush=True)

        B = build_binary(P); del P; gc.collect()
        E = lsa(B); del B; gc.collect()
        pc = per_chrom(E, y, groups); del E; gc.collect()
        d["D_per_chromosome_recall"] = pc
        top = sorted(pc.items(), key=lambda kv: -kv[1])[:5]
        nz = sum(1 for v in pc.values() if v > 0.15)
        d["D_n_chrom_above_0.15"] = nz
        print(f"  D per-chromosome recall: {nz}/22 chromosomes above 0.15 | top5 {top}", flush=True)
        res[panel] = dict(label=label, **d)
        json.dump(res, open(os.path.join(HERE, "results", "coocc_diagnose.json"), "w"), indent=1)
        gc.collect()

    print("\n=== READING ===")
    fg, ts = res.get("coexpr_devel", {}), res.get("coexpr", {})
    a = fg.get("B_adjacency", {})
    if a and a.get("adjacent_in_file", 0) > 1.5 * a.get("same_chr_<10Mb", 1):
        print("  B: fetal-gut ADJACENT-IN-FILE correlation exceeds genomic-neighbour correlation")
        print("     -> technical smoothing along file column order, NOT genomic co-regulation. ARTIFACT.")
    for p, nm in [("coexpr_devel", "fetal gut"), ("coexpr", "Tabula Sapiens")]:
        c = res.get(p, {}).get("C_frequency_nonlinear", {}).get("group")
        if c is not None:
            print(f"  C: {nm} frequency-only (non-linear) group = {c:.3f}"
                  + ("  <- frequency alone explains much of it" if c > 0.2 else ""))
    print("\n[done] -> results/coocc_diagnose.json")


if __name__ == "__main__":
    main()
