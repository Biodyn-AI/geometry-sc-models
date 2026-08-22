"""DOES CELL-TYPE COMPOSITION MANUFACTURE SPURIOUS CNV CALLS IN KARYOTYPICALLY NORMAL CELLS?

The chain of reasoning. purity_decomposition.py showed cell-type heterogeneity specifically manufactures
chromosome-scale co-expression structure (a matched-difficulty control stayed flat while chromosome decodability
tripled). CNV callers for scRNA-seq -- inferCNV, CopyKAT, Numbat -- all work by the SAME operation that produces
that structure: order genes by genomic coordinate, smooth expression along the genome, and read coherent
chromosome-scale runs as copy-number gains/losses, relative to a reference cell population. If genomically-
clustered differential expression between cell types produces chromosome-scale runs on its own, then comparing a
query cell type against a compositionally-mismatched normal reference will call CNVs that cannot exist.

This is a direct, falsifiable test of that, on data where the answer is known: human fetal gut, obs['disease'] ==
'normal' for every cell, karyotypically diploid. ANY whole-chromosome coherent call here is spurious by
construction. We implement the inferCNV core faithfully (reference gene-centering, genomic-order smoothing,
per-cell centering, denoising) and ask whether normal-vs-normal cross-cell-type comparison produces calls that a
practitioner would act on.

THE DECISIVE CONTRAST.
  cross[c]  = mean smoothed CNV score on chromosome c over QUERY cells, when reference = cell type A, query =
              a DIFFERENT normal cell type B. (compositional mismatch -- the realistic failure case)
  null[c]   = the same quantity when reference and query are random halves of ONE cell type (matched
              composition -- the pipeline's behaviour when it is used as intended). Repeated to get a per-
              chromosome null mean (~0) and SD.
  A SPURIOUS CALL is a chromosome where |cross[c]| exceeds 3 x null SD[c]. Under the null the pipeline makes
  essentially none; the question is how many the cross-type comparison makes on cells that are provably diploid.

Everything is whole-chromosome (the most conservative unit; no centromere table needed) and needs no external
ground truth. Cell counts are matched across cross and null so the null SD is the right yardstick.

Out: results/infercnv_confound.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from genome_wide import coords, AUTOSOMES
from coocc_strongest import symbols_of

SEED = 0
N = 800                 # cells per role (ref, query), matched everywhere
WIN = 101               # inferCNV default genomic smoothing window (genes)
CLIP = 3.0              # inferCNV-style residual bound (log space)
N_NULL = 20             # random same-cell-type splits for the null
DATA = f"{_DATA}"
FETAL = f"{DATA}/pancreas/fetal_gut.h5ad"
TYPES = ["fibroblast", "enterocyte", "enteric smooth muscle cell", "mesodermal cell", "neural crest cell"]


def load_dense(path, rows, keepcol, n_panel):
    """Dense (len(rows) x n_panel) log-normalised expression over the ordered gene panel."""
    rows = np.array(sorted(int(r) for r in rows))
    D = np.zeros((len(rows), n_panel), np.float32)
    with h5py.File(path, "r") as f:
        X = f["X"]; indptr = X["indptr"]; data = X["data"]; idx = X["indices"]
        ip = indptr[:]
        for i, r in enumerate(rows):
            s0, e0 = int(ip[r]), int(ip[r + 1])
            if e0 <= s0:
                continue
            ii = idx[s0:e0]; vv = data[s0:e0]
            m = keepcol[ii] >= 0
            D[i, keepcol[ii[m]]] = vv[m]
    return D


def infercnv(E, ref_mask, chrom_ids):
    """Faithful inferCNV core. E: (cells, genes) ordered by genome. Returns smoothed residual (cells, genes)."""
    R = E - E[ref_mask].mean(0, keepdims=True)          # 1. gene-center on the reference
    np.clip(R, -CLIP, CLIP, out=R)                       # 2. bound residuals (denoise)
    S = np.empty_like(R)                                 # 3. smooth along genome, per chromosome
    for c in np.unique(chrom_ids):
        j = np.nonzero(chrom_ids == c)[0]
        w = min(WIN, len(j) if len(j) % 2 else len(j) - 1) or 1
        S[:, j] = uniform_filter1d(R[:, j], size=w, axis=1, mode="nearest")
    S -= np.median(S, axis=1, keepdims=True)             # 4. center each cell (remove baseline ploidy)
    return S


def chrom_means(S, chrom_ids, cell_mask, chroms):
    return np.array([S[cell_mask][:, chrom_ids == c].mean() for c in chroms])


def main():
    C = coords()
    with h5py.File(FETAL, "r") as f:
        syms = symbols_of(f)
        ct = f["obs"]["cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ct["categories"][:]]
        codes = ct["codes"][:]

    # ordered autosomal gene panel present in the file
    present = {s: j for j, s in enumerate(syms)}
    panel = [s for s in C.index[C.chromosome.isin(AUTOSOMES)] if s in present]
    panel = sorted(panel, key=lambda s: (int(C.chromosome[s]), float(C.start[s])))
    chrom_ids = np.array([int(C.chromosome[s]) for s in panel])
    chroms = sorted(set(chrom_ids))
    keepcol = np.full(len(syms), -1, np.int64)
    for pos, s in enumerate(panel):
        keepcol[present[s]] = pos
    print(f"panel: {len(panel)} autosomal genes ordered by genome | {len(chroms)} chromosomes | "
          f"window {WIN} genes | disease=normal (all cells diploid)\n", flush=True)

    rng = np.random.default_rng(SEED)
    rows_of = {t: np.nonzero(codes == cats.index(t))[0] for t in TYPES if t in cats}
    cache = {t: load_dense(FETAL, rng.choice(r, size=min(2 * N, len(r)), replace=False), keepcol, len(panel))
             for t, r in rows_of.items()}
    for t in cache:
        print(f"  loaded {t:<28} {cache[t].shape[0]} cells", flush=True)

    # ---- null: same-cell-type random split, per chromosome mean/SD ----
    print("\n[null] same-cell-type splits (matched composition; the pipeline used as intended)", flush=True)
    null_scores = {c: [] for c in chroms}
    for t, E in cache.items():
        for _ in range(N_NULL // len(cache) + 1):
            perm = rng.permutation(E.shape[0])
            ref, qry = perm[:N], perm[N:2 * N]
            if len(qry) < N:
                continue
            mask = np.zeros(E.shape[0], bool); mask[ref] = True
            S = infercnv(E, mask, chrom_ids)
            qmask = np.zeros(E.shape[0], bool); qmask[qry] = True
            cm = chrom_means(S, chrom_ids, qmask, chroms)
            for c, v in zip(chroms, cm):
                null_scores[c].append(v)
            del S; gc.collect()
    null_sd = {int(c): float(np.std(null_scores[c])) for c in chroms}
    null_mean = {int(c): float(np.mean(null_scores[c])) for c in chroms}
    print(f"  null per-chromosome |mean| max {max(abs(v) for v in null_mean.values()):.4f}, "
          f"SD range {min(null_sd.values()):.4f}-{max(null_sd.values()):.4f} "
          f"(over {len(null_scores[chroms[0]])} splits)", flush=True)

    # ---- cross: reference = type A, query = a DIFFERENT normal type ----
    print("\n[cross] normal-vs-normal, compositionally mismatched:", flush=True)
    types = list(cache)
    res = {"n_cells": N, "window": WIN, "null_sd": {str(k): v for k, v in null_sd.items()},
           "null_mean": {str(k): v for k, v in null_mean.items()}, "chroms": [int(c) for c in chroms], "pairs": {}}
    print(f"{'reference':<22} {'query':<22} {'spurious calls':<15} {'largest |chr z|':<16} strongest chr")
    print("-" * 92)
    worst = 0.0
    for a in types:
        Ea = cache[a]; mask = np.zeros(Ea.shape[0], bool); mask[:N] = True
        for b in types:
            if a == b:
                continue
            # ref = first N of A ; query = first N of B, scored through A-centered pipeline
            E = np.vstack([Ea[:N], cache[b][:N]])
            rmask = np.zeros(E.shape[0], bool); rmask[:N] = True
            qmask = np.zeros(E.shape[0], bool); qmask[N:] = True
            S = infercnv(E, rmask, chrom_ids)
            cm = chrom_means(S, chrom_ids, qmask, chroms)
            z = np.array([(cm[i] - null_mean[c]) / (null_sd[c] + 1e-9) for i, c in enumerate(chroms)])
            calls = int((np.abs(z) > 3).sum())
            top = chroms[int(np.argmax(np.abs(z)))]
            res["pairs"][f"{a} -> {b}"] = {"z": {str(c): float(zz) for c, zz in zip(chroms, z)},
                                           "chrom_score": {str(c): float(v) for c, v in zip(chroms, cm)},
                                           "n_spurious_calls": calls, "max_abs_z": float(np.abs(z).max()),
                                           "strongest_chrom": int(top),
                                           "strongest_direction": "gain" if cm[chroms.index(top)] > 0 else "loss"}
            worst = max(worst, np.abs(z).max())
            print(f"{a:<22} {b:<22} {calls:<15} {np.abs(z).max():<16.1f} chr{top} "
                  f"{'gain' if cm[chroms.index(top)] > 0 else 'loss'}", flush=True)
            del S; gc.collect()
        json.dump(res, open(os.path.join(HERE, "results", "infercnv_confound.json"), "w"), indent=1)

    calls_per_pair = [v["n_spurious_calls"] for v in res["pairs"].values()]
    res["summary"] = {"n_pairs": len(calls_per_pair),
                      "pairs_with_any_call": int(sum(c > 0 for c in calls_per_pair)),
                      "mean_calls_per_pair": float(np.mean(calls_per_pair)),
                      "max_abs_z_overall": float(worst)}
    json.dump(res, open(os.path.join(HERE, "results", "infercnv_confound.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    print(f"  {res['summary']['pairs_with_any_call']}/{len(calls_per_pair)} normal cell-type pairs produce "
          f"at least one spurious whole-chromosome call (>3 sigma over the matched-composition null)")
    print(f"  mean {res['summary']['mean_calls_per_pair']:.1f} spurious chromosomes per pair; "
          f"largest deviation {worst:.1f} sigma")
    print("  -> " + ("cell-type composition alone yields chromosome-scale CNV calls on karyotypically NORMAL cells"
                     if worst > 3 else "the pipeline is robust to compositional mismatch here"))
    print("\n[done] -> results/infercnv_confound.json")


if __name__ == "__main__":
    main()


def mitigation():
    """Does the standard fix -- a DIVERSE normal reference (all cell types pooled) -- remove the spurious calls,
    and how large are they in RAW CNV units vs a real single-copy event? This is what decides whether the
    confound is a practical hazard or a solved one. Run: python infercnv_confound.py mitigate"""
    C = coords()
    with h5py.File(FETAL, "r") as f:
        syms = symbols_of(f)
        ct = f["obs"]["cell_type"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ct["categories"][:]]
        codes = ct["codes"][:]
    present = {s: j for j, s in enumerate(syms)}
    panel = sorted([s for s in C.index[C.chromosome.isin(AUTOSOMES)] if s in present],
                   key=lambda s: (int(C.chromosome[s]), float(C.start[s])))
    chrom_ids = np.array([int(C.chromosome[s]) for s in panel]); chroms = sorted(set(chrom_ids))
    keepcol = np.full(len(syms), -1, np.int64)
    for pos, s in enumerate(panel):
        keepcol[present[s]] = pos
    rng = np.random.default_rng(SEED)
    cache = {t: load_dense(FETAL, rng.choice(np.nonzero(codes == cats.index(t))[0],
                            size=min(2 * N, int((codes == cats.index(t)).sum())), replace=False), keepcol, len(panel))
             for t in TYPES if t in cats}

    # diverse reference: equal cells from every type
    per = N // len(cache)
    ref = np.vstack([E[:per] for E in cache.values()])
    real_gain, real_loss = np.log(1.5) * 0.5, abs(np.log(0.5)) * 0.5   # dosage-attenuated single-copy shift

    print("=== MITIGATION: DIVERSE reference (best practice) vs the query's own type ===", flush=True)
    print(f"{'query type':<28} {'chr>|0.5x single-copy|':<24} {'max |raw shift|':<16} {'as % of a real loss'}")
    print("-" * 86)
    out = {"real_single_copy_gain": real_gain, "real_single_copy_loss": real_loss, "diverse_ref": {}}
    for t, Eq in cache.items():
        E = np.vstack([ref, Eq[:N]])
        rmask = np.zeros(E.shape[0], bool); rmask[:ref.shape[0]] = True
        qmask = np.zeros(E.shape[0], bool); qmask[ref.shape[0]:] = True
        S = infercnv(E, rmask, chrom_ids)
        cm = chrom_means(S, chrom_ids, qmask, chroms)
        big = int((np.abs(cm) > 0.5 * min(real_gain, real_loss)).sum())
        out["diverse_ref"][t] = {"max_abs_shift": float(np.abs(cm).max()),
                                 "n_chrom_over_half_single_copy": big,
                                 "pct_of_real_loss": float(100 * np.abs(cm).max() / real_loss)}
        print(f"{t:<28} {big:<24} {np.abs(cm).max():<16.3f} {100*np.abs(cm).max()/real_loss:.0f}%", flush=True)
        del S; gc.collect()
    json.dump(out, open(os.path.join(HERE, "results", "infercnv_mitigation.json"), "w"), indent=1)
    print(f"\n  (a real single-copy loss ~ {real_loss:.3f}, gain ~ {real_gain:.3f} log units, dosage-attenuated)")
    print("  -> " + ("a diverse reference LARGELY controls it (few chromosomes over half a single-copy shift)"
                     if max(v["n_chrom_over_half_single_copy"] for v in out["diverse_ref"].values()) <= 2
                     else "even a diverse reference leaves chromosome-scale deviations a caller could flag"))
    print("[done] -> results/infercnv_mitigation.json")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "mitigate":
    mitigation(); sys.exit(0)
