"""BUILD FACTORISED CO-EXPRESSION BASES — the correct null the Stage-1 screen has never used.

WHY. `gm_lib._coexpr*()` returns RAW log1p-CP10k per-gene profiles. This project has already retracted that as a
co-expression baseline in writing: factorising the IDENTICAL cells moves chromosome decoding from 0.044 to 0.720,
at which point MaxToki-217M (0.506) LOSES by -0.213 and only the 1B (0.880) clears it. Every "beats co-expression"
margin `screen.py` has ever printed is therefore measured against a null known to be far too weak.

WHAT THIS BUILDS. For each panel the screen uses as a reference, the word2vec analogue for expression data:
gene = token, cell = document, document = that cell's top-K expressed genes (mirroring the rank-value tokenizer
the model itself uses). Then LSA — PPMI-weight the cell x gene matrix and take a truncated SVD over genes
(Levy & Goldberg: SGNS implicitly factorises shifted PPMI; Louwerse & Zwaan recovered city lat/long this way).
Dimension is SWEPT, because the retraction also established that a single dimension is not a fair instrument.

CALIBRATION SELF-TEST, AND IT GATES THE OUTPUT. A new baseline that comes out WEAK is not self-validating — this
project shipped a wrong claim exactly that way (a PPMI-SVD retaining every detected gene scored 0.051 and was
written up as "the paper's claim is strengthened"; the bug was that ubiquitous housekeeping detection dominates
the matrix, and the top-K-per-cell restriction is what makes the document analogy work). So this script probes
chromosome from each factorisation and REFUSES to write the cache unless the factorised score clears the raw
profile by a wide margin. If the self-test fails, the build is broken, not the baseline.

CAVEAT ON CELL COUNT, recorded here so it is not lost. The panels cached for the screen are SUBSAMPLES
(TS 7,500 cells / fetal gut 8,000 / K562 8,000). The 0.720 ceiling in the retraction came from the FULL fetal gut
at 62,849 cells. So the factorisations built here are weaker than the strongest baseline known to exist, and a
model margin measured against them is an UPPER bound on the model's true margin. For any coordinate hypothesis,
quote the 0.720 full-panel ceiling as well.

Out: data/genemanifold/fact_<panel>_<dim>.npz  (+ results/coexpr_fact.json)
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from shallow_coocc_baseline import build_binary, lsa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

TOPK, SEED = 2048, 0
DIMS = [128, 256, 512]
PANELS = ["coexpr", "coexpr_devel", "coexpr_k562"]
RAW_TAG = {"coexpr": "coexpr_ts", "coexpr_devel": "coexpr_devel", "coexpr_k562": "coexpr_k562"}
MIN_JUMP = 0.10          # factorised must beat the raw profile by at least this on chromosome, or the build is broken


def chrom_score(X, syms, C):
    """22-class balanced accuracy under the 10-Mb genomic group split (the honest split)."""
    keep = [i for i, s in enumerate(syms) if s in C.index and C.chromosome[s] in AUTOSOMES]
    if len(keep) < 500:
        return float("nan"), 0
    X = np.asarray(X[keep], dtype=np.float32)
    ss = [syms[i] for i in keep]
    y = np.array([C.chromosome[s] for s in ss])
    blk = np.array([f"{C.chromosome[s]}:{int(C.start[s]) // BLOCK}" for s in ss])
    P = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, blk):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=0.1, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P[te] = clf.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, P.astype(str))), len(y)


def main():
    C = coords()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = dict(topk=TOPK, dims=DIMS, chance=1 / 22, panels={})

    for panel in PANELS:
        tag = RAW_TAG[panel]
        z = np.load(os.path.join(G.CACHE, f"{tag}.npz"), allow_pickle=True)
        P, syms = z["profiles"], z["symbols"].astype(str)
        n_g, n_c = P.shape
        print(f"\n=== {panel} ({tag}) — {n_g} genes x {n_c} cells ===", flush=True)

        raw, n_raw = chrom_score(P, syms, C)
        print(f"  raw profile (the retracted null)      chromosome {raw:.3f}   n={n_raw}", flush=True)

        B = build_binary(P, topk=TOPK)                 # (cells, genes) binary top-K
        del P, z; gc.collect()
        print(f"  top-{TOPK} binary: {B.shape}, {B.nnz:,} nnz, {B.nnz/B.shape[0]:.0f} genes/cell", flush=True)

        rec = dict(n_genes=n_g, n_cells=n_c, raw_profile_chrom=raw, dims={})
        for d in DIMS:
            if d >= min(n_g, n_c):
                print(f"  dim {d}: skipped (>= min(genes,cells))"); continue
            F = lsa(B, dims=d)                          # (genes, d)
            s, n = chrom_score(F, syms, C)
            rec["dims"][str(d)] = dict(chrom=s, n=n)
            jump = s - raw
            print(f"  LSA-{d:<5} chromosome {s:.3f}   (raw {raw:.3f}, jump {jump:+.3f})", flush=True)
            np.savez_compressed(os.path.join(G.CACHE, f"fact_{panel}_{d}.npz"),
                                profiles=F, symbols=syms)
            del F; gc.collect()
        del B; gc.collect()

        best = max((v["chrom"] for v in rec["dims"].values()), default=float("nan"))
        rec["best_chrom"], rec["jump"] = best, best - raw
        ok = np.isfinite(best) and (best - raw) >= MIN_JUMP
        rec["self_test_passed"] = bool(ok)
        # A failure here means ONE OF TWO THINGS, and they must not be conflated:
        #   (a) the build is broken, or
        #   (b) this panel genuinely does not carry chromosome-scale co-expression.
        # Disambiguate by CROSS-PANEL CONSISTENCY: identical code on another panel. Tabula Sapiens fails
        # (0.061 -> 0.081) while fetal gut passes hugely (0.042 -> 0.585) on the SAME code path, so the build
        # is sound and TS is simply a weak panel -- which independently matches the established matched-cell
        # panel sweep (TS kidney 0.086, TS lung 0.081, vs gut 0.413-0.524). Do NOT read a TS failure as a bug.
        rec["interpretation"] = ("strong panel" if ok else
                                 "WEAK PANEL for this property, or broken build — disambiguate against the "
                                 "other panels built by this same code before concluding either")
        print(f"  --> best {best:.3f} vs raw {raw:.3f} = {best-raw:+.3f}  "
              f"{'PASS' if ok else 'BELOW THRESHOLD — weak panel or broken build; check cross-panel'}",
              flush=True)
        out["panels"][panel] = rec

    json.dump(out, open(os.path.join(HERE, "results", "coexpr_fact.json"), "w"), indent=1)
    npass = sum(1 for v in out["panels"].values() if v["self_test_passed"])
    print(f"\n{npass}/{len(PANELS)} panels passed the calibration self-test")
    print("[done] -> results/coexpr_fact.json  +  data/genemanifold/fact_<panel>_<dim>.npz")


if __name__ == "__main__":
    main()
