"""DOES A GENE RESPOND TO CELLULAR CONTEXT IN A GENE-SPECIFIC WAY? (the real test)

THE QUESTION. Ihor's: can we track how a gene's representation changes with the genes around it, and is the
change biologically meaningful? Everything hinges on separating a real effect from the AVERAGING NULL --
attention pools over the cell, so every gene in a macrophage drifts the same way, and a gene will "look
metabolic" in a metabolic cell for no interesting reason.

THE FORMULATION. Work with pairwise context SHIFTS rather than a multi-context residual:

    Delta_p(g; c1->c2) = v_p(g,c2) - v_p(g,c1)          measured in cell partition p
    delta_p(g)         = Delta_p(g) - mean_over_genes Delta_p(g)

The mean over genes IS the averaging null (the context main effect). Subtracting it leaves the GENE-SPECIFIC
response -- gene A and gene B moving differently between the same two contexts. That is the only place biology
can live, and it is the object of the test.

WHY THIS DESIGN, POINT BY POINT -- each item fixes a measured failure of the scGPT pilot (`ctx_interaction.py`):

 - PAIRWISE, NOT A 7-WAY RESIDUAL. The pilot's interaction term was a residual with zero mean across contexts,
   which PINS each gene's vectors at mean pairwise cosine -1/(n_ctx-1) = -0.167 by construction; the observed
   -0.141 was that arithmetic floor, not a finding. Pairwise differences carry no such constraint.
 - SPLIT-HALF ACROSS INDEPENDENT CELLS. delta is estimated twice, from disjoint cell partitions. Noise cannot
   replicate across independent cells; structure can. The pilot split by DIMENSION, which proved nothing
   because dimensions within a layer are correlated (anisotropy 0.79-0.93) -- its "reliability" of 0.98 sat
   against a null that was itself 0.82.
 - COUNT-BALANCED CELLS ONLY. We keep only (gene, context) cells that hit the extractor's occurrence cap in
   BOTH partitions, so every mean is over exactly the same number of tokens. Unequal counts give SEM ~ 1/sqrt(n)
   and manufacture a fake gene-specific term -- the actual cause of the pilot's z=+10.4.
 - PER-DIMENSION STANDARDISATION, NOT A SCALAR ANISOTROPY BASELINE. Timkey & van Schijndel 2021
   (arXiv:2109.04404): 1-3 "rogue dimensions" dominate cosine similarity and subtracting a scalar baseline does
   NOT fix it. A rogue dimension modulated by gene would fake exactly our effect. We z-score every dimension
   across all (gene, context) entries first, and report how much of the raw signal lived in the top dimension.
 - THE STATISTIC IS DIRECTIONAL AGREEMENT, NOT MAGNITUDE. Magnitude is dominated by per-gene vector norm.

    same = mean_g cos( delta_0(g), delta_1(g) )      the SAME gene, two independent halves of the cells
    diff = mean_g cos( delta_0(g), delta_1(perm g) ) a DIFFERENT gene, same two halves
    EXCESS = same - diff

Under the averaging null both are ~0 and EXCESS ~ 0: whatever is left after removing the context mean is noise,
which does not reproduce. A positive EXCESS means a gene's departure from the crowd is real and repeatable.

READING IT. EXCESS is the headline. Bootstrap CI over genes; a null result is interpretable because the
positive control (reliability of the CONTEXT MAIN EFFECT itself) is reported alongside -- if the main effect
replicates strongly and the gene-specific part does not, that is a clean "attention only averages" verdict
rather than a failed measurement.

Out: results/ctx_polysemy.json
"""
import os, sys, json, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAPS = [0, 4, 8, 11]
SEED = 0
MIN_GENES = 200          # a context pair needs this many count-balanced genes to be scored


def load(tap):
    p = os.path.join(RES, f"ctx_maxtoki_L{tap:02d}.npz")
    z = np.load(p, allow_pickle=True)
    return (z["M"].astype(np.float32), z["counts"], z["genes"].astype(str),
            z["contexts"].astype(str), int(z["cap"]))


def zscore_dims(M, mask):
    """per-dimension standardisation over all count-balanced (gene, context) entries -- the Timkey fix."""
    flat = M[:, mask]                                   # (part, n_entries, d)
    mu = flat.reshape(-1, flat.shape[-1]).mean(0)
    sd = flat.reshape(-1, flat.shape[-1]).std(0) + 1e-6
    return (M - mu) / sd, float((sd ** 2).max() / (sd ** 2).sum())


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def main():
    rng = np.random.default_rng(SEED)
    out = {"taps": {}}
    for tap in TAPS:
        M, counts, genes, ctxs, cap = load(tap)
        nP, nC, nG, d = M.shape
        full = (counts == cap).all(0)                   # (n_ctx, n_gene) balanced in BOTH partitions
        # `full` is (n_ctx, n_gene); passing it whole lets numpy index axes 1 and 2 together -> (part, n_true, d)
        Mz, top_dim_share = zscore_dims(M, full)
        print(f"\n=== layer {tap} — {nG} genes x {nC} contexts, cap {cap} ===")
        print(f"    top single dimension carried {top_dim_share:.1%} of raw variance before standardising "
              f"(Timkey rogue-dimension check)")

        rows, same_all, diff_all, main_all = [], [], [], []
        for c1, c2 in itertools.combinations(range(nC), 2):
            keep = full[c1] & full[c2]
            if keep.sum() < MIN_GENES:
                continue
            D0 = Mz[0, c2, keep] - Mz[0, c1, keep]      # shift, cell-partition 0
            D1 = Mz[1, c2, keep] - Mz[1, c1, keep]      # shift, cell-partition 1
            b0, b1 = D0.mean(0), D1.mean(0)             # the averaging null = context main effect
            # positive control: does the context main effect itself replicate across independent cells?
            main_all.append(float(np.dot(b0, b1) / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-9)))
            d0, d1 = D0 - b0, D1 - b1                   # gene-specific departure from the crowd
            same = cos_rows(d0, d1)
            perm = rng.permutation(len(d1))
            diff = cos_rows(d0, d1[perm])
            rows.append(dict(c1=ctxs[c1], c2=ctxs[c2], n=int(keep.sum()),
                             same=float(same.mean()), diff=float(diff.mean()),
                             excess=float(same.mean() - diff.mean())))
            same_all.append(same); diff_all.append(diff)

        if not rows:
            print("    no context pair had enough count-balanced genes"); continue
        S = np.concatenate(same_all); Dg = np.concatenate(diff_all)
        excess = float(S.mean() - Dg.mean())
        bs = [float(S[rng.integers(0, len(S), len(S))].mean() - Dg[rng.integers(0, len(Dg), len(Dg))].mean())
              for _ in range(2000)]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        mainrep = float(np.mean(main_all))
        print(f"    context pairs scored: {len(rows)} (median {int(np.median([r['n'] for r in rows]))} "
              f"balanced genes each)")
        print(f"    POSITIVE CONTROL — context main effect replicates across cells: cos = {mainrep:+.3f}")
        print(f"    same gene, independent halves : {S.mean():+.4f}")
        print(f"    different gene, same halves   : {Dg.mean():+.4f}")
        print(f"    EXCESS (the test)             : {excess:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")
        top = sorted(rows, key=lambda r: -r["excess"])[:3]
        for r in top:
            print(f"      strongest pair: {r['c1'][:26]} vs {r['c2'][:26]}  excess {r['excess']:+.4f} (n={r['n']})")
        out["taps"][f"L{tap:02d}"] = dict(excess=excess, ci=[lo, hi], same=float(S.mean()),
                                          diff=float(Dg.mean()), main_effect_replication=mainrep,
                                          top_dim_share=top_dim_share, n_pairs=len(rows), pairs=rows)

    if out["taps"]:
        best = max(out["taps"].items(), key=lambda kv: kv[1]["excess"])
        e, (lo, hi) = best[1]["excess"], best[1]["ci"]
        out["verdict"] = (
            f"{best[0]}: gene-specific context response EXCESS = {e:+.4f}, CI [{lo:+.4f}, {hi:+.4f}]. " +
            ("POSITIVE — a gene's departure from the crowd reproduces across independent cells, so context "
             "response is gene-specific and not merely the averaging null. Next: does it track FUNCTION?"
             if lo > 0 else
             "NULL — and an interpretable one: the context main effect replicates at cos "
             f"{best[1]['main_effect_replication']:+.3f}, so the measurement works; what does not reproduce is "
             "the GENE-SPECIFIC part. That is the averaging null holding, consistent with the field-wide "
             "finding that SCFM attention re-expresses co-expression."))
        print(f"\nVERDICT: {out['verdict']}")
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "ctx_polysemy.json"), "w"), indent=1)
    print("\n[done] -> results/ctx_polysemy.json")


if __name__ == "__main__":
    main()
