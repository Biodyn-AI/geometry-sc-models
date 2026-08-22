"""Route Q — is the saddle's BIOLOGY stable across the (r, lambda) solutions that fit equally well?

arms.py + the rank/lambda control showed the arm split is NOT identified: at (r=8, lam=1e-3) the arms are
individually harmful (irreducible saddle), at (r=32, lam=1e-2) they are individually useful, and both fit
the held-out data equally well (dQ +0.0274 vs +0.0238). And the top-15 gene lists barely overlap (1/15).

So the gene-LEVEL read is not identified either. This script asks the weaker, defensible question:
is the PROGRAM-level read stable? Gene sets are fixed a priori (standard membership, not data-derived).
We z-score the saddle score s_g = c_hat^T Q c_hat over all genes and report each set's mean, per config.
Passing = the sign of each set's mean is the same across every config.

Out: results/program_stability_scgpt_setty.json
"""
import os, re, json
import numpy as np, torch, qfit

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")

RIBIO = set("PES1 BRIX1 WDR12 WDR75 NIP7 NOL11 UTP15 UTP18 UTP20 DIMT1 BOP1 RRS1 GNL3 NOP56 NOP58 "
            "FBL DDX21 RRP9 TSR1 RIOK2 MPHOSPH10 NOL6 WDR43 WDR3 WDR36 PWP2 HEATR1 RRP1 RRP12 NOC2L".split())
QUIES = set("FOXO3 TLE4 CDKN1B CDKN1C MECOM HLF GATA2 MLLT3 PRDM1 BACH2 KLF4 EGR1 JUN FOS NR4A1 "
            "TXNIP MEIS1 PBX1 ZFP36 KLF2".split())
CFGS = [(8, 1e-3), (8, 1e-2), (16, 1e-3), (32, 1e-3), (32, 1e-2), (4, 1e-3)]
SEEDS = (0, 1)


def main(model="scgpt", dataset="setty"):
    X, y = qfit.load(model, dataset)
    xh, c, _, _ = qfit.preprocess(X, X, True)
    lin = qfit.LinearPart(xh, c, y)
    z = np.load(os.path.join(RES, "gene_centroids_scgpt.npz"), allow_pickle=True)
    names = np.array([str(g) for g in z["names"]])
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-8] = 1.0
    Cs = (z["cent"] - mu) / sd
    Cs /= np.linalg.norm(Cs, axis=1, keepdims=True) + 1e-12

    sets = [
        ("ribosome-biogenesis (nucleolar)", np.array([g in RIBIO for g in names])),
        ("mito / OXPHOS biogenesis", np.array([bool(re.match(r"^(MRPS|MRPL|TIMM|TOMM|NDUF|UQCR|ATP5|ATPAF|MIPEP)", g)) for g in names])),
        ("cytoplasmic RPS/RPL (abundance confound)", np.array([bool(re.match(r"^RP[SL]\d", g)) for g in names])),
        ("HSC quiescence / TF", np.array([g in QUIES for g in names])),
    ]
    print("set sizes:", {n: int(m.sum()) for n, m in sets}, flush=True)

    scores = {}
    for r, lam in CFGS:
        ss = [np.einsum("ij,jk,ik->i", Cs, qfit.fit_q(lin, xh, y, r, lam, seed=qfit.SEED + 7 * s).Q(), Cs)
              for s in SEEDS]
        s = np.mean(ss, 0)
        scores[(r, lam)] = (s - s.mean()) / s.std()
        print(f"  fitted r={r} lam={lam}", flush=True)

    out = {}
    hdr = "".join(f"  r{r}/{lam:g}" for r, lam in CFGS)
    print(f"\n{'z-scored mean saddle score':45s}{hdr}")
    for name, m in sets:
        vals = [float(scores[k][m].mean()) for k in CFGS]
        consistent = bool(all(v > 0 for v in vals) or all(v < 0 for v in vals))
        out[name] = dict(n_genes=int(m.sum()), values=vals, sign_consistent=consistent,
                         mean=float(np.mean(vals)))
        print(f"{name:45s}" + "".join(f"  {v:+8.3f}" for v in vals) +
              f"   consistent={consistent}")
    json.dump(dict(model=model, dataset=dataset, configs=[list(k) for k in CFGS], sets=out),
              open(os.path.join(RES, f"program_stability_{model}_{dataset}.json"), "w"), indent=1)
    print(f"\n-> results/program_stability_{model}_{dataset}.json")


if __name__ == "__main__":
    main()
