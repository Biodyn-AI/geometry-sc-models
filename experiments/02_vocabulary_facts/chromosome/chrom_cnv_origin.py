"""DID THE MODEL LEARN CHROMOSOME FROM COPY-NUMBER VARIATION? (Ihor, 2026-07-20)

THE HYPOTHESIS ([[cnv-alternative-mechanism]], still the sharpest open threat). The model may have learned
"which chromosome" not from chromatin biology but from DOSAGE: in aneuploid cells (cancer lines, and a lot of
public scRNA-seq is cancer) a whole chromosome is gained or lost, so ALL of its genes shift together. A model
trained to predict expression would find "these ~800 genes move as a block" a genuinely useful regularity.

WHY THIS IS NOW WORTH TESTING DIRECTLY. `steer_locality.py` just showed the CAUSAL response is a CLIFF at the
chromosome boundary and FLAT within it (near−far +0.16 n.s.; same−other +1.97). Uniform-whole-chromosome is the
dosage signature; graded/local would have been the chromatin-domain signature. So the causal shape already
points at CNV. This asks whether the DATA contains the block structure that would teach it.

THE MEASUREMENT (expression data only for the biology side; no model forward passes).
  BLOCK COHERENCE of chromosome c = how much c's genes fluctuate TOGETHER across cells, above independent noise.
    aggregate_c(cell) = mean over c's genes of gene-centred log1p-CP10k expression
    coherence_c        = Var_cells(aggregate_c) / Var_cells(aggregate of a MATCHED RANDOM gene set)
  The null draws the same number of genes, matched on expression decile, so gene count and abundance are held
  fixed and only the CHROMOSOME GROUPING varies. Independent genes -> ratio ~1. A chromosome behaving as a
  dosage block -> ratio >> 1.

TWO PREDICTIONS, both falsifiable:
  P1  ANEUPLOID cells should show much higher block coherence than NORMAL cells. (K562 is a near-triploid
      cancer line with well-known whole-chromosome and arm-level changes; fetal gut is karyotypically normal.)
      If P1 fails, aneuploidy is not creating block structure in this kind of data at all.
  P2  If the model learned FROM that structure, per-chromosome block coherence should predict the model's
      per-chromosome CAUSAL STEERING STRENGTH (read from results/steer_propagation_chromosome_1b_seed0.json).
      Correlated -> CNV-origin supported. Uncorrelated -> the model's chromosome variable is not simply a
      readout of dosage blocks, and the CNV story weakens even though the causal SHAPE looks dosage-like.

Run: ../../.venv/bin/python -u chrom_cnv_origin.py        (light venv; no torch needed)
Out: results/chrom_cnv_origin.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES

K562 = (f"{_DATA}/"
        "state_activations/replogle_k562_subset.h5ad")
N_CELLS = 3000
N_NULL = 60
MIN_GENES = 25
SEED = 0


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def load(path, gene_key_candidates=("feature_name", "gene_name_index", "index", "_index")):
    with h5py.File(path, "r") as f:
        syms = None
        for k in gene_key_candidates:
            if k in f["var"]:
                v = f["var"][k]
                if isinstance(v, h5py.Group) and "categories" in v:
                    syms = _dec(v["categories"][:]).astype(str)[v["codes"][:]]
                else:
                    syms = _dec(v[:]).astype(str)
                break
        X = f["X"]
        shape = X.attrs.get("shape")
        shape = tuple(int(v) for v in shape) if shape is not None else X.shape
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], min(N_CELLS, shape[0]), replace=False))
        if isinstance(X, h5py.Group):                       # CSR
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            E = np.zeros((len(sel), shape[1]), np.float32)
            for i, r in enumerate(sel):
                a, b = int(indptr[r]), int(indptr[r + 1]); E[i, idx[a:b]] = data[a:b]
        else:
            E = np.stack([np.asarray(X[int(i), :], dtype=np.float32) for i in sel])
    return np.char.upper(syms.astype(str)), E


def coherence(E, syms, C, label):
    """Block coherence per chromosome: Var(chromosome aggregate) / Var(matched random-gene aggregate)."""
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    L = np.log1p(E / tot * 1e4)
    L = L - L.mean(0)                                        # centre each gene
    sd = L.std(0)
    ok = sd > 1e-8
    dec = np.zeros(L.shape[1], int)
    dec[ok] = np.digitize(L[:, ok].std(0), np.percentile(L[:, ok].std(0), np.arange(10, 100, 10)))
    gchr = np.array([C.loc[s, "chromosome"] if s in C.index else "" for s in syms], dtype=object)
    rng = np.random.default_rng(SEED + 3)
    by_dec = {d: np.where(ok & (dec == d))[0] for d in range(10)}
    out = {}
    for c in AUTOSOMES:
        idx = np.where(ok & (gchr == c))[0]
        if len(idx) < MIN_GENES:
            continue
        obs = float(L[:, idx].mean(1).var())
        comp = [dec[i] for i in idx]
        nulls = []
        for _ in range(N_NULL):
            pick = [rng.choice(by_dec.get(d, np.where(ok)[0])) for d in comp]
            nulls.append(float(L[:, np.array(pick)].mean(1).var()))
        nm = float(np.mean(nulls))
        out[c] = dict(n_genes=int(len(idx)), obs_var=obs, null_var=nm, coherence=obs / (nm + 1e-12))
    print(f"  [{label}] {len(out)} chromosomes; median coherence "
          f"{np.median([v['coherence'] for v in out.values()]):.2f}x")
    return out


def main():
    C = coords()
    res = {}
    print("BLOCK COHERENCE = Var(chromosome aggregate) / Var(matched random gene set). ~1 = independent genes.\n")
    sets = [("fetal_gut (karyotypically normal)", G.FETAL_GUT), ("K562 (aneuploid cancer line)", K562)]
    coh = {}
    for label, path in sets:
        try:
            syms, E = load(path)
            coh[label] = coherence(E, syms, C, label)
        except Exception as e:
            print(f"  [{label}] FAILED: {repr(e)[:90]}")
    res["coherence"] = {k: v for k, v in coh.items()}

    # ---- P1: aneuploid vs normal
    keys = list(coh)
    if len(keys) == 2:
        a, b = keys
        common = sorted(set(coh[a]) & set(coh[b]), key=lambda x: int(x))
        va = np.array([coh[a][c]["coherence"] for c in common])
        vb = np.array([coh[b][c]["coherence"] for c in common])
        rng = np.random.default_rng(SEED)
        d = vb - va
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        print(f"\n=== P1: does ANEUPLOIDY create block structure? ===")
        print(f"  {a:<38} median {np.median(va):.2f}x")
        print(f"  {b:<38} median {np.median(vb):.2f}x")
        print(f"  difference (aneuploid − normal): {d.mean():+.2f}  "
              f"CI[{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]  "
              f"{int((d>0).sum())}/{len(d)} chromosomes higher")
        res["P1"] = dict(normal_median=float(np.median(va)), aneuploid_median=float(np.median(vb)),
                         diff=float(d.mean()), n_higher=int((d > 0).sum()), n=len(d))

    # ---- P2: does coherence predict the model's causal steering strength?
    p = os.path.join(HERE, "results", "steer_propagation_chromosome_1b_seed0.json")
    if os.path.exists(p) and coh:
        sw = json.load(open(p))["sweep"][-1]["per_cat"]
        steer = {str(x["cat"]): float(x["specific"]) for x in sw}
        from scipy.stats import spearmanr
        print(f"\n=== P2: does block coherence predict the model's per-chromosome steering strength? ===")
        res["P2"] = {}
        # GENE-COUNT CONFOUND, and it is large in BOTH directions, so the raw correlation is uninterpretable:
        # steering strength rises with chromosome gene count (more readout genes = more mass to gain,
        # measured Spearman +0.49) while coherence FALLS with it (a mean over fewer genes averages out less,
        # measured -0.36). Report the PARTIAL correlation with log(gene count) regressed out of both.
        def _resid(y, x):
            A = np.vstack([np.ones_like(x), np.log(x)]).T
            return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]

        for label in coh:
            cc = sorted(set(coh[label]) & set(steer), key=lambda x: int(x))
            x = np.array([coh[label][c]["coherence"] for c in cc])
            y = np.array([steer[c] for c in cc])
            ng = np.array([coh[label][c]["n_genes"] for c in cc], dtype=float)
            rho = float(spearmanr(x, y).statistic)
            rho_p = float(spearmanr(_resid(x, ng), _resid(y, ng)).statistic)
            rng2 = np.random.default_rng(SEED + 7)
            null = np.array([spearmanr(_resid(x, ng), _resid(y, ng)[rng2.permutation(len(y))]).statistic
                             for _ in range(20000)])
            pv = float(((null >= rho_p).sum() + 1) / (len(null) + 1))
            print(f"  coherence({label:<38}) vs steering: raw {rho:+.3f}  "
                  f"PARTIAL(gene count out) {rho_p:+.3f}  p={pv:.4f}  (n={len(cc)})")
            res["P2"][label] = dict(rho_raw=rho, rho_partial=rho_p, p=pv, n=len(cc),
                                    rho=rho_p)   # verdict uses the partial
        best = max(res["P2"].values(), key=lambda v: v["rho"])
        print(f"\n  -> {'CNV-ORIGIN SUPPORTED: chromosomes that behave as dosage blocks are the ones the model represents most strongly' if best['p'] < 0.05 and best['rho'] > 0 else 'NOT SUPPORTED: block coherence does not predict which chromosomes the model uses -- the causal shape looks dosage-like, but the model is not simply reading dosage blocks'}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "chrom_cnv_origin.json"), "w"), indent=1)
    print("\n[done] -> results/chrom_cnv_origin.json")


if __name__ == "__main__":
    main()
