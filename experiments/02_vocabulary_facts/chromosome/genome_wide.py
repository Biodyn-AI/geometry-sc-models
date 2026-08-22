"""IS THE HOX FINDING A TANDEM-ARRAY QUIRK, OR DOES THE TABLE KNOW THE GENOME? (Ihor, 2026-07-17)

WHY THIS AND NOT THE UCE TEST I PROPOSED. The plan was "UCE has no learned gene table, so locus could only
appear in its ACTIVATIONS -- if it does, the mechanism is contextual inference from expression." **That test is
confounded and cannot be run.** `route_uce/uce_loader.py:138` builds each cell sentence as

    ordered[i] = int(chrom) + CHROM_TOKEN_OFFSET     # <- chrom-OPEN token IS the chromosome's identity
    ...
    sbs = np.argsort(chosen_starts[loc])             # <- genes sorted by GENOMIC START within the chromosome

so UCE is handed BOTH HOX coordinates explicitly, every forward pass: cluster = the chrom-open token (chr7's
differs from chr17's; `rng.shuffle(uq)` only permutes block ORDER, not identity), paralog = the within-chromosome
positional sort (HOX genes are physically ordered 1..13 along the chromosome). A positive in UCE's layers would
be the model reading its own input format. The proposed dichotomy had a third branch and that branch is true.

THE TEST THAT ACTUALLY ATTACKS THE MECHANISM. HOX is n=39 on ONE tandem array, and `hox_specific.py` showed the
protocadherin array too (0.569) -- so the open question is whether the table encodes GENOME ORGANISATION
BROADLY or only tandem arrays. `species_chrom.csv` (UCE's own artifact) gives chromosome + genomic start for
19,844 human genes. That turns the question into a ~17,000-gene, 22-class problem: **no small-n artifact
(RESULTS.md section 10), enormous power, and a direct read on the mechanism.**

  T1  CHROMOSOME CLASSIFICATION, genome-wide. Predict which autosome a gene sits on from its embedding.
      Metric = BALANCED accuracy (macro recall), because chromosome sizes are wildly unequal (chr1 has ~5x
      chr21's genes) and plain accuracy would reward "always guess chr1". Chance = 1/22 = 0.045.
  T2  THE SAME, RESTRICTED TO HOX-FREE GENES -- drops all 39 HOX genes and the protocadherin array, so a
      positive cannot be the known tandem arrays carrying the whole result.
  T3  GENOMIC-DISTANCE DECAY. For same-chromosome pairs, does embedding similarity fall off with genomic
      distance? Reported as Spearman(cosine, |start_i - start_j|) over sampled pairs.

CHOICES THAT MATTER.
  * AUTOSOMES ONLY (1..22). chrX carries X-inactivation and chrMT a huge expression signature -- both are
    trivially decodable from expression and would flatter every basis. Reported separately, never pooled.
  * The verdict rule is unchanged (gm_lib.py:20): a model basis must beat coexpr AND esm2.
  * Null permutes the chromosome label over the same genes (gm_lib's mandated null; NEVER feature-shuffle).

Run: ../../.venv/bin/python -u genome_wide.py
Out: results/genome_wide.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

_HF_HUB = _os.path.join(_os.environ.get(
    "HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface")), "hub")
CHROM_CSV = _os.path.join(
    _HF_HUB, "models--minwoosun--uce-misc", "snapshots",
    "bffb91084e4476698984e7e01f6170ce291f4074", "species_chrom.csv")
BASES = ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "geneformer_v1_we", "scgpt_we",
         "coexpr", "coexpr_devel", "esm2"]
MODELS = ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "geneformer_v1_we", "scgpt_we"]
REFS = ["coexpr", "coexpr_devel", "esm2"]
AUTOSOMES = [str(i) for i in range(1, 23)]
ALPHA = 1.0e3
N_PERM = 5           # n~17k over 22 classes: the null concentrates hard on 1/22; 5 draws pin it
SEED = 0
N_PAIRS = 200_000


def coords():
    d = pd.read_csv(CHROM_CSV)
    d = d[d.species == "human"].copy()
    d["gene_symbol"] = d.gene_symbol.astype(str).str.upper()
    d = d[~d.gene_symbol.duplicated(keep="first")]
    d["chromosome"] = d.chromosome.astype(str)
    return d.set_index("gene_symbol")


def bal_acc(X, y, folds, seed=SEED):
    """Out-of-fold balanced accuracy of a linear classifier. Balanced == macro recall, so unequal
    chromosome sizes cannot be gamed."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = RidgeClassifier(alpha=ALPHA).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def run(X, y, tag, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    obs = bal_acc(X, y, folds)
    rng = np.random.default_rng(seed)
    null = np.array([bal_acc(X, rng.permutation(y), folds) for _ in range(N_PERM)])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    print(f"    {tag:<22} balanced acc = {obs:.4f}   null {null.mean():.4f}+-{null.std():.4f}   "
          f"z={z:+.1f}   ({obs / (null.mean() + 1e-12):.1f}x chance)", flush=True)
    return dict(bal_acc=obs, null_mean=float(null.mean()), null_sd=float(null.std()), z=float(z),
                n=len(y), n_classes=int(len(set(y))))


def dist_decay(M, syms, C, rng):
    """T3: for SAME-chromosome pairs, does embedding cosine fall off with genomic distance?"""
    ch = C.loc[syms, "chromosome"].values
    st = C.loc[syms, "start"].values.astype(float)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    i = rng.integers(0, len(syms), N_PAIRS * 3)
    j = rng.integers(0, len(syms), N_PAIRS * 3)
    ok = (ch[i] == ch[j]) & (i != j)
    i, j = i[ok][:N_PAIRS], j[ok][:N_PAIRS]
    if len(i) < 1000:
        return None
    cos = np.sum(Mn[i] * Mn[j], axis=1)
    gd = np.abs(st[i] - st[j])
    r = spearmanr(cos, gd).statistic
    return dict(rho=float(r), n_pairs=int(len(i)))


OUT = os.path.join(HERE, "results", "genome_wide.json")


def main():
    C = coords()
    hox = set(S.H["hox_grid"]["genes"])

    # RESUME. Every basis is saved the moment it finishes: the coexpr bases cost ~10 min each (d=7500/8000)
    # and a crash mid-sweep previously threw away the whole run. Everything here is seeded, so a resumed
    # basis reproduces bit-for-bit -- the JSON is a cache, not a shortcut.
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if res:
        print(f"[resume] {len(res)} basis/bases already done: {', '.join(res)}\n")

    for b in BASES:
        if b in res:
            r = res[b]
            print(f"=== {b}  [cached] all={r['all']['bal_acc']:.4f}  "
                  f"no_arrays={r['no_arrays']['bal_acc']:.4f}", flush=True)
            continue
        try:
            M, syms = G.basis(b)
        except Exception as e:
            print(f"{b:<18} LOAD FAILED {repr(e)[:60]}"); continue
        keep = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in syms])
        M2, s2 = M[keep], syms[keep]
        y = C.loc[s2, "chromosome"].values.astype(str)
        print(f"\n=== {b}  (n={len(s2)} autosomal genes, d={M2.shape[1]}, {len(set(y))} chromosomes) ===",
              flush=True)
        rec = {"n": int(len(s2)), "d": int(M2.shape[1])}

        rec["all"] = run(M2, y, "T1 all autosomal")

        # T2 -- drop the known tandem arrays so they cannot be carrying the result
        pcdh = {s for s in s2 if s.startswith("PCDHA") or s.startswith("PCDHB") or s.startswith("PCDHG")}
        drop = np.array([(s in hox) or (s in pcdh) for s in s2])
        rec["n_dropped"] = int(drop.sum())
        rec["no_arrays"] = run(M2[~drop], y[~drop], f"T2 minus HOX+PCDH ({int(drop.sum())})")

        rec["decay"] = dist_decay(M2, s2, C, np.random.default_rng(SEED))
        if rec["decay"]:
            print(f"    {'T3 genomic-distance':<22} Spearman(cosine, |Δstart|) = {rec['decay']['rho']:+.4f}"
                  f"  over {rec['decay']['n_pairs']:,} same-chromosome pairs", flush=True)
        res[b] = rec
        json.dump(res, open(OUT, "w"), indent=1)          # persist per basis -- survive a crash

    print(f"\n{'=' * 100}\nVERDICT -- genome-wide chromosome, autosomes only. beat coexpr AND esm2 (gm_lib.py:20)")
    print(f"{'=' * 100}")
    for key, lbl in [("all", "all autosomal genes"), ("no_arrays", "HOX + protocadherin REMOVED")]:
        print(f"\n  [{lbl}]")
        refs = {r: res[r][key]["bal_acc"] for r in REFS if r in res}
        for b in MODELS:
            if b not in res:
                continue
            v = res[b][key]["bal_acc"]
            ok = all(v > x for x in refs.values())
            print(f"    {b:<18} {v:.4f}  (z={res[b][key]['z']:+.1f})   "
                  f"{'** beats every reference **' if ok else 'no'}")
        print("    refs: " + "  ".join(f"{r}={v:.4f}" for r, v in refs.items())
              + f"   | chance = {1 / 22:.4f}")

    json.dump(res, open(OUT, "w"), indent=1)
    print("\n[done] -> results/genome_wide.json")


if __name__ == "__main__":
    main()
