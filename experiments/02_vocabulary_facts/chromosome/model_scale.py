"""DOES THE GENOMIC COORDINATE SHARPEN WITH MODEL SCALE? — matched 217M vs 1B comparison.

position_improve.py found the 1B output table reads position far better than the 217M one (0.626 vs 0.412), but
the two runs used slightly different gene subsets (embedding-based de-duplication keeps different genes per
model), so the comparison was not matched. This script fixes that and adds the missing chromosome measurement.

Both models are evaluated on the IDENTICAL gene set (genes present in both output tables with a curated autosome
coordinate), with every metric and its leakage control stated explicitly:

  chromosome_random   22-class balanced accuracy, random 5-fold                     (chance = 1/22 = 0.045)
  chromosome_group    22-class balanced accuracy, 10-Mb genomic GroupKFold          (neighbourhood held out)
  position_rho        within-chromosome Spearman rho, near-duplicates removed,
                      random folds                                                  (leakage-clean, artefact-free)

Out: results/model_scale.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from genome_position_geometry import dedup
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

MSETUP = f"{_DATA}/maxtoki/setup"
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
ALPHAS = np.logspace(0, 5, 12)
BLOCK, MINCHR, SEED = 10e6, 200, 0


def load_table(which):
    """which='217M' -> gm_lib maxtoki_lmhead ; '1B' -> MaxToki-1B lm_head, same symbol mapping."""
    if which == "217M":
        return G.basis("maxtoki_lmhead")
    R = G.ST_Reader(f"{MSETUP}/MaxToki-1B-HF/model.safetensors")
    W = R.get("lm_head.weight").astype(np.float32)
    tok = json.load(open(f"{MSETUP}/token_dictionary.json"))
    e2s = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    rows, syms = [], []
    for ens, tid in tok.items():
        s = e2s.get(ens)
        if s is not None and tid < W.shape[0]:
            rows.append(tid); syms.append(s)
    o = np.argsort(syms); rows, syms = np.array(rows)[o], np.array(syms)[o]
    _, keep = np.unique(syms, return_index=True)
    return W[rows[keep]], syms[keep]


def chrom_acc(X, y, groups=None):
    """22-class balanced accuracy; grouped folds if groups given (neighbourhood holdout)."""
    if groups is None:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    else:
        splits = list(GroupKFold(5).split(X, y, groups=groups))
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def position_rho(M, syms, C):
    pi = {s: i for i, s in enumerate(syms)}
    rr = []
    for c in AUTOSOMES:
        g = [s for s in C.index[C.chromosome == c] if s in pi]
        if len(g) < MINCHR:
            continue
        Xf = M[[pi[s] for s in g]]; start = C.loc[g, "start"].values.astype(float)
        keep = dedup(Xf, start)
        if keep.sum() < 120:
            continue
        X, y = Xf[keep], start[keep]
        P = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            sc = StandardScaler().fit(X[tr])
            P[te] = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        r = spearmanr(P, y).statistic
        rr.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(rr)), len(rr)


def main():
    C = coords()
    M217, s217 = load_table("217M"); print(f"[load] 217M table {M217.shape}")
    M1B, s1B = load_table("1B");     print(f"[load] 1B  table {M1B.shape}")

    # MATCHED gene set: present in both tables and carrying an autosome coordinate
    common = sorted(set(s217) & set(s1B) & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    i217 = {s: i for i, s in enumerate(s217)}; i1B = {s: i for i, s in enumerate(s1B)}
    A = M217[[i217[s] for s in common]]; B = M1B[[i1B[s] for s in common]]
    syms = np.array(common)
    chrom = np.array([C.chromosome[s] for s in syms])
    start = C.loc[list(syms), "start"].values.astype(float)
    blocks = np.array([f"{c}_{int(st // BLOCK)}" for c, st in zip(chrom, start)])
    print(f"[matched] {len(syms)} genes evaluated in BOTH models\n")

    res = {"n_genes_matched": len(syms), "chance_chrom": 1/22}
    for name, X in [("maxtoki_217M", A), ("maxtoki_1B", B)]:
        cr = chrom_acc(X, chrom)
        cg = chrom_acc(X, chrom, groups=blocks)
        pr, nch = position_rho(X, syms, C)
        res[name] = dict(dim=int(X.shape[1]), chromosome_random=cr, chromosome_group=cg,
                         position_rho=pr, n_chr_position=nch)
        print(f"{name:<14} dim={X.shape[1]:<5} chromosome: random {cr:.3f} | 10-Mb group {cg:.3f} "
              f"|| position rho {pr:+.3f} ({nch} chr)", flush=True)

    a, b = res["maxtoki_217M"], res["maxtoki_1B"]
    print(f"\nSCALING (matched genes, identical probes):")
    print(f"  chromosome random   {a['chromosome_random']:.3f} -> {b['chromosome_random']:.3f} "
          f"({b['chromosome_random']-a['chromosome_random']:+.3f})")
    print(f"  chromosome group    {a['chromosome_group']:.3f} -> {b['chromosome_group']:.3f} "
          f"({b['chromosome_group']-a['chromosome_group']:+.3f})")
    print(f"  position rho        {a['position_rho']:+.3f} -> {b['position_rho']:+.3f} "
          f"({b['position_rho']-a['position_rho']:+.3f})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "model_scale.json"), "w"), indent=1)
    print("\n[done] -> results/model_scale.json")


if __name__ == "__main__":
    main()
