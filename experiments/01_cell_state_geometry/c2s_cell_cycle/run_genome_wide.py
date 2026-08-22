"""GENOME-WIDE chromosome test on C2S-Scale — port of route_genemanifold/genome_wide.py.

Does the C2S context-aware activation basis know a gene's genomic location, beyond coexpr + esm2?
  T1  chromosome classification (balanced accuracy, autosomes 1-22, chance = 1/22)
  T2  the same with HOX + protocadherin arrays REMOVED (so tandem arrays can't carry it)
  T3  genomic-distance decay: does same-chromosome embedding cosine fall off with |Δstart|?
Verdict: a c2s_ctx basis must beat BOTH references. Null permutes the chromosome label (never feature-shuffle).

NB user override (2026-07-24): the standing 'no chromosome' directive is set aside for this direct request.
Out: results/genome_wide.json
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2s_gm_lib as G
import gene_sets as S
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

BASES_DIR = os.environ.get("C2S_GM_BASES", os.path.join(os.path.dirname(__file__), "bases"))
MODELS = sorted(f"c2s_ctx_L{int(os.path.basename(p)[9:11]):02d}"
                for p in glob.glob(os.path.join(BASES_DIR, "c2s_ctx_L*.npz")))
REFS = ["coexpr", "esm2"]
BASES = MODELS + REFS
AUTOSOMES = [str(i) for i in range(1, 23)]
ALPHA = 1.0e3
N_PERM = int(os.environ.get("C2S_GW_NPERM", "8"))
SEED = 0
N_PAIRS = 200_000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "genome_wide.json")


def bal_acc(X, y, folds):
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
    print(f"    {tag:<24} bal acc = {obs:.4f}  null {null.mean():.4f}+-{null.std():.4f}  "
          f"z={z:+.1f}  ({obs / (1/22):.1f}x chance)", flush=True)
    return dict(bal_acc=obs, null_mean=float(null.mean()), null_sd=float(null.std()), z=float(z),
                n=len(y), n_classes=int(len(set(y))))


def dist_decay(M, syms, C, rng):
    ch = C.loc[syms, "chromosome"].values
    st = C.loc[syms, "start"].values.astype(float)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    i = rng.integers(0, len(syms), N_PAIRS * 3); j = rng.integers(0, len(syms), N_PAIRS * 3)
    ok = (ch[i] == ch[j]) & (i != j)
    i, j = i[ok][:N_PAIRS], j[ok][:N_PAIRS]
    if len(i) < 1000:
        return None
    cos = np.sum(Mn[i] * Mn[j], axis=1); gd = np.abs(st[i] - st[j])
    return dict(rho=float(spearmanr(cos, gd).statistic), n_pairs=int(len(i)))


def main():
    C = G.coords()
    hox = set(S.H["hox_grid"]["genes"])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for b in BASES:
        if b in res:
            print(f"=== {b} [cached] ===", flush=True); continue
        try:
            M, syms = G.basis(b)
        except Exception as e:
            print(f"{b} LOAD FAILED {repr(e)[:70]}", flush=True); continue
        keep = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in syms])
        M2, s2 = M[keep], syms[keep]
        if len(s2) < 100:
            print(f"{b}: only {len(s2)} autosomal genes — skip", flush=True); continue
        y = C.loc[s2, "chromosome"].values.astype(str)
        print(f"\n=== {b}  (n={len(s2)} autosomal genes, d={M2.shape[1]}, {len(set(y))} chr) ===", flush=True)
        rec = {"n": int(len(s2)), "d": int(M2.shape[1])}
        rec["all"] = run(M2, y, "T1 all autosomal")
        pcdh = {s for s in s2 if s.startswith(("PCDHA", "PCDHB", "PCDHG"))}
        drop = np.array([(s in hox) or (s in pcdh) for s in s2])
        rec["n_dropped"] = int(drop.sum())
        rec["no_arrays"] = run(M2[~drop], y[~drop], f"T2 minus HOX+PCDH ({int(drop.sum())})")
        rec["decay"] = dist_decay(M2, s2, C, np.random.default_rng(SEED))
        if rec["decay"]:
            print(f"    {'T3 dist-decay':<24} Spearman(cos,|Δstart|) = {rec['decay']['rho']:+.4f} "
                  f"over {rec['decay']['n_pairs']:,} pairs", flush=True)
        res[b] = rec
        json.dump(res, open(OUT, "w"), indent=1)

    print(f"\n{'=' * 90}\nVERDICT — chromosome, autosomes only. beat coexpr AND esm2\n{'=' * 90}")
    for key, lbl in [("all", "all autosomal"), ("no_arrays", "HOX+PCDH removed")]:
        print(f"\n  [{lbl}]")
        refs = {r: res[r][key]["bal_acc"] for r in REFS if r in res}
        for b in MODELS:
            if b not in res:
                continue
            v = res[b][key]["bal_acc"]
            ok = refs and all(v > x for x in refs.values())
            print(f"    {b:<18} {v:.4f} (z={res[b][key]['z']:+.1f})  "
                  f"{'** beats every reference **' if ok else 'no'}", flush=True)
        print("    refs: " + "  ".join(f"{r}={v:.4f}" for r, v in refs.items()) + f"  | chance={1/22:.4f}")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n[done] -> results/genome_wide.json", flush=True)


if __name__ == "__main__":
    main()
