"""CAN A BETTER CLASSIFIER READ THE COORDINATE MORE ACCURATELY? — sweep for CHROMOSOME (and position).

The chromosome task has only ever been run with one regularised multinomial logistic regression (C=0.1). Given
what the geometry says — the code is LINEAR but lives in LOW-VARIANCE, CORRELATED directions, nearly orthogonal
to the top principal components — two families should in principle do better than plain logistic regression:

  * SHRINKAGE LDA, which models the within-class covariance explicitly (Ledoit-Wolf), the classical tool for
    "signal in low-variance directions of a correlated space";
  * WHITENING before a linear probe, which rescales those low-variance directions up instead of leaving them
    buried (per-feature StandardScaler does not decorrelate).

Every method is reported under BOTH splits, because a method can look better on random folds purely by finding a
gene's tandem duplicate:
   random 5-fold      -- can exploit near-duplicate genes
   10-Mb group split  -- whole neighbourhood held out; the honest number

Memory-lean by design (float32; no full pairwise distance matrices).
Run: ../../.venv/bin/python -u classifier_sweep.py [--basis 217M|1B]
Out: results/classifier_sweep.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from table_grid import load
from model_scale import BLOCK
from genome_wide import coords, AUTOSOMES
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score

SEED = 0
WHITEN_K = 400          # components kept for the whitening probe


def splits(X, y, groups):
    return {"random": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)),
            "group": list(GroupKFold(5).split(X, y, groups=groups))}


def run(make, X, y, folds, whiten=False):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
        if whiten:
            p = PCA(n_components=min(WHITEN_K, A.shape[0] - 1, A.shape[1]), whiten=True,
                    svd_solver="randomized", random_state=SEED).fit(A)
            A, B = p.transform(A), p.transform(B)
        m = make().fit(A, y[tr])
        pred[te] = m.predict(B)
    return float(balanced_accuracy_score(y, pred.astype(str)))


def main():
    which = "1B" if "--basis" in sys.argv and sys.argv[sys.argv.index("--basis") + 1] == "1B" else "217M"
    C = coords(); M, s = load(which, "output"); M = M.astype(np.float32)
    pi = {q: i for i, q in enumerate(s)}
    common = sorted(set(s) & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    X = M[[pi[q] for q in common]]
    y = np.array([C.chromosome[q] for q in common])
    start = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(st // BLOCK)}" for c, st in zip(y, start)])
    fl = splits(X, y, groups)
    print(f"basis {which} output | n={len(y)} genes, d={X.shape[1]} | chance = {1/22:.3f}\n")
    print(f"{'classifier':<34} {'random 5-fold':<15} {'10-Mb group split'}")
    print("-" * 70)

    methods = [
        ("logistic C=0.1 (paper baseline)", lambda: LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1), False),
        ("logistic C=1",                    lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1), False),
        ("logistic C=0.01",                 lambda: LogisticRegression(max_iter=2000, C=0.01, n_jobs=-1), False),
        ("LDA (Ledoit-Wolf shrinkage)",     lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), False),
        ("LinearSVC",                       lambda: LinearSVC(C=0.01, max_iter=5000), False),
        (f"whitened({WHITEN_K}) + logistic", lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1), True),
        (f"whitened({WHITEN_K}) + LDA",      lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), True),
    ]
    res = {"basis": which, "n_genes": len(y), "dim": int(X.shape[1]), "chance": 1 / 22, "methods": {}}
    for name, make, wh in methods:
        try:
            r = run(make, X, y, fl["random"], wh)
            g = run(make, X, y, fl["group"], wh)
            res["methods"][name] = dict(random=r, group=g)
            print(f"{name:<34} {r:<15.3f} {g:.3f}", flush=True)
        except Exception as e:
            print(f"{name:<34} FAILED: {repr(e)[:60]}", flush=True)

    base = res["methods"].get("logistic C=0.1 (paper baseline)", {})
    if base:
        best_r = max(res["methods"].items(), key=lambda kv: kv[1]["random"])
        best_g = max(res["methods"].items(), key=lambda kv: kv[1]["group"])
        print(f"\nbest on random split : {best_r[0]} {best_r[1]['random']:.3f} "
              f"(baseline {base['random']:.3f}, {best_r[1]['random']-base['random']:+.3f})")
        print(f"best on GROUP split  : {best_g[0]} {best_g[1]['group']:.3f} "
              f"(baseline {base['group']:.3f}, {best_g[1]['group']-base['group']:+.3f})")
        print("  (the group-split column is the honest one: it holds out each gene's whole neighbourhood)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"classifier_sweep_{which}.json"), "w"), indent=1)
    print(f"\n[done] -> results/classifier_sweep_{which}.json")


if __name__ == "__main__":
    main()
