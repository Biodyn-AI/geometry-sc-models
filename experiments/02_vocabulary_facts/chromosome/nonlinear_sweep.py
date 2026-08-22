"""NON-LINEAR CLASSIFIERS for the 22-class chromosome task.

The earlier chromosome sweep was entirely linear (logistic / LDA / LinearSVC / whitening); the non-linear tests
in this project were run on the POSITION regression, not on chromosome classification. This closes that gap.

Families tested: instance-based (k-NN, cosine), kernel (Nystroem RBF -> linear), neural (MLP, two capacities),
and trees (random forest, histogram gradient boosting).

Every method is reported under BOTH splits, which is the point:
   random 5-fold      -- a method can score well by finding a gene's tandem DUPLICATE
   10-Mb group split  -- whole neighbourhood held out; the honest number
k-NN is included precisely because it should expose that gap if it exists.

Reference (linear, same genes): logistic C=0.1 -> 0.485 / 0.368 ; LinearSVC C=0.01 -> 0.507 / 0.397.

Run: ../../.venv/bin/python -u nonlinear_sweep.py
Out: results/nonlinear_sweep.json
"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from table_grid import load
from model_scale import BLOCK
from genome_wide import coords, AUTOSOMES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

SEED = 0
BASE = {"logistic C=0.1": (0.485, 0.368), "LinearSVC C=0.01": (0.507, 0.397)}


def evaluate(make, X, y, folds):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = make().fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def main():
    C = coords(); M, s = load("217M", "output"); M = M.astype(np.float32)
    pi = {q: i for i, q in enumerate(s)}
    common = sorted(set(s) & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    X = M[[pi[q] for q in common]]
    y = np.array([C.chromosome[q] for q in common])
    start = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(st // BLOCK)}" for c, st in zip(y, start)])
    fr = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    fg = list(GroupKFold(5).split(X, y, groups=groups))
    print(f"n={len(y)} genes, d={X.shape[1]}, 22 classes, chance={1/22:.3f}")
    print("reference linear:  logistic 0.485/0.368   LinearSVC 0.507/0.397\n")
    print(f"{'non-linear classifier':<34} {'random':<9} {'group':<9} {'gap (inflation)':<16} {'mins'}")
    print("-" * 82)

    methods = [
        ("kNN k=1 (cosine)",   lambda: KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)),
        ("kNN k=15 (cosine)",  lambda: KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1)),
        ("Nystroem RBF + SVC", lambda: make_pipeline(
            Nystroem(gamma=1.0 / X.shape[1], n_components=1500, random_state=SEED),
            LinearSVC(C=0.01, max_iter=4000))),
        ("MLP (256,)",         lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0, max_iter=400,
                                                     early_stopping=True, random_state=SEED)),
        ("MLP (512,128)",      lambda: MLPClassifier(hidden_layer_sizes=(512, 128), alpha=1.0, max_iter=400,
                                                     early_stopping=True, random_state=SEED)),
        ("RandomForest(300)",  lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                                              min_samples_leaf=2)),
        ("HistGradientBoost",  lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                                                      max_depth=6, random_state=SEED)),
    ]
    res = {}
    for name, make in methods:
        t0 = time.time()
        try:
            r = evaluate(make, X, y, fr)
            g = evaluate(make, X, y, fg)
            mins = (time.time() - t0) / 60
            res[name] = dict(random=r, group=g, gap=r - g, minutes=round(mins, 1))
            print(f"{name:<34} {r:<9.3f} {g:<9.3f} {r-g:<+16.3f} {mins:.1f}", flush=True)
        except Exception as e:
            print(f"{name:<34} FAILED: {repr(e)[:52]}", flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "nonlinear_sweep.json"), "w"), indent=1)

    if res:
        bg = max(res.items(), key=lambda kv: kv[1]["group"])
        print(f"\nbest NON-LINEAR on the honest group split: {bg[0]} {bg[1]['group']:.3f}")
        print(f"best LINEAR                              : LinearSVC 0.397")
        print(f"-> non-linear {'BEATS' if bg[1]['group'] > 0.397 else 'does NOT beat'} the linear probe")
        print("   (a large random-minus-group gap = the method is exploiting tandem duplicates, not learning locus)")
    print("\n[done] -> results/nonlinear_sweep.json")


if __name__ == "__main__":
    main()
