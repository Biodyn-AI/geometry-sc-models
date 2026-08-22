"""FINAL PROBE GRID — best linear vs best non-linear, on both models AND both baselines, matched genes.

Two questions this settles:
  (1) Does the BEAT-BOTH margin survive at higher probe capacity? The +0.06 MLP gain was measured only on
      MaxToki; a more flexible probe might lift ESM2 / co-expression too, which would erode the margin. The
      baselines must be run at the same capacity before any MLP number is quoted as an improvement.
  (2) Does the non-linear gain carry to the LARGER model, with the classifier scaled to its width?

MLP hidden sizes are SCALED TO EACH TABLE'S WIDTH (h1 ~ 0.42d capped at 1536, h2 = h1/4) so a 2304-wide or
5120-wide table is not handicapped by a probe sized for 1232.

All four tables are evaluated on the identical gene set, both splits:
   random 5-fold      -- can exploit tandem duplicates
   10-Mb group split  -- whole neighbourhood held out; the honest number

Out: results/final_probe_grid.json
"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from table_grid import load as load_mt
from model_scale import BLOCK
from genome_wide import coords, AUTOSOMES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

SEED = 0


def mlp_for(d):
    h1 = int(np.clip(d * 0.42, 256, 1536)); return (h1, max(64, h1 // 4))


def evaluate(make, X, y, folds):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        pred[te] = make().fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def main():
    C = coords()
    tabs = {}
    m217, s217 = load_mt("217M", "output"); tabs["MaxToki-217M"] = (m217.astype(np.float32), s217)
    m1b, s1b = load_mt("1B", "output");     tabs["MaxToki-1B"] = (m1b.astype(np.float32), s1b)
    for b, nm in [("esm2", "ESM2 (sequence)"), ("coexpr_devel", "co-expression (data)")]:
        M, s = G.basis(b); tabs[nm] = (M.astype(np.float32), s)
    for k, (M, s) in tabs.items():
        print(f"[load] {k:<22} {M.shape}")

    common = sorted(set.intersection(*[set(s) for _, s in tabs.values()])
                    & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    y = np.array([C.chromosome[q] for q in common])
    start = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(st // BLOCK)}" for c, st in zip(y, start)])
    fr = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    fg = list(GroupKFold(5).split(np.zeros(len(y)), y, groups=groups))
    print(f"\n[matched] {len(common)} genes present in ALL FOUR tables | chance = {1/22:.3f}\n")

    print(f"{'table':<22} {'probe':<22} {'random':<9} {'group':<9} {'mins'}")
    print("-" * 74)
    res = {"n_genes": len(common), "chance": 1 / 22, "grid": {}}
    for name, (M, s) in tabs.items():
        pi = {q: i for i, q in enumerate(s)}
        X = M[[pi[q] for q in common]]
        d = X.shape[1]; hl = mlp_for(d)
        for pname, make in [("LinearSVC C=0.01", lambda: LinearSVC(C=0.01, max_iter=4000)),
                            (f"MLP {hl}", lambda hl=hl: MLPClassifier(hidden_layer_sizes=hl, alpha=1.0,
                                                                      max_iter=600, early_stopping=True,
                                                                      random_state=SEED))]:
            t0 = time.time()
            try:
                r = evaluate(make, X, y, fr); g = evaluate(make, X, y, fg)
                res["grid"][f"{name} | {pname}"] = dict(table=name, probe=pname, width=d, random=r, group=g)
                print(f"{name:<22} {pname:<22} {r:<9.3f} {g:<9.3f} {(time.time()-t0)/60:.1f}", flush=True)
            except Exception as e:
                print(f"{name:<22} {pname:<22} FAILED {repr(e)[:40]}", flush=True)
            json.dump(res, open(os.path.join(HERE, "results", "final_probe_grid.json"), "w"), indent=1)

    print("\n=== BEAT-BOTH CHECK at each probe capacity (group split, the honest number) ===")
    for pk in ["LinearSVC", "MLP"]:
        rows = {v["table"]: v for k, v in res["grid"].items() if v["probe"].startswith(pk)}
        if len(rows) < 4:
            continue
        for mdl in ["MaxToki-217M", "MaxToki-1B"]:
            m = rows[mdl]["group"]; e = rows["ESM2 (sequence)"]["group"]; c = rows["co-expression (data)"]["group"]
            ok = m > e and m > c
            print(f"  {pk:<10} {mdl:<14} {m:.3f}  vs esm2 {e:.3f}, coexpr {c:.3f}  -> "
                  f"{'BEATS BOTH' if ok else 'MARGIN LOST'}")
    print("\n[done] -> results/final_probe_grid.json")


if __name__ == "__main__":
    main()
