"""BEST PROBE vs BEST PROBE -- the comparison that actually decides it.

coocc_final.py probed BOTH the model tables and the shallow LSA baseline with LogisticRegression C=0.1 and
concluded the 217M loses (0.424 vs 0.692) and the 1B wins (0.796). Same probe on both sides, so not rigged --
but it is the WEAKEST probe available, and final_probe_grid.py shows the models gain a lot from stronger ones
(1B: 0.796 -> 0.880 with LinearSVC; 217M: 0.424 -> 0.506 with an MLP). The LSA baseline was never given the
same opportunity, so the verdict rests on an untested assumption: that probe capacity helps both sides equally.

If it does, the ranking survives. If the models gain and the LSA baseline does not -- plausible, since LSA-256 is
already an optimally-compressed dense code with little left for a bigger probe to extract, whereas the models'
coordinate is distributed over many low-variance directions where extra capacity pays -- then the retraction is
too harsh and has to be walked back.

Identical gene set (15,135), identical folds, three probes per representation. MLP hidden sizes scale with the
representation's own width (final_probe_grid.mlp_for) so a 256-dim baseline is not handicapped against a
2304-dim table, nor flattered.

NOTE ON POSITION: no equivalent re-run is needed there. The position comparison already used RidgeCV on both
sides, and this project established that the position code is LINEAR (gradient boost 0.26 and MLP 0.07 both lose
to linear 0.41), so the linear probe is already the model's best. Position stands as reported.

Out: results/coocc_bestprobe.json
"""
import os, sys, json, gc, time, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from table_grid import load as load_mt
from shallow_coocc_baseline import lsa
from coocc_fair import stream_binary
from final_probe_grid import mlp_for, evaluate
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

SEED = 0


def main():
    C = coords()
    tabs = {nm: load_mt(w, "output") for nm, w in [("MaxToki-217M", "217M"), ("MaxToki-1B", "1B")]}
    _, sd = G.basis("coexpr_devel")
    sd = [s for s in sd if s in C.index and C.chromosome[s] in AUTOSOMES]

    common = sorted(set.intersection(*[set(s) for _, s in tabs.values()], set(sd))
                    & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    y = np.array([C.chromosome[q] for q in common])
    st = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])
    fr = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    fg = list(GroupKFold(5).split(np.zeros(len(y)), y, groups=groups))
    print(f"{len(common)} matched genes | chance {1/22:.3f}\n")

    def sub(M, syms):
        pi = {q: i for i, q in enumerate(syms)}
        return M[[pi[q] for q in common]]

    reps = {}
    for nm, (M, s) in tabs.items():
        reps[nm] = sub(M.astype(np.float32), s)
    print("[build] streaming full 62,849-cell fetal-gut corpus ...", flush=True)
    B = stream_binary(G.FETAL_GUT, sd, n_cells=None)
    for d in (256, 512):
        reps[f"shallow LSA-{d} (full corpus)"] = sub(lsa(B, dims=d), sd)
    del B; gc.collect()

    res = {"n_genes": len(common), "chance": 1 / 22, "grid": {}}
    print(f"{'representation':<34} {'probe':<20} {'random':<9} {'group':<9} {'mins'}")
    print("-" * 80)
    for nm, X in reps.items():
        hl = mlp_for(X.shape[1])
        for pn, make in [("logistic C=0.1", lambda: LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1)),
                         ("LinearSVC C=0.01", lambda: LinearSVC(C=0.01, max_iter=4000)),
                         (f"MLP {hl}", lambda hl=hl: MLPClassifier(hidden_layer_sizes=hl, alpha=1.0,
                                                                   max_iter=600, early_stopping=True,
                                                                   random_state=SEED))]:
            t0 = time.time()
            try:
                r = evaluate(make, X, y, fr); g = evaluate(make, X, y, fg)
                res["grid"][f"{nm} | {pn}"] = dict(rep=nm, probe=pn, width=int(X.shape[1]), random=r, group=g)
                print(f"{nm:<34} {pn:<20} {r:<9.3f} {g:<9.3f} {(time.time()-t0)/60:.1f}", flush=True)
            except Exception as e:
                print(f"{nm:<34} {pn:<20} FAILED {repr(e)[:34]}", flush=True)
            json.dump(res, open(os.path.join(HERE, "results", "coocc_bestprobe.json"), "w"), indent=1)
        gc.collect()

    print("\n=== BEST PROBE PER REPRESENTATION (group split) ===")
    best = {}
    for v in res["grid"].values():
        if v["group"] > best.get(v["rep"], {"group": -1})["group"]:
            best[v["rep"]] = v
    for nm, v in sorted(best.items(), key=lambda kv: -kv[1]["group"]):
        print(f"  {nm:<34} {v['group']:.3f}  ({v['probe']})")
    bl = max((v["group"] for k, v in best.items() if k.startswith("shallow")), default=0)
    print(f"\n  strongest shallow baseline: {bl:.3f}")
    for nm in tabs:
        m = best[nm]["group"]
        print(f"  {nm:<14} {m:.3f} -> {'BEATS baseline by %+.3f' % (m - bl) if m > bl else 'LOSES by %+.3f' % (m - bl)}")
    print("\n[done] -> results/coocc_bestprobe.json")


if __name__ == "__main__":
    main()
