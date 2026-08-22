"""THE MATCHED VERDICT: shallow co-occurrence factorisation vs MaxToki, same genes, same probe, plus the null
that decides whether the fetal-gut signal is co-expression structure at all.

coocc_diagnose.py killed the four cheap explanations for LSA(fetal gut) = 0.551 group-split (preprocessing,
file-order smoothing, frequency, single-chromosome concentration: 21/22 chromosomes carry it). So it is probably
real, and if it is real the paper's central comparison is in trouble -- our co-expression control was a RAW
per-gene profile scoring 0.043 on the very same cells that, once FACTORISED, give 0.551. That is exactly
Barenholtz's argument, and it would mean MaxToki does not beat a properly-constructed data baseline.

Before rewriting the paper around that, three things have to hold.

  1  SHUFFLE NULL (the decisive one). Permute each gene's expression INDEPENDENTLY across cells. This destroys
     all co-expression while preserving every gene's marginal (detection rate, mean, variance) and the matrix
     geometry the SVD sees. If LSA still decodes chromosome, the signal is an artifact of the factorisation
     acting on marginals, not co-expression. It must collapse to chance.
     Also: permute gene LABELS (trivial sanity, must be chance).

  2  MATCHED GENES. The 0.551 vs 0.368 comparison used different gene sets (15,156 vs ~18,000). Redo on the
     intersection of {MaxToki-217M, MaxToki-1B, fetal gut, Tabula Sapiens, coordinates}, identical probe.

  3  DIMENSION. LSA used 256 dims, MaxToki-217M has 1232. Low dimension could be helping (denoising) or hurting.
     Sweep 64/256/1024 on both panels: if Tabula Sapiens merely needs more dimensions, the panel gap is about
     capacity, not biology; if it stays flat, the two tissues genuinely differ.

Out: results/coocc_matched.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import BLOCK
from table_grid import load as load_mt
from shallow_coocc_baseline import build_binary, lsa, probe

SEED = 0
DIM_SWEEP = [64, 256, 1024]


def panel_matrix(panel, C):
    P, syms = G.basis(panel)
    keep = [i for i, s in enumerate(syms) if s in C.index and C.chromosome[s] in AUTOSOMES]
    return np.asarray(P[keep], dtype=np.float32), np.array(syms)[keep]


def sub(M, syms, common):
    pi = {q: i for i, q in enumerate(syms)}
    return M[[pi[q] for q in common]]


def main():
    C = coords()
    res = {"chance": 1 / 22}
    rng = np.random.default_rng(SEED)

    # ---------- 1. shuffle nulls, on the panel that produced the surprise ----------
    print("=== 1. NULLS on fetal gut (must collapse to chance ~0.045) ===", flush=True)
    P, sy = panel_matrix("coexpr_devel", C)
    y = np.array([C.chromosome[s] for s in sy])
    st = C.loc[list(sy), "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])

    Pn = P.copy()
    for i in range(Pn.shape[0]):                     # independent permutation per gene: kills co-expression,
        Pn[i] = Pn[i, rng.permutation(Pn.shape[1])]  # preserves every marginal exactly
    En = lsa(build_binary(Pn)); del Pn; gc.collect()
    res["null_cell_shuffle"] = probe(En, y, groups); del En; gc.collect()
    print(f"  cell-shuffled expression   random {res['null_cell_shuffle']['random']:.3f}"
          f"  group {res['null_cell_shuffle']['group']:.3f}", flush=True)

    E = lsa(build_binary(P))
    yp = y[rng.permutation(len(y))]
    res["null_label_shuffle"] = probe(E, yp, groups)
    print(f"  shuffled chromosome labels random {res['null_label_shuffle']['random']:.3f}"
          f"  group {res['null_label_shuffle']['group']:.3f}", flush=True)
    del E, P; gc.collect()

    # ---------- 2 & 3. matched genes, dimension sweep ----------
    print("\n=== 2+3. MATCHED GENE SET ===", flush=True)
    tabs = {}
    for nm, w in [("MaxToki-217M", "217M"), ("MaxToki-1B", "1B")]:
        M, s = load_mt(w, "output"); tabs[nm] = (M.astype(np.float32), s)
    panels = {}
    for panel, nm in [("coexpr_devel", "fetal gut"), ("coexpr", "Tabula Sapiens")]:
        panels[nm] = panel_matrix(panel, C)

    common = sorted(set.intersection(*[set(s) for _, s in tabs.values()],
                                     *[set(s) for _, s in panels.values()])
                    & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    y = np.array([C.chromosome[q] for q in common])
    st = C.loc[common, "start"].values.astype(float)
    groups = np.array([f"{c}_{int(v // BLOCK)}" for c, v in zip(y, st)])
    print(f"  {len(common)} genes present in ALL tables and both panels\n", flush=True)
    print(f"{'representation':<44} {'dim':<7} {'random':<9} {'group'}")
    print("-" * 72)

    res["n_genes"] = len(common); res["rows"] = {}

    def emit(name, X):
        r = probe(X, y, groups)
        res["rows"][name] = dict(dim=int(X.shape[1]), **r)
        print(f"{name:<44} {X.shape[1]:<7} {r['random']:<9.3f} {r['group']:.3f}", flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "coocc_matched.json"), "w"), indent=1)

    for nm, (M, s) in tabs.items():
        emit(f"{nm} output table (the model)", sub(M, s, common))
    for nm, (Pp, sp_) in panels.items():
        emit(f"raw expression profile -- {nm}", sub(Pp, sp_, common))
    for nm, (Pp, sp_) in panels.items():
        B = build_binary(Pp)
        for d in DIM_SWEEP:
            E = lsa(B, dims=d)
            emit(f"LSA co-occurrence factorisation -- {nm}", sub(E, sp_, common))
            res["rows"][f"LSA co-occurrence factorisation -- {nm}"]["panel"] = nm
            res["rows"][f"LSA-{d} -- {nm}"] = res["rows"].pop(f"LSA co-occurrence factorisation -- {nm}")
            del E; gc.collect()
        del B; gc.collect()

    json.dump(res, open(os.path.join(HERE, "results", "coocc_matched.json"), "w"), indent=1)
    print("\n=== VERDICT ===")
    nulls_ok = max(res["null_cell_shuffle"]["group"], res["null_label_shuffle"]["group"]) < 0.08
    print(f"  nulls collapse to chance: {nulls_ok}")
    best_lsa = max(((k, v) for k, v in res["rows"].items() if k.startswith("LSA")),
                   key=lambda kv: kv[1]["group"], default=(None, {"group": 0}))
    mt = res["rows"].get("MaxToki-217M output table (the model)", {}).get("group", 0)
    mt1 = res["rows"].get("MaxToki-1B output table (the model)", {}).get("group", 0)
    print(f"  best shallow factorisation : {best_lsa[0]} {best_lsa[1]['group']:.3f}")
    print(f"  MaxToki-217M / 1B          : {mt:.3f} / {mt1:.3f}")
    if nulls_ok and best_lsa[1]["group"] > max(mt, mt1):
        print("  -> A SHALLOW FACTORISATION OF EXPRESSION DATA BEATS THE MODEL on matched genes.")
        print("     The paper's co-expression control was under-powered; the model-specific claim must be retracted.")
    elif nulls_ok:
        print("  -> shallow factorisation is a much stronger baseline than the raw profile, but the model still leads.")
    else:
        print("  -> NULLS DID NOT COLLAPSE: the LSA signal is an artifact of the factorisation, not co-expression.")
    print("\n[done] -> results/coocc_matched.json")


if __name__ == "__main__":
    main()
