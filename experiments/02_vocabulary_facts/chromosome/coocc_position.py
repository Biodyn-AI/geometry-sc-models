"""SAME DEFECT, SECOND CLAIM: within-chromosome POSITION against a properly-constructed data baseline.

The chromosome claim has just been shown to rest on an under-powered null: a RAW per-gene expression profile
scores 0.044 where a co-occurrence FACTORISATION of the identical cells scores 0.692 (coocc_final.py). The
position claim was validated against exactly the same raw-profile baseline (MaxToki +0.396 vs co-expression
+0.063), so it inherits the defect and cannot be trusted until re-run.

Identical probe to model_scale.position_rho -- per-chromosome RidgeCV, embedding-cosine dedup, mean Spearman rho
over autosomes -- applied to the shallow LSA embedding of the full 62,849-cell fetal-gut corpus, the
representation that broke the chromosome claim.

Out: results/coocc_position.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from model_scale import position_rho
from table_grid import load as load_mt
from shallow_coocc_baseline import lsa
from coocc_fair import stream_binary


def main():
    C = coords()
    res = {}
    print("=== within-chromosome position (mean Spearman rho, chance 0) ===\n", flush=True)

    for nm, w in [("MaxToki-217M", "217M"), ("MaxToki-1B", "1B")]:
        M, s = load_mt(w, "output")
        r, n = position_rho(M.astype(np.float32), list(s), C)
        res[nm] = {"rho": r, "n_chrom": n}
        print(f"  {nm:<34} rho {r:+.3f}  ({n} chromosomes)", flush=True)
        del M; gc.collect()

    Pd, sd = G.basis("coexpr_devel")
    keep = [i for i, s in enumerate(sd) if s in C.index and C.chromosome[s] in AUTOSOMES]
    Pd = np.asarray(Pd[keep], np.float32); sd = list(np.array(sd)[keep])
    r, n = position_rho(Pd, sd, C)
    res["raw profile -- fetal gut"] = {"rho": r, "n_chrom": n}
    print(f"  {'raw expression profile (the old null)':<34} rho {r:+.3f}  ({n} chromosomes)", flush=True)
    del Pd; gc.collect()

    B = stream_binary(G.FETAL_GUT, sd, n_cells=None)
    for d in [256, 512]:
        E = lsa(B, dims=d)
        r, n = position_rho(E, sd, C)
        res[f"LSA-{d} full corpus"] = {"rho": r, "n_chrom": n}
        print(f"  {f'shallow LSA-{d}, 62,849 cells':<34} rho {r:+.3f}  ({n} chromosomes)", flush=True)
        del E; gc.collect()
        json.dump(res, open(os.path.join(HERE, "results", "coocc_position.json"), "w"), indent=1)
    del B; gc.collect()

    best = max((v["rho"] for k, v in res.items() if k.startswith("LSA")), default=0)
    print("\n=== VERDICT ===")
    for nm in ("MaxToki-217M", "MaxToki-1B"):
        print(f"  {nm}: {res[nm]['rho']:+.3f} vs shallow baseline {best:+.3f} -> "
              f"{'model leads' if res[nm]['rho'] > best else 'SHALLOW BASELINE LEADS'}")
    print(f"  the old raw-profile null was {res['raw profile -- fetal gut']['rho']:+.3f}"
          f" -- understated by {best - res['raw profile -- fetal gut']['rho']:+.3f}")
    json.dump(res, open(os.path.join(HERE, "results", "coocc_position.json"), "w"), indent=1)
    print("\n[done] -> results/coocc_position.json")


if __name__ == "__main__":
    main()
