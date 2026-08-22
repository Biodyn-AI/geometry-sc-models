"""INPUT vs OUTPUT table, 217M vs 1B — the full 2x2 grid, at native width AND dimension-controlled.

Four learned per-gene tables, all evaluated on the IDENTICAL gene set with identical probes:
    217M input  (model.embed_tokens.weight, d=1232)   217M output (lm_head.weight, d=1232)
    1B   input  (model.embed_tokens.weight, d=2304)   1B   output (lm_head.weight, d=2304)

Metrics (stated explicitly; they are NOT comparable to one another):
    chrom_random  22-class BALANCED ACCURACY, random 5-fold                 (chance = 1/22 = 0.045)
    chrom_group   22-class BALANCED ACCURACY, 10-Mb genomic GroupKFold      (whole neighbourhood held out)
    position_rho  within-chromosome SPEARMAN rho, near-duplicates removed, random folds

Widths: NATIVE (as deployed), plus isotropic Gaussian RANDOM PROJECTION of all four tables to a common width
(512, 1024). Random projection, not PCA truncation: the coordinate occupies low-variance directions, which PCA
truncation would preferentially discard (a biased control).

Out: results/table_grid.json
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
from model_scale import chrom_acc, position_rho, BLOCK

MSETUP = f"{_DATA}/maxtoki/setup"
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
PROJ_DIMS = [512, 1024]
SEED = 0


def load(model, table):
    """model in {217M,1B}; table in {input,output}. Returns (matrix, symbols)."""
    key = "model.embed_tokens.weight" if table == "input" else "lm_head.weight"
    path = f"{MSETUP}/MaxToki-{'217M' if model == '217M' else '1B'}-HF/model.safetensors"
    W = G.ST_Reader(path).get(key).astype(np.float32)
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


def measure(X, syms, chrom, blocks, C):
    return dict(chrom_random=chrom_acc(X, chrom),
                chrom_group=chrom_acc(X, chrom, groups=blocks),
                position_rho=position_rho(X, syms, C)[0])


def main():
    C = coords()
    tabs, symsets = {}, []
    for m in ("217M", "1B"):
        for t in ("input", "output"):
            W, s = load(m, t); tabs[(m, t)] = (W, s); symsets.append(set(s))
            print(f"[load] {m:<5} {t:<7} {W.shape}")
    # identical gene set for all four tables
    common = sorted(set.intersection(*symsets) & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    syms = np.array(common)
    chrom = np.array([C.chromosome[s] for s in syms])
    start = C.loc[list(syms), "start"].values.astype(float)
    blocks = np.array([f"{c}_{int(st // BLOCK)}" for c, st in zip(chrom, start)])
    print(f"[matched] {len(syms)} genes evaluated in ALL FOUR tables\n")

    X = {}
    for k, (W, s) in tabs.items():
        pi = {q: i for i, q in enumerate(s)}
        X[k] = W[[pi[q] for q in common]]

    res = {"n_genes": len(syms), "chance_chrom": 1 / 22, "native": {}, "projected": {}}

    print("=== NATIVE WIDTH (as deployed) ===")
    print(f"{'model':<6} {'table':<8} {'width':<7} {'chrom random':<14} {'chrom group':<13} {'position rho'}")
    for m in ("217M", "1B"):
        for t in ("input", "output"):
            r = measure(X[(m, t)], syms, chrom, blocks, C)
            res["native"][f"{m}_{t}"] = dict(width=int(X[(m, t)].shape[1]), **r)
            print(f"{m:<6} {t:<8} {X[(m,t)].shape[1]:<7} {r['chrom_random']:<14.3f} "
                  f"{r['chrom_group']:<13.3f} {r['position_rho']:+.3f}", flush=True)

    rng = np.random.default_rng(SEED)
    for D in PROJ_DIMS:
        print(f"\n=== RANDOM PROJECTION to D={D} (all four tables, isotropic) ===")
        print(f"{'model':<6} {'table':<8} {'chrom random':<14} {'chrom group':<13} {'position rho'}")
        res["projected"][str(D)] = {}
        for m in ("217M", "1B"):
            for t in ("input", "output"):
                A = X[(m, t)]
                R = rng.standard_normal((A.shape[1], D)).astype(np.float32) / np.sqrt(D)
                Xp = (A - A.mean(0)) @ R
                r = measure(Xp, syms, chrom, blocks, C)
                res["projected"][str(D)][f"{m}_{t}"] = r
                print(f"{m:<6} {t:<8} {r['chrom_random']:<14.3f} {r['chrom_group']:<13.3f} "
                      f"{r['position_rho']:+.3f}", flush=True)

    # summary contrasts
    print("\n=== CONTRASTS ===")
    for D, tag in [(None, "native")] + [(d, f"D={d}") for d in PROJ_DIMS]:
        blk = res["native"] if D is None else res["projected"][str(D)]
        for m in ("217M", "1B"):
            i, o = blk[f"{m}_input"], blk[f"{m}_output"]
            print(f"  {tag:<8} {m:<5} output-minus-input: chrom_random {o['chrom_random']-i['chrom_random']:+.3f}"
                  f" | position {o['position_rho']-i['position_rho']:+.3f}")
        for t in ("input", "output"):
            a, b = blk[f"217M_{t}"], blk[f"1B_{t}"]
            print(f"  {tag:<8} {t:<6} 1B-minus-217M   : chrom_random {b['chrom_random']-a['chrom_random']:+.3f}"
                  f" | position {b['position_rho']-a['position_rho']:+.3f}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "table_grid.json"), "w"), indent=1)
    print("\n[done] -> results/table_grid.json")


if __name__ == "__main__":
    main()
