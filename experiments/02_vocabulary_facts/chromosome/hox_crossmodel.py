"""DOES THE HOX-CLUSTER FINDING GENERALISE ACROSS THE PORTFOLIO? (Ihor, 2026-07-17)

THE PORTFOLIO IS FIVE MODELS, BUT ONLY THREE CAN BE ASKED THIS QUESTION.

  scGPT        learned gene-ID embedding table                      -> testable
  Geneformer   learned gene-ID embedding table (BERT MLM)           -> testable, NEVER RUN until now
  MaxToki      learned gene-ID embedding table (Llama CLM)          -> the finding (0.884 / 0.931)
  UCE          NO learned gene table -- tokens ARE frozen ESM2      -> NOT testable by construction
  STATE-SE     NO learned gene table -- same ESM2-token family      -> NOT testable by construction

UCE/STATE are not "untested", they are ALREADY ANSWERED: gm_lib's `esm2` basis IS their gene vocabulary
(19,790 x 5,120, byte-identical to route_uce/MODEL_NOTES.md), and it scores 0.023 on HOX cluster -- below its
own noise floor. A model whose gene representation is a frozen protein embedding cannot encode genomic locus in
that representation, because protein sequence does not carry it. Their row in the table below is the esm2 row.

So the real question is GENEFORMER: an independently trained, expression-only model with a learned gene table,
a different objective (masked rank-value LM vs causal LM), and a different corpus. It shares MaxToki's
tokenizer (20,274/20,275 identical mappings), so HOX genes land on the same token IDs -- no re-alignment, and
token-ID leakage is already excluded (RESULTS.md section 4: token_id alone -> 0.036).

  IF Geneformer separates the clusters -> the finding is a property of expression-trained gene tables in
     general, and the mechanism question gets much sharper.
  IF it does NOT -> the finding is MaxToki-specific (scGPT is already only 0.569), which points at MaxToki's
     corpus or objective rather than at "SCFMs learn genome structure".

Every number is scored with run_probe.py's own probe/folds, against BOTH references (gm_lib.py:20's
beat-both rule), and against each basis's OWN pure-noise floor -- because ties in the coordinate raise that
floor and it is not comparable across n (see RESULTS.md section 10).

Run: ../../.venv/bin/python -u hox_crossmodel.py
Out: results/hox_crossmodel.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
from run_probe import _folds, score, perm_p

GENE_TABLE_MODELS = ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "geneformer_v1_we", "scgpt_we"]
REFS = ["coexpr", "coexpr_devel", "esm2"]
N_FLOOR = 100
SEED = 0

h = S.H["hox_grid"]
GENES = np.array(h["genes"])
PAR = np.asarray(h["coord"][0], float)
CLU = np.asarray(h["coord"][1], float)


def noise_floor(n, d, coord, folds, reps=N_FLOOR):
    """Pure-noise |rho| for THIS n, THIS d and THIS coordinate vector (ties raise the floor)."""
    rng = np.random.default_rng(SEED)
    v = np.array([score("order", rng.standard_normal((n, d)), coord, folds) for _ in range(reps)])
    return float(v.mean()), float(v.std())


def main():
    res = {}
    for b in GENE_TABLE_MODELS + REFS:
        try:
            M, ok = G.subset(b, GENES)
        except Exception as e:
            print(f"  {b:<18} LOAD FAILED: {repr(e)[:70]}"); continue
        n, d = int(ok.sum()), M.shape[1]
        if n < 8:
            print(f"  {b:<18} skipped (n={n} < 8)"); continue
        folds = _folds(n)
        rec = {"n": n, "d": d}
        print(f"\n=== {b}  (n={n}/39 HOX genes, d={d}) ===", flush=True)
        for nm, coord in [("cluster", CLU[ok]), ("paralog", PAR[ok])]:
            s = score("order", M, coord, folds)
            p = perm_p("order", M, coord, folds, s)
            fm, fs = noise_floor(n, d, coord, folds)
            z = (s - fm) / (fs + 1e-12)
            rec[nm] = dict(stat=float(s), p=float(p), floor_mean=fm, floor_sd=fs, z=float(z))
            print(f"  {nm:<8} probe={s:.3f}  p={p:.3f}  noise floor={fm:.3f}+-{fs:.3f}  z={z:+.1f}", flush=True)
        res[b] = rec

    # ---- verdict, per model, on CLUSTER: beat BOTH references (gm_lib.py:20)
    print(f"\n{'=' * 96}\nDOES IT GENERALISE? CLUSTER (genomic locus) -- must beat coexpr AND esm2\n{'=' * 96}")
    print(f"  {'model':<18} {'cluster':<9} {'z vs floor':<11} {'beats coexpr?':<15} {'beats esm2?':<13} verdict")
    ref_scores = {r: res[r]["cluster"]["stat"] for r in REFS if r in res}
    for b in GENE_TABLE_MODELS:
        if b not in res:
            continue
        v, z = res[b]["cluster"]["stat"], res[b]["cluster"]["z"]
        bc = all(v > ref_scores[r] for r in ref_scores if r.startswith("coexpr"))
        be = v > ref_scores.get("esm2", 1e9)
        ok = bc and be and res[b]["cluster"]["p"] < 0.05
        print(f"  {b:<18} {v:<9.3f} {z:<+11.1f} {str(bc):<15} {str(be):<13} "
              f"{'** MODEL **' if ok else 'no'}")
        res[b]["cluster"]["verdict"] = bool(ok)
    print(f"\n  references: " + "  ".join(f"{r}={v:.3f}" for r, v in ref_scores.items()))
    print("  UCE / STATE-SE: their gene table IS the esm2 row above -- no learned table to test.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "hox_crossmodel.json"), "w"), indent=1)
    print("\n[done] -> results/hox_crossmodel.json")


if __name__ == "__main__":
    main()
