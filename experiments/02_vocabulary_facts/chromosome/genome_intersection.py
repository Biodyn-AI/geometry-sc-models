"""GENOME-WIDE CHROMOSOME, SCORED ON A COMMON GENE SET (Ihor, 2026-07-17).

WHY THIS EXISTS. genome_wide.py scores every basis on its OWN intersection with the coordinate table (n ranges
15,156-18,864). Comparing balanced accuracy across DIFFERENT gene sets is exactly the error RESULTS.md section
11's METHOD ADDENDUM warns about -- and here it is not harmless: esm2 covers ~4,000 extra protein-family genes
(Ig/TCR V-segments, olfactory receptors) that are tandem arrays on a few chromosomes, which INFLATE esm2's
own-set score (0.190) relative to the models. On the COMMON gene set the sequence control drops to ~0.105 and
the model-vs-sequence ordering changes: this is the number the beat-both verdict (gm_lib.py:20) must use.

Scores EVERY basis on the genes shared by ALL bases, with identical StratifiedKFold folds.

Run: ../../.venv/bin/python -u genome_intersection.py
Out: results/genome_intersection.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES, bal_acc, BASES, MODELS, REFS
from sklearn.model_selection import StratifiedKFold


def main():
    C = coords()
    loaded, common = {}, None
    for b in BASES:
        M, s = G.basis(b)
        keep = np.array([x in C.index and C.loc[x, "chromosome"] in AUTOSOMES for x in s])
        d = {sym: M[i] for i, sym in zip(np.where(keep)[0], s[keep])}
        loaded[b] = d
        common = set(d) if common is None else (common & set(d))
    common = sorted(common)
    y = C.loc[common, "chromosome"].values.astype(str)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(common)), y))
    print(f"common gene set: n={len(common)} shared by all {len(BASES)} bases, {len(set(y))} chromosomes\n")

    sc = {}
    for b in BASES:
        X = np.stack([loaded[b][g] for g in common]).astype(float)
        sc[b] = bal_acc(X, y, folds)
        print(f"  {b:<18} balanced acc = {sc[b]:.4f}   ({sc[b] / (1/22):.1f}x chance)", flush=True)

    print(f"\nVERDICT on the COMMON gene set -- beat coexpr AND esm2 (gm_lib.py:20):")
    refmax = max(sc[r] for r in REFS)
    res = {"n": len(common), "chance": 1 / 22, "scores": sc, "verdict": {}}
    for b in MODELS:
        wins = sc[b] > refmax
        margin = sc[b] - sc["esm2"]
        print(f"  {b:<18} {sc[b]:.4f}   margin over esm2 = {margin:+.4f}   "
              f"{'** beats every reference **' if wins else 'does NOT beat sequence'}")
        res["verdict"][b] = dict(score=sc[b], beats_all=bool(wins), margin_over_esm2=float(margin))
    print(f"\n  refs: " + "  ".join(f"{r}={sc[r]:.4f}" for r in REFS))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_intersection.json"), "w"), indent=1)
    print("\n[done] -> results/genome_intersection.json")


if __name__ == "__main__":
    main()
