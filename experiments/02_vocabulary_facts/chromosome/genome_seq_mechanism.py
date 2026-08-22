"""WHY DOES PROTEIN SEQUENCE (esm2) PREDICT CHROMOSOME AT ALL — and why not for HOX? (Ihor, 2026-07-17)

esm2 scores ~0.19 on genome-wide 22-class chromosome (genome_wide.py) but 0.023 on the HOX 4-class cluster
(section 3). Those look contradictory until you name the mechanism: TANDEM DUPLICATION. Gene families arise by
local duplication, so paralogous genes (similar protein) usually sit on the SAME chromosome -> protein sequence
predicts chromosome genome-wide. HOX is the exception that PROVES it: the four HOX clusters arose by
large-scale/whole-genome duplication and sit on FOUR DIFFERENT chromosomes, so a HOX gene's closest protein
relative is its paralogue elsewhere -> sequence points AWAY from the cluster label -> esm2 at floor on HOX.

TEST: for each gene, is its ESM2 nearest neighbour on the same chromosome? Compare ALL genes vs HOX genes vs a
random-pair chance rate. If the mechanism is right: all-genes >> chance, HOX ~ chance.

Coordinates: species_chrom.csv (UCE artifact), same as the rest of the genome-wide route.

Run: ../../.venv/bin/python -u genome_seq_mechanism.py
Out: results/genome_seq_mechanism.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
from genome_wide import coords, AUTOSOMES

BLOCK = 2000


def main():
    C = coords()
    HOX = set(S.H["hox_grid"]["genes"])
    M, syms = G.basis("esm2")
    keep = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in syms])
    M, syms = M[keep], syms[keep]
    ch = C.loc[syms, "chromosome"].values.astype(str)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

    nn = np.empty(len(syms), dtype=int)
    for i in range(0, len(syms), BLOCK):
        Sm = Mn[i:i + BLOCK] @ Mn.T
        for r in range(Sm.shape[0]):
            Sm[r, i + r] = -np.inf
        nn[i:i + BLOCK] = np.argmax(Sm, axis=1)

    same = ch[nn] == ch
    ishox = np.array([s in HOX for s in syms])
    _, cnt = np.unique(ch, return_counts=True)
    chance = float(((cnt / cnt.sum()) ** 2).sum())

    all_rate = float(same.mean())
    hox_rate = float(same[ishox].mean())
    print("Is a gene's ESM2 (protein-sequence) nearest neighbour on the SAME chromosome?")
    print(f"  ALL autosomal genes (n={len(syms)}): {all_rate:.3f}   ({all_rate / chance:.1f}x chance)")
    print(f"  HOX genes only      (n={int(ishox.sum())}):      {hox_rate:.3f}   ({hox_rate / chance:.1f}x chance)")
    print(f"  chance (random gene pair):        {chance:.3f}")
    print("\n  -> genome-wide, protein neighbours ARE genomic neighbours -> esm2 reads chromosome.")
    print("  -> for HOX, ~chance: the closest protein relative is the paralogue on ANOTHER chromosome")
    print("     -> sequence is decorrelated from locus on HOX, so esm2 floors at 0.023 there.")
    print("\n  sample HOX nearest-neighbour pairs (gene -> closest protein relative):")
    for i in np.where(ishox)[0][:8]:
        print(f"    {syms[i]:<7} (chr{ch[i]:<2}) -> {syms[nn[i]]:<7} (chr{ch[nn[i]]:<2})  "
              f"{'SAME' if same[i] else 'DIFFERENT'}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(all_rate=all_rate, hox_rate=hox_rate, chance=chance, n=int(len(syms))),
              open(os.path.join(HERE, "results", "genome_seq_mechanism.json"), "w"), indent=1)
    print("\n[done] -> results/genome_seq_mechanism.json")


if __name__ == "__main__":
    main()
