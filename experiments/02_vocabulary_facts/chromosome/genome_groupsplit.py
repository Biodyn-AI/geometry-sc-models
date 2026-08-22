"""IS IT LOCUS KNOWLEDGE, OR DUPLICATE LEAKAGE? — the decisive control (Ihor, 2026-07-17).

THE THREAT. A random 5-fold split puts HOXA9 in train and HOXA10 in test. They are tandem duplicates with
near-identical embeddings, so the probe can score "chr7" by memorising the twin, not by knowing the genome.
Genome-wide, the same applies to every gene family clustered in cis. So the 22-way chromosome accuracy
(genome_wide.py / genome_intersection.py) could be mostly within-neighbourhood memorisation.

THE TEST. GroupKFold where the group is a GENOMIC BLOCK (default 10 Mb): a gene's ENTIRE local neighbourhood --
its tandem array, its duplicates, its TAD -- is held out together, while the chromosome is still represented by
blocks elsewhere. The question becomes exactly the right one: given genes from OTHER parts of chr7, can you
tell that a gene whose whole neighbourhood you have never seen is on chr7?

  * If the signal is duplicate/neighbourhood leakage, it collapses toward chance.
  * If the table knows chromosome membership abstractly, it survives.

Coordinates: species_chrom.csv (UCE artifact), the SAME source as genome_wide.py -- not a second, unverified
coordinate table. Autosomes only. Balanced accuracy. The verdict-relevant contrast is model vs esm2 (esm2 is
the purest gene-family / protein-similarity representation, so if the whole effect were family clustering, esm2
would survive and the models would not).

Run: ../../.venv/bin/python -u genome_groupsplit.py
Out: results/genome_groupsplit.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES, bal_acc
from sklearn.model_selection import StratifiedKFold, GroupKFold

BASES = ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "scgpt_we", "coexpr_devel", "esm2"]
BLOCK_MB = 10.0
SEED = 0


def main():
    C = coords()
    res = {"block_mb": BLOCK_MB}
    print(f"GroupKFold with {BLOCK_MB:.0f}-Mb genomic blocks (whole neighbourhood held out together)\n")
    print(f"  {'basis':<16} {'random split':<14} {'group split':<13} {'AUROC?':<8} {'% local (lost)'}")
    for b in BASES:
        M, syms = G.basis(b)
        keep = np.array([s in C.index and C.loc[s, "chromosome"] in AUTOSOMES for s in syms])
        M2, s2 = M[keep], syms[keep]
        ch = C.loc[s2, "chromosome"].values.astype(str)
        st = C.loc[s2, "start"].values.astype(float)
        y = ch
        # group = chromosome + 10-Mb block index -> a contiguous genomic window
        blk = np.array([f"{c}:{int(p // (BLOCK_MB * 1e6))}" for c, p in zip(ch, st)])

        f_rand = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(M2, y))
        a_rand = bal_acc(M2, y, f_rand)

        # GroupKFold needs >= n_splits distinct groups per fold; it has thousands here
        f_grp = list(GroupKFold(5).split(M2, y, blk))
        a_grp = bal_acc(M2, y, f_grp)

        lost = (a_rand - a_grp) / (a_rand - 1 / 22 + 1e-12)      # fraction of above-chance signal that was local
        print(f"  {b:<16} {a_rand:<14.4f} {a_grp:<13.4f} {'':<8} {lost * 100:>5.0f}%", flush=True)
        res[b] = dict(random=float(a_rand), group=float(a_grp), n=int(len(s2)),
                      n_blocks=int(len(set(blk))), frac_local=float(lost))

    print(f"\n  chance = {1/22:.4f}.  esm2 is the family/sequence control: if the effect were gene-family")
    print("  clustering, esm2 would survive the group split. The verdict is whether the MODELS survive and")
    print("  esm2 does NOT.")
    m = res.get("maxtoki_lmhead", {}); e = res.get("esm2", {})
    if m and e:
        print(f"\n  MaxToki: {m['random']:.3f} -> {m['group']:.3f} (retains {(1-m['frac_local'])*100:.0f}% of "
              f"signal).  ESM2: {e['random']:.3f} -> {e['group']:.3f} (retains "
              f"{(1-e['frac_local'])*100:.0f}%).")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_groupsplit.json"), "w"), indent=1)
    print("\n[done] -> results/genome_groupsplit.json")


if __name__ == "__main__":
    main()
