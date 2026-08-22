"""IS THE MODEL'S CHROMOSOME ADVANTAGE USEFUL STRUCTURE, OR MEMORISATION? (Ihor, 2026-07-20)

THE QUESTION. The 1B beats the strong LSA co-occurrence baseline at supervised chromosome decoding (0.880 vs
0.720, group-split fair). Ihor: is that +0.16 *usable* genome-organisation structure, or did the model just
memorise more per-gene facts? The distinction decides whether this is a tool (infer genome layout from
expression where coordinates are UNKNOWN -- non-model organisms, unannotated genomes) or only a scientific
curiosity about a human genome whose coordinates are already a lookup.

THE TEST -- assumption-free, so memorisation is impossible. NO probe is trained; nothing is fit to the labels.
For a pair of genes, similarity = cosine of their embeddings. AUROC asks: is embedding similarity higher for
SAME-chromosome pairs than DIFFERENT-chromosome pairs? This is exactly the unannotated-genome use case: all you
have is "genes that cluster in embedding space are probably syntenic".

Two regimes:
  ALL pairs        : same-chr (any distance) vs different-chr. The raw number.
  FAR pairs (>= FAR_MB) : same-chromosome pairs must be >= FAR_MB apart, i.e. NOT neighbours. This is the
    decisive one: it removes the tandem-duplication / local co-expression confound (adjacent genes look alike
    for reasons that are not "the model knows chromosome"), the exact confound the 10-Mb group-split was built
    to kill. A basis that only knows "neighbours are similar" scores ~0.5 here; a basis that knows genome-wide
    chromosome membership scores > 0.5.

BASES: MaxToki-1B, MaxToki-217M (both lm_head), the shallow LSA co-occurrence baseline (the thing to beat),
raw co-expression profile, and ESM2 (sequence control). If 1B > LSA in the FAR regime, the advantage is real
transferable organisation. If 1B collapses to LSA there, the supervised win was memorisation.

Run: ../../.venv/bin/python -u synteny_transfer.py
Out: results/synteny_transfer.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from table_grid import load as load_mt
from shallow_coocc_baseline import lsa
from coocc_fair import stream_binary

SEED = 0
FAR_MB = 20.0
N_PAIRS = 400_000


def auroc(sim, label):
    """AUROC of `sim` predicting the boolean `label`, via the Mann-Whitney U statistic (no sklearn needed)."""
    order = np.argsort(sim)
    ranks = np.empty(len(sim)); ranks[order] = np.arange(1, len(sim) + 1)
    pos = label.sum(); neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pair_auroc(M, chrom, start, rng, far_mb):
    """Cosine-similarity AUROC for same- vs different-chromosome pairs, ALL and FAR regimes."""
    Mn = (M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    n = len(M)
    i = rng.integers(0, n, N_PAIRS * 3); j = rng.integers(0, n, N_PAIRS * 3)
    ok = i != j
    i, j = i[ok], j[ok]
    same = chrom[i] == chrom[j]
    dist = np.where(same, np.abs(start[i] - start[j]) / 1e6, np.inf)
    # batched cosine -- materialising Mn[i]*Mn[j] for all pairs at once is ~24 GB at hidden=5120 and OOMs
    cos = np.empty(len(i), np.float32)
    for a in range(0, len(i), 200_000):
        b = slice(a, a + 200_000)
        cos[b] = np.einsum("kd,kd->k", Mn[i[b]], Mn[j[b]])

    # ALL: balance same vs different by subsampling to the smaller class, up to N_PAIRS
    def balanced(mask_pos, mask_neg):
        p = np.where(mask_pos)[0]; q = np.where(mask_neg)[0]
        k = min(len(p), len(q), N_PAIRS // 2)
        sel = np.concatenate([rng.choice(p, k, replace=False), rng.choice(q, k, replace=False)])
        lab = np.zeros(len(sel), bool); lab[:k] = True
        return auroc(cos[sel], lab)

    a_all = balanced(same, ~same)
    a_far = balanced(same & (dist >= far_mb), ~same)
    return a_all, a_far, int((same & (dist >= far_mb)).sum())


def main():
    C = coords()
    print("[bases] loading MaxToki 1B/217M tables, LSA co-occurrence, co-expression, ESM2 ...", flush=True)
    bases = {}
    for nm, mdl in [("MaxToki-1B", "1B"), ("MaxToki-217M", "217M")]:
        M, s = load_mt(mdl, "output")
        bases[nm] = (M.astype(np.float32), list(s))
    for nm, key in [("co-expression (raw)", "coexpr_devel"), ("ESM2 (sequence)", "esm2")]:
        M, s = G.basis(key)
        bases[nm] = (M.astype(np.float32), list(s))

    # common gene set across the model/data/sequence bases, autosomal, with coordinates
    _, sd = G.basis("coexpr_devel")
    common = sorted(set.intersection(*[set(s) for _, s in bases.values()])
                    & set(sd) & set(C.index[C.chromosome.isin(AUTOSOMES)]))
    print(f"[bases] {len(common)} common autosomal genes")

    # LSA on exactly these genes (built on the full corpus, then subset)
    print("[lsa] streaming full 62,849-cell fetal-gut corpus ...", flush=True)
    B = stream_binary(G.FETAL_GUT, common, n_cells=None)
    bases["shallow LSA-256"] = (lsa(B, dims=256), list(common))
    del B; gc.collect()

    chrom = np.array([str(C.chromosome[g]) for g in common])
    start = np.array([float(C.loc[g, "start"]) for g in common])

    def sub(M, syms):
        pi = {q: i for i, q in enumerate(syms)}
        return M[[pi[q] for q in common]]

    print(f"\n=== SAME-CHROMOSOME retrieval AUROC (no training -> nothing memorised) ===")
    print(f"  regime FAR = same-chromosome pairs >= {FAR_MB:.0f} Mb apart (the memorisation-proof one)\n")
    print(f"  {'basis':<24} {'ALL pairs':>10} {'FAR pairs':>11}")
    res = {}
    rng = np.random.default_rng(SEED)
    for nm, (M, s) in bases.items():
        Msub = sub(M.astype(np.float32), s)
        a_all, a_far, n_far = pair_auroc(Msub, chrom, start, np.random.default_rng(SEED), FAR_MB)
        res[nm] = dict(auroc_all=a_all, auroc_far=a_far, n_far=n_far)
        print(f"  {nm:<24} {a_all:>10.4f} {a_far:>11.4f}")

    print(f"\n  (chance = 0.5000; FAR regime uses {res['MaxToki-1B']['n_far']:,} same-chr pairs)")
    m1b, lsa_r = res["MaxToki-1B"]["auroc_far"], res["shallow LSA-256"]["auroc_far"]
    m217 = res["MaxToki-217M"]["auroc_far"]
    print(f"\n=== VERDICT (FAR regime -- the usable, memorisation-proof number) ===")
    print(f"  MaxToki-1B   {m1b:.4f}")
    print(f"  LSA-256      {lsa_r:.4f}   (the strong baseline)")
    print(f"  MaxToki-217M {m217:.4f}")
    print(f"  1B − LSA = {m1b - lsa_r:+.4f}")
    if m1b > lsa_r + 0.02:
        print("  -> the model's genome-organisation structure BEATS co-occurrence factorisation with NO "
              "training: it is transferable and usable for synteny inference where labels are unknown.")
    elif m1b > lsa_r - 0.02:
        print("  -> 1B ~ LSA: the unsupervised structure ties the baseline; the supervised +0.16 is NOT "
              "extra usable organisation, it is what a trained probe extracts (partly memorisation).")
    else:
        print("  -> 1B < LSA unsupervised: the supervised advantage does NOT transfer to the label-free "
              "regime -- it was memorisation, not usable structure.")

    json.dump(dict(far_mb=FAR_MB, n_common=len(common), bases=res,
                   verdict_1b_minus_lsa_far=float(m1b - lsa_r)),
              open(os.path.join(HERE, "results", "synteny_transfer.json"), "w"), indent=1)
    print("\n[done] -> results/synteny_transfer.json")


if __name__ == "__main__":
    main()
