"""S4d - does the model form a group ABSTRACTION, or only pairwise affinity?

Where this comes from. S4: the 20-way probe on the gene table is at chance (0.046-0.058) at every
consistency level while PPMI on the same corpus reaches 0.497. S4b: 8x more training changes nothing
(W_E 0.045 -> 0.048 from 1500 to 12000 steps while val_corr rises 0.806 -> 0.857). So the model
genuinely does not put a linear group code in its table. S4c asks whether pairwise affinity is there
instead. This asks the question that actually matters:

  Can the model relate two genes it has NEVER seen together?

That is the difference between memorising co-occurrence and forming an abstraction, and it is the
synthetic analogue of the real chromosome result's decisive control -- the 10-Mb neighbourhood
holdout, which stopped a gene being placed by recognising its near-identical neighbour.

Design. Split every group's genes into two halves H1 and H2. Every expression program draws from H1
only or from H2 only, never both. So a within-group CROSS-HALF pair never co-occurs in any cell,
while within-group same-half pairs co-occur constantly.

  held-out pairs  : same group, different half   -> never co-occur
  observed pairs  : same group, same half        -> co-occur (the memorisable ones)
  control pairs   : different group              -> never co-occur, and unrelated

Statistic: mean cosine in W_E for held-out pairs minus control pairs, against a permutation null
that shuffles group labels. If it is positive, the model linked genes that never appeared together,
which pairwise memorisation cannot do. If it is zero while observed-pair cosine is high, the model
memorised co-occurrence and never abstracted.

The gap between observed and held-out is the whole result, so both are reported.
"""
import os
import json, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import tokenize, train, gene_table  # noqa: E402
from s4_vocab_facts import probe, ppmi_svd, OUT, N_TRAIN, STEPS  # noqa: E402


def split_half_corpus(n_cells, n_genes=1000, n_groups=20, n_programs=40,
                      genes_per_program=20, lib=3000, seed=0):
    """Group-structured corpus where programs never mix a group's two halves."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, n_genes)
    half = np.zeros(n_genes, dtype=int)
    for g in range(n_groups):
        idx = np.where(group == g)[0]
        rng.shuffle(idx)
        half[idx[len(idx) // 2:]] = 1                      # H1 = 0, H2 = 1

    programs = []
    for _ in range(n_programs):
        g, h = rng.integers(0, n_groups), rng.integers(0, 2)
        pool = np.where((group == g) & (half == h))[0]
        if len(pool) < 3:
            continue
        programs.append(rng.choice(pool, size=min(genes_per_program, len(pool)), replace=False))

    base = rng.gamma(2.0, 1.0, n_genes) + 0.1
    counts = np.zeros((n_cells, n_genes), dtype=np.uint16)
    for i in range(n_cells):
        rate = base.copy()
        for p in rng.choice(len(programs), size=3, replace=False):
            rate[programs[p]] *= 8.0
        counts[i] = rng.multinomial(lib, rate / rate.sum()).astype(np.uint16)
    return counts, group, half, {"n_genes": n_genes, "n_groups": n_groups,
                                 "n_programs": len(programs),
                                 "genes_per_program": genes_per_program, "seed": seed}


def pair_cosines(X, group, half, n_null=200, seed=0):
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    C = Z @ Z.T
    iu = np.triu_indices(len(Z), 1)
    c = C[iu]
    gi, gj = group[iu[0]], group[iu[1]]
    hi, hj = half[iu[0]], half[iu[1]]

    same_g = gi == gj
    observed = same_g & (hi == hj)          # co-occur in programs
    heldout = same_g & (hi != hj)           # NEVER co-occur
    control = ~same_g

    def stat(g_lab):
        s = g_lab[iu[0]] == g_lab[iu[1]]
        ho = s & (hi != hj)
        return c[ho].mean() - c[~s].mean()

    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(group)) for _ in range(n_null)])
    real = c[heldout].mean() - c[control].mean()
    return {"observed_cos": float(c[observed].mean()),
            "heldout_cos": float(c[heldout].mean()),
            "control_cos": float(c[control].mean()),
            "observed_minus_control": float(c[observed].mean() - c[control].mean()),
            "heldout_minus_control": float(real),
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "z": float((real - null.mean()) / max(null.std(), 1e-12)),
            "p_one_sided": float((null >= real).mean()),
            "n_observed": int(observed.sum()), "n_heldout": int(heldout.sum()),
            "n_control": int(control.sum())}


def run(seed=0, n_programs=40):
    t0 = time.time()
    counts, group, half, meta = split_half_corpus(N_TRAIN, n_programs=n_programs, seed=seed)
    data = tokenize(counts, seed=seed)
    model, hist = train(data, meta["n_genes"], steps=STEPS, seed=seed, quiet=True)
    W = gene_table(model)
    P = ppmi_svd(counts, seed=seed)

    r = {"seed": seed, "n_programs": meta["n_programs"], "val_corr": hist[-1]["val_corr"],
         "probe_WE": probe(W, group, seed=seed),
         "probe_ppmi": probe(P, group, seed=seed),
         "WE": pair_cosines(W, group, half, seed=seed),
         "PPMI": pair_cosines(P, group, half, seed=seed),
         "secs": round(time.time() - t0, 1)}

    for tag in ("WE", "PPMI"):
        d = r[tag]
        print(f"  [{tag:4s}] observed-pair cos {d['observed_minus_control']:+.4f} | "
              f"HELD-OUT pair cos {d['heldout_minus_control']:+.4f} "
              f"(null {d['null_mean']:+.4f}+-{d['null_sd']:.4f}, z {d['z']:+.1f}, "
              f"p {d['p_one_sided']:.3f})")
    print(f"  probe: W_E {r['probe_WE']:.3f}  PPMI {r['probe_ppmi']:.3f}  "
          f"(chance 0.050, val {r['val_corr']:+.3f}, {r['secs']:.0f}s)")
    return r


def main(seeds=(0, 1, 2)):
    print("S4d: can the model relate genes it never saw together?\n")
    rows = []
    for s in seeds:
        print(f"seed {s}")
        rows.append(run(s))
        json.dump(rows, open(f"{OUT}/s4d_abstraction.json", "w"), indent=1)

    print("\n=== SUMMARY (mean over seeds) ===")
    for tag in ("WE", "PPMI"):
        o = np.mean([r[tag]["observed_minus_control"] for r in rows])
        h = np.mean([r[tag]["heldout_minus_control"] for r in rows])
        z = np.mean([r[tag]["z"] for r in rows])
        print(f"  {tag:4s}  observed {o:+.4f}   held-out {h:+.4f}   mean z {z:+.1f}")
    print(f"\nwrote {OUT}/s4d_abstraction.json")


if __name__ == "__main__":
    main()
