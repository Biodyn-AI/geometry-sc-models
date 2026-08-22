"""S4b - is the model's failure to learn the group fact a TRAINING BUDGET artifact?

S4 shows that at consistency 0.2 a PPMI factorisation of the corpus recovers group identity
(0.138-0.266 vs chance 0.050) while the model's gene table sits AT chance (0.044-0.053). Read
naively that says a training-free factorisation has a lower detection threshold than the model.

That reading is worthless until training length is ruled out. This programme has already been
burned by exactly this: in the hematopoietic line the entire operator ranking INVERTED between 220
and 2000 epochs -- at the short budget a random matrix beat the real operator, at the long budget it
did not. A fixed step count is not a neutral choice.

So: hold the corpus fixed, sweep the number of training steps, and watch the gene-table probe. If it
climbs with budget, the S4 null at low consistency is undertraining and must not be reported as a
model property.
"""
import os
import json, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import group_corpus, tokenize, train, gene_table  # noqa: E402
from s4_vocab_facts import probe, ppmi_svd, OUT, N_TRAIN  # noqa: E402

STEP_GRID = (1500, 3000, 6000, 12000)


def main(levels=(0.2, 0.6), seeds=(0,), n_groups=20):
    print(f"S4b: consistency {levels} x steps {STEP_GRID} x seeds {seeds}, chance 0.050\n")
    rows = []
    for cons in levels:
        for seed in seeds:
            counts, group, meta = group_corpus(N_TRAIN, n_groups=n_groups,
                                               consistency=cons, seed=seed)
            data = tokenize(counts, seed=seed)
            base = probe(ppmi_svd(counts, seed=seed), group, seed=seed)
            print(f"  consistency {cons}  seed {seed}  PPMI baseline {base:.3f}")
            for steps in STEP_GRID:
                t0 = time.time()
                model, hist = train(data, meta["n_genes"], steps=steps, seed=seed, quiet=True)
                acc = probe(gene_table(model), group, seed=seed)
                r = {"consistency": cons, "seed": seed, "steps": steps,
                     "val_corr": hist[-1]["val_corr"], "probe_model_WE": acc,
                     "probe_ppmi_baseline": base, "margin": acc - base,
                     "secs": round(time.time() - t0, 1)}
                rows.append(r)
                json.dump(rows, open(f"{OUT}/s4b_training_budget.json", "w"), indent=1)
                print(f"    steps {steps:6d}  val {r['val_corr']:+.3f}  "
                      f"W_E {acc:.3f}  margin {acc - base:+.3f}  ({r['secs']:.0f}s)")

    print("\n=== does the gene table improve with budget? ===")
    for cons in levels:
        g = [r for r in rows if r["consistency"] == cons]
        if len(g) < 2:
            continue
        s = np.array([r["steps"] for r in g], float)
        a = np.array([r["probe_model_WE"] for r in g])
        rho = float(np.corrcoef(np.log(s), a)[0, 1])
        print(f"  cons {cons}: W_E {' -> '.join(f'{v:.3f}' for v in a)} "
              f"(PPMI {g[0]['probe_ppmi_baseline']:.3f})  corr(log steps, acc) = {rho:+.2f}")
    print(f"\nwrote {OUT}/s4b_training_budget.json")


if __name__ == "__main__":
    main()
