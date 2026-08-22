"""S4e - is the gene-embedding table load-bearing in this architecture at all?

S4/S4b/S4c say the table carries no group code (probe 0.046, nonlinear probe 0.059, cosine z +2.0
with an effect 247x smaller than PPMI's) and, strikingly, almost no gene-abundance code either
(W_E -> per-gene mean expression R2 = +0.037). A table that encodes essentially nothing linearly
raises an obvious worry: maybe the model barely uses it, and "the model does not store the fact in
W_E" is then a statement about a vestigial component, not about how models store vocabulary facts.

Decisive test: break the table and see whether the model still works.

  real      trained model, untouched
  zeroed    gene_emb set to 0 at inference -- the model loses gene identity entirely
  shuffled  gene_emb rows permuted -- gene identity preserved as a code but scrambled per gene
  frozen    a model trained from scratch with gene_emb frozen at its random init

If val_corr barely moves under zeroed/shuffled, the table is not load-bearing and S4 must be
redesigned before anything is concluded from it. If val_corr collapses, the table matters and the
S4 result stands as a real dissociation.

`frozen` is the strongest form: if a model with a permanently random gene table matches a trained
one, gene identity is not being used to solve this task.
"""
import os
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import group_corpus, tokenize, train, batch_of, DEV  # noqa: E402
from s4_vocab_facts import OUT, N_TRAIN, STEPS  # noqa: E402


@torch.no_grad()
def val_corr(model, data, n_val=1000, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.arange(min(n_val, data["gid"].shape[0]))
    gid, vin, pad, mask, val = batch_of(data, idx, DEV, gen, 0.4)
    mf = mask.detach().cpu().numpy().astype(bool)     # index on CPU, not MPS
    p = model(gid, vin, pad).float().cpu().numpy()[mf]
    t = val.float().cpu().numpy()[mf]
    return float(np.corrcoef(p, t)[0, 1])


def main(cons=1.0, seed=0):
    print(f"S4e: is W_E load-bearing?  consistency {cons}, seed {seed}\n")
    counts, group, meta = group_corpus(N_TRAIN, consistency=cons, seed=seed)
    data = tokenize(counts, seed=seed)
    V = meta["n_genes"]

    model, hist = train(data, V, steps=STEPS, seed=seed, quiet=True)
    W = model.gene_emb.weight.data.clone()
    out = {"consistency": cons, "seed": seed, "real": val_corr(model, data)}
    print(f"  real                       val_corr {out['real']:+.4f}")

    model.gene_emb.weight.data.zero_()
    out["zeroed"] = val_corr(model, data)
    print(f"  gene_emb ZEROED            val_corr {out['zeroed']:+.4f}  "
          f"(drop {out['real'] - out['zeroed']:+.4f})")

    g = torch.Generator().manual_seed(0)
    model.gene_emb.weight.data = W[torch.randperm(W.shape[0], generator=g)]
    out["shuffled"] = val_corr(model, data)
    print(f"  gene_emb ROW-SHUFFLED      val_corr {out['shuffled']:+.4f}  "
          f"(drop {out['real'] - out['shuffled']:+.4f})")
    model.gene_emb.weight.data = W

    # strongest arm: never let the table train at all
    t0 = time.time()
    frozen, fh = train(data, V, steps=STEPS, seed=seed, quiet=True, freeze_gene_emb=True)
    out["frozen_gene_emb"] = fh[-1]["val_corr"]
    print(f"  gene_emb FROZEN AT INIT    val_corr {out['frozen_gene_emb']:+.4f}  "
          f"(vs real {out['real']:+.4f})  ({time.time() - t0:.0f}s)")

    out["verdict"] = ("table is load-bearing" if out["real"] - out["zeroed"] > 0.05
                      else "TABLE IS VESTIGIAL - S4 must be redesigned")
    print(f"\n  VERDICT: {out['verdict']}")
    json.dump(out, open(f"{OUT}/s4e_is_the_table_used.json", "w"), indent=1)
    print(f"wrote {OUT}/s4e_is_the_table_used.json")


if __name__ == "__main__":
    main()
