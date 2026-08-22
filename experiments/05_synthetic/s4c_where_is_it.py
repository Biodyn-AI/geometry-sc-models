"""S4c - if the group fact is not a linear code in W_E, where is it?

S4 finding: at every consistency level the 20-way probe on the gene table sits at chance (0.046-0.058)
while a PPMI factorisation of the SAME corpus climbs to 0.497. Two deflationary explanations are
already excluded:
  * task degeneracy   - a per-gene-mean predictor scores val_corr 0.47 vs the model's 0.84, so the
                        model is doing real contextual work
  * baseline mismatch - PPMI restricted to the model's own top-128 input still scores 0.453, so the
                        structure is in what the model actually sees
Undertraining is being tested separately in S4b.

That leaves the interesting possibility: the model may encode "these two genes go together"
PAIRWISE, in attention and FFN weights, without ever forming a group ABSTRACTION that a linear
readout of the embedding table can see. A 20-class probe cannot detect pairwise affinity; a
same-group-vs-different-group cosine test can.

Battery, on one trained model per consistency level:
  a probe_WE          reproduce the S4 number
  b cosine_WE         same-group minus different-group cosine in the table, vs a label-permutation
                      null (sensitive to pairwise affinity, blind to nothing)
  c probe_WE_mlp      nonlinear probe - is the group code there but not linear?
  d probe_context     probe on context-averaged gene-token hidden states. NOTE this is NOT the same
                      claim: hidden states see which other genes are in the cell, so a positive here
                      is contextual inference, not a stored table fact.
  e regress_mean      what the table DID learn - R^2 predicting each gene's mean expression bin
  f cosine_ppmi       the same cosine test on the PPMI baseline, as a reference ceiling
"""
import os
import json, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import group_corpus, tokenize, train, gene_table, cell_embeddings, batch_of, DEV  # noqa
from s4_vocab_facts import probe, ppmi_svd, OUT, N_TRAIN, STEPS  # noqa


def cosine_test(X, group, n_null=200, seed=0):
    """mean cosine(same group) - mean cosine(different group), against a label-permutation null."""
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    C = Z @ Z.T
    iu = np.triu_indices(len(Z), 1)
    c = C[iu]

    def stat(g):
        same = g[iu[0]] == g[iu[1]]
        return c[same].mean() - c[~same].mean()

    real = stat(group)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(group)) for _ in range(n_null)])
    return {"delta_cos": float(real), "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "z": float((real - null.mean()) / max(null.std(), 1e-12)),
            "p_one_sided": float((null >= real).mean())}


@torch.no_grad()
def context_gene_states(model, data, V, n_cells=4000, bs=256):
    """Mean hidden state per gene, averaged over the cells it appears in (final block)."""
    model.eval()
    acc = np.zeros((V, model.d), dtype=np.float64)
    cnt = np.zeros(V, dtype=np.int64)
    gen = torch.Generator(device=DEV).manual_seed(0)
    for s in range(0, min(n_cells, data["gid"].shape[0]), bs):
        idx = torch.arange(s, min(s + bs, n_cells))
        gid, vin, pad, _, _ = batch_of(data, idx, DEV, gen, 0.0)
        x = model.gene_emb(gid) + model.val_emb(vin)
        for b in model.blocks:
            x = b(x, pad)
        g = gid.cpu().numpy().ravel()
        h = x.float().cpu().numpy().reshape(-1, model.d)
        m = (~pad).cpu().numpy().ravel() & (g < V)
        np.add.at(acc, g[m], h[m])
        np.add.at(cnt, g[m], 1)
    return acc / np.clip(cnt, 1, None)[:, None], cnt


def run(cons, seed=0, n_groups=20):
    t0 = time.time()
    counts, group, meta = group_corpus(N_TRAIN, n_groups=n_groups, consistency=cons, seed=seed)
    data = tokenize(counts, seed=seed)
    model, hist = train(data, meta["n_genes"], steps=STEPS, seed=seed, quiet=True)
    V = meta["n_genes"]
    W = gene_table(model)
    P = ppmi_svd(counts, seed=seed)

    # what did the table learn instead?
    mask = np.arange(data["gid"].shape[1])[None, :] < data["n"].numpy()[:, None]
    g = data["gid"].numpy()[mask]; v = data["val"].numpy()[mask]
    mu = np.zeros(V + 1); cn = np.zeros(V + 1)
    np.add.at(mu, g, v); np.add.at(cn, g, 1)
    mu = (mu / np.clip(cn, 1, None))[:V]

    H, seen = context_gene_states(model, data, V)
    ok = seen > 20

    r = {"consistency": cons, "seed": seed, "val_corr": hist[-1]["val_corr"],
         "chance": 1.0 / n_groups,
         "a_probe_WE": probe(W, group, seed=seed),
         "b_cosine_WE": cosine_test(W, group, seed=seed),
         "c_probe_WE_mlp": float(np.mean(cross_val_score(
             MLPClassifier((256,), max_iter=800, random_state=seed),
             (W - W.mean(0)) / (W.std(0) + 1e-8), group,
             cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="balanced_accuracy"))),
         "d_probe_context": probe(H[ok], group[ok], seed=seed),
         "d_n_genes_scored": int(ok.sum()),
         "e_regress_mean_r2": float(np.mean(cross_val_score(
             RidgeCV(alphas=np.logspace(-2, 3, 12)),
             (W - W.mean(0)) / (W.std(0) + 1e-8), mu, cv=5, scoring="r2"))),
         "f_cosine_ppmi": cosine_test(P, group, seed=seed),
         "f_probe_ppmi": probe(P, group, seed=seed),
         "secs": round(time.time() - t0, 1)}

    print(f"\n  consistency {cons}  (chance {r['chance']:.3f}, val_corr {r['val_corr']:+.3f})")
    print(f"    a  probe on W_E                    {r['a_probe_WE']:.3f}")
    print(f"    c  nonlinear probe on W_E          {r['c_probe_WE_mlp']:.3f}")
    b = r["b_cosine_WE"]
    print(f"    b  same-vs-diff cosine in W_E      {b['delta_cos']:+.5f}  "
          f"null {b['null_mean']:+.5f}+-{b['null_sd']:.5f}  z {b['z']:+.1f}  p {b['p_one_sided']:.3f}")
    f = r["f_cosine_ppmi"]
    print(f"    f  same-vs-diff cosine in PPMI     {f['delta_cos']:+.5f}  z {f['z']:+.1f}"
          f"   (probe {r['f_probe_ppmi']:.3f})")
    print(f"    d  probe on CONTEXT gene states    {r['d_probe_context']:.3f}  "
          f"(n={r['d_n_genes_scored']}) -- contextual, not a table fact")
    print(f"    e  W_E -> per-gene mean expression R2 {r['e_regress_mean_r2']:+.3f}")
    return r


def main(levels=(0.4, 1.0)):
    print("S4c: where does the group fact live?\n")
    rows = []
    for c in levels:
        rows.append(run(c))
        json.dump(rows, open(f"{OUT}/s4c_where_is_it.json", "w"), indent=1)
    print(f"\nwrote {OUT}/s4c_where_is_it.json")


if __name__ == "__main__":
    main()
