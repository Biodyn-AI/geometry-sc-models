"""S4 - when does a model learn a fact about the VOCABULARY, and when does it use it?

The synthetic analogue of the chromosome result, with ground truth. Every gene is assigned to one
of `n_groups` arbitrary groups. Group membership appears in NO single cell -- it is recoverable only
from corpus-wide co-occurrence, exactly like a gene's chromosome. The knob is `consistency`: the
probability that an expression program draws its genes from within one group rather than at random.

Three levels, the programme's own ladder, all against ground truth:

  (a) decodable      20-way probe on the learned gene table W_E, split OVER GENES
  (b) beyond baseline same probe on a PPMI+SVD factorisation of the identical corpus at the same
                      dimensionality -- the training-free competitor the real chromosome result had
                      to beat (LSA-256 scored 0.720 there)
  (c) causally used   push a random half of a cell's gene embeddings toward group c and read the
                      model's predicted expression at the OTHER, untouched half; compare to a
                      norm-matched random push. This is steer_propagation's split-half design.

What it answers: 4.1 (what predicts appearance), 4.2 (what predicts causal use), 0.3 (is there a
threshold), and whether decodability and causal use come on at the same point -- which no real
dataset can tell us, because consistency is not a knob there.
"""
import os
import json, sys, time
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import group_corpus, tokenize, train, gene_table, batch_of, DEV  # noqa: E402

OUT = os.environ.get("GEOMSC_RESULTS",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
N_TRAIN, STEPS, DIM = 20000, 3000, 192


def probe(X, y, seed=0, folds=5):
    """Balanced accuracy of a 20-way linear probe, split over genes. Chance = 1/n_groups."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(np.mean([np.mean(p[y[te] == c] == c) for c in np.unique(y[te])]))
    return float(np.mean(accs))


def ppmi_svd(counts, dim=DIM, seed=0):
    """Training-free co-occurrence factorisation of the same corpus (the LSA-style baseline)."""
    B = (counts > 0).astype(np.float64)
    co = B.T @ B
    tot = co.sum()
    pi = co.sum(1, keepdims=True) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co / tot) / (pi @ pi.T))
    pmi[~np.isfinite(pmi)] = 0.0
    np.maximum(pmi, 0, out=pmi)
    return TruncatedSVD(n_components=min(dim, pmi.shape[0] - 1),
                        random_state=seed).fit_transform(pmi)


@torch.no_grad()
def steer_effect(model, data, group, n_groups, alphas=(0.5, 1.0, 2.0), n_cells=1500,
                 n_rand=5, seed=0):
    """Split-half causal test, per group, with the main effect removed and a dose sweep.

    For each group c: push a random half of every cell's gene embeddings along the group-c
    direction and read the model's predicted expression at the OTHER half. The statistic is the
    group-c-vs-rest contrast in the read half, MINUS the same contrast with no push -- so the
    standing "group c is predicted higher anyway" main effect cancels. The random arm uses
    `n_rand` norm-matched random directions averaged, giving a matched zero.

    Returns per-alpha dicts with the per-group paired differences, so a sign test is possible.
    """
    model.eval()
    W = model.gene_emb.weight.detach()
    scale = W.norm(dim=1).mean()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    gt = torch.from_numpy(group).to(DEV)

    dirs = torch.stack([(W[:-1][gt == c].mean(0) - W[:-1].mean(0)) for c in range(n_groups)])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)

    idx = torch.from_numpy(rng.choice(data["gid"].shape[0], n_cells, replace=False))
    gid, vin, pad, _, _ = batch_of(data, idx, DEV, gen, 0.0)
    valid = ~pad
    push = torch.from_numpy(rng.random(gid.shape) < 0.5).to(DEV) & valid
    read = (~push) & valid
    gid_group = gt[gid.clamp(max=len(gt) - 1)]
    x0 = model.gene_emb(gid) + model.val_emb(vin)

    def contrast(delta, c):
        x = x0 if delta is None else x0 + push.unsqueeze(-1) * delta
        for b in model.blocks:
            x = b(x, pad)
        pred = model.dec(x).squeeze(-1)
        m = (gid_group == c) & read
        o = read & ~m
        if m.sum() < 10 or o.sum() < 10:
            return None
        return float(pred[m].mean() - pred[o].mean())

    base = {c: contrast(None, c) for c in range(n_groups)}
    out = {}
    for a in alphas:
        real_d, rand_d = [], []
        for c in range(n_groups):
            if base[c] is None:
                continue
            r = contrast(a * scale * dirs[c], c)
            rs = [contrast(a * scale * (v / v.norm()), c)
                  for v in torch.randn(n_rand, W.shape[1], generator=gen, device=DEV)]
            rs = [v for v in rs if v is not None]
            if r is None or not rs:
                continue
            real_d.append(r - base[c])
            rand_d.append(float(np.mean(rs)) - base[c])
        real_d, rand_d = np.array(real_d), np.array(rand_d)
        diff = real_d - rand_d
        out[a] = {"real": float(real_d.mean()), "random": float(rand_d.mean()),
                  "diff": float(diff.mean()), "diff_sd": float(diff.std()),
                  "n_groups_scored": int(len(diff)),
                  "frac_groups_positive": float((diff > 0).mean())}
    return out


def run_one(consistency, seed, n_groups=20):
    t0 = time.time()
    counts, group, meta = group_corpus(N_TRAIN, n_groups=n_groups,
                                       consistency=consistency, seed=seed)
    data = tokenize(counts, seed=seed)
    model, hist = train(data, meta["n_genes"], steps=STEPS, seed=seed, quiet=True)

    acc_model = probe(gene_table(model), group, seed=seed)
    acc_base = probe(ppmi_svd(counts, seed=seed), group, seed=seed)
    steer = steer_effect(model, data, group, n_groups, seed=seed)
    top = steer[max(steer)]

    r = {"consistency": consistency, "seed": seed, "n_groups": n_groups,
         "chance": 1.0 / n_groups, "val_corr": hist[-1]["val_corr"],
         "probe_model_WE": acc_model, "probe_ppmi_baseline": acc_base,
         "margin_over_baseline": acc_model - acc_base,
         "steer_by_alpha": {str(k): v for k, v in steer.items()},
         "steer_minus_random": top["diff"], "steer_frac_pos": top["frac_groups_positive"],
         "secs": round(time.time() - t0, 1)}
    doses = " ".join(f"a{a}:{steer[a]['diff']:+.3f}" for a in sorted(steer))
    print(f"  cons {consistency:.1f} seed {seed}  val {r['val_corr']:+.3f} | "
          f"W_E {acc_model:.3f} vs PPMI {acc_base:.3f} (margin {acc_model - acc_base:+.3f}) | "
          f"steer-rand {doses}  frac+ {top['frac_groups_positive']:.2f}  ({r['secs']:.0f}s)")
    return r


def main(seeds=(0, 1), levels=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    print(f"S4: {len(levels)} consistency levels x {len(seeds)} seeds, chance = 0.050\n")
    rows = []
    for c in levels:
        for s in seeds:
            rows.append(run_one(c, s))
            json.dump(rows, open(f"{OUT}/s4_vocab_facts.json", "w"), indent=1)

    print("\n=== SUMMARY (mean over seeds) ===")
    print("  cons |  W_E   PPMI  margin | steer-rand")
    for c in levels:
        g = [r for r in rows if r["consistency"] == c]
        print(f"  {c:4.1f} | {np.mean([r['probe_model_WE'] for r in g]):.3f} "
              f"{np.mean([r['probe_ppmi_baseline'] for r in g]):.3f} "
              f"{np.mean([r['margin_over_baseline'] for r in g]):+.3f} | "
              f"{np.mean([r['steer_minus_random'] for r in g]):+.4f} "
              f"(frac+ {np.mean([r['steer_frac_pos'] for r in g]):.2f})")
    print(f"\nwrote {OUT}/s4_vocab_facts.json")


if __name__ == "__main__":
    main(seeds=tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0, 1))
