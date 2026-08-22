"""S4g - the S4 steering arm, with a null that matches the DIRECTION CONSTRUCTION, not just the norm.

Why this is a rewrite rather than a rerun. In S4 the group direction is built from real embedding
rows,

    d_c = mean(W_E[group == c]) - mean(W_E),   normalised

while the control was an isotropic `randn` vector matched only on NORM. On the vanilla transformer
that control fails visibly: at consistency 0.0, where the corpus contains NO group structure and the
true effect is exactly zero, the contrast reads

    seed 0:  -0.342 / -0.682 / -0.762   (alpha 0.5 / 1 / 2),  35% of groups positive
    seed 1:  -0.617 / -1.259 / -1.205,                        40% positive

A large, dose-increasing effect where there can be none. The cause is anisotropy: any direction built
by averaging real embedding rows lives in the dominant part of the embedding distribution, and moving
along it does something systematically different from moving along an isotropic random vector, whether
or not it encodes anything. Norm-matching does not fix that; construction-matching does.

First attempt: build the direction the same way from SHUFFLED labels. That was NOT enough -- the
calibration point still read -0.28. The reason is alignment. The real arm pushes along group c's mean
and READS group c's genes, so push-set and read-set are the same set; a null that pushes along a
shuffled group c but still reads the REAL group c is structurally different (misaligned) whatever the
model learned.

The working null therefore gives each draw its OWN shuffled labelling and uses it for BOTH the
direction AND the readout, each deflated by its own no-push baseline. Both arms then have aligned
push and read sets, and the only remaining difference is whether the grouping carries real
co-occurrence structure.

Consistency 0.0 is the calibration point. If it does not read ~0, the instrument is still wrong and
nothing is reported from it.

The probe arm of S4 is unaffected and is not repeated here.
"""
import os
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_lib import group_corpus, tokenize, train, batch_of, DEV  # noqa: E402
from s4_vocab_facts import OUT, N_TRAIN, STEPS  # noqa: E402

ALPHAS = (0.5, 1.0, 2.0)
N_NULL = 5          # shuffled-label draws averaged for the null
N_CELLS = 1500


def group_dirs(W, labels, n_groups):
    """mean(rows of group c) - global mean, normalised. Built identically for real and shuffled."""
    d = torch.stack([W[labels == c].mean(0) - W.mean(0) for c in range(n_groups)])
    return d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)


@torch.no_grad()
def steer(model, data, group, n_groups, seed=0):
    model.eval()
    W = model.gene_emb.weight.detach()[:-1]        # drop pad row
    scale = W.norm(dim=1).mean()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    gt = torch.from_numpy(group).to(DEV)

    # The null must match the ALIGNMENT between push-set and read-set, not just the direction
    # construction. The real arm pushes along group c's mean and reads group c's genes. An earlier
    # version pushed along a SHUFFLED group c but still read the REAL group c, so the arms differed
    # structurally (aligned vs misaligned) regardless of learned content -- which is why the
    # consistency-0.0 calibration point read -0.28 instead of 0. Each null draw therefore carries its
    # own shuffled labelling, used for BOTH the direction and the readout.
    real_d = group_dirs(W, gt, n_groups)
    null_lab = [torch.from_numpy(rng.permutation(group)).to(DEV) for _ in range(N_NULL)]
    null_d = [group_dirs(W, lab, n_groups) for lab in null_lab]

    idx = torch.from_numpy(rng.choice(data["gid"].shape[0], N_CELLS, replace=False))
    gid, vin, pad, _, _ = batch_of(data, idx, DEV, gen, 0.0)
    valid = ~pad
    push = torch.from_numpy(rng.random(gid.shape) < 0.5).to(DEV) & valid
    read = (~push) & valid
    gid_group = gt[gid.clamp(max=len(gt) - 1)]
    x0 = model.gene_emb(gid) + model.val_emb(vin)

    def contrast(delta, c, labels=None):
        """labels=None uses the real grouping; a null draw passes ITS OWN labelling."""
        lab = gid_group if labels is None else labels[gid.clamp(max=len(group) - 1)]
        x = x0 if delta is None else x0 + push.unsqueeze(-1) * delta
        for b in model.blocks:
            x = b(x, pad)
        if hasattr(model, "norm"):
            x = model.norm(x)
        pred = model.dec(x).squeeze(-1)
        m = (lab == c) & read
        o = read & ~m
        if m.sum() < 10 or o.sum() < 10:
            return None
        return float(pred[m].mean() - pred[o].mean())

    base = {c: contrast(None, c) for c in range(n_groups)}
    base_null = [{c: contrast(None, c, lab) for c in range(n_groups)} for lab in null_lab]
    out = {}
    for a in ALPHAS:
        rd, nd = [], []
        for c in range(n_groups):
            if base[c] is None:
                continue
            r = contrast(a * scale * real_d[c], c)
            ns = []
            for k, nd_ in enumerate(null_d):
                v = contrast(a * scale * nd_[c], c, null_lab[k])
                b0 = base_null[k][c]
                if v is not None and b0 is not None:
                    ns.append(v - b0)          # each null draw deflated by ITS OWN baseline
            if r is None or not ns:
                continue
            rd.append(r - base[c]); nd.append(float(np.mean(ns)))
        rd, nd = np.array(rd), np.array(nd)
        diff = rd - nd
        out[a] = {"real": float(rd.mean()), "null": float(nd.mean()),
                  "diff": float(diff.mean()), "diff_sd": float(diff.std()),
                  "frac_positive": float((diff > 0).mean()), "n_groups": int(len(diff))}
    return out


def run(cons, seed, n_groups=20):
    t0 = time.time()
    counts, group, meta = group_corpus(N_TRAIN, n_groups=n_groups, consistency=cons, seed=seed)
    data = tokenize(counts, seed=seed)
    model, hist = train(data, meta["n_genes"], steps=STEPS, seed=seed, quiet=True)
    s = steer(model, data, group, n_groups, seed=seed)
    doses = "  ".join(f"a{a}:{s[a]['diff']:+.3f}" for a in ALPHAS)
    top = s[ALPHAS[-1]]
    print(f"  cons {cons:.1f} seed {seed}  val {hist[-1]['val_corr']:+.3f} | {doses} | "
          f"frac+ {top['frac_positive']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"consistency": cons, "seed": seed, "val_corr": hist[-1]["val_corr"],
            "by_alpha": {str(k): v for k, v in s.items()}}


def main(levels=(0.0, 0.4, 1.0), seeds=(0, 1)):
    print("S4g: steering with a CONSTRUCTION-MATCHED null (shuffled group labels)")
    print("     consistency 0.0 is the calibration point -- it MUST read ~0\n")
    rows = []
    for c in levels:
        for s in seeds:
            rows.append(run(c, s))
            json.dump(rows, open(f"{OUT}/s4g_steering_matched_null.json", "w"), indent=1)
    print("\n=== SUMMARY (mean over seeds, alpha=2.0) ===")
    for c in levels:
        g = [r["by_alpha"]["2.0"] for r in rows if r["consistency"] == c]
        print(f"  cons {c:.1f}  diff {np.mean([x['diff'] for x in g]):+.4f}  "
              f"frac+ {np.mean([x['frac_positive'] for x in g]):.2f}")
    print(f"\nwrote {OUT}/s4g_steering_matched_null.json")


if __name__ == "__main__":
    main()
