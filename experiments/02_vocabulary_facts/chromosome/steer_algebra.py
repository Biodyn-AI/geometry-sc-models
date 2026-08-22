"""IS THE CHROMOSOME CHANNEL A LINEAR AXIS? bidirectionality, additivity, cross-talk (Ihor, 2026-07-20).

Everything so far pushes ONE chromosome in ONE direction and asks whether the effect exists. This asks what
KIND of object the channel is, which constrains what it can be used for:

  1. BIDIRECTIONAL?  Push -alpha*d_C. A genuine signed axis should SUPPRESS chr-C ("simulate a loss") as
     cleanly as +alpha raises it. If only the positive direction works, it is an additive "switch this feature
     on" effect, not an axis -- and you could never use it to model a deletion. An early smoke run hinted at an
     asymmetry (5/6 chromosomes bidirectional, one one-way), so it needs a proper test.
  2. ADDITIVE?  Steer chr-A and chr-B simultaneously. If the joint response equals the sum of the individual
     responses, these are independent linear channels that COMPOSE, and multi-region edits are possible.
     *** MEASURED IN LOGIT SPACE, NOT PROBABILITY SPACE. *** Softmax is normalised, so raising two chromosomes'
     probability necessarily makes them compete -- probability-space sublinearity would be an artefact of the
     readout, not a fact about the model. Logits are pre-normalisation, so additivity is testable there.
  3. CROSS-TALK?  Does pushing chr-A move chr-B's genes more than it moves an average other chromosome's?
     Independent lookups -> no. A shared low-dimensional code -> yes, with structure.

Statistic for additivity: regress the joint per-gene response on the sum of the individual responses across all
genes. Slope ~1 with high R^2 = composes linearly. Slope < 1 = saturating.

Run: ../../.venv_state/bin/python -u steer_algebra.py [n_cells] [n_pairs] [alpha] [model]
Out: results/steer_algebra_<model>.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import steer_lib as SL
import steer_classifier as SC
from genome_wide import coords, AUTOSOMES

SEED = 0


def main(n_cells=12, n_pairs=8, alpha=0.5, model="1b"):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    vocab = EMB.shape[0]
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    push = alpha * mean_norm

    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok2chr = {int(t): str(C.loc[s, "chromosome"]) for ens, t in tokmap.items()
               if (s := ens2sym.get(ens)) in C.index and C.loc[s, "chromosome"] in AUTOSOMES and int(t) < vocab}
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    dirs, test_idx, chroms = {}, {}, []
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        v = EMB[tids[m]].mean(0) - gcen
        dirs[c] = v / (np.linalg.norm(v) + 1e-12)                 # unit
        test_idx[c] = np.array(tids[(tchr == c) & (~is_tr)], dtype=np.int64)
        chroms.append(c)
    pairs = [(chroms[i], chroms[(i + 7) % len(chroms)]) for i in range(min(n_pairs, len(chroms)))]
    print(f"[setup] model={model}, {len(chroms)} chromosomes, {len(pairs)} pairs, {n_cells} cells, "
          f"alpha={alpha} (push {push:.2f})\n")

    def D(vec, name):
        return SL.Direction(vec=vec, name=name, basis="embed_tokens")

    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    ridx = {c: torch.as_tensor(test_idx[c]) for c in chroms}

    def run(ids, read_pos, vec=None, mask=None, scale=1.0):
        """Return per-gene LOGIT vector at the read positions (mean over positions)."""
        if vec is None:
            lg = st.logits(ids)[read_pos]
        else:
            nv = float(np.linalg.norm(vec))
            with st.steering(D(vec, "combo"), alpha=push * nv * scale, positions=mask, site="embed"):
                lg = st.logits(ids)[read_pos]
        return lg.mean(0).numpy()

    # ---- collect
    up = {c: [] for c in chroms}; dn = {c: [] for c in chroms}
    joint = {p: [] for p in pairs}; solo = {c: [] for c in set([x for p in pairs for x in p])}
    prng = np.random.default_rng(SEED + 9)
    for i, sq in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], sq, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(sq))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        base = run(ids, read_pos)
        for c in chroms:
            up[c].append(run(ids, read_pos, dirs[c], mask) - base)
            dn[c].append(run(ids, read_pos, -dirs[c], mask) - base)
        for (a, b) in pairs:
            joint[(a, b)].append(run(ids, read_pos, dirs[a] + dirs[b], mask) - base)
        if (i + 1) % 3 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)
    up = {c: np.mean(v, 0) for c, v in up.items()}
    dn = {c: np.mean(v, 0) for c, v in dn.items()}
    joint = {p: np.mean(v, 0) for p, v in joint.items()}

    # ---- 1. bidirectionality
    print("\n=== 1. BIDIRECTIONAL? (does a negative push SUPPRESS the chromosome?) ===")
    bid = []
    for c in chroms:
        u = float(up[c][test_idx[c]].mean()); d = float(dn[c][test_idx[c]].mean())
        bid.append(dict(chrom=c, up=u, down=d, symmetric=bool(d < 0 < u)))
    n_bi = sum(b["symmetric"] for b in bid)
    mu, md = np.mean([b["up"] for b in bid]), np.mean([b["down"] for b in bid])
    print(f"  mean Δlogit on own genes: +push {mu:+.4f}   −push {md:+.4f}")
    print(f"  bidirectional (up>0 and down<0): {n_bi}/{len(bid)} chromosomes")
    print(f"  asymmetry |down|/|up| = {abs(md)/(abs(mu)+1e-12):.2f}  "
          f"({'symmetric axis' if 0.5 < abs(md)/(abs(mu)+1e-12) < 2 else 'ASYMMETRIC -- not a clean signed axis'})")

    # ---- 2. additivity, in logit space
    print("\n=== 2. ADDITIVE? (joint push vs sum of individual pushes, LOGIT space) ===")
    add = []
    for (a, b) in pairs:
        s = up[a] + up[b]
        j = joint[(a, b)]
        m = np.zeros(len(j), bool); m[tids] = True                # score on coordinate-bearing genes
        x, y = s[m], j[m]
        slope = float(np.polyfit(x, y, 1)[0])
        r = float(np.corrcoef(x, y)[0, 1])
        add.append(dict(pair=f"{a}+{b}", slope=slope, r=r))
        print(f"  chr{a}+chr{b:<4} slope {slope:+.3f}   r {r:+.3f}")
    sl = np.array([a["slope"] for a in add]); rr = np.array([a["r"] for a in add])
    print(f"  mean slope {sl.mean():.3f} (1.0 = perfectly additive)   mean r {rr.mean():.3f}")
    print(f"  -> {'COMPOSES LINEARLY: multi-region edits are possible' if sl.mean() > 0.8 and rr.mean() > 0.9 else 'SUB-ADDITIVE: the channels saturate/compete when pushed together'}")

    # ---- 3. cross-talk
    print("\n=== 3. CROSS-TALK? (does pushing A move B more than an average other chromosome?) ===")
    ct = []
    for (a, b) in pairs:
        on_b = float(up[a][test_idx[b]].mean())
        others = [float(up[a][test_idx[c]].mean()) for c in chroms if c not in (a, b)]
        ct.append(on_b - float(np.mean(others)))
    ct = np.array(ct)
    bs = np.array([ct[rng.integers(0, len(ct), len(ct))].mean() for _ in range(5000)])
    print(f"  Δ(effect of A on B's genes − on an average other chromosome) = {ct.mean():+.4f}  "
          f"CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
    print(f"  -> {'structured cross-talk between specific chromosome pairs' if np.percentile(bs,2.5) > 0 else 'no pair-specific cross-talk: the lookups are effectively independent'}")

    json.dump(dict(model=model, alpha=alpha, n_cells=len(seqs),
                   bidirectional=dict(per_chrom=bid, n_symmetric=n_bi, n=len(bid),
                                      mean_up=float(mu), mean_down=float(md),
                                      asymmetry=float(abs(md) / (abs(mu) + 1e-12))),
                   additivity=dict(pairs=add, mean_slope=float(sl.mean()), mean_r=float(rr.mean())),
                   crosstalk=dict(mean=float(ct.mean()),
                                  ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])),
              open(os.path.join(HERE, "results", f"steer_algebra_{model}.json"), "w"), indent=1)
    print(f"\n[done] -> results/steer_algebra_{model}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
         sys.argv[4] if len(sys.argv) > 4 else "1b")
