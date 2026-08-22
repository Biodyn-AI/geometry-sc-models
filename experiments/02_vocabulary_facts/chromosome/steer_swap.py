"""GENE SWAP — the assumption-free CEILING for Ihor's operator objection (2026-07-20).

THE OBJECTION AND WHERE IT STANDS. Ihor: chromosome is a CLUSTERING with non-parallel offsets (§5), so adding
one global vector d_C may be the wrong operation. `steer_relative.py` tested the source-relative fix
(d[g] = centroid(C) − centroid(chrom of g)) and it did NOT beat global (+0.0129 vs +0.0147 destination gap).
But BOTH assume the blobs are related by LINEAR ARITHMETIC. If that assumption is itself wrong, both operators
fail the same way and comparing them cannot reveal it.

THE ASSUMPTION-FREE VERSION. Do not do arithmetic at all: REPLACE a pushed gene's token with a REAL chromosome-C
gene, matched on abundance. No vector, no linearity, no centroid.

WHAT THIS IS AND IS NOT. It is NOT a cleaner test of "the model has a chromosome variable" -- inserting real C
genes raises C predictions partly by ordinary gene-gene CO-EXPRESSION, which the model plainly has. So the
native readout here is confounded BY DESIGN. What it IS: a CEILING. It is the strongest possible "make this
context more chr-C" intervention. If even the swap yields only a weak cell-type destination alignment, then the
weak alignment was never about the steering operator, and the objection is closed. If the swap is much
stronger, the linear-arithmetic assumption was the bottleneck.

DESIGN
  swap_C    : each pushed position's gene -> a chromosome-C gene (TRAIN half), matched on abundance decile
  swap_rand : each pushed position's gene -> a RANDOM gene, matched the same way   <- isolates C-identity
  readout   : native chr-C mass on C's HELD-OUT (test-half) genes at UNSTEERED positions, + the 18-class
              cell_type head; destination scored against real expression enrichment as everywhere else.
Abundance matching uses the tokenizer's own per-gene median (the statistic it normalises by), so a swap never
silently trades a common gene for a rare one.

Run: ../../.venv_state/bin/python -u steer_swap.py [n_cells]
Out: results/steer_swap.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import steer_lib as SL
import steer_classifier as SC
import steer_mechanism as SM
from genome_wide import coords, AUTOSOMES

SEED = 0


def main(n_cells=24):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer()
    with open(os.path.join(SC.CACHE_DIR, "steer_heads.pkl"), "rb") as f:
        heads = pickle.load(f)
    H = heads["cell_type"]; classes = np.array(H["classes"])

    EMB = SL._embed_matrix(st.xt, "embed")
    vocab = EMB.shape[0]
    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok2chr, tok2ens = {}, {}
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and t < vocab:
            tok2chr[t] = str(C.loc[s, "chromosome"]); tok2ens[t] = ens
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    is_tr = rng.random(len(tids)) < 0.5

    # abundance decile from the tokenizer's OWN per-gene median (what it normalises by)
    med = st.tok.gene_median
    ab = np.array([med.get(tok2ens[t], 0.0) for t in tids], dtype=float)
    lab = np.log1p(ab)
    cuts = np.percentile(lab, np.arange(10, 100, 10))
    dec = np.digitize(lab, cuts)
    tok_dec = {int(t): int(d) for t, d in zip(tids, dec)}

    chroms, read_idx, pool_C = [], {}, {}
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        chroms.append(c)
        read_idx[c] = np.array(tids[(tchr == c) & (~is_tr)], dtype=np.int64)
        pool_C[c] = {d: tids[m & (dec == d)] for d in range(10)}
    pool_any = {d: tids[is_tr & (dec == d)] for d in range(10)}
    print(f"[setup] {len(chroms)} chromosomes; abundance-matched swap pools built from the TRAIN half")

    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    y = np.asarray(labels["cell_type"]).astype(str)
    keep = np.array([v in set(classes) for v in y])
    ridx = {c: torch.as_tensor(read_idx[c]) for c in chroms}
    print(f"[cells] {len(seqs)}\n")

    def swap_ids(ids, push_pos, pool, srng):
        """Replace the gene at each pushed position with an abundance-matched gene from `pool`."""
        out = ids.copy()
        for p in push_pos:
            t = int(ids[p])
            d = tok_dec.get(t)
            if d is None:
                continue
            cand = pool.get(d)
            if cand is None or len(cand) == 0:
                for dd in sorted(range(10), key=lambda x: abs(x - d)):
                    cand = pool.get(dd)
                    if cand is not None and len(cand):
                        break
            if cand is not None and len(cand):
                out[p] = int(srng.choice(cand))
        return out

    base_Z, base_mass = [], {c: [] for c in chroms}
    Zc, mc = {c: [] for c in chroms}, {c: [] for c in chroms}
    Zr, mr = {c: [] for c in chroms}, {c: [] for c in chroms}
    prng = np.random.default_rng(SEED + 9)
    srng = np.random.default_rng(SEED + 77)

    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        push_pos, read_pos = gp[sh[:half]], gp[sh[half:]]

        h, lg = st.forward_both(ids)
        p = torch.softmax(lg[read_pos], -1)
        base_Z.append(SL.Steerer.pool(h, gp))
        for c in chroms:
            base_mass[c].append(float(p[:, ridx[c]].sum(-1).mean()))

        for c in chroms:
            idc = swap_ids(ids, push_pos, pool_C[c], srng)
            h, lg = st.forward_both(idc)
            p = torch.softmax(lg[read_pos], -1)
            Zc[c].append(SL.Steerer.pool(h, gp)); mc[c].append(float(p[:, ridx[c]].sum(-1).mean()))

            idr = swap_ids(ids, push_pos, pool_any, srng)
            h, lg = st.forward_both(idr)
            p = torch.softmax(lg[read_pos], -1)
            Zr[c].append(SL.Steerer.pool(h, gp)); mr[c].append(float(p[:, ridx[c]].sum(-1).mean()))
        if (i + 1) % 4 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    # ---- NATIVE (confounded by co-expression BY DESIGN -- reported as a ceiling, not as evidence)
    d_c = np.array([np.mean(mc[c]) - np.mean(base_mass[c]) for c in chroms])
    d_r = np.array([np.mean(mr[c]) - np.mean(base_mass[c]) for c in chroms])
    spec = d_c - d_r
    bs = np.array([spec[rng.integers(0, len(spec), len(spec))].mean() for _ in range(5000)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(f"\n=== NATIVE (CEILING; co-expression confounded by design) ===")
    print(f"  swap->chrC {d_c.mean():+.5f}   swap->random {d_r.mean():+.5f}   "
          f"SPECIFIC {spec.mean():+.5f} CI[{ci[0]:+.5f},{ci[1]:+.5f}]  {int((spec>0).sum())}/{len(spec)} chr +")

    # ---- DESTINATION alignment
    ENR, chrs_e, types_e = SM.build_enrichment(list(classes))
    ti = [i for i, t in enumerate(classes) if t in types_e]
    tj = [types_e.index(classes[i]) for i in ti]
    ci_ = [i for i, c in enumerate(chroms) if c in chrs_e]
    cj = [chrs_e.index(chroms[i]) for i in ci_]
    Em = ENR[np.ix_(cj, tj)]

    def dest(Zs):
        P = H["clf"].predict_proba(H["scaler"].transform(np.stack(Zs)))[keep]
        return np.bincount(P.argmax(1), minlength=len(classes)) / len(P)

    def align(Zdict, tag):
        D = np.stack([dest(Zdict[c]) for c in chroms])[np.ix_(ci_, ti)]
        zr = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        Dz = np.stack([zr(r) for r in D]); Ez = np.stack([zr(r) for r in Em])
        Cc = Dz @ Ez.T / D.shape[1]
        matched = np.diag(Cc); off = Cc[~np.eye(len(matched), dtype=bool)]
        gap = float(matched.mean() - off.mean())
        null = np.array([np.diag(Cc[rng.permutation(len(matched))]).mean() - off.mean() for _ in range(20000)])
        p = float(((null >= gap).sum() + 1) / (len(null) + 1))
        print(f"  {tag:<12} matched {matched.mean():+.4f}  mismatched {off.mean():+.4f}  GAP {gap:+.4f}  p={p:.4f}")
        return dict(matched=float(matched.mean()), mismatched=float(off.mean()), gap=gap, p=p)

    print(f"\n=== DESTINATION alignment with real expression enrichment ===")
    a_c = align(Zc, "swap->chrC")
    a_r = align(Zr, "swap->random")

    VEC_GLOBAL, VEC_REL = 0.0147, 0.0129        # steer_relative.py, same cells/protocol
    print(f"\n  CEILING COMPARISON (destination gap):")
    print(f"    gene swap (no arithmetic) : {a_c['gap']:+.4f}")
    print(f"    vector, global            : {VEC_GLOBAL:+.4f}")
    print(f"    vector, source-relative   : {VEC_REL:+.4f}")
    better = a_c["gap"] > max(VEC_GLOBAL, VEC_REL) + 0.01
    print(f"  -> {'SWAP IS CLEARLY STRONGER: linear arithmetic WAS the bottleneck' if better else 'the swap does NOT clear the vector operators: linear arithmetic was NOT the bottleneck -- the weak cell-type alignment is a property of the effect, not of how we pushed'}")

    json.dump(dict(n_cells=int(keep.sum()), n_chrom=len(chroms),
                   native=dict(swap_c=float(d_c.mean()), swap_rand=float(d_r.mean()),
                               specific=float(spec.mean()), ci=ci),
                   dest=dict(swap_c=a_c, swap_rand=a_r, vec_global=VEC_GLOBAL, vec_relative=VEC_REL),
                   swap_beats_vectors=bool(better)),
              open(os.path.join(HERE, "results", "steer_swap.json"), "w"), indent=1)
    print("\n[done] -> results/steer_swap.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
