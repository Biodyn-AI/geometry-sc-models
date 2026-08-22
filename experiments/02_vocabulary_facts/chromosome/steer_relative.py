"""IS THE STEERING OPERATOR ITSELF WRONG? global vs SOURCE-RELATIVE push (Ihor's objection, 2026-07-20).

IHOR'S POINT. Chromosome is stored as a CLUSTERING, not an ordered axis: §5 TEST 1 showed the cluster label is
invariant to random re-coding (unordered blobs), and §5 TEST 2 showed the blob-to-blob offsets are only ~0.2
aligned where a genuine grid needs ~1.0. If that is right, then "add one global vector d_C to every pushed
gene" is the WRONG OPERATION. d_C = centroid(C) - global_centroid. Adding it to a gene already in C's blob
pushes it deeper in; adding the SAME vector to a gene sitting in chr3's blob does not make it C-like, it just
translates it somewhere. The correct displacement should depend on WHERE THE GENE STARTS.

This also predicts the exact asymmetry observed: the NATIVE readout worked (a crude uniform shove still makes
the context look C-ish in aggregate) while the fine-grained CELL-TYPE alignment failed (that needs the pushed
genes to actually behave like specific C genes). So the null may be a fact about my operator, not the model.

THE TEST. Same cells, same targets, same magnitude, three operators:
  global    d[g] = centroid(C) - centroid(all)                 <- what every previous run used
  relative  d[g] = centroid(C) - centroid(chromosome of g)     <- Ihor's operator: source-dependent
  sham      d[g] = centroid(random gene set) - centroid(all)   <- the usual in-manifold, meaning-free control
All are unit-normalised per gene and applied at the same alpha, so magnitude is held fixed and only the
GEOMETRY of the operation differs. Read both readouts:
  NATIVE   softmax mass on target-C HELD-OUT genes at UNSTEERED positions (split-half; must cross attention)
  DEST     the 18-class cell_type head on the pooled final state, scored against real expression enrichment
If Ihor is right: relative >= global on NATIVE, and clearly better on the destination alignment.

Run: ../../.venv_state/bin/python -u steer_relative.py [n_cells] [alpha]
Out: results/steer_relative.json
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


def main(n_cells=24, alpha=0.5):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer()
    with open(os.path.join(SC.CACHE_DIR, "steer_heads.pkl"), "rb") as f:
        heads = pickle.load(f)
    H = heads["cell_type"]; classes = np.array(H["classes"])

    EMB = SL._embed_matrix(st.xt, "embed")
    vocab, hid = EMB.shape
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok2chr = {int(t): str(C.loc[s, "chromosome"]) for ens, t in tokmap.items()
               if (s := ens2sym.get(ens)) in C.index and C.loc[s, "chromosome"] in AUTOSOMES and int(t) < vocab}
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    chrom_cen, read_idx, chroms = {}, {}, []
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        chrom_cen[c] = EMB[tids[m]].mean(0)
        read_idx[c] = np.array(tids[(tchr == c) & (~is_tr)], dtype=np.int64)
        chroms.append(c)

    # ---- the three operators, as (vocab, hidden) per-token displacement matrices, each row unit-normalised
    def unit(M):
        n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
        return M / n

    # COMPACT CODEBOOKS: row 0 = "no displacement" (genes with no autosome label), rows 1.. = one per source
    # chromosome. token_row maps a token id to its source-chromosome row. This is ~100 KB instead of ~200 MB.
    src_list = chroms
    src_row = {c: i + 1 for i, c in enumerate(src_list)}
    token_row = np.zeros(vocab, dtype=np.int64)
    for t, ch in tok2chr.items():
        if ch in src_row:
            token_row[t] = src_row[ch]

    PT = {}
    for c in chroms:
        g = chrom_cen[c] - gcen
        gu = g / (np.linalg.norm(g) + 1e-12)
        cb_g = np.zeros((len(src_list) + 1, hid)); cb_g[1:] = gu          # global: same vector for every gene
        cb_r = np.zeros((len(src_list) + 1, hid))                          # relative: depends on source chrom
        for s in src_list:
            cb_r[src_row[s]] = chrom_cen[c] - chrom_cen[s]
        PT[c] = dict(global_=(unit(cb_g), token_row), relative=(unit(cb_r), token_row))
    sham = {}
    srng = np.random.default_rng(SEED + 31)
    for c in chroms:
        grp = srng.choice(tids[is_tr], size=int(((tchr == c) & is_tr).sum()), replace=False)
        v = EMB[grp].mean(0) - gcen
        cb_s = np.zeros((len(src_list) + 1, hid)); cb_s[1:] = v / (np.linalg.norm(v) + 1e-12)
        sham[c] = (unit(cb_s), token_row)

    push = alpha * mean_norm
    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    y = np.asarray(labels["cell_type"]).astype(str)
    keep = np.array([v in set(classes) for v in y])
    ridx = {c: torch.as_tensor(read_idx[c]) for c in chroms}
    print(f"[setup] {len(chroms)} chromosomes, {len(seqs)} cells, alpha={alpha} (push {push:.2f})")
    print(f"[ops]   global / relative / sham -- all unit per-gene, identical magnitude\n")

    ops = ["global_", "relative", "sham"]
    Z = {o: {c: [] for c in chroms} for o in ops}
    mass = {o: {c: [] for c in chroms} for o in ops}
    base_Z, base_mass = [], {c: [] for c in chroms}
    prng = np.random.default_rng(SEED + 9)

    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]

        h, lg = st.forward_both(ids)
        p = torch.softmax(lg[read_pos], -1)
        base_Z.append(SL.Steerer.pool(h, gp))
        for c in chroms:
            base_mass[c].append(float(p[:, ridx[c]].sum(-1).mean()))

        for c in chroms:
            for o in ops:
                M = PT[c][o] if o in ("global_", "relative") else sham[c]
                with st.steering(None, alpha=push, positions=mask, site="embed",
                                 per_token=M, input_ids=ids):
                    h, lg = st.forward_both(ids)
                p = torch.softmax(lg[read_pos], -1)
                Z[o][c].append(SL.Steerer.pool(h, gp))
                mass[o][c].append(float(p[:, ridx[c]].sum(-1).mean()))
        if (i + 1) % 4 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    # ---- NATIVE
    print(f"\n=== NATIVE: chr-C mass at UNSTEERED positions (vs sham) ===")
    nat = {}
    for o in ops:
        d = np.array([np.mean(mass[o][c]) - np.mean(base_mass[c]) for c in chroms])
        nat[o] = d
    for o in ["global_", "relative"]:
        spec = nat[o] - nat["sham"]
        bs = np.array([spec[rng.integers(0, len(spec), len(spec))].mean() for _ in range(5000)])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"  {o:<9} Δmass {nat[o].mean():+.5f}   vs sham {spec.mean():+.5f}  "
              f"CI[{ci[0]:+.5f},{ci[1]:+.5f}]  {int((spec>0).sum())}/{len(spec)} chr +")
    print(f"  {'sham':<9} Δmass {nat['sham'].mean():+.5f}")

    # ---- DESTINATION alignment vs real expression enrichment
    ENR, chrs_e, types_e = None, None, None
    import steer_mechanism as SM
    ENR, chrs_e, types_e = SM.build_enrichment(list(classes))
    ti = [i for i, t in enumerate(classes) if t in types_e]
    tj = [types_e.index(classes[i]) for i in ti]
    ci_ = [i for i, c in enumerate(chroms) if c in chrs_e]
    cj = [chrs_e.index(chroms[i]) for i in ci_]
    Em = ENR[np.ix_(cj, tj)]

    def dest(Zs):
        P = H["clf"].predict_proba(H["scaler"].transform(np.stack(Zs)))[keep]
        return np.bincount(P.argmax(1), minlength=len(classes)) / len(P)

    print(f"\n=== DESTINATION alignment with real expression enrichment (matched vs mismatched) ===")
    out_ops = {}
    for o in ops:
        D = np.stack([dest(Z[o][c]) for c in chroms])[np.ix_(ci_, ti)]
        zr = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        Dz = np.stack([zr(r) for r in D]); Ez = np.stack([zr(r) for r in Em])
        Cc = Dz @ Ez.T / D.shape[1]
        matched = np.diag(Cc); off = Cc[~np.eye(len(matched), dtype=bool)]
        gap = float(matched.mean() - off.mean())
        null = np.array([np.diag(Cc[rng.permutation(len(matched))]).mean() - off.mean() for _ in range(20000)])
        p = float(((null >= gap).sum() + 1) / (len(null) + 1))
        print(f"  {o:<9} matched {matched.mean():+.4f}  mismatched {off.mean():+.4f}  "
              f"GAP {gap:+.4f}  p={p:.4f}")
        out_ops[o] = dict(matched=float(matched.mean()), mismatched=float(off.mean()), gap=gap, p=p,
                          native=float(nat[o].mean()))

    g, r = out_ops["global_"]["gap"], out_ops["relative"]["gap"]
    print(f"\n  VERDICT: relative − global on destination alignment = {r - g:+.4f}")
    print(f"  -> {'SOURCE-RELATIVE IS BETTER: the operator was the problem' if r > g + 0.01 else 'the operator is NOT the explanation: relative does not beat global'}")
    json.dump(dict(alpha=alpha, n_cells=int(keep.sum()), n_chrom=len(chroms), ops=out_ops),
              open(os.path.join(HERE, "results", "steer_relative.json"), "w"), indent=1)
    print("\n[done] -> results/steer_relative.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
