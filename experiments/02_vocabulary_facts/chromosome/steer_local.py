"""LOCAL GENOMIC DOMAINS — the fine-grained version of the failed mechanism test (Ihor, 2026-07-18).

WHY. steer_mechanism.py asked "does chr-C steering send cells to the cell type that over-expresses chr-C
genes?" and the answer was NO (0/22 argmax; the sharpened version p=0.115). But a whole chromosome is a
grab-bag of ~1000 unrelated genes -- co-regulation does not operate at that scale. It operates at TAD /
chromatin-domain scale, ~0.5-2 Mb. So the honest reading of that negative was "the lens was too coarse", and
this is the finer lens.

THE UNIT. Split each autosome into WINDOW-Mb bins. A bin with enough genes is a candidate co-regulated domain.
Build its steering direction exactly as before (centroid of the bin's genes minus the global centroid, in INPUT
embed space), but from ~20 local genes rather than ~1000 scattered ones.

THREE MEASUREMENTS, in order -- the first one gates the rest:
  1. NATIVE POSITIVE CONTROL (new; the chromosome test never needed it because chromosomes obviously worked).
     Steer toward bin B, and read the model's own logit mass on B's HELD-OUT genes (bin genes are split
     train/test: the direction is built from train, the readout uses test). If a local direction cannot even
     raise its own domain's mass above a sham push, local domains are not represented as usable directions and
     everything downstream is underpowered -- we would want to know that BEFORE interpreting a null.
  2. DESTINATION. Where does the 18-class cell_type head send cells under each bin's push? Is the destination
     bin-specific (agreement across bins vs across shams)?
  3. THE MECHANISM TEST. Does bin B's destination match the cell type that genuinely over-expresses bin B's
     genes (expression only, no model)? Matched-vs-mismatched permutation, exactly as steer_mechanism.py, but
     now over HUNDREDS of bins instead of 22 chromosomes -- far more power, and an enrichment side built from
     ~20 co-located genes, which should not be degenerate the way whole-chromosome averages were.

Run: ../../.venv_state/bin/python -u steer_local.py [n_cells] [window_mb] [n_bins] [alpha]
Out: results/steer_local.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import steer_lib as SL
import steer_classifier as SC
from genome_wide import coords, AUTOSOMES

MIN_GENES = 16           # a bin needs this many vocab genes to give a stable direction AND enrichment
SEED = 0
N_CELLS_ENRICH = 6000


def build_bins(window_mb, n_bins, rng):
    """Local genomic windows -> (bin_id -> dict(train_tokens, test_tokens, symbols, chrom))."""
    C = coords()
    tokmap = json.load(open(SL.TOKMAP))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    recs = []
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES:
            recs.append((t, s, str(C.loc[s, "chromosome"]), float(C.loc[s, "start"])))
    bins = {}
    for t, s, c, p in recs:
        key = f"{c}:{int(p // (window_mb * 1e6))}"
        bins.setdefault(key, []).append((t, s, c))
    good = {k: v for k, v in bins.items() if len(v) >= MIN_GENES}
    keys = sorted(good)
    if n_bins and len(keys) > n_bins:            # spread the sample across the genome, not one chromosome
        keys = list(np.array(keys)[np.sort(rng.choice(len(keys), n_bins, replace=False))])
    out = {}
    for k in keys:
        v = good[k]
        idx = rng.permutation(len(v)); half = len(v) // 2
        out[k] = dict(chrom=v[0][2],
                      train=[v[i][0] for i in idx[:half]], test=[v[i][0] for i in idx[half:]],
                      syms=[v[i][1] for i in idx])
    return out


def bin_enrichment(bins, classes):
    """ENRICH[bin, celltype] from expression only -- per-gene z across cell types, averaged over the BIN's
    genes (~20 co-located genes, not ~1000 scattered ones)."""
    with h5py.File(G.FETAL_GUT, "r") as f:
        fn = f["var"]["feature_name"]
        syms = SC._dec(fn["categories"][:]).astype(str)[fn["codes"][:]] if isinstance(fn, h5py.Group) \
            else SC._dec(fn[:]).astype(str)
        ct = SC._cat(f, "cell_type")
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], min(N_CELLS_ENRICH, shape[0]), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            a, b = int(indptr[r]), int(indptr[r + 1]); E[i, idx[a:b]] = data[a:b]
    ct = ct[sel]; up = np.char.upper(syms.astype(str))
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    L = np.log1p(E / tot * 1e4)
    keep_t = [t for t in classes if (ct == t).sum() >= 10]
    M = np.stack([L[ct == t].mean(0) for t in keep_t])
    mu, sd = M.mean(0), M.std(0); ok = sd > 1e-8
    Zg = np.zeros_like(M); Zg[:, ok] = (M[:, ok] - mu[ok]) / sd[ok]
    pos = {}
    for i, s in enumerate(up):
        pos.setdefault(s, i)
    rows, used = [], []
    for k, b in bins.items():
        ii = np.array([pos[s] for s in b["syms"] if s in pos and ok[pos[s]]], dtype=int)
        if len(ii) < 8:
            continue
        rows.append(Zg[:, ii].mean(1)); used.append(k)
    return (np.stack(rows) if rows else np.zeros((0, len(keep_t)))), used, keep_t


def main(n_cells=30, window_mb=5.0, n_bins=48, alpha=0.5, model="217m"):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    with open(SC.heads_path(model), "rb") as f:
        heads = pickle.load(f)
    H = heads["cell_type"]; classes = np.array(H["classes"])

    bins = build_bins(window_mb, n_bins, rng)
    EMB = SL._embed_matrix(st.xt, "embed")
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    gcen = EMB[np.array(sorted({t for b in bins.values() for t in b["train"]}))].mean(0)

    dirs, read_tok = {}, {}
    for k, b in bins.items():
        v = EMB[np.array(b["train"])].mean(0) - gcen
        dirs[k] = SL.Direction(vec=v, name=f"bin:{k}", basis="embed_tokens")
        read_tok[k] = np.array(b["test"], dtype=np.int64)
    keys = sorted(dirs)
    # sham bins: same construction, same sizes, random gene groupings
    all_tr = np.array(sorted({t for b in bins.values() for t in b["train"]}))
    shams = {}
    for i, k in enumerate(keys):
        grp = rng.choice(all_tr, size=len(bins[k]["train"]), replace=False)
        shams[f"sham{i}"] = SL.Direction(vec=EMB[grp].mean(0) - gcen, name=f"sham{i}", basis="embed_tokens")
    sham_keys = sorted(shams)
    print(f"[bins] {len(keys)} local windows of {window_mb} Mb (>= {MIN_GENES} genes), "
          f"median {np.median([len(bins[k]['syms']) for k in keys]):.0f} genes/bin")
    print(f"[setup] alpha={alpha} (push {alpha*mean_norm:.2f}), {n_cells} cells, "
          f"{len(sham_keys)} matched shams\n")

    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    y = np.asarray(labels["cell_type"]).astype(str)
    keep = np.array([v in set(classes) for v in y])
    push = alpha * mean_norm
    ridx = {k: torch.as_tensor(read_tok[k]) for k in keys}

    def run(ids, gp, read_pos, d=None, mask=None):
        if d is None:
            h, lg = st.forward_both(ids)
        else:
            with st.steering(d, alpha=push, positions=mask, site="embed"):
                h, lg = st.forward_both(ids)
        p = torch.softmax(lg[read_pos], -1)
        return SL.Steerer.pool(h, gp), p

    base_Z, base_mass = [], {k: [] for k in keys}
    bin_Z, bin_mass = {k: [] for k in keys}, {k: [] for k in keys}
    sham_Z, sham_mass = {s: [] for s in sham_keys}, {k: [] for k in keys}
    prng = np.random.default_rng(SEED + 9)
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]

        z, p = run(ids, gp, read_pos)
        base_Z.append(z)
        for k in keys:
            base_mass[k].append(float(p[:, ridx[k]].sum(-1).mean()))
        for k in keys:
            z, p = run(ids, gp, read_pos, dirs[k], mask)
            bin_Z[k].append(z); bin_mass[k].append(float(p[:, ridx[k]].sum(-1).mean()))
        for j, sk in enumerate(sham_keys):
            z, p = run(ids, gp, read_pos, shams[sk], mask)
            sham_Z[sk].append(z)
            sham_mass[keys[j]].append(float(p[:, ridx[keys[j]]].sum(-1).mean()))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    # ---- 1. NATIVE POSITIVE CONTROL: does a local push raise its OWN domain's held-out mass?
    d_bin = np.array([np.mean(bin_mass[k]) - np.mean(base_mass[k]) for k in keys])
    d_sham = np.array([np.mean(sham_mass[k]) - np.mean(base_mass[k]) for k in keys])
    spec = d_bin - d_sham
    bs = np.array([spec[rng.integers(0, len(spec), len(spec))].mean() for _ in range(5000)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(f"\n=== 1. NATIVE control: steer toward a local domain, read ITS OWN held-out genes ===")
    print(f"  Δmass local push {d_bin.mean():+.5f}   sham {d_sham.mean():+.5f}   "
          f"SPECIFIC {spec.mean():+.5f}  CI[{ci[0]:+.5f},{ci[1]:+.5f}]  {int((spec>0).sum())}/{len(spec)} bins +")
    local_works = ci[0] > 0
    print(f"  -> local domain directions {'DO' if local_works else 'DO NOT'} function as usable directions")

    # ---- 2. destinations
    def dest_of(Zs):
        P = H["clf"].predict_proba(H["scaler"].transform(np.stack(Zs)))[keep]
        a = P.argmax(1)
        return np.bincount(a, minlength=len(classes)) / len(a), P
    base_dist, P_base = dest_of(base_Z)
    D = np.stack([dest_of(bin_Z[k])[0] for k in keys])
    Dsh = np.stack([dest_of(sham_Z[s])[0] for s in sham_keys])

    def agreement(A):
        v = [float(np.minimum(A[i], A[j]).sum()) for i in range(len(A)) for j in range(i + 1, len(A))]
        return float(np.mean(v)) if v else float("nan")
    print(f"\n=== 2. destinations ===")
    print(f"  destination agreement: local bins {agreement(D):.3f}   shams {agreement(Dsh):.3f}")

    # ---- 3. THE MECHANISM TEST at local scale
    ENR, used, types_e = bin_enrichment({k: bins[k] for k in keys}, list(classes))
    ki = [keys.index(k) for k in used]
    ti = [i for i, t in enumerate(classes) if t in types_e]
    tj = [types_e.index(classes[i]) for i in ti]
    Dm = D[np.ix_(ki, ti)]; Em = ENR[:, tj]

    def zr(v):
        s = v.std(); return (v - v.mean()) / s if s > 1e-12 else v * 0
    Dz = np.stack([zr(r) for r in Dm]); Ez = np.stack([zr(r) for r in Em])
    Cc = Dz @ Ez.T / Dm.shape[1]
    matched = np.diag(Cc); off = Cc[~np.eye(len(matched), dtype=bool)]
    gap = float(matched.mean() - off.mean())
    null = np.array([np.diag(Cc[rng.permutation(len(matched))]).mean() - off.mean() for _ in range(20000)])
    p_corr = float(((null >= gap).sum() + 1) / (len(null) + 1))
    dtop, etop = Dm.argmax(1), Em.argmax(1)
    agree = float((dtop == etop).mean())
    nulla = np.array([float((dtop[rng.permutation(len(dtop))] == etop).mean()) for _ in range(20000)])
    p_arg = float(((nulla >= agree).sum() + 1) / (len(nulla) + 1))
    n_distinct = len(set(etop.tolist())); modal = float(np.bincount(etop, minlength=Dm.shape[1]).max() / len(etop))

    print(f"\n=== 3. MECHANISM at local scale: destination vs real expression enrichment ===")
    print(f"  units: {Dm.shape[0]} local domains x {Dm.shape[1]} cell types")
    print(f"  enrichment distinct top-types: {n_distinct}/{len(etop)} (one type takes {modal:.0%})"
          + ("   <-- DEGENERATE" if modal > 0.5 else ""))
    print(f"  matched {matched.mean():+.4f}  mismatched {off.mean():+.4f}  GAP {gap:+.4f}  p={p_corr:.4f}")
    print(f"  argmax agreement {agree:.3f} (null {nulla.mean():.3f})  p={p_arg:.4f}")
    # VERDICT LOGIC -- rewritten 2026-07-20, and the reason is recorded so this is not goalpost-moving.
    # The first version gated on `gap > 0.05`, a threshold picked while diagnosing the CHROMOSOME-scale test,
    # whose real problem was a DEGENERATE enrichment matrix (one cell type was top for 82% of chromosomes).
    # Applied at local scale that arbitrary cut mislabels a p=0.0000, non-degenerate, 96-unit result as
    # "inconclusive". The defensible criteria are: (a) is the enrichment side non-degenerate, so the comparison
    # can carry chromosome/domain-specific information at all; (b) is the matched-vs-mismatched gap significant.
    # Effect SIZE is then reported descriptively rather than used as a pass/fail gate -- a small effect that
    # replicates is still an effect, and calling it "strong" would be the real error.
    degenerate = modal > 0.5
    size = "weak" if gap < 0.05 else ("moderate" if gap < 0.15 else "strong")
    verdict = ("INCONCLUSIVE: enrichment side is degenerate -- comparison cannot carry domain-specific info"
               if degenerate else
               "NOT SUPPORTED at local scale" if p_corr >= 0.05 else
               f"SUPPORTED at local scale ({size} effect): destination tracks domain-specific expression")
    print(f"  effect size: gap {gap:+.4f} ({size});  enrichment {'DEGENERATE' if degenerate else 'ok'}")
    print(f"  VERDICT: {verdict}")

    # ---- 4. TARGETED test: "does it steer in the INTENDED direction?" (Ihor, 2026-07-20)
    # The profile correlation above is aggregate and indirect. The direct question: for each domain B, take the
    # cell type T_B that the DATA says most over-expresses B's genes, and ask whether steering toward B raises
    # p(T_B) -- in probability units. The control is the SAME T_B under MISMATCHED domains, which subtracts off
    # "T_B is simply a common attractor" (a raw rise in p(T_B) proves nothing if everything rises there).
    # *** FLOOR ASYMMETRY -- the reason this metric needs a guard (found 2026-07-20). ***
    # tgt = D[B,T_B] - mean_other(D[o,T_B]) (the baseline cancels). If T_B is a cell type that essentially
    # never receives steered cells, mean_other ~ 0 and D[B,T_B] >= 0, so tgt >= 0 BY CONSTRUCTION -- those
    # domains can only push the mean UP and can never contribute a negative. Measured: 49/96 domains had such
    # a target, and restricting to REACHABLE targets moved the CI onto zero. Any headline must use the
    # restricted set. (It also explains why the effect looked biggest in the LEAST coherent windows: noisier
    # enrichment -> more idiosyncratic, rarely-reached T_B -> more floor-bounded, non-negative cases.)
    Tb = etop                                                     # data-predicted destination per domain
    base_t = base_dist[np.array(ti)]                              # baseline p over the aligned type set
    matched_d = np.array([Dm[b, Tb[b]] - base_t[Tb[b]] for b in range(Dm.shape[0])])
    mism_d = np.array([np.mean([Dm[o, Tb[b]] for o in range(Dm.shape[0]) if o != b]) - base_t[Tb[b]]
                       for b in range(Dm.shape[0])])
    tgt = matched_d - mism_d
    commonness = np.array([np.mean([Dm[o, Tb[b]] for o in range(Dm.shape[0]) if o != b])
                           for b in range(Dm.shape[0])])
    reach = commonness >= 0.01                                    # target actually reachable under steering
    bs2 = np.array([tgt[rng.integers(0, len(tgt), len(tgt))].mean() for _ in range(5000)])
    ci2 = [float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))]
    print(f"\n=== 4. TARGETED: does steering toward B raise p(the cell type biology predicts for B)? ===")
    print(f"  Δp(T_B) steering toward B      : {matched_d.mean():+.4f}")
    print(f"  Δp(T_B) steering elsewhere     : {mism_d.mean():+.4f}   (same target type, mismatched domains)")
    # The MEAN alone is not enough: this distribution is heavily skewed (a minority of domains with large
    # positive effects can carry a positive mean while the TYPICAL domain moves the wrong way). Report the
    # median and a sign test, and require the mean, the median AND the sign test to agree before claiming a
    # reliable intended-direction push.
    n_pos = int((tgt > 1e-9).sum()); n_neg = int((tgt < -1e-9).sum())
    med = float(np.median(tgt))
    from math import comb
    n_mv = n_pos + n_neg
    p_sign = (sum(comb(n_mv, i) for i in range(n_pos, n_mv + 1)) / 2 ** n_mv) if n_mv else 1.0
    print(f"  INTENDED-DIRECTION effect      : mean {tgt.mean():+.4f}  CI[{ci2[0]:+.4f},{ci2[1]:+.4f}]  "
          f"median {med:+.4f}")
    print(f"  consistency                    : {n_pos} positive / {n_neg} negative domains  "
          f"sign-test p={p_sign:.3f}")
    # the honest number: floor-bounded targets removed
    sub = tgt[reach]
    if len(sub) >= 8:
        bs3 = np.array([sub[rng.integers(0, len(sub), len(sub))].mean() for _ in range(5000)])
        ci3 = [float(np.percentile(bs3, 2.5)), float(np.percentile(bs3, 97.5))]
        print(f"  FLOOR GUARD: {int((~reach).sum())}/{len(tgt)} domains have a target that is never reached "
              f"(tgt >= 0 by construction)")
        print(f"  restricted to REACHABLE targets  : n={len(sub)}  mean {sub.mean():+.4f}  "
              f"median {float(np.median(sub)):+.4f}  CI[{ci3[0]:+.4f},{ci3[1]:+.4f}]")
    else:
        ci3 = [float("nan"), float("nan")]
    reliable = len(sub) >= 8 and ci3[0] > 0 and float(np.median(sub)) > 0
    if reliable:
        print(f"  -> RELIABLE intended-direction push ({sub.mean():+.3f} prob. units) -- a TILT, not control")
    else:
        print("  -> NOT DEMONSTRATED: the unrestricted mean is inflated by floor-bounded targets; on reachable "
              "targets the interval covers zero. No reliable push toward the predicted cell type.")

    out_extra = dict(matched_dp=float(matched_d.mean()), mismatched_dp=float(mism_d.mean()),
                     intended_effect=float(tgt.mean()), ci=ci2, n_pos=int((tgt > 0).sum()),
                     n=int(len(tgt)))

    out = dict(window_mb=window_mb, n_bins=len(keys), n_cells=int(keep.sum()), alpha=alpha,
               native=dict(d_bin=float(d_bin.mean()), d_sham=float(d_sham.mean()),
                           specific=float(spec.mean()), ci=ci, works=bool(local_works),
                           n_pos=int((spec > 0).sum())),
               dest_agreement=dict(bins=agreement(D), shams=agreement(Dsh)),
               mechanism=dict(n_units=int(Dm.shape[0]), matched=float(matched.mean()),
                              mismatched=float(off.mean()), gap=gap, p_corr=p_corr,
                              argmax=agree, p_argmax=p_arg, enrich_distinct=n_distinct,
                              enrich_modal_share=modal, verdict=verdict),
               targeted=out_extra,
               # save the matrices so future questions never need another 40-minute re-run
               matrices=dict(bins=[keys[i] for i in ki], types=[str(classes[i]) for i in ti],
                             dest=Dm.tolist(), enrich=Em.tolist(), base=base_t.tolist()))
    json.dump(out, open(os.path.join(HERE, "results", f"steer_local_{model}.json"), "w"), indent=1)
    print("\n[done] -> results/steer_local.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30,
         float(sys.argv[2]) if len(sys.argv) > 2 else 5.0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 48,
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.5,
         sys.argv[5] if len(sys.argv) > 5 else "217m")
