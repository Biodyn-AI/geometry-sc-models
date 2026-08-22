"""DOSAGE OR PROGRAM? — the shape of the steering response along the chromosome (Ihor, 2026-07-20).

WHY THIS QUESTION. Every steering readout so far collapsed the response into ONE number (total softmax mass on
chr-C). That throws away the entire spatial profile, which is exactly where the two competing mechanisms differ:

  CNV / DOSAGE hypothesis (the live alternative, [[cnv-alternative-mechanism]]): an aneuploid chromosome shifts
    ALL of its genes roughly uniformly, regardless of function or neighbourhood. If the model learned chromosome
    from copy-number variation in its corpus, the steering response should be FLAT along the chromosome.
  CHROMATIN-DOMAIN / PROGRAM hypothesis: co-regulation acts on megabase blocks, so the response should be
    SPATIALLY CLUMPY -- neighbouring genes move together, distant same-chromosome genes do not.

Opposite, quantitative predictions about the same measurement. And it uses the STRONG, replicated channel (the
model's own gene predictions, +0.15 on the 1B, 22/22 chromosomes) rather than the weak cell-type readout.

THE MEASUREMENT. Steer toward chromosome C at the usual split-half protocol, and record the change in the
model's predicted log-probability for EVERY gene at the unsteered read positions. Then, within chromosome C:
  * bin the chromosome into WINDOW-Mb windows, take the mean response per window;
  * BETWEEN-WINDOW VARIANCE is the statistic. Dosage -> ~0 (uniform lift). Domains -> large.
  * NULL: shuffle the gene->position assignment WITHIN the chromosome and recompute. That destroys spatial
    structure while holding the response values, the gene set and the window sizes fixed -- so it isolates
    "is the response spatially organised" from "how big is the response".
  * Report the null-corrected excess and a per-chromosome breakdown, plus the spill-over onto OTHER chromosomes
    (a dosage variable should not move them; a program variable might, via shared programs).

Also reported: the response's uniformity (CV) and whether it is bidirectional (steering -C should push chr-C
DOWN if it is a real signed axis rather than a one-way artifact).

Run: ../../.venv_state/bin/python -u steer_dosage.py [n_cells] [alpha] [model] [window_mb]
Out: results/steer_dosage_<model>.json
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
MIN_GENES_WIN = 6          # a window needs this many genes for a stable per-window mean


def main(n_cells=20, alpha=0.5, model="1b", window_mb=5.0):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    vocab = EMB.shape[0]
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())

    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok_chr, tok_pos = {}, {}
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and t < vocab:
            tok_chr[t] = str(C.loc[s, "chromosome"]); tok_pos[t] = float(C.loc[s, "start"])
    tids = np.array(sorted(tok_chr)); tchr = np.array([tok_chr[t] for t in tids])
    tpos = np.array([tok_pos[t] for t in tids])
    is_tr = rng.random(len(tids)) < 0.5              # direction built on TRAIN; profile read on TEST genes
    gcen = EMB[tids[is_tr]].mean(0)

    dirs, chroms = {}, []
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        v = EMB[tids[m]].mean(0) - gcen
        dirs[c] = SL.Direction(vec=v, name=f"chr:{c}", basis="embed_tokens")
        chroms.append(c)
    sham = []
    srng = np.random.default_rng(SEED + 31)
    for c in chroms[:3]:
        grp = srng.choice(tids[is_tr], size=int(((tchr == c) & is_tr).sum()), replace=False)
        sham.append(SL.Direction(vec=EMB[grp].mean(0) - gcen, name="sham", basis="embed_tokens"))

    push = alpha * mean_norm
    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    print(f"[setup] model={model} alpha={alpha} (push {push:.2f}); {len(chroms)} chromosomes; "
          f"{len(seqs)} cells; windows {window_mb} Mb\n")

    def logprob(ids, read_pos, d=None, mask=None, sign=1.0):
        if d is None:
            lg = st.logits(ids)[read_pos]
        else:
            with st.steering(d, alpha=sign * push, positions=mask, site="embed"):
                lg = st.logits(ids)[read_pos]
        return torch.log_softmax(lg, -1).mean(0).numpy()          # (vocab,) mean over read positions

    # accumulate per-gene Δlogprob for each steered chromosome, + a sham and a NEGATIVE push
    D = {c: [] for c in chroms}
    Dneg = {c: [] for c in chroms[:6]}                            # bidirectionality on a subset (cost)
    Dsham = []
    prng = np.random.default_rng(SEED + 9)
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        base = logprob(ids, read_pos)
        for c in chroms:
            D[c].append(logprob(ids, read_pos, dirs[c], mask) - base)
        for c in Dneg:
            Dneg[c].append(logprob(ids, read_pos, dirs[c], mask, sign=-1.0) - base)
        Dsham.append(logprob(ids, read_pos, sham[0], mask) - base)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)
    D = {c: np.mean(v, 0) for c, v in D.items()}
    Dneg = {c: np.mean(v, 0) for c, v in Dneg.items()}
    Dsham = np.mean(Dsham, 0)

    # ---------------- the spatial-structure test, per steered chromosome
    print("\n=== IS THE RESPONSE UNIFORM (dosage) OR SPATIALLY CLUMPY (domains)? ===")
    print(f"  {'chr':<5} {'n':>5} {'mean Δ':>9} {'CV':>7} {'between-win var':>16} {'null':>9} {'excess z':>9}")
    rows, zs = [], []
    for c in chroms:
        m = (tchr == c) & (~is_tr)                                # TEST genes of the steered chromosome
        idx = tids[m]; pos = tpos[m]
        if len(idx) < 40:
            continue
        r = D[c][idx]
        win = (pos // (window_mb * 1e6)).astype(int)
        uw, cnt = np.unique(win, return_counts=True)
        keep = uw[cnt >= MIN_GENES_WIN]
        if len(keep) < 4:
            continue
        wm = np.array([r[win == w].mean() for w in keep])
        obs = float(wm.var())
        nul = []
        for _ in range(400):                                      # shuffle gene->position within the chrom
            rp = r[np.random.default_rng(None).permutation(len(r))]
            nul.append(np.var([rp[win == w].mean() for w in keep]))
        nm, ns = float(np.mean(nul)), float(np.std(nul))
        z = (obs - nm) / (ns + 1e-12)
        cv = float(r.std() / (abs(r.mean()) + 1e-12))
        rows.append(dict(chrom=c, n=int(len(idx)), mean=float(r.mean()), cv=cv, between_var=obs,
                         null_var=nm, z=float(z), n_win=int(len(keep))))
        zs.append(z)
        print(f"  {c:<5} {len(idx):>5} {r.mean():>+9.4f} {cv:>7.2f} {obs:>16.3e} {nm:>9.2e} {z:>+9.2f}")

    zs = np.array(zs)
    print(f"\n  spatial-structure z, mean over {len(zs)} chromosomes: {zs.mean():+.2f}   "
          f"{int((zs > 2).sum())}/{len(zs)} with z>2")
    verdict = ("SPATIALLY STRUCTURED -> domain/program-like, NOT a uniform dosage lift"
               if zs.mean() > 2 else
               "UNIFORM -> dosage-like: the response does not respect megabase structure")
    print(f"  VERDICT: {verdict}")

    # ---------------- bidirectionality and spill-over
    print("\n=== bidirectional? (a signed axis should push chr-C DOWN under a negative push) ===")
    bid = []
    for c in Dneg:
        m = (tchr == c) & (~is_tr); idx = tids[m]
        up, dn = float(D[c][idx].mean()), float(Dneg[c][idx].mean())
        bid.append((c, up, dn))
        print(f"  chr{c:<4} +push {up:+.4f}   −push {dn:+.4f}   {'BIDIRECTIONAL' if dn < 0 < up else 'one-way'}")

    print("\n=== spill-over: response on OTHER chromosomes (dosage should not move them) ===")
    off = []
    for c in chroms:
        m_on = (tchr == c) & (~is_tr); m_off = (tchr != c) & (~is_tr)
        off.append(float(D[c][tids[m_off]].mean() - Dsham[tids[m_off]].mean()))
    on = float(np.mean([D[c][tids[(tchr == c) & (~is_tr)]].mean() for c in chroms]))
    print(f"  on-target mean Δlogprob {on:+.4f}   off-target (sham-corrected) {np.mean(off):+.4f}   "
          f"ratio {abs(on / (np.mean(off) + 1e-12)):.1f}x")

    out = dict(model=model, alpha=alpha, window_mb=window_mb, n_cells=len(seqs),
               spatial=rows, mean_z=float(zs.mean()), n_z_gt2=int((zs > 2).sum()), verdict=verdict,
               bidirectional=[dict(chrom=c, up=u, down=d) for c, u, d in bid],
               on_target=on, off_target=float(np.mean(off)))
    json.dump(out, open(os.path.join(HERE, "results", f"steer_dosage_{model}.json"), "w"), indent=1)
    print(f"\n[done] -> results/steer_dosage_{model}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5,
         sys.argv[3] if len(sys.argv) > 3 else "1b",
         float(sys.argv[4]) if len(sys.argv) > 4 else 5.0)
