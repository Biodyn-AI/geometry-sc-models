"""HOW FAR DOES A LOCAL PUSH REACH? — the decay curve (Ihor, 2026-07-20).

THE QUESTION, and why it is the right one. Every previous readout summed the response into a single
chromosome-level number, which cannot distinguish two very different models of what the variable IS:

  CHROMOSOME-LEVEL (dosage-like): the model knows "this gene is on chr7" and nothing finer. Steering a 5 Mb
    window inside chr7 should lift the WHOLE of chr7 about equally -- flat within the chromosome, a cliff at
    the chromosome boundary.
  REGIONAL / DOMAIN-LEVEL: the model knows WHERE along the chromosome. Steering a 5 Mb window should lift
    genes NEAR that window most, decaying with genomic distance -- a graded curve, not a cliff.

That is a decay curve, and it is directly biologically meaningful: co-regulation (TADs, shared enhancers,
replication domains) acts at ~0.5-2 Mb, so a response that decays over a few Mb is the signature of a learned
regional program, while a flat-within-chromosome response is the signature of a bare chromosome label.

It also uses the STRONG channel: the model's own next-gene predictions (+0.15 on the 1B, 22/22 chromosomes),
not the weak cell-type readout that has produced mostly nulls.

DESIGN
  * pick N local 5 Mb windows spread across the genome (>= MIN_GENES vocab genes each), split each window's
    genes TRAIN (build the direction) / TEST (never read as part of the response);
  * steer toward the window at the usual split-half protocol; record the change in predicted log-probability
    for every gene at the UNSTEERED read positions;
  * bin all genes by |genomic distance| from the window centre ON THE SAME CHROMOSOME, plus an
    "other chromosome" bin; average over windows;
  * SHAM control (same construction, random gene grouping) subtracted at every distance, so a generic
    "everything moves" lift cannot masquerade as locality.
  * Stability: only genes whose BASELINE probability is above the median are scored -- log-ratios of
    near-zero-probability genes are dominated by tail noise (measured: it swamped the signal in a first pass).

READ IT AS: near-bin >> far-same-chromosome  => regional variable (graded, domain-like).
            near-bin ~= far-same-chromosome >> other-chromosome => chromosome-level label only.

Run: ../../.venv_state/bin/python -u steer_locality.py [n_cells] [n_windows] [alpha] [model]
Out: results/steer_locality_<model>.json
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
WINDOW_MB = 5.0
MIN_GENES = 16
# distance bins in Mb from the steered window's centre, same chromosome
EDGES = [0, 2.5, 7.5, 15, 30, 60, 1e9]
LABELS = ["<2.5Mb", "2.5-7.5", "7.5-15", "15-30", "30-60", ">60Mb"]


def main(n_cells=20, n_windows=30, alpha=0.5, model="1b"):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    vocab = EMB.shape[0]
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())

    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    rec = []
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and t < vocab:
            rec.append((t, str(C.loc[s, "chromosome"]), float(C.loc[s, "start"])))
    tids = np.array([r[0] for r in rec]); tchr = np.array([r[1] for r in rec]); tpos = np.array([r[2] for r in rec])
    o = np.argsort(tids); tids, tchr, tpos = tids[o], tchr[o], tpos[o]
    tok_row = {int(t): i for i, t in enumerate(tids)}

    # local windows
    wkey = np.array([f"{c}:{int(p // (WINDOW_MB * 1e6))}" for c, p in zip(tchr, tpos)])
    uw, cnt = np.unique(wkey, return_counts=True)
    good = uw[cnt >= MIN_GENES]
    sel = good[np.sort(rng.choice(len(good), min(n_windows, len(good)), replace=False))]
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)

    wins = {}
    for w in sel:
        m = wkey == w
        tr = m & is_tr
        if tr.sum() < 6:
            continue
        v = EMB[tids[tr]].mean(0) - gcen
        wins[w] = dict(dir=SL.Direction(vec=v, name=f"win:{w}", basis="embed_tokens"),
                       chrom=tchr[m][0], centre=float(tpos[m].mean()))
    wkeys = sorted(wins)
    shams = []
    srng = np.random.default_rng(SEED + 31)
    for w in wkeys[:5]:
        n_tr = int(((wkey == w) & is_tr).sum())
        grp = srng.choice(tids[is_tr], size=max(n_tr, 6), replace=False)
        shams.append(SL.Direction(vec=EMB[grp].mean(0) - gcen, name="sham", basis="embed_tokens"))

    push = alpha * mean_norm
    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    print(f"[setup] model={model}, {len(wkeys)} windows of {WINDOW_MB} Mb, {len(seqs)} cells, "
          f"alpha={alpha} (push {push:.2f})\n")

    def lp(ids, read_pos, d=None, mask=None):
        if d is None:
            lg = st.logits(ids)[read_pos]
        else:
            with st.steering(d, alpha=push, positions=mask, site="embed"):
                lg = st.logits(ids)[read_pos]
        return torch.log_softmax(lg, -1).mean(0).numpy()

    acc = {w: [] for w in wkeys}
    acc_sham = []
    base_lp = []
    prng = np.random.default_rng(SEED + 9)
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        b = lp(ids, read_pos); base_lp.append(b)
        for w in wkeys:
            acc[w].append(lp(ids, read_pos, wins[w]["dir"], mask) - b)
        acc_sham.append(np.mean([lp(ids, read_pos, sd, mask) - b for sd in shams], 0))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    base_lp = np.mean(base_lp, 0)
    Dsham = np.mean(acc_sham, 0)
    # stability filter: only genes with above-median baseline probability
    bp = base_lp[tids]
    stable = bp >= np.median(bp)
    print(f"[filter] scoring the {int(stable.sum())}/{len(tids)} genes with above-median baseline probability "
          f"(log-ratios of near-zero-probability genes are tail-noise dominated)\n")

    # ---- decay curve
    per_bin = {L: [] for L in LABELS}
    other = []
    for w in wkeys:
        d = np.mean(acc[w], 0) - Dsham                     # sham-corrected response, per vocab token
        r = d[tids]
        same = (tchr == wins[w]["chrom"]) & (~is_tr) & stable
        oth = (tchr != wins[w]["chrom"]) & (~is_tr) & stable
        dist = np.abs(tpos - wins[w]["centre"]) / 1e6
        for k, L in enumerate(LABELS):
            m = same & (dist >= EDGES[k]) & (dist < EDGES[k + 1])
            if m.sum() >= 5:
                per_bin[L].append(float(r[m].mean()))
        if oth.sum() >= 50:
            other.append(float(r[oth].mean()))

    print("=== DECAY CURVE: response vs genomic distance from the steered 5 Mb window ===")
    print(f"  {'bin':<10} {'n windows':>10} {'mean Δlogprob (sham-corrected)':>32}")
    curve = {}
    for L in LABELS:
        v = np.array(per_bin[L])
        if len(v) == 0:
            continue
        curve[L] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std()))
        print(f"  {L:<10} {len(v):>10} {v.mean():>32.4f}")
    om = float(np.mean(other))
    curve["other_chrom"] = dict(n=len(other), mean=om)
    print(f"  {'other chr':<10} {len(other):>10} {om:>32.4f}")

    # ---- the decisive contrasts, paired over windows
    def paired(a, b):
        n = min(len(per_bin[a]), len(per_bin[b]))
        if n < 6:
            return None
        d = np.array(per_bin[a][:n]) - np.array(per_bin[b][:n])
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
        return dict(diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    n=n, n_pos=int((d > 0).sum()))

    near_far = paired("<2.5Mb", ">60Mb")
    print("\n=== IS IT REGIONAL OR JUST CHROMOSOME-LEVEL? ===")
    if near_far:
        print(f"  near (<2.5Mb) − far same-chrom (>60Mb): {near_far['diff']:+.4f}  "
              f"CI[{near_far['ci'][0]:+.4f},{near_far['ci'][1]:+.4f}]  {near_far['n_pos']}/{near_far['n']} windows +")
    far = np.mean(per_bin[">60Mb"]) if per_bin[">60Mb"] else float("nan")
    print(f"  far same-chrom (>60Mb) − other chromosome: {far - om:+.4f}")
    regional = bool(near_far and near_far["ci"][0] > 0)
    verdict = ("REGIONAL: the push decays with genomic distance -> the model carries a graded position "
               "variable, not just a chromosome label" if regional else
               "CHROMOSOME-LEVEL ONLY: the push lifts the whole chromosome about equally -> a bare chromosome "
               "label, no usable sub-chromosomal locality in the response")
    print(f"\n  VERDICT: {verdict}")

    json.dump(dict(model=model, alpha=alpha, window_mb=WINDOW_MB, n_windows=len(wkeys), n_cells=len(seqs),
                   curve=curve, near_minus_far=near_far, far_minus_other=float(far - om),
                   regional=regional, verdict=verdict),
              open(os.path.join(HERE, "results", f"steer_locality_{model}.json"), "w"), indent=1)
    print(f"\n[done] -> results/steer_locality_{model}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         int(sys.argv[2]) if len(sys.argv) > 2 else 30,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
         sys.argv[4] if len(sys.argv) > 4 else "1b")
