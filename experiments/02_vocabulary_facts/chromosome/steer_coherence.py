"""ARE THE DOMAINS THAT WORK THE REAL ONES? (Ihor, 2026-07-20)

THE OBSERVATION. `steer_local.py` section 4: steering toward a 5 Mb domain raises the probability of the cell
type that genuinely over-expresses that domain's genes -- mean +0.026, CI excludes 0 -- BUT only 26 of 96
domains move that way; the median domain is slightly negative. The mean is carried by a minority.

THE HYPOTHESIS. Most 5 Mb windows are arbitrary chunks of DNA with no reason to behave as a unit. A "domain"
is only real where its genes are actually CO-REGULATED. So the failures may not be failures of the mechanism --
they may be windows that are not domains. If so, the intended-direction effect should CONCENTRATE in windows
whose genes are genuinely co-expressed, and the 70 "negatives" are mostly noise windows diluting the signal.

THE TEST (expression data only -- no model, no forward passes).
  COHERENCE[bin] = how co-expressed are this window's genes, above a matched null.
     mean pairwise Pearson r among the window's genes across cells, MINUS the mean of the same statistic on
     random gene sets of the same size matched on expression decile (highly-expressed genes correlate more,
     so an unmatched null would confound coherence with abundance).
  EFFECT[bin]    = the per-domain intended-direction effect, recomputed from the SAVED matrices.
  Then: does EFFECT rise with COHERENCE?

HONESTY. This is POST-HOC on the same 96 domains that produced the observation, so it is exploratory, not
confirmatory. It is reported as a continuous relationship (correlation across all 96 domains + tertiles), never
as a hand-picked threshold, and with a permutation p. A positive result here is a lead worth a fresh
pre-registered run on new domains -- not a finished claim.

Run: ../../.venv/bin/python -u steer_coherence.py       (light venv; no torch needed)
Out: results/steer_coherence.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES

MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
TOKMAP = f"{MAXTOKI_SETUP}/token_dictionary.json"
N_CELLS = 6000
WINDOW_MB, N_BINS, SEED = 5.0, 96, 0
N_NULL = 40
MIN_GENES = 16


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def rebuild_bins():
    """Reproduce steer_local.build_bins EXACTLY (same seed, same params) so the gene lists match the run."""
    rng = np.random.default_rng(SEED)
    C = coords()
    tokmap = json.load(open(TOKMAP))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    recs = []
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES:
            recs.append((t, s, str(C.loc[s, "chromosome"]), float(C.loc[s, "start"])))
    bins = {}
    for t, s, c, p in recs:
        bins.setdefault(f"{c}:{int(p // (WINDOW_MB * 1e6))}", []).append((t, s, c))
    good = {k: v for k, v in bins.items() if len(v) >= MIN_GENES}
    keys = sorted(good)
    if N_BINS and len(keys) > N_BINS:
        keys = list(np.array(keys)[np.sort(rng.choice(len(keys), N_BINS, replace=False))])
    out = {}
    for k in keys:
        v = good[k]
        idx = rng.permutation(len(v))
        out[k] = [v[i][1] for i in idx]
    return out


def main():
    res = json.load(open(os.path.join(HERE, "results", "steer_local.json")))
    M = res["matrices"]
    D = np.array(M["dest"]); E = np.array(M["enrich"]); base = np.array(M["base"])
    bin_keys = list(M["bins"])
    Tb = E.argmax(1)
    matched = np.array([D[b, Tb[b]] - base[Tb[b]] for b in range(len(D))])
    mism = np.array([np.mean([D[o, Tb[b]] for o in range(len(D)) if o != b]) - base[Tb[b]]
                     for b in range(len(D))])
    effect = matched - mism
    print(f"[effect] {len(effect)} domains; mean {effect.mean():+.4f}  median {np.median(effect):+.4f}  "
          f"{int((effect > 0).sum())} positive")

    bins = rebuild_bins()
    missing = [k for k in bin_keys if k not in bins]
    assert not missing, f"bin regeneration mismatch: {missing[:5]}"
    print(f"[bins] regenerated deterministically; all {len(bin_keys)} match the run")

    # ---- expression
    with h5py.File(G.FETAL_GUT, "r") as f:
        fn = f["var"]["feature_name"]
        syms = _dec(fn["categories"][:]).astype(str)[fn["codes"][:]] if isinstance(fn, h5py.Group) \
            else _dec(fn[:]).astype(str)
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], min(N_CELLS, shape[0]), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        Ex = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            a, b = int(indptr[r]), int(indptr[r + 1]); Ex[i, idx[a:b]] = data[a:b]
    up = np.char.upper(syms.astype(str))
    tot = Ex.sum(1, keepdims=True); tot[tot == 0] = 1
    L = np.log1p(Ex / tot * 1e4)
    pos = {}
    for i, s in enumerate(up):
        pos.setdefault(s, i)

    mean_expr = L.mean(0)
    sd = L.std(0)
    usable = np.where(sd > 1e-8)[0]
    dec = np.zeros(L.shape[1], int)
    dec[usable] = np.digitize(mean_expr[usable], np.percentile(mean_expr[usable], np.arange(10, 100, 10)))
    by_dec = {d: usable[dec[usable] == d] for d in range(10)}

    def mean_pair_r(cols):
        if len(cols) < 3:
            return np.nan
        Z = L[:, cols]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        Cm = (Z.T @ Z) / len(Z)
        iu = np.triu_indices(len(cols), 1)
        return float(Cm[iu].mean())

    print(f"[coherence] matched null: {N_NULL} random gene sets per domain, matched on expression decile\n")
    coh, eff, used_keys = [], [], []
    nrng = np.random.default_rng(SEED + 5)
    for bi, k in enumerate(bin_keys):
        cols = np.array([pos[s] for s in bins[k] if s in pos and sd[pos[s]] > 1e-8], dtype=int)
        if len(cols) < 8:
            continue
        obs = mean_pair_r(cols)
        # matched null: same size, same expression-decile composition
        comp = [dec[c] for c in cols]
        nulls = []
        for _ in range(N_NULL):
            pick = []
            for d in comp:
                cand = by_dec.get(d, usable)
                pick.append(nrng.choice(cand))
            nulls.append(mean_pair_r(np.array(pick, dtype=int)))
        coh.append(obs - float(np.nanmean(nulls)))
        eff.append(effect[bi]); used_keys.append(k)
    coh = np.array(coh); eff = np.array(eff)
    print(f"[coherence] computed for {len(coh)} domains  (mean excess r {coh.mean():+.4f})")

    # ---- does the effect rise with coherence?
    from scipy.stats import spearmanr, mannwhitneyu
    rho = float(spearmanr(coh, eff).statistic)
    prng = np.random.default_rng(SEED + 11)
    null_rho = np.array([spearmanr(coh, eff[prng.permutation(len(eff))]).statistic for _ in range(20000)])
    p_rho = float(((null_rho >= rho).sum() + 1) / (len(null_rho) + 1))

    q = np.quantile(coh, [1 / 3, 2 / 3])
    lo, mid, hi = eff[coh <= q[0]], eff[(coh > q[0]) & (coh <= q[1])], eff[coh > q[1]]
    pos_m, neg_m = coh[eff > 0], coh[eff <= 0]
    u_p = float(mannwhitneyu(pos_m, neg_m, alternative="greater").pvalue) if len(pos_m) and len(neg_m) else 1.0

    print("\n=== does the intended-direction effect concentrate in COHERENT domains? ===")
    print(f"  Spearman(coherence, effect) = {rho:+.3f}   permutation p = {p_rho:.4f}")
    print(f"  effect by coherence tertile:  low {lo.mean():+.4f} (n={len(lo)})   "
          f"mid {mid.mean():+.4f} (n={len(mid)})   high {hi.mean():+.4f} (n={len(hi)})")
    print(f"  coherence of WORKING vs failing domains: {pos_m.mean():+.4f} vs {neg_m.mean():+.4f}  "
          f"(Mann-Whitney one-sided p = {u_p:.4f})")

    supported = (p_rho < 0.05 and rho > 0) or (u_p < 0.05)
    verdict = ("SUPPORTED (exploratory): the effect concentrates in genuinely co-regulated windows -- the "
               "failures look like windows that are not domains"
               if supported else
               "NOT SUPPORTED: working domains are no more co-regulated than failing ones -- the minority that "
               "works is not explained by domain coherence")
    print(f"\n  VERDICT: {verdict}")
    print("  (POST-HOC on the same 96 domains -- a lead, not a finished claim; confirm on fresh domains.)")

    json.dump(dict(n=len(coh), rho=rho, p_rho=p_rho, tertiles=dict(low=float(lo.mean()), mid=float(mid.mean()),
              high=float(hi.mean())), coh_pos=float(pos_m.mean()), coh_neg=float(neg_m.mean()), p_mwu=u_p,
              verdict=verdict, post_hoc=True),
              open(os.path.join(HERE, "results", "steer_coherence.json"), "w"), indent=1)
    print("\n[done] -> results/steer_coherence.json")


if __name__ == "__main__":
    main()
