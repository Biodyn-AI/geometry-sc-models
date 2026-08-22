"""FINE POSITION, LEAKAGE-CONTROLLED **AND ARTIFACT-CONTROLLED** — the corrected version of genome_position.py.

genome_position.py's BLOCK column is DEGENERATE. Verified on pure noise (zero position signal): block |rho| =
0.35-0.54 (vs random 0.06-0.11). The mechanism is the block form of the session's mean-reversion artifact
(cf. hox_within.py T1, run_probe.py size-1 folds): under GroupKFold by 10-Mb position block, RidgeCV predicts a
held-out block ~ the MEAN position of the training blocks, which is ANTI-correlated with the held-out block's
own position; `abs(Spearman)` then reads that -1 as a large positive. So coexpr_devel's 0.069 -> 0.559 is the
artifact, not a position signal, and every "block |rho|" and "retained %" in that file is uninterpretable.

THE FIX. Per (basis, chromosome), compare the observed block statistic to a PERMUTATION NULL that shuffles the
position labels across genes (X fixed). The null preserves the position distribution and the fold structure --
hence the full mean-reversion artifact -- and destroys only the embedding->position signal. So:

    excess = observed_block_rho - null_mean       (signal above the artifact)
    z      = (observed - null_mean) / null_sd

Real sub-chromosomal position knowledge shows as excess > 0 with z >> 0; the artifact cancels. Verdict rule
unchanged (gm_lib.py:20): a model basis must beat coexpr AND esm2 on the null-corrected statistic.

Run: ../../.venv/bin/python -u genome_position2.py
Out: results/genome_position2.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from genome_position import oof_rho, ALPHAS, BLOCK_MB, MIN_PER_CHR   # reuse the EXACT probe machinery
from sklearn.model_selection import KFold, GroupKFold

BASES = ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "scgpt_we", "coexpr_devel", "esm2"]
N_PERM = 40
SEED = 0


def block_rho(X, start, seed=SEED):
    """SIGNED out-of-fold Spearman under 10-Mb GroupKFold. SIGNED is essential: the block mean-reversion
    artifact is systematically NEGATIVE (a held-out block is predicted ~ the mean of the OTHER blocks, which
    anti-correlates with its own position), while genuine position signal is POSITIVE. Under abs() the two
    entangle; signed, they ADD, so (observed - null) isolates the real signal."""
    blk = (start // (BLOCK_MB * 1e6)).astype(int)
    nf = min(5, len(np.unique(blk)))
    if nf < 2:
        return None
    return oof_rho(X, start, list(GroupKFold(nf).split(X, start, blk)))     # SIGNED, not abs


def _excess(X, start, rng):
    """observed signed block-rho minus the within-chr position-shuffle null mean (= the artifact)."""
    obs = block_rho(X, start)
    if obs is None:
        return None
    null = np.array([block_rho(X, rng.permutation(start)) for _ in range(N_PERM)], dtype=float)
    null = null[np.isfinite(null)]
    return obs, float(null.mean()), float(null.std())


def _validate():
    """The statistic must (a) return ~0 excess on pure noise and (b) recover a planted monotone signal."""
    rng = np.random.default_rng(1)
    n, d, span = 800, 512, 200e6
    start = np.sort(rng.uniform(0, span, n))
    # (a) pure noise
    Xn = rng.standard_normal((n, d))
    o, nm, _ = _excess(Xn, start, np.random.default_rng(2))
    exc_noise = o - nm
    # (b) planted: one feature carries position (monotone) + noise
    Xp = rng.standard_normal((n, d)); Xp[:, 0] = (start / span) * 3 + rng.standard_normal(n) * 0.5
    o2, nm2, _ = _excess(Xp, start, np.random.default_rng(3))
    exc_plant = o2 - nm2
    print(f"[validate] pure-noise excess = {exc_noise:+.3f} (want ~0)   "
          f"planted-signal excess = {exc_plant:+.3f} (want >0)\n")
    return exc_noise, exc_plant


def main():
    C = coords()
    _validate()
    res = {"block_mb": BLOCK_MB, "n_perm": N_PERM}
    print(f"Within-chromosome position, NULL-CORRECTED (block artifact removed by within-chr position shuffle)")
    print(f"  observed block |rho| vs {N_PERM}-perm null; excess and z reported (mean over chromosomes)\n")
    print(f"  {'basis':<16} {'obs_block':<10} {'null_mean':<10} {'EXCESS':<9} {'mean_z':<8} "
          f"{'chr sig>null95':<14} {'n_chr'}")
    print("  " + "-" * 82)
    for b in BASES:
        M, syms = G.basis(b)
        pos_i = {s: i for i, s in enumerate(syms)}
        obs_l, exc_l, z_l, nsig = [], [], [], 0
        for c in AUTOSOMES:
            g = [s for s in C.index[C.chromosome == c] if s in pos_i]
            if len(g) < MIN_PER_CHR:
                continue
            X = M[[pos_i[s] for s in g]]
            start = C.loc[g, "start"].values.astype(float)
            obs = block_rho(X, start)
            if obs is None:
                continue
            rng = np.random.default_rng(hash((b, c)) % (2**32))
            null = np.array([block_rho(X, rng.permutation(start)) for _ in range(N_PERM)])
            null = null[np.array([v is not None for v in null], dtype=bool)].astype(float)
            nm, nsd = float(null.mean()), float(null.std())
            obs_l.append(obs); exc_l.append(obs - nm)
            z_l.append((obs - nm) / (nsd + 1e-9))
            if obs > np.quantile(null, 0.95):
                nsig += 1
        if not obs_l:
            continue
        mo, me, mz = float(np.mean(obs_l)), float(np.mean(exc_l)), float(np.mean(z_l))
        print(f"  {b:<16} {mo:<10.3f} {mo - me:<10.3f} {me:<+9.3f} {mz:<+8.2f} "
              f"{nsig}/{len(obs_l):<12} {len(obs_l)}", flush=True)
        res[b] = dict(mean_obs=mo, mean_null=mo - me, mean_excess=me, mean_z=mz,
                      n_chr_sig=int(nsig), n_chr=len(obs_l))
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        json.dump(res, open(os.path.join(HERE, "results", "genome_position2.json"), "w"), indent=1)

    print("\n  VERDICT (gm_lib.py:20 -- model beats coexpr AND esm2 on the NULL-CORRECTED excess):")
    mods = {b: res[b]["mean_excess"] for b in ["maxtoki_lmhead", "maxtoki_we", "geneformer_we", "scgpt_we"]
            if b in res}
    refs = {b: res[b]["mean_excess"] for b in ["coexpr_devel", "esm2"] if b in res}
    if mods and refs:
        mb_ = max(mods, key=mods.get)
        ok = all(mods[mb_] > v for v in refs.values())
        rstr = "  ".join(f"{r.split('_')[0][:6]}:{v:+.3f}" for r, v in refs.items())
        print(f"    best model {mb_} excess {mods[mb_]:+.3f}  vs  {rstr}  -> "
              f"{'** MODEL: encodes sub-chromosomal position **' if ok else 'NO'}")
        res["verdict"] = dict(best_model=mb_, excess=mods[mb_], refs=refs, passes=bool(ok))
        json.dump(res, open(os.path.join(HERE, "results", "genome_position2.json"), "w"), indent=1)
    print("\n[done] -> results/genome_position2.json")


if __name__ == "__main__":
    main()
