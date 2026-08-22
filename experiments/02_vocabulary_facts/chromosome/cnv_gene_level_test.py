"""DOES THE MODEL'S CHROMOSOME KNOWLEDGE PATTERN-MATCH THE CNV SOURCE OR THE NORMAL-TISSUE SOURCE?

WHY THIS EXISTS. `cnv_mechanism_test.py` asked whether the chromosomes the model decodes best are the ones
with the loudest copy-number signal. That test correlates 21 numbers against 21 numbers and is hopeless:
every bootstrap CI spans roughly +-0.5. It cannot support or refute anything, and it should not be cited
either way.

THE FIX IS TO CHANGE THE UNIT OF ANALYSIS FROM CHROMOSOME TO GENE. n goes from 21 to ~2500.

THE LOGIC. Every basis, asked "which chromosome is this gene on?", gets some genes right and some wrong.
That per-gene profile of success is a FINGERPRINT of the information the basis is using. If MaxToki learned
chromosome from copy-number co-variation, its fingerprint should resemble the ANEUPLOID panel's fingerprint --
the same genes should be easy, the same genes hard -- over and above any resemblance to the NORMAL panel.
If instead it learned chromosome from normal chromatin-domain co-regulation, the resemblance should run the
other way.

    score_b(g) = out-of-fold probability that basis b assigns to gene g's TRUE chromosome

then compare, across genes:
    agree(model, k562)  vs  agree(model, TS-normal)
both raw, and each PARTIALLED on the other, so we ask what each source explains that the other does not.

TWO CONFOUNDS, BOTH CONTROLLED.
 1. CHROMOSOME PRIOR. Genes on gene-dense chromosomes are easier for every basis, which manufactures
    agreement from nothing. Controlled by also computing every correlation WITHIN chromosome (rank-residualise
    each score on its chromosome mean), so only within-chromosome gene-to-gene variation is compared.
 2. GENE DETECTABILITY. Well-measured genes are easier for both expression panels. This inflates
    agree(model,k562) and agree(model,TS) roughly equally, so the CONTRAST between them -- which is what the
    verdict reads -- is largely protected. The partial correlations sharpen this further.

READING IT. The verdict is the CONTRAST, not either number alone:
    agree(model,k562 | TS)  >>  agree(model,TS | k562)   -> supports the CNV account
    agree(model,TS | k562)  >>  agree(model,k562 | TS)   -> supports the co-regulation account
    the two are comparable                               -> the fingerprints do not discriminate; no verdict

POWER CHECK. Unlike the per-chromosome test, this one reports bootstrap CIs on the CONTRAST itself, so a null
result is interpretable as "no difference" rather than "no power". If the CI on the contrast is wide, say so.

Memory-lean: one basis at a time, float32, gm_lib cache cleared between bases.
Out: results/cnv_gene_level_test.json
"""
import os, sys, json, gc, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr, rankdata

N_GENES, SEED, N_BOOT = 2500, 0, 4000
BASES = ["maxtoki_lmhead", "coexpr_k562", "coexpr"]
NICE = {"maxtoki_lmhead": "model", "coexpr_k562": "K562 (aneuploid)", "coexpr": "TS (normal)"}


def common_symbols(C):
    keep = None
    for b in BASES:
        _, syms = G.basis(b)
        s = set(syms)
        keep = s if keep is None else (keep & s)
        G._cache.clear(); gc.collect()
    keep &= set(C.index[C.chromosome.isin(AUTOSOMES)])
    return sorted(keep)


def true_class_prob(X, y):
    """out-of-fold probability assigned to each gene's TRUE chromosome (the per-gene 'fingerprint')."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=0.1, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P = clf.predict_proba(sc.transform(X[te]))
        idx = {c: i for i, c in enumerate(clf.classes_)}
        p[te] = [P[i, idx[c]] for i, c in enumerate(y[te])]
    return p


def within_resid(v, chrom):
    """rank-residualise within chromosome: removes the 'some chromosomes are easy' component entirely."""
    r = rankdata(v).astype(float)
    out = r.copy()
    for c in np.unique(chrom):
        m = chrom == c
        out[m] = r[m] - r[m].mean()
    return out


def partial(a, b, ctrl):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(ctrl)
    A = np.vstack([np.ones_like(rc), rc]).T
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(spearmanr(res(ra), res(rb)).statistic)


def main():
    C = coords()
    syms_all = common_symbols(C)
    rng = np.random.default_rng(SEED)
    syms = sorted(rng.choice(syms_all, min(N_GENES, len(syms_all)), replace=False))
    chrom = np.array([C.chromosome[s] for s in syms])
    print(f"[setup] {len(syms)} genes, {len(set(chrom))} autosomes — unit of analysis is the GENE (n={len(syms)}), "
          f"not the chromosome (n={len(set(chrom))})\n", flush=True)

    sc_ = {}
    for b in BASES:
        M, ss = G.basis(b)
        pi = {s: i for i, s in enumerate(ss)}
        X = np.asarray(M[[pi[s] for s in syms]], dtype=np.float32)
        del M; G._cache.clear(); gc.collect()
        sc_[b] = true_class_prob(X, chrom)
        print(f"  {NICE[b]:<18} mean P(true chromosome) = {sc_[b].mean():.4f}   (chance = {1/len(set(chrom)):.4f})",
              flush=True)
        del X; gc.collect()

    mo, k5, ts = sc_["maxtoki_lmhead"], sc_["coexpr_k562"], sc_["coexpr"]
    res = dict(n_genes=len(syms), n_chrom=len(set(chrom)),
               mean_true_prob={NICE[b]: float(sc_[b].mean()) for b in BASES}, tests={})

    for tag, f in [("raw", lambda v: v), ("within-chromosome", lambda v: within_resid(v, chrom))]:
        Mo, K5, Ts = f(mo), f(k5), f(ts)
        a_k, a_t = float(spearmanr(Mo, K5).statistic), float(spearmanr(Mo, Ts).statistic)
        p_k, p_t = partial(Mo, K5, Ts), partial(Mo, Ts, K5)
        contrast = p_k - p_t
        # bootstrap the CONTRAST -- a null is only meaningful if this CI is tight
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(Mo), len(Mo))
            try:
                bs.append(partial(Mo[i], K5[i], Ts[i]) - partial(Mo[i], Ts[i], K5[i]))
            except Exception:
                pass
        lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"),) * 2
        res["tests"][tag] = dict(agree_model_k562=a_k, agree_model_ts=a_t,
                                 partial_k562_given_ts=p_k, partial_ts_given_k562=p_t,
                                 contrast=contrast, contrast_ci=[lo, hi])
        print(f"\n--- {tag} ---")
        print(f"  agreement  model ~ K562 (aneuploid)      {a_k:+.3f}")
        print(f"  agreement  model ~ TS   (normal)         {a_t:+.3f}")
        print(f"  partial    model ~ K562 | TS             {p_k:+.3f}")
        print(f"  partial    model ~ TS   | K562           {p_t:+.3f}")
        print(f"  CONTRAST   (K562|TS) - (TS|K562)         {contrast:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   {'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")

    w = res["tests"]["within-chromosome"]
    c, (lo, hi) = w["contrast"], w["contrast_ci"]
    if lo <= 0 <= hi:
        v = ("NO DISCRIMINATION — the contrast CI includes zero. The model's per-gene chromosome fingerprint "
             "matches the aneuploid and the normal panel about equally well, so this test does not favour "
             "either mechanism. Note this is now a well-powered null (n~2500 genes), not an underpowered one.")
    elif c > 0:
        v = ("SUPPORTS THE CNV ACCOUNT — the model's per-gene fingerprint resembles the aneuploid panel's "
             "beyond what the normal panel explains. The paper's chromatin-domain mechanism needs to share "
             "the stage, or yield.")
    else:
        v = ("SUPPORTS THE CO-REGULATION ACCOUNT — the model's per-gene fingerprint resembles the NORMAL "
             "panel's beyond what the aneuploid panel explains, which is the opposite of what the CNV "
             "mechanism predicts.")
    res["verdict"] = v
    print(f"\nVERDICT: {v}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "cnv_gene_level_test.json"), "w"), indent=1)
    print("[done] -> results/cnv_gene_level_test.json")


if __name__ == "__main__":
    main()
