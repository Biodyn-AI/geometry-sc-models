"""CONTROL FOR THE DOMAIN-SHIFT CONFOUND IN THE GENE-LEVEL FINGERPRINT TEST.

`cnv_gene_level_test.py` found that MaxToki's per-gene chromosome-placement fingerprint resembles the NORMAL
panel's beyond what the aneuploid panel explains (within-chromosome contrast -0.091, CI [-0.152,-0.033]) --
the opposite of the CNV prediction.

THE ALTERNATIVE EXPLANATION THIS SCRIPT KILLS OR CONFIRMS. MaxToki and Tabula Sapiens are both built from
broad NORMAL human tissue; K562 is a single aneuploid line profiled under CRISPRi. So model~TS agreement could
reflect nothing about chromosome at all -- just SHARED MEASUREMENT QUALITY. Genes that are broadly and highly
expressed have well-estimated profiles in both MaxToki's corpus and TS, so they are easy to place in both;
a different set of genes is well-measured in K562. That alone would produce model~TS > model~K562 with no
chromosome mechanism involved, and it would be a domain artefact masquerading as a mechanistic result.

THE CONTROL. Recompute both partial correlations while additionally conditioning on per-gene detectability in
BOTH panels (row mean and row dispersion of each expression matrix -- how well-measured each gene is in TS,
and how well-measured it is in K562). If the contrast survives conditioning on all four detectability
covariates, "shared measurement quality" is not what is driving it.

    contrast = rho(model, K562 | TS, detectability...) - rho(model, TS | K562, detectability...)

SECOND CONTROL, cheaper and independent: a MATCHED-STRENGTH check. TS is a much weaker chromosome decoder than
K562 (mean P(true chr) 0.087 vs 0.315). One might worry the comparison is unfair in some direction. Note the
observed effect runs OPPOSITE to any naive strength bias -- the WEAKER decoder agrees with the model MORE --
so strength bias cannot manufacture this result, but we report both decoders' strength alongside so the reader
can see it.

Out: results/cnv_gene_level_control.json
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


def true_class_prob(X, y):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=0.1, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P = clf.predict_proba(sc.transform(X[te]))
        idx = {c: i for i, c in enumerate(clf.classes_)}
        p[te] = [P[i, idx[c]] for i, c in enumerate(y[te])]
    return p


def within_resid(v, chrom):
    r = rankdata(v).astype(float); out = r.copy()
    for c in np.unique(chrom):
        m = chrom == c
        out[m] = r[m] - r[m].mean()
    return out


def mpartial(a, b, ctrls):
    """Spearman(a,b) after rank-residualising BOTH on every control in ctrls (multiple regression on ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    A = np.column_stack([np.ones(len(ra))] + [rankdata(c) for c in ctrls])
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    r = spearmanr(res(ra), res(rb)).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def main():
    C = coords()
    keep = None
    for b in BASES:
        _, syms = G.basis(b)
        s = set(syms); keep = s if keep is None else (keep & s)
        G._cache.clear(); gc.collect()
    keep &= set(C.index[C.chromosome.isin(AUTOSOMES)])
    rng = np.random.default_rng(SEED)
    syms = sorted(rng.choice(sorted(keep), min(N_GENES, len(keep)), replace=False))
    chrom = np.array([C.chromosome[s] for s in syms])
    print(f"[setup] {len(syms)} genes, {len(set(chrom))} autosomes\n", flush=True)

    fp, det = {}, {}
    for b in BASES:
        M, ss = G.basis(b)
        pi = {s: i for i, s in enumerate(ss)}
        X = np.asarray(M[[pi[s] for s in syms]], dtype=np.float32)
        del M; G._cache.clear(); gc.collect()
        fp[b] = true_class_prob(X, chrom)
        if b != "maxtoki_lmhead":                       # detectability covariates from the two panels
            det[f"{b}_mean"] = X.mean(1).astype(float)
            det[f"{b}_sd"] = X.std(1).astype(float)
        print(f"  {NICE[b]:<18} mean P(true chr) = {fp[b].mean():.4f}", flush=True)
        del X; gc.collect()

    mo, k5, ts = fp["maxtoki_lmhead"], fp["coexpr_k562"], fp["coexpr"]
    cov = list(det.values())
    print(f"\n  conditioning on {len(cov)} detectability covariates: {list(det)}")

    res = dict(n_genes=len(syms), mean_true_prob={NICE[b]: float(fp[b].mean()) for b in BASES},
               covariates=list(det), tests={})

    for tag, f in [("raw", lambda v: v), ("within-chromosome", lambda v: within_resid(v, chrom))]:
        Mo, K5, Ts = f(mo), f(k5), f(ts)
        for lab, extra in [("no detectability control", []), ("+ detectability control", cov)]:
            p_k = mpartial(Mo, K5, [Ts] + extra)
            p_t = mpartial(Mo, Ts, [K5] + extra)
            con = p_k - p_t
            bs = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(Mo), len(Mo))
                try:
                    bs.append(mpartial(Mo[i], K5[i], [Ts[i]] + [c[i] for c in extra])
                              - mpartial(Mo[i], Ts[i], [K5[i]] + [c[i] for c in extra]))
                except Exception:
                    pass
            lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan,) * 2
            res["tests"][f"{tag} / {lab}"] = dict(partial_k562=p_k, partial_ts=p_t,
                                                  contrast=con, ci=[lo, hi])
            sig = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
            print(f"\n  [{tag} / {lab}]")
            print(f"     model~K562 | TS...   {p_k:+.3f}      model~TS | K562...   {p_t:+.3f}")
            print(f"     CONTRAST {con:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {sig}")

    a = res["tests"]["within-chromosome / no detectability control"]
    b = res["tests"]["within-chromosome / + detectability control"]
    shrink = (abs(b["contrast"]) / abs(a["contrast"])) if a["contrast"] else float("nan")
    survives = (b["ci"][1] < 0) and shrink > 0.5
    res["verdict"] = (
        f"contrast {a['contrast']:+.3f} -> {b['contrast']:+.3f} after conditioning on detectability "
        f"({shrink:.0%} retained). " +
        ("SURVIVES: the model's fingerprint resembles the NORMAL panel's for reasons not reducible to shared "
         "measurement quality, which is evidence against the CNV account."
         if survives else
         "DOES NOT SURVIVE cleanly: much of the apparent normal-panel advantage is explained by shared "
         "gene detectability, so the gene-level test should NOT be cited as evidence against CNV."))
    print(f"\nVERDICT: {res['verdict']}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "cnv_gene_level_control.json"), "w"), indent=1)
    print("[done] -> results/cnv_gene_level_control.json")


if __name__ == "__main__":
    main()
