"""CAN WE IMPROVE THE POSITION READ-OUT? — Ihor's "different linear directions at different intervals" + others.

Baseline: one global ridge on the OUTPUT table (lm_head), leakage-clean (near-duplicate genes removed by
embedding cosine, random 5-fold), mean signed Spearman over the 22 autosomes = 0.412.

(A) DIAGNOSTIC — do the directions actually differ by interval?  Split each chromosome into 3 position tertiles,
    fit a ridge inside each, and measure the pairwise |cos| between the three weight vectors. This is meaningless
    without calibration: with d >> n per tertile the estimates are noisy, so *any* split gives low cosines. We
    therefore compare against a NULL that splits the same genes into 3 RANDOM groups of the same sizes. If the
    position-tertile cosines are no lower than the random-split cosines, the apparent "different directions" is
    estimation noise and one global direction suffices.

(B) PREDICTIVE — does anything beat the single global ridge (all on identical folds)?
      ridge          one global direction (baseline)
      local_linear   fit a ridge on each test gene's k nearest TRAINING genes in embedding space
                     (a locally-linear / piecewise model: different direction in different regions)
      two_stage      classify the position tertile, then a within-tertile ridge
      drop_topPC     project out the top-K principal components first (position is ~orthogonal to them,
                     so removing high-variance nuisance may denoise)
      pls            partial least squares (supervised direction finding, unlike PCA)
      concat         concatenate the output table with the input table
      1B             the same read-out on the larger MaxToki-1B output table

Out: results/position_improve.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from genome_position_geometry import dedup
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

ALPHAS = np.logspace(0, 5, 12)
MINCHR, SEED, KNN = 200, 0, 150
MSETUP = f"{_DATA}/maxtoki/setup"
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"


def sp(a, b):
    r = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def load_1b():
    R = G.ST_Reader(f"{MSETUP}/MaxToki-1B-HF/model.safetensors")
    W = R.get("lm_head.weight").astype(np.float32)
    tok = json.load(open(f"{MSETUP}/token_dictionary.json"))
    e2s = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    rows, syms = [], []
    for ens, tid in tok.items():
        s = e2s.get(ens)
        if s is not None and tid < W.shape[0]:
            rows.append(tid); syms.append(s)
    o = np.argsort(syms); rows, syms = np.array(rows)[o], np.array(syms)[o]
    _, keep = np.unique(syms, return_index=True)
    return W[rows[keep]], syms[keep]


def per_chrom(M, syms, C, methods, Mextra=None):
    pi = {s: i for i, s in enumerate(syms)}
    out = {m: [] for m in methods}
    for c in AUTOSOMES:
        g = [s for s in C.index[C.chromosome == c] if s in pi]
        if len(g) < MINCHR:
            continue
        idx = [pi[s] for s in g]
        Xf = M[idx]; start = C.loc[g, "start"].values.astype(float)
        keep = dedup(Xf, start)
        if keep.sum() < 120:
            continue
        X = Xf[keep]; y = start[keep]; n = len(y)
        Xe = Mextra[idx][keep] if Mextra is not None else None
        fl = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)))

        preds = {m: np.zeros(n) for m in methods}
        for tr, te in fl:
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if "ridge" in methods:
                preds["ridge"][te] = RidgeCV(alphas=ALPHAS).fit(Xtr, y[tr]).predict(Xte)
            if "drop_topPC" in methods:
                U, s, Vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
                K = min(20, Vt.shape[0] - 1)
                P = Vt[:K]                                    # project OUT the top-K directions
                Xtr2 = Xtr - (Xtr @ P.T) @ P; Xte2 = Xte - (Xte @ P.T) @ P
                preds["drop_topPC"][te] = RidgeCV(alphas=ALPHAS).fit(Xtr2, y[tr]).predict(Xte2)
            if "pls" in methods:
                nc = min(12, Xtr.shape[1], len(tr) - 1)
                preds["pls"][te] = PLSRegression(n_components=nc).fit(Xtr, y[tr]).predict(Xte).ravel()
            if "concat" in methods and Xe is not None:
                sce = StandardScaler().fit(Xe[tr])
                Ctr = np.hstack([Xtr, sce.transform(Xe[tr])]); Cte = np.hstack([Xte, sce.transform(Xe[te])])
                preds["concat"][te] = RidgeCV(alphas=ALPHAS).fit(Ctr, y[tr]).predict(Cte)
            if "local_linear" in methods:
                Ntr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
                Nte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
                simm = Nte @ Ntr.T                              # (n_te, n_tr) cosine
                k = min(KNN, len(tr) - 1)
                for i in range(len(te)):
                    nb = np.argpartition(-simm[i], k)[:k]
                    preds["local_linear"][te[i]] = Ridge(alpha=100.0).fit(Xtr[nb], y[tr][nb]).predict(Xte[i:i+1])[0]
            if "two_stage" in methods:
                q = np.quantile(y[tr], [1/3, 2/3]); ttr = np.digitize(y[tr], q)
                clf = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ttr)
                tte = clf.predict(Xte)
                pr = np.zeros(len(te))
                for t in np.unique(ttr):
                    m_tr = ttr == t; m_te = tte == t
                    if m_te.sum() == 0:
                        continue
                    if m_tr.sum() >= 20:
                        pr[m_te] = Ridge(alpha=100.0).fit(Xtr[m_tr], y[tr][m_tr]).predict(Xte[m_te])
                    else:
                        pr[m_te] = y[tr][m_tr].mean() if m_tr.sum() else y[tr].mean()
                    pr[m_te] += 0.0
                # add the tertile's central offset so cross-tertile ordering is preserved
                cent = {t: np.median(y[tr][ttr == t]) for t in np.unique(ttr)}
                pr = pr * 0.5 + np.array([cent.get(t, y[tr].mean()) for t in tte]) * 0.5
                preds["two_stage"][te] = pr
        for m in methods:
            out[m].append(sp(preds[m], y))
    return {m: float(np.mean(v)) for m, v in out.items()}, {m: len(v) for m, v in out.items()}


def direction_diagnostic(M, syms, C, n_null=5):
    """(A) do tertile directions differ more than random-split directions?"""
    pi = {s: i for i, s in enumerate(syms)}
    tert, rand = [], []
    rng = np.random.default_rng(SEED)
    for c in AUTOSOMES:
        g = [s for s in C.index[C.chromosome == c] if s in pi]
        if len(g) < MINCHR:
            continue
        idx = [pi[s] for s in g]; Xf = M[idx]; start = C.loc[g, "start"].values.astype(float)
        keep = dedup(Xf, start)
        if keep.sum() < 150:
            continue
        X = StandardScaler().fit_transform(Xf[keep]); y = start[keep]

        def dirs(groups):
            ws = []
            for gi in np.unique(groups):
                m = groups == gi
                if m.sum() < 40:
                    return None
                w = Ridge(alpha=100.0).fit(X[m], y[m]).coef_
                ws.append(w / (np.linalg.norm(w) + 1e-12))
            return ws

        q = np.quantile(y, [1/3, 2/3]); wt = dirs(np.digitize(y, q))
        if wt is None:
            continue
        tert.append(np.mean([abs(float(wt[i] @ wt[j])) for i in range(len(wt)) for j in range(i+1, len(wt))]))
        rr = []
        for _ in range(n_null):
            wr = dirs(rng.integers(0, 3, len(y)))
            if wr:
                rr.append(np.mean([abs(float(wr[i] @ wr[j])) for i in range(len(wr)) for j in range(i+1, len(wr))]))
        if rr:
            rand.append(float(np.mean(rr)))
    return float(np.mean(tert)), float(np.mean(rand)), len(tert)


def main():
    C = coords()
    M, syms = G.basis("maxtoki_lmhead")
    Memb, syms_e = G.basis("maxtoki_we")
    assert list(syms) == list(syms_e), "table gene order mismatch"

    print("(A) DIAGNOSTIC — do linear directions differ by position interval?")
    t, r, nchr = direction_diagnostic(M, syms, C)
    print(f"    mean |cos| between POSITION-tertile directions : {t:.3f}")
    print(f"    mean |cos| between RANDOM-split directions     : {r:.3f}   ({nchr} chromosomes)")
    print(f"    -> {'tertile directions differ MORE than chance (local structure)' if t < r - 0.02 else 'no more different than a random split: ONE global direction'}\n")

    methods = ["ridge", "local_linear", "two_stage", "drop_topPC", "pls", "concat"]
    print("(B) PREDICTIVE — leakage-clean mean signed rho over chromosomes (baseline ridge = 0.412)")
    res, ns = per_chrom(M, syms, C, methods, Mextra=Memb)
    for m in methods:
        print(f"    {m:<13} {res[m]:+.3f}   ({ns[m]} chr)")

    out = dict(diagnostic=dict(tertile_cos=t, random_cos=r, n_chr=nchr), predictive=res, n_chr=ns)

    # bigger model
    try:
        print("\n(C) LARGER MODEL — MaxToki-1B output table")
        M1, s1 = load_1b()
        r1, n1 = per_chrom(M1, s1, C, ["ridge"])
        print(f"    1B ridge {r1['ridge']:+.3f}   ({n1['ridge']} chr)   vs 217M {res['ridge']:+.3f}")
        out["model_1b_ridge"] = r1["ridge"]
    except Exception as e:
        print("    1B skipped:", repr(e)[:120])

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "position_improve.json"), "w"), indent=1)
    print("\n[done] -> results/position_improve.json")


if __name__ == "__main__":
    main()
