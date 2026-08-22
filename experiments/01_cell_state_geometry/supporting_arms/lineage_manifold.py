"""Supervised bilinear concept-manifold probing + LEACE-style erasure on a BRANCHING
hematopoietic lineage manifold (scGPT L11, Tabula-Sapiens immune, 1000 cells).

The cell-cycle test (route_topology) found a *cyclic* concept that the model LINEARLY EMBEDS
(linear reads it optimally; bilinear/kNN add nothing). Cell-cycle is a simple 1-cycle. The
source hematopoietic-manifold paper's nonlinearity was a BRANCHING differentiation tree -- the
case cell-cycle didn't cover. This script runs the book's Ch4 §5 method (supervised, NOT the
unsupervised reconstruction AEs that gave "bilinear ≈ linear" everywhere):

  A. CURVATURE   linear vs quad(bilinear) vs kNN decoding of lineage branch in a MATCHED dim.
                 kNN >> linear ⇒ curved decision manifold; ≈ ⇒ flat / linearly embedded.
  B. ERASURE     LEACE-style. Ladder of {no / linear / bilinear} erasure × {linear / kNN} decode.
                 Residual that survives LINEAR erasure but dies under BILINEAR erasure =
                 irreducibly-nonlinear (2nd-order) lineage structure a linear probe provably can't reach.
                 MANDATORY CONTROL: linear-decode after linear-erasure ≈ chance (guaranteed by LEACE);
                 if the nonlinear residual is also ≈ chance, the branching manifold is linearly embedded.
  C. CONCEPT MATRIX  fit σ(D(Lx⊙Rx)) softmax CP probe; eigendecompose per-branch concept matrix W_c
                 = Σ_k D_ck l_k r_kᵀ. Low-rank & signed (curved) or dominated by one linear direction (flat)?
                 Held-out bilinear vs linear accuracy: does the quadratic form actually buy separation?
  D. GEOMETRY    intrinsic dim (TwoNN), H0/H1 persistence vs matched-covariance Gaussian null.

Reps: cached scGPT L11 (route_geometry/results/emb_cache/scgpt_L11.npz): linear_sae (linear baseline),
atomic (bilinear AE). CPU-only, threads capped. All caches read-only.

Run:  ../.venv/bin/python lineage_manifold.py
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
import torch
torch.set_num_threads(3)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lineage_graph import assign  # noqa: E402

RG = os.path.abspath(os.path.join(HERE, "..", "route_geometry", "results"))
EMB = os.path.join(RG, "emb_cache", "scgpt_L11.npz")
RULER = os.path.join(RG, "scgpt_L11_ruler.npz")
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
RNG = 0

# 5 well/decently-sampled branches; drop stem (n=8, branch point but too few to CV).
BRANCHES = ["myeloid", "T", "B", "NK", "erythroid"]


# ---------------- data ----------------
def load(rep):
    E = np.load(EMB, allow_pickle=True)
    d = np.load(RULER, allow_pickle=True)
    tis = d["tissue"]; ct = d["cell_type"]
    im = tis == "immune"
    X = E[rep].astype(np.float64)[im]
    br, mj, dp = assign(ct[im])
    keep = np.isin(br, BRANCHES)
    y = np.array([BRANCHES.index(b) for b in br[keep]])
    return X[keep], y, mj[keep], dp[keep]


def pca_reduce(X, d):
    Xc = X - X.mean(0)
    return PCA(min(d, Xc.shape[1]), random_state=RNG).fit_transform(Xc)


def chance(y):
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / c.sum())          # majority-class accuracy


# ---------------- A. curvature ----------------
def cv_acc(clf, X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr]); p = clf.predict(X[te])
        accs.append((p == y[te]).mean()); f1s.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def curvature(X, y, d=20):
    Xr = StandardScaler().fit_transform(pca_reduce(X, d))
    lin = cv_acc(LogisticRegression(max_iter=2000, C=1.0), Xr, y)
    knn = cv_acc(KNeighborsClassifier(15), Xr, y)
    Xq = StandardScaler().fit_transform(
        PolynomialFeatures(2, include_bias=False).fit_transform(pca_reduce(X, min(d, 20))))
    quad = cv_acc(LogisticRegression(max_iter=2000, C=0.5), Xq, y)
    return dict(dim=d, chance=chance(y),
                linear_acc=lin[0], linear_f1=lin[1],
                quad_acc=quad[0], quad_f1=quad[1],
                knn_acc=knn[0], knn_f1=knn[1],
                curvature_acc=knn[0] - lin[0], quad_gain_acc=quad[0] - lin[0])


# ---------------- B. LEACE erasure ----------------
def leace_fit(X, Z, eps=1e-6):
    """Closed-form LEACE eraser (Belrose 2023): guarantees zero cross-cov(erased X, Z),
    i.e. NO linear/affine predictor of Z survives. Returns callable eraser."""
    mu = X.mean(0); Xc = X - mu
    Zc = Z - Z.mean(0)
    Sig = np.cov(Xc, rowvar=False)
    w, U = np.linalg.eigh(Sig + eps * np.eye(Sig.shape[0]))
    w = np.clip(w, eps, None)
    Wh = U @ np.diag(w ** -0.5) @ U.T          # Sigma^{-1/2}
    Wi = U @ np.diag(w ** 0.5) @ U.T           # Sigma^{+1/2}
    Sxz = Xc.T @ Zc / (len(X) - 1)             # d×k cross-cov
    M = Wh @ Sxz
    Uu, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-8 * (s[0] + 1e-12)).sum())
    P = Uu[:, :r]                              # concept subspace in whitened coords
    A = Wi @ (np.eye(P.shape[0]) - P @ P.T) @ Wh
    return (lambda Xn: mu + (Xn - mu) @ A.T), r


def onehot(y):
    Z = np.zeros((len(y), y.max() + 1)); Z[np.arange(len(y)), y] = 1.0; return Z


def _decode_pair(X, y, seed):
    """(linear acc, kNN acc) with a fresh split; fit eraser only on train, apply to test would
    require passing eraser — here erasure is precomputed globally (transductive), we just decode."""
    lin = cv_acc(LogisticRegression(max_iter=2000, C=1.0), X, y, seed)[0]
    knn = cv_acc(KNeighborsClassifier(15), X, y, seed)[0]
    return lin, knn


def erasure(X, y, d_quad=25, shuffle_null=True):
    """LEACE erasure ladder, eraser refit INSIDE each CV fold (no leakage).

    Everything happens in a d_quad-dim reduced x-space and its degree-2 (bilinear) lift Φ.
    Two erasures:
      LINEAR   : LEACE in x-space  -> kills any affine-in-x predictor (guaranteed).
      BILINEAR : LEACE in Φ-space  -> kills any affine-in-Φ = quadratic-in-x predictor (guaranteed).
    Three decoders applied after each erasure:
      lin_x  : logistic on erased x         (linear probe)
      bil    : logistic on Φ(erased x)      (BILINEAR probe = the matched test for bilinear erasure)
      knn    : kNN on Φ(erased x)           (stronger, >2nd-order-sensitive probe)
    Controls (must hold): linear-erase+lin_x ≈ chance;  bilinear-erase+bil ≈ chance.
    Honesty caveat baked into interpretation: LEACE only guarantees LINEAR guardedness, so a
    positive bilinear residual after linear erasure is EXPECTED even for a linearly-embedded
    (redundantly-encoded) concept -- it is NOT by itself evidence of linear-inaccessibility.
    The decisive linear-vs-nonlinear-at-baseline test lives in curvature() / cp_probe()."""
    ch = chance(y)
    poly = PolynomialFeatures(2, include_bias=False)
    Xb = pca_reduce(X, d_quad)          # reduced x-space (erase & decode here)

    def decode(Xtr, Xte, ytr, yte):
        Ptr, Pte = poly.fit_transform(Xtr), poly.transform(Xte)
        s1 = StandardScaler().fit(Xtr); Xtr_s, Xte_s = s1.transform(Xtr), s1.transform(Xte)
        s2 = StandardScaler().fit(Ptr); Ptr_s, Pte_s = s2.transform(Ptr), s2.transform(Pte)
        lin_x = (LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, ytr).predict(Xte_s) == yte).mean()
        bil = (LogisticRegression(max_iter=2000, C=0.5).fit(Ptr_s, ytr).predict(Pte_s) == yte).mean()
        knn = (KNeighborsClassifier(15).fit(Ptr_s, ytr).predict(Pte_s) == yte).mean()
        return np.array([lin_x, bil, knn])

    def ladder(yv):
        Z = onehot(yv)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        base, lin_er, bil_er = [], [], []
        for tr, te in skf.split(Xb, yv):
            # baseline
            base.append(decode(Xb[tr], Xb[te], yv[tr], yv[te]))
            # linear erasure in x-space
            erx, _ = leace_fit(Xb[tr], Z[tr])
            lin_er.append(decode(erx(Xb[tr]), erx(Xb[te]), yv[tr], yv[te]))
            # bilinear erasure in Φ-space: erase, then decode ON the erased Φ directly
            Ptr, Pte = poly.fit_transform(Xb[tr]), poly.transform(Xb[te])
            erp, _ = leace_fit(Ptr, Z[tr])
            Ep_tr, Ep_te = erp(Ptr), erp(Pte)
            s = StandardScaler().fit(Ep_tr); Etr_s, Ete_s = s.transform(Ep_tr), s.transform(Ep_te)
            bl = (LogisticRegression(max_iter=2000, C=0.5).fit(Etr_s, yv[tr]).predict(Ete_s) == yv[te]).mean()
            bk = (KNeighborsClassifier(15).fit(Etr_s, yv[tr]).predict(Ete_s) == yv[te]).mean()
            bil_er.append(np.array([np.nan, bl, bk]))
        with np.errstate(invalid="ignore"):
            return np.nanmean(base, 0), np.nanmean(lin_er, 0), np.nanmean(bil_er, 0)

    base, lin_er, bil_er = ladder(y)
    out = dict(
        chance=ch, d_quad=d_quad,
        baseline=dict(lin_x=float(base[0]), bilinear=float(base[1]), knn=float(base[2])),
        linear_erase=dict(lin_x=float(lin_er[0]), bilinear=float(lin_er[1]), knn=float(lin_er[2])),
        bilinear_erase=dict(bilinear=float(bil_er[1]), knn=float(bil_er[2])),
        # matched-decoder residuals over chance (the book's target quantity):
        resid_bilinear_after_linear_erase=float(lin_er[1] - ch),   # 2nd-order info surviving LINEAR erasure
        resid_bilinear_after_bilinear_erase=float(bil_er[1] - ch), # should be ~0 (control)
        irreducibly_2nd_order=float((lin_er[1] - ch) - (bil_er[1] - ch)),
        # stronger kNN probe (>2nd-order-sensitive):
        resid_knn_after_linear_erase=float(lin_er[2] - ch),
        resid_knn_after_bilinear_erase=float(bil_er[2] - ch),
    )
    if shuffle_null:
        rng = np.random.default_rng(0)
        ys = y.copy(); rng.shuffle(ys)
        _, ls, bs = ladder(ys)
        out["shuffle_null"] = dict(resid_bilinear_after_linear_erase=float(ls[1] - ch),
                                   resid_knn_after_linear_erase=float(ls[2] - ch))
    return out


def linear_control(X, d_quad=25, k=5, seed=0):
    """DECISIVE CONTROL: a concept that is LINEARLY EMBEDDED BY CONSTRUCTION.
    Labels y_syn = argmax(x @ W_rand) are a linear (argmax) function of x -> provably linearly
    decodable and containing ZERO nonlinear structure. Running the SAME erasure ladder: if a big
    'bilinear survives linear erasure' residual appears HERE too, then that residual is an artifact
    of LEACE guaranteeing only linear guardedness -- NOT evidence of irreducible nonlinearity.
    This is the control that reveals the 'linearly embedded' answer if that's the truth."""
    Xb = pca_reduce(X, d_quad)
    rng = np.random.default_rng(seed)
    # bias the random directions so argmax classes are reasonably balanced
    for _ in range(20):
        W = rng.standard_normal((Xb.shape[1], k))
        y_syn = np.argmax(StandardScaler().fit_transform(Xb) @ W, axis=1)
        _, c = np.unique(y_syn, return_counts=True)
        if len(c) == k and c.min() >= 15:
            break
    return erasure(X, y_syn, d_quad, shuffle_null=False)


# ---------------- C. bilinear CP concept probe ----------------
class CPHead(torch.nn.Module):
    def __init__(self, d, k, rank):
        super().__init__()
        self.L = torch.nn.Linear(d, rank, bias=False)
        self.R = torch.nn.Linear(d, rank, bias=False)
        self.D = torch.nn.Linear(rank, k, bias=True)   # σ(D(Lx⊙Rx))

    def forward(self, x):
        return self.D(self.L(x) * self.R(x))

    def concept_matrices(self):
        """W_c = Σ_k D_ck l_k r_kᵀ  (per class c). Returns array [k_classes,d,d]."""
        Lw = self.L.weight.detach().numpy(); Rw = self.R.weight.detach().numpy()  # rank×d
        Dw = self.D.weight.detach().numpy()                                       # classes×rank
        Ws = np.einsum("cr,rd,re->cde", Dw, Lw, Rw)
        return Ws


def cp_probe(X, y, d=50, rank=16, epochs=300, seed=0):
    Xr = StandardScaler().fit_transform(pca_reduce(X, d))
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    tr, te = next(iter(skf.split(Xr, y)))
    Xt = torch.tensor(Xr[tr], dtype=torch.float32); yt = torch.tensor(y[tr])
    Xv = torch.tensor(Xr[te], dtype=torch.float32); yv = torch.tensor(y[te])
    k = int(y.max() + 1)
    m = CPHead(Xr.shape[1], k, rank)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xt), yt); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        bil_acc = float((m(Xv).argmax(1) == yv).float().mean())
    # linear baseline in same space
    lin_acc = cv_acc(LogisticRegression(max_iter=2000), Xr, y, seed)[0]
    # eigen-analyze concept matrices
    Ws = m.concept_matrices()
    eig = {}
    for c, name in enumerate(BRANCHES):
        S = (Ws[c] + Ws[c].T) / 2.0
        ev = np.linalg.eigvalsh(S)
        absev = np.sort(np.abs(ev))[::-1]
        pr = float((absev.sum() ** 2) / (np.sum(absev ** 2) + 1e-12))   # participation ratio (eff. rank)
        eig[name] = dict(eff_rank=pr, top_pos=float(ev.max()), top_neg=float(ev.min()),
                         signed_ratio=float(abs(ev.min()) / (abs(ev.max()) + 1e-12)),
                         top5_abs=[float(v) for v in absev[:5]])
    return dict(dim=d, rank=rank, chance=chance(y),
                bilinear_heldout_acc=bil_acc, linear_heldout_acc=lin_acc,
                bilinear_gain=bil_acc - lin_acc, concept_eig=eig)


# ---------------- D. geometry ----------------
def twonn_id(X, frac=0.9):
    """TwoNN intrinsic-dimension estimator (Facco 2017)."""
    from scipy.spatial.distance import cdist
    D = cdist(X, X); np.fill_diagonal(D, np.inf)
    r1 = np.sort(D, 1)[:, 0]; r2 = np.sort(D, 1)[:, 1]
    mu = r2 / (r1 + 1e-12); mu = mu[np.isfinite(mu) & (mu > 1)]
    mu = np.sort(mu)[: int(frac * len(mu))]
    F = np.arange(1, len(mu) + 1) / len(mu)
    x = np.log(mu); ylog = -np.log(1 - F + 1e-12)
    d = float(np.sum(x * ylog) / (np.sum(x * x) + 1e-12))   # slope through origin
    return d


def top_h_persistence(E, dim, n_sub=700, d_pca=15, seed=0):
    from ripser import ripser
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(E), min(n_sub, len(E)), replace=False)
    Y = PCA(min(d_pca, E.shape[1]), random_state=0).fit_transform((E - E.mean(0))[idx])
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    dgm = ripser(Y, maxdim=dim)["dgms"][dim]
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(dgm) == 0:
        return 0.0
    return float(np.sort(dgm[:, 1] - dgm[:, 0])[::-1][0])


def geometry(X, y):
    Xr = pca_reduce(X, 50)
    idd = twonn_id(Xr)
    # PCA dim for 90% variance
    Xc = X - X.mean(0)
    ev = PCA(min(200, Xc.shape[1]), random_state=0).fit(Xc).explained_variance_ratio_
    d90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    # H0/H1 vs matched-cov Gaussian null
    def null_stat(dim):
        real = top_h_persistence(X, dim, seed=0)
        Ec = X - X.mean(0); cov = np.cov(Ec, rowvar=False)
        Lc = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
        nl = [top_h_persistence(np.random.default_rng(100 + s).standard_normal(X.shape) @ Lc.T, dim, seed=s)
              for s in range(5)]
        nl = np.array(nl)
        return dict(real=real, null_mean=float(nl.mean()), null_std=float(nl.std()),
                    z=float((real - nl.mean()) / (nl.std() + 1e-9)))
    return dict(twonn_intrinsic_dim=idd, ambient_dim=int(X.shape[1]),
                pca_dim_90pct=d90, H0=null_stat(0), H1=null_stat(1))


# ---------------- driver ----------------
def run(rep):
    X, y, mj, dp = load(rep)
    print(f"\n===== scGPT L11 / {rep} — lineage manifold (n={len(y)}, "
          f"branches={dict(zip(BRANCHES, np.bincount(y)))}) =====", flush=True)
    out = {"rep": rep, "n": int(len(y)), "branches": BRANCHES,
           "counts": {b: int(c) for b, c in zip(BRANCHES, np.bincount(y))}}
    out["curvature_20d"] = curvature(X, y, 20)
    c = out["curvature_20d"]
    print(f"  [A curvature 20D] chance={c['chance']:.3f} | lin={c['linear_acc']:.3f} "
          f"quad={c['quad_acc']:.3f} knn={c['knn_acc']:.3f} | curv(knn-lin)={c['curvature_acc']:+.3f} "
          f"quad_gain={c['quad_gain_acc']:+.3f}", flush=True)
    out["erasure"] = erasure(X, y)
    e = out["erasure"]
    print(f"  [B erasure] chance={e['chance']:.3f} | baseline lin/bil/knn="
          f"{e['baseline']['lin_x']:.3f}/{e['baseline']['bilinear']:.3f}/{e['baseline']['knn']:.3f}", flush=True)
    print(f"      LINEAR-erase  -> lin_x={e['linear_erase']['lin_x']:.3f}(ctrl≈chance) "
          f"bilinear={e['linear_erase']['bilinear']:.3f} knn={e['linear_erase']['knn']:.3f}", flush=True)
    print(f"      BILINEAR-erase-> bilinear={e['bilinear_erase']['bilinear']:.3f}(ctrl≈chance) "
          f"knn={e['bilinear_erase']['knn']:.3f}", flush=True)
    print(f"      resid(bilinear probe): after-LIN={e['resid_bilinear_after_linear_erase']:+.3f} "
          f"after-BIL={e['resid_bilinear_after_bilinear_erase']:+.3f} "
          f"=> irreducibly-2nd-order={e['irreducibly_2nd_order']:+.3f} | "
          f"shuffle-null={e.get('shuffle_null',{}).get('resid_bilinear_after_linear_erase',float('nan')):+.3f}",
          flush=True)
    out["linear_control_erase"] = linear_control(X)
    lc = out["linear_control_erase"]
    print(f"  [B-ctrl LINEARLY-EMBEDDED synthetic concept] resid(bilinear probe) after-LIN-erase="
          f"{lc['resid_bilinear_after_linear_erase']:+.3f} (if ~as large as real => residual is NOT "
          f"evidence of nonlinearity)", flush=True)
    out["cp_probe"] = cp_probe(X, y)
    p = out["cp_probe"]
    print(f"  [C CP probe] chance={p['chance']:.3f} | linear_heldout={p['linear_heldout_acc']:.3f} "
          f"bilinear_heldout={p['bilinear_heldout_acc']:.3f} | bilinear_gain={p['bilinear_gain']:+.3f}", flush=True)
    for b in BRANCHES:
        g = p["concept_eig"][b]
        print(f"      W[{b:9s}] eff_rank={g['eff_rank']:.1f} signed(|neg|/|pos|)={g['signed_ratio']:.2f} "
              f"top+={g['top_pos']:+.2f} top-={g['top_neg']:+.2f}", flush=True)
    out["geometry"] = geometry(X, y)
    g = out["geometry"]
    print(f"  [D geometry] TwoNN ID={g['twonn_intrinsic_dim']:.1f} (ambient {g['ambient_dim']}, "
          f"PCA-dim@90%={g['pca_dim_90pct']}) | H0 z={g['H0']['z']:+.1f} | H1 z={g['H1']['z']:+.1f}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, f"scgpt_L11_{rep}_lineage.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    reps = sys.argv[1:] or ["linear_sae", "atomic"]
    for r in reps:
        run(r)
    print("\ndone.")
