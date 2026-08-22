"""Does the developmental trajectory's tangent ROTATE, and does that make bilinear steering stay on-manifold?

The program's curvature numbers are *decodability gaps* (kNN/quad minus linear R^2 for pseudotime). That is
the wrong statistic for steering: a chord through a curved arc decodes arc-length almost perfectly yet leaves
the manifold everywhere. Steering is governed by the rotation of the local tangent, and by whether the ascent
direction is constant or position-dependent:

    linear decoder    f(x) = b·x + c        =>  ascent dir = b            (CONSTANT)
    quadratic decoder f(x) = xQx + b·x + c  =>  ascent dir = 2Qx + b      (a FIELD; rotates with x)

If Q ~ 0 the two coincide and every gap below collapses to 0 — the test is self-normalizing.

Part 1 (H1) — tangent rotation. Fit a local linear pseudotime probe inside each pseudotime quintile; measure
  the angle between the first and last bin's direction (`theta_far`), and whether pairwise angle grows with bin
  separation (`monotonicity`). Every null below is a target whose TRUE tangent is constant, so whatever theta_far
  it shows is finite-sample noise; the statistic is rotation_excess = theta_far - theta_far_null.
    - pre-registered null: SNR-matched linear-in-X target along a RANDOM direction (weakest; bins sit elsewhere)
    - NULL A `null_rotation_stats`: rank-matched linear score (bins match the real bins) x 20 draws
    - NULL B `knn_smoother`: NULL A with the target kNN-smoothed, same operator on real and null (label noise)
    - NULL C `null_resid_perm_stats`: linear fit + residuals permuted WITHIN bins (matches per-bin noise
      variance exactly -- the control against a null that is quieter than the real target). This is the one to
      cite. See route_branchpoint/FULL_SPACE_SIGNIFICANCE_RESULTS.md for why a depressed null is worthless.
  theta_far is also, unlike the program's decodability-gap `curv_poly`, free of the linear-R^2 headroom confound
  (Spearman +0.04 vs -0.93 across the 16 model x tissue cells) -- see RESULTS.md section 3.

Part 2 (H2) — off-manifold steering. From held-out early cells, take equal-length unit steps along
  (a) the constant linear direction, (b) the quadratic gradient field 2Qx+b, (c) an `oracle_knn` reference
  field that uses real labels. At each budget measure distance to the nearest REAL cell (/ d0, the median
  real-cell NN distance) and the pseudotime advance judged by a kNN regressor fit on real cells only (never by
  the decoder used to steer). Same synthetic-linear control: with a linear target the bilinear steer must show
  no advantage.

All in the project-standard 20-D whitened PCA space; Q, b, judge and manifold reference fit on train only.
Criteria pre-registered in PREREGISTRATION.md before any number was computed.

Run:  ../../.venv/bin/python tangent_diagnostic.py [setty|lung|gut] [model ...]
Out:  results/tangent_diagnostic_<dataset>.json (+ printed tables)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{_DATA}"
DATA, MAXA = os.path.join(ROOT, "branchpoint"), os.path.join(ROOT, "maxtoki_acts")
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

MODELS = ["scgpt", "geneformer", "state", "maxtoki"]
CAP, DIM, NBINS, SEED = 3000, 20, 5, 0
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BUDGETS = list(range(13))            # steps of h = 0.25 * d0  => T=8 is the pre-registered comparison
T_STAR = 8


def path(model, sfx):
    a = os.path.join(DATA, f"{model}_{sfx}.npz")
    b = os.path.join(MAXA, f"{model}_{sfx}.npz")
    return a if os.path.exists(a) else b


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ang(u, v):
    """Angle in degrees between two vectors (signed sense preserved: no abs, a reversal must read as 180)."""
    return float(np.degrees(np.arccos(np.clip(unit(u) @ unit(v), -1.0, 1.0))))


def prep(X, d=DIM):
    """Project-standard space: center -> PCA(d) -> whiten. Unsupervised; basis only, no target leakage."""
    Xc = X - X.mean(0)
    Xr = PCA(min(d, Xc.shape[1], Xc.shape[0] - 1), random_state=SEED).fit_transform(Xc)
    return StandardScaler().fit_transform(Xr)


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-9)


def cv_linear_r2(Xz, y):
    p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xz):
        p[te] = Ridge(alpha=10.0).fit(Xz[tr], y[tr]).predict(Xz[te])
    return float(r2_score(y, p))


def rank_match(score, y_ref):
    """Push `score` through the empirical marginal of `y_ref` (identical distribution, identical quantile bins)."""
    return np.sort(y_ref)[np.argsort(np.argsort(score))]


def knn_smoother(Xz, k=15):
    """Operator that averages a target over each cell's k nearest neighbours in the embedding.

    LABEL-NOISE CONTROL. The calibrated null injects HOMOscedastic noise, but Palantir/dpt pseudotime error is
    heteroscedastic along the trajectory; unevenly-noisy bins yield unevenly-noisy local probes, which inflates
    theta_far beyond what the null predicts. Smoothing suppresses per-cell label noise while preserving any
    genuine curvature (a smooth nonlinear function stays nonlinear). Applied IDENTICALLY to real and null
    targets, so it cannot advantage either."""
    _, idx = NearestNeighbors(n_neighbors=k).fit(Xz).kneighbors(Xz)
    return lambda v: v[idx].mean(1)


def null_rotation_stats(Xz, y, w_glob, lin_r2, nbins=NBINS, n_seeds=20, smooth=None):
    """THE CALIBRATED NULL for H1, over n_seeds draws.

    Target = (real pseudotime's own best linear score) + matched noise, then RANK-MATCHED onto the real
    pseudotime's empirical marginal. Two properties make this the tightest possible 'no rotation' null:
      (a) a monotone reparametrisation of a linear score leaves the local tangent DIRECTION equal to w_glob
          everywhere -- so the null's true rotation is exactly zero -- while pseudotime itself is only defined
          up to monotone reparametrisation, i.e. the null is the honest statement of the linear hypothesis;
      (b) it has the real target's marginal, so quantile bins hold the same cells with the same within-bin
          target spread, equalising local probe SNR (the random-direction null does not).
    """
    r = float(np.clip(lin_r2, 0.05, 0.98))
    sigma = float(np.sqrt(max(1.0 / r - 1.0, 1e-6)))
    th, rh = [], []
    for s_ in range(n_seeds):
        rng = np.random.default_rng(1000 + s_)
        yn = rank_match(zscore(Xz @ w_glob) + sigma * rng.standard_normal(len(y)), y)
        if smooth is not None:
            yn = smooth(yn)                       # identical operator applied to real and null targets
        rr, _, _ = tangent_rotation(Xz, yn, nbins=nbins, seed=s_)
        th.append(rr["theta_far"]); rh.append(rr["monotonicity"])
    yn0 = rank_match(zscore(Xz @ w_glob) + sigma * np.random.default_rng(1000).standard_normal(len(y)), y)
    if smooth is not None:
        yn0 = smooth(yn0)
    return np.array(th), np.array(rh), float(cv_linear_r2(Xz, zscore(yn0)))


def null_resid_perm_stats(Xz, y, nbins=NBINS, n_seeds=20):
    """NULL C — heteroscedastic residual-permutation null. The tightest available control on clause A.

    Motivation (cf. route_branchpoint/FULL_SPACE_SIGNIFICANCE_RESULTS.md finding 1: a null whose statistic is
    depressed relative to the real one is worthless). NULL A/B above inject HOMOscedastic Gaussian noise, but
    real dpt/Palantir error varies along the trajectory. If real bins are noisier than null bins, their local
    probes are noisier, and theta_far is inflated for reasons that have nothing to do with curvature.

    Construction: fit the global linear model, take its residual r = y - yhat, and permute r WITHIN each
    pseudotime bin. This preserves each bin's residual variance EXACTLY (so per-bin probe noise is matched cell
    for cell) while destroying any systematic dependence of the residual on x inside the bin -- which is
    precisely the curvature. The null's true tangent is therefore constant (= w_glob) with real, heteroscedastic
    noise. It is conservative: it retains as much within-bin residual variance as the real target, curvature
    included."""
    yz = zscore(y)
    yhat = Ridge(alpha=10.0).fit(Xz, yz).predict(Xz)
    r = yz - yhat
    q = np.quantile(yz, np.linspace(0, 1, nbins + 1))
    bins = [np.where((yz >= q[b]) & ((yz <= q[b + 1]) if b == nbins - 1 else (yz < q[b + 1])))[0]
            for b in range(nbins)]
    th, rh = [], []
    for s_ in range(n_seeds):
        rng = np.random.default_rng(2000 + s_)
        rp = r.copy()
        for idx in bins:                                       # permute residuals within each bin
            rp[idx] = r[rng.permutation(idx)]
        rr, _, _ = tangent_rotation(Xz, yhat + rp, nbins=nbins, seed=s_)
        th.append(rr["theta_far"]); rh.append(rr["monotonicity"])
    return np.array(th), np.array(rh)


def sig(real, null):
    """Empirical one-sided calibration of `real` against the null draws (branchpoint_controls.py convention)."""
    return dict(real=float(real), null_mean=float(null.mean()), null_sd=float(null.std()),
                null_p95=float(np.percentile(null, 95)),
                z=float((real - null.mean()) / (null.std() + 1e-9)),
                p_value=float((np.sum(null >= real) + 1) / (len(null) + 1)),
                above_null_p95=bool(real > np.percentile(null, 95)))


def synth_linear_target(Xz, real_lin_r2, seed=SEED, direction=None):
    """SNR-matched target that is a GLOBALLY LINEAR function of the same features (true tangent constant).

    direction=None -> random w (the pre-registered null).
    direction=w_glob -> the STRICTER 'aligned' null: the target is the real pseudotime's own best linear
    predictor plus matched noise, so its quantile bins contain nearly the same cells as the real bins. This
    removes the objection that the random-w null bins a different region of the manifold and therefore faces
    a different within-bin covariance / estimation-noise regime.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(Xz.shape[1]) if direction is None else direction
    s = zscore(Xz @ w)
    r = float(np.clip(real_lin_r2, 0.05, 0.98))
    sigma = float(np.sqrt(max(1.0 / r - 1.0, 1e-6)))          # R^2 = 1/(1+sigma^2)
    return zscore(s + sigma * rng.standard_normal(len(s)))


# ------------------------------------------------------------------ Part 1: tangent rotation
def tangent_rotation(Xz, y, nbins=NBINS, seed=SEED):
    y = zscore(y)                                              # matched ridge shrinkage across real/synth
    q = np.quantile(y, np.linspace(0, 1, nbins + 1))
    dirs, halves, ns = [], [], []
    for b in range(nbins):
        m = (y >= q[b]) & ((y <= q[b + 1]) if b == nbins - 1 else (y < q[b + 1]))
        idx = np.where(m)[0]
        dirs.append(unit(Ridge(alpha=10.0).fit(Xz[idx], y[idx]).coef_))
        ns.append(len(idx))
        p = np.random.default_rng(seed + b).permutation(idx)   # within-bin split-half noise context
        h1, h2 = p[: len(p) // 2], p[len(p) // 2:]
        halves.append(ang(Ridge(alpha=10.0).fit(Xz[h1], y[h1]).coef_,
                          Ridge(alpha=10.0).fit(Xz[h2], y[h2]).coef_))
    dirs = np.array(dirs)
    theta = np.array([[ang(dirs[i], dirs[j]) for j in range(nbins)] for i in range(nbins)])
    sep = np.array([abs(i - j) for i in range(nbins) for j in range(i + 1, nbins)])
    off = np.array([theta[i, j] for i in range(nbins) for j in range(i + 1, nbins)])
    rho = float(spearmanr(sep, off).statistic)
    w_glob = unit(Ridge(alpha=10.0).fit(Xz, y).coef_)
    return dict(bin_n=[int(v) for v in ns], theta_far=float(theta[0, -1]),
                theta_matrix=theta.round(1).tolist(), monotonicity=rho,
                split_half_mean=float(np.mean(halves)),
                angle_to_global=[ang(w_glob, d) for d in dirs]), dirs, w_glob


# ------------------------------------------------------------------ Part 2: steering
def fit_quadratic(Xz, y):
    """Ridge on raw degree-2 features (NOT re-standardized, so Q is recoverable). Returns symmetric Q, b, cv R^2."""
    pf = PolynomialFeatures(2, include_bias=False).fit(Xz)
    P = pf.transform(Xz)
    best, best_a = -np.inf, ALPHAS[0]
    for a in ALPHAS:
        p = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(P):
            p[te] = Ridge(alpha=a).fit(P[tr], y[tr]).predict(P[te])
        r = r2_score(y, p)
        if r > best:
            best, best_a = r, a
    coef = Ridge(alpha=best_a).fit(P, y).coef_
    d = Xz.shape[1]
    Q, b = np.zeros((d, d)), np.zeros(d)
    for j, pw in enumerate(pf.powers_):
        nz = np.nonzero(pw)[0]
        if pw.sum() == 1:
            b[nz[0]] += coef[j]
        elif len(nz) == 1:                 # x_i^2
            Q[nz[0], nz[0]] += coef[j]
        else:                              # x_i x_k  ->  split across the symmetric pair
            Q[nz[0], nz[1]] += coef[j] / 2.0
            Q[nz[1], nz[0]] += coef[j] / 2.0
    return Q, b, float(best), best_a


def steer(X0, dir_fn, h, budgets):
    """Equal-path-length integration: unit-norm step each iteration. Returns {T: points}."""
    x = X0.copy(); out = {0: x.copy()}
    for t in range(1, max(budgets) + 1):
        D = dir_fn(x)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        x = x + h * D
        if t in budgets:
            out[t] = x.copy()
    return out


def oracle_field(Xtr, ytr, k=50):
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x):
        _, idx = nn.kneighbors(x)
        D = np.zeros_like(x)
        for i in range(len(x)):
            nb = idx[i]
            hi = nb[ytr[nb] > np.median(ytr[nb])]          # neighbours ahead in pseudotime
            hi = hi if len(hi) else nb
            D[i] = Xtr[hi].mean(0) - x[i]
        return D
    return f


def local_tangent_alignment(Xtr, pts, dirs, k=50, m=5):
    """Fraction of the step direction lying in the local manifold tangent (top-m PCs of the k nearest real cells)."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    _, idx = nn.kneighbors(pts)
    out = []
    for i in range(len(pts)):
        N = Xtr[idx[i]]
        V = np.linalg.svd(N - N.mean(0), full_matrices=False)[2][:m]
        out.append(float(np.linalg.norm(V @ unit(dirs[i]))))
    return float(np.mean(out))


def steering(Xz, y, seed=SEED, with_oracle=True):
    y = zscore(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    Xtr, ytr, Xte, yte = Xz[tr], y[tr], Xz[te], y[te]

    Q, b, quad_r2, alpha = fit_quadratic(Xtr, ytr)
    w_lin = unit(Ridge(alpha=10.0).fit(Xtr, ytr).coef_)
    lin_r2 = cv_linear_r2(Xtr, ytr)
    judge = KNeighborsRegressor(15).fit(Xtr, ytr)             # independent of both steering decoders

    nn_tr = NearestNeighbors(n_neighbors=2).fit(Xtr)
    d0 = float(np.median(nn_tr.kneighbors(Xtr)[0][:, 1]))     # real-cell NN distance scale
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    h = 0.25 * d0

    src = Xte[yte <= np.quantile(yte, 1.0 / 3.0)]             # held-out early / progenitor cells
    fields = {"linear": lambda x: np.tile(w_lin, (len(x), 1)),
              "bilinear": lambda x: 2.0 * (x @ Q) + b}
    if with_oracle:
        fields["oracle_knn"] = oracle_field(Xtr, ytr)

    res = {}
    for name, fn in fields.items():
        pts = steer(src, fn, h, BUDGETS)
        base = judge.predict(pts[0])
        rows = {}
        for T in BUDGETS:
            dnn = nn1.kneighbors(pts[T])[0][:, 0]
            rows[T] = dict(off_manifold_ratio=float(dnn.mean() / d0),
                           advance=float((judge.predict(pts[T]) - base).mean()),
                           displacement=float(np.linalg.norm(pts[T] - pts[0], axis=1).mean() / d0))
        D = fn(pts[T_STAR])
        rows["tangent_alignment_at_Tstar"] = local_tangent_alignment(Xtr, pts[T_STAR], D)
        res[name] = rows
    return dict(n_train=len(tr), n_source=len(src), d0=d0, step=h, alpha=alpha,
                linear_r2=lin_r2, quad_r2=quad_r2,
                grad_field_rotation=ang(2.0 * (Xtr[ytr <= np.quantile(ytr, 0.2)].mean(0) @ Q) + b,
                                        2.0 * (Xtr[ytr >= np.quantile(ytr, 0.8)].mean(0) @ Q) + b),
                curves=res)


# ------------------------------------------------------------------ driver
def run(model, sfx):
    p = path(model, sfx)
    if not os.path.exists(p):
        print(f"[skip] {model}/{sfx}: no cache"); return None
    z = np.load(p, allow_pickle=True)
    X = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y); X, y = X[ok], y[ok]
    if len(y) > CAP:
        idx = np.random.default_rng(SEED).choice(len(y), CAP, replace=False)
        X, y = X[idx], y[idx]
    Xz = prep(X)
    lin_r2 = cv_linear_r2(Xz, zscore(y))
    y_syn = synth_linear_target(Xz, lin_r2)

    rot_real, _, w_glob = tangent_rotation(Xz, y)
    rot_syn, _, _ = tangent_rotation(Xz, y_syn)
    excess = rot_real["theta_far"] - rot_syn["theta_far"]

    # CALIBRATED null (added after the first run; strictly harder than the pre-registered one, so it can only
    # weaken H1): 20 draws of a rank-matched linear-score target whose true tangent is constant everywhere.
    th_n, rh_n, null_r2 = null_rotation_stats(Xz, y, w_glob, lin_r2)
    theta_sig = sig(rot_real["theta_far"], th_n)
    rho_sig = sig(rot_real["monotonicity"], rh_n)
    excess_strict = rot_real["theta_far"] - float(th_n.mean())

    # LABEL-NOISE CONTROL: kNN-smooth the target (real and null alike) and re-run the whole comparison
    sm = knn_smoother(Xz)
    y_sm = sm(y)
    w_sm = unit(Ridge(alpha=10.0).fit(Xz, zscore(y_sm)).coef_)
    rot_sm, _, _ = tangent_rotation(Xz, y_sm)
    th_sm, rh_sm, _ = null_rotation_stats(Xz, y, w_sm, cv_linear_r2(Xz, zscore(y_sm)), smooth=sm)
    smoothed = dict(theta_far=sig(rot_sm["theta_far"], th_sm),
                    monotonicity=sig(rot_sm["monotonicity"], rh_sm))

    # NULL C: heteroscedastic residual-permutation (noise matched per-bin to the real target)
    th_c, rh_c = null_resid_perm_stats(Xz, y)
    resid_perm = dict(theta_far=sig(rot_real["theta_far"], th_c),
                      monotonicity=sig(rot_real["monotonicity"], rh_c))

    # robustness of theta_far to the two arbitrary knobs (PCA dim, bin count), vs the calibrated null each time
    robust = {}
    for dd in (20, 50):
        Xd = prep(X, dd)
        r2d = cv_linear_r2(Xd, zscore(y))
        wg = unit(Ridge(alpha=10.0).fit(Xd, zscore(y)).coef_)
        for nb in (4, 5, 6):
            tr_, _, _ = tangent_rotation(Xd, y, nbins=nb)
            tn_, rn_, _ = null_rotation_stats(Xd, y, wg, r2d, nbins=nb, n_seeds=8)
            robust[f"d{dd}_b{nb}"] = dict(real=tr_["theta_far"], null_mean=float(tn_.mean()),
                                          excess=tr_["theta_far"] - float(tn_.mean()),
                                          rho=tr_["monotonicity"], rho_null_mean=float(rn_.mean()))

    st_real = steering(Xz, y)
    st_syn = steering(Xz, y_syn, with_oracle=False)

    def gap(st):
        c = st["curves"]
        return c["linear"][T_STAR]["off_manifold_ratio"] - c["bilinear"][T_STAR]["off_manifold_ratio"]

    def bil_le_lin_everywhere(st):
        c = st["curves"]
        return all(c["bilinear"][T]["off_manifold_ratio"] <= c["linear"][T]["off_manifold_ratio"] + 1e-9
                   for T in BUDGETS)

    out = dict(model=model, dataset=sfx, n=int(len(y)), d=int(X.shape[1]), dim=DIM, linear_r2_20pc=lin_r2,
               rotation=dict(real=rot_real, synth_null=rot_syn,
                             calibrated_null=dict(theta_far=theta_sig, monotonicity=rho_sig,
                                                  achieved_lin_r2=null_r2, real_lin_r2=lin_r2),
                             smoothed=smoothed, resid_perm_null=resid_perm,
                             rotation_excess=excess, rotation_excess_strict=excess_strict,
                             robustness=robust),
               steering=dict(real=st_real, synth_null=st_syn,
                             gap_real=gap(st_real), gap_synth=gap(st_syn),
                             bilinear_le_linear_all_budgets=bil_le_lin_everywhere(st_real)))

    c, s = st_real["curves"], st_syn["curves"]
    print(f"\n===== {model} / {sfx}  (n={len(y)}, d={X.shape[1]}, 20-PC linear R2={lin_r2:.3f}) =====")
    print(f"  [H1 rotation]  theta_far={rot_real['theta_far']:.1f}deg  random-null={rot_syn['theta_far']:.1f}deg"
          f"  => EXCESS={excess:+.1f}deg | monotonicity rho={rot_real['monotonicity']:+.2f}"
          f" (null {rot_syn['monotonicity']:+.2f}) | split-half={rot_real['split_half_mean']:.1f}deg")
    print(f"      CALIBRATED null (20 rank-matched draws, lin R2 {null_r2:.3f} vs real {lin_r2:.3f}):"
          f" theta_far {theta_sig['null_mean']:.1f}+-{theta_sig['null_sd']:.1f}deg"
          f" => EXCESS={excess_strict:+.1f}deg  z={theta_sig['z']:+.1f}  p={theta_sig['p_value']:.3f}")
    print(f"      monotonicity rho: real {rho_sig['real']:+.2f} vs null {rho_sig['null_mean']:+.2f}"
          f"+-{rho_sig['null_sd']:.2f}  z={rho_sig['z']:+.1f}  p={rho_sig['p_value']:.3f}")
    tc = resid_perm["theta_far"]
    print(f"      NULL C (resid-perm, heteroscedastic): theta null {tc['null_mean']:.1f}+-{tc['null_sd']:.1f}"
          f" => EXCESS={tc['real']-tc['null_mean']:+.1f}deg  z={tc['z']:+.1f}  p={tc['p_value']:.3f}")
    ts, rs = smoothed["theta_far"], smoothed["monotonicity"]
    print(f"      LABEL-NOISE ctrl (kNN-smoothed target, same smoothing on null):"
          f" theta {ts['real']:.1f} vs {ts['null_mean']:.1f}+-{ts['null_sd']:.1f} (z={ts['z']:+.1f}, p={ts['p_value']:.3f})"
          f" | rho {rs['real']:+.2f} vs {rs['null_mean']:+.2f} (p={rs['p_value']:.3f})")
    print(f"      robustness real/excess: "
          + "  ".join(f"{k}:{v['real']:.0f}/{v['excess']:+.0f}" for k, v in robust.items()))
    print(f"  [H2 steering]  quad R2={st_real['quad_r2']:.3f} vs linear {st_real['linear_r2']:.3f}"
          f" | grad-field rotation early->late = {st_real['grad_field_rotation']:.1f}deg")
    print(f"      {'T':>3} | {'off-manifold ratio (nn dist / d0)':^34} | {'pseudotime advance (SD)':^30}")
    print(f"      {'':>3} | {'linear':>10} {'bilinear':>10} {'oracle':>10} | {'linear':>9} {'bilinear':>9} {'oracle':>9}")
    for T in (0, 2, 4, 6, 8, 12):
        print(f"      {T:>3} | {c['linear'][T]['off_manifold_ratio']:>10.2f} {c['bilinear'][T]['off_manifold_ratio']:>10.2f}"
              f" {c['oracle_knn'][T]['off_manifold_ratio']:>10.2f} | {c['linear'][T]['advance']:>9.3f}"
              f" {c['bilinear'][T]['advance']:>9.3f} {c['oracle_knn'][T]['advance']:>9.3f}")
    print(f"      gap@T{T_STAR} (lin-bil) = {out['steering']['gap_real']:+.3f}"
          f" | SYNTH-linear control gap = {out['steering']['gap_synth']:+.3f}"
          f" | bil<=lin at every budget: {out['steering']['bilinear_le_linear_all_budgets']}")
    print(f"      tangent alignment @T{T_STAR}: linear={c['linear']['tangent_alignment_at_Tstar']:.3f}"
          f" bilinear={c['bilinear']['tangent_alignment_at_Tstar']:.3f}"
          f" oracle={c['oracle_knn']['tangent_alignment_at_Tstar']:.3f}")
    print(f"      synth-null steering: lin_off={s['linear'][T_STAR]['off_manifold_ratio']:.2f}"
          f" bil_off={s['bilinear'][T_STAR]['off_manifold_ratio']:.2f}")
    return out


def main():
    sfx = sys.argv[1] if len(sys.argv) > 1 else "setty"
    models = sys.argv[2:] or MODELS
    out = {}
    for m in models:
        r = run(m, sfx)
        if r:
            out[m] = r
    f = os.path.join(RESULTS, f"tangent_diagnostic_{sfx}.json")
    json.dump(out, open(f, "w"), indent=1)

    print(f"\n================ VERDICT ({sfx}) — criteria fixed in PREREGISTRATION.md ================")
    h1 = [m for m, r in out.items() if r["rotation"]["rotation_excess"] >= 15.0
          and r["rotation"]["real"]["monotonicity"] >= 0.5]
    # the two clauses of H1, reported SEPARATELY against the calibrated null (they behave very differently)
    h1s = [m for m, r in out.items() if r["rotation"]["rotation_excess_strict"] >= 15.0]
    h1r = [m for m, r in out.items() if r["rotation"]["calibrated_null"]["monotonicity"]["p_value"] <= 0.05]
    h1sm = [m for m, r in out.items() if r["rotation"]["smoothed"]["theta_far"]["above_null_p95"]]
    h1c = [m for m, r in out.items() if r["rotation"]["resid_perm_null"]["theta_far"]["above_null_p95"]]
    h2 = [m for m, r in out.items()
          if r["steering"]["gap_real"] >= 0.25 and r["steering"]["bilinear_le_linear_all_budgets"]
          and r["steering"]["gap_synth"] <= 0.05
          and r["steering"]["real"]["curves"]["bilinear"][T_STAR]["advance"]
          >= r["steering"]["real"]["curves"]["linear"][T_STAR]["advance"]]
    print(f"  {'model':<12} {'theta':>7} {'null':>13} {'excess':>8} {'z':>6} {'p':>7} | {'rho':>6} {'rho_null':>9} {'p':>7}")
    for m, r in out.items():
        cn = r["rotation"]["calibrated_null"]; t, rr = cn["theta_far"], cn["monotonicity"]
        print(f"  {m:<12} {t['real']:>6.1f}d {t['null_mean']:>6.1f}+-{t['null_sd']:<4.1f} {t['real']-t['null_mean']:>+7.1f}d"
              f" {t['z']:>+6.1f} {t['p_value']:>7.3f} | {rr['real']:>+6.2f} {rr['null_mean']:>+9.2f} {rr['p_value']:>7.3f}")
    print(f"  H1 as pre-registered (excess>=15 & rho>=0.5, random null): {len(h1)}/{len(out)} {h1} "
          f"-> {'CONFIRMED' if len(h1) >= 3 else 'REFUTED'}")
    print(f"  .. clause A only: theta excess>=15deg vs CALIBRATED null:  {len(h1s)}/{len(out)} {h1s} "
          f"-> {'CONFIRMED' if len(h1s) >= 3 else 'REFUTED'}")
    print(f"  .. clause A on kNN-SMOOTHED target (label-noise control):  {len(h1sm)}/{len(out)} {h1sm} "
          f"-> {'CONFIRMED' if len(h1sm) >= 3 else 'REFUTED'}")
    print(f"  .. clause A vs NULL C (heteroscedastic resid-perm):        {len(h1c)}/{len(out)} {h1c} "
          f"-> {'CONFIRMED' if len(h1c) >= 3 else 'REFUTED'}")
    print(f"  .. clause B only: rho significant vs CALIBRATED null:      {len(h1r)}/{len(out)} {h1r} "
          f"-> {'CONFIRMED' if len(h1r) >= 3 else 'REFUTED'}")
    print(f"  H2 (gap>=0.25, monotone, advance kept, synth gap<=0.05): {len(h2)}/{len(out)} models {h2} "
          f"-> {'CONFIRMED' if len(h2) >= 3 else 'REFUTED'}")
    print(f"\n[done] -> {f}")


if __name__ == "__main__":
    main()
