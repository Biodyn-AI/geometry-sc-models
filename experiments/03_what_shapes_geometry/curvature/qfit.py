"""Route Q — supervised low-rank symmetric Q for the developmental-trajectory nonlinearity.

Core estimator + nested-CV harness. See PREREGISTRATION.md (written before any fit).

Model (prereg §3):
    linear :  y ~ b.x_hat + w*||x_s|| + c                              (ridge, alpha CV'd on train)
    quad   :  y ~ x_hat^T Q x_hat + b.x_hat + w*||x_s|| + c            Q = M diag(s) M^T
              (symmetric by construction, rank <= r, s unconstrained so Q may be INDEFINITE)

FAIRNESS (important -- an under-fit linear baseline manufactures fake Q-gain):
the linear part is *profiled out in closed form* rather than learned by the same optimizer as Q.
For fixed design A = [x_hat, ||x_s||] the ridge hat matrix H = A(A'A + aI)^-1 A' does not depend on
Q, so minimizing over (b,w,c) for every Q is exact and free. We therefore fit Q against the
*deflated* target (I-H)y, i.e. Q sees only the residual no linear readout of x can explain.
Consequences: (i) the linear baseline is the optimal ridge, never under-fit; (ii) r=0 gives
delta_q == 0 identically; (iii) any positive delta_q is supra-linear structure by construction.

Preprocessing (prereg §2): train-fold center -> train-fold per-dim std -> L2-normalize rows.
||x_s|| (pre-normalization norm) is an explicit linear covariate for BOTH models, so the
abundance/total-counts axis is available to the baseline and cannot masquerade as nonlinear gain.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

SEED = 20260710
DATA = f"{_DATA}/branchpoint"
MAXA = f"{_DATA}/maxtoki_acts"

RANKS = (1, 2, 4, 8, 16, 32)
LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
# reduced grid for the (expensive) null distributions -- same *procedure*, coarser grid
RANKS_NULL = (1, 4, 16)
LAMBDAS_NULL = (1e-4, 1e-3, 1e-2)

ALPHAS = np.logspace(-3, 4, 15)
STEPS = 300
LR = 5e-2


def load(model, dataset):
    """Return (X, y) with non-finite pseudotime dropped."""
    p = os.path.join(DATA, f"{model}_{dataset}.npz")
    if not os.path.exists(p) and model == "maxtoki":
        p = os.path.join(MAXA, f"maxtoki_{dataset}.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p, allow_pickle=True)
    X = z["emb"].astype(np.float64)
    y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y)
    return X[ok], y[ok]


def preprocess(Xtr, Xte, use_abundance=True):
    """Prereg §2. Fit center/std on TRAIN only. Returns xhat and the abundance covariate."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0

    def go(X):
        xs = (X - mu) / sd
        nrm = np.linalg.norm(xs, axis=1, keepdims=True)
        nrm[nrm < 1e-8] = 1.0
        return xs / nrm, nrm[:, 0]

    xh_tr, n_tr = go(Xtr)
    xh_te, n_te = go(Xte)
    if use_abundance:
        m, s = n_tr.mean(), n_tr.std() + 1e-8
        c_tr, c_te = (n_tr - m) / s, (n_te - m) / s
    else:
        c_tr, c_te = np.zeros_like(n_tr), np.zeros_like(n_te)
    return xh_tr, c_tr, xh_te, c_te


class LinearPart:
    """Ridge on A=[x_hat, cov] with intercept, plus the projector needed to deflate a quad feature."""

    def __init__(self, xh_tr, c_tr, y_tr):
        A = np.c_[xh_tr, c_tr]
        self.a_mu = A.mean(0)
        self.y_mu = y_tr.mean()
        Ac = A - self.a_mu
        yc = y_tr - self.y_mu
        self.alpha = float(RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(Ac, yc).alpha_)
        # R = (A'A + aI)^-1 A'   -> ridge coefs for any target: b = R @ target_centered
        G = Ac.T @ Ac + self.alpha * np.eye(Ac.shape[1])
        self.R = np.linalg.solve(G, Ac.T)
        self.Ac = Ac
        self.b = self.R @ yc

    def deflate(self, v_centered):
        """(I - H) v  where H is the ridge hat matrix. Used to strip the linear-explainable part."""
        return v_centered - self.Ac @ (self.R @ v_centered)

    def design(self, xh, c):
        return np.c_[xh, c] - self.a_mu

    def predict_linear(self, xh, c):
        return self.design(xh, c) @ self.b + self.y_mu

    def refit_with_quad(self, y_tr, q_tr):
        """Given the final quad feature on train, the optimal linear part is ridge on (y - q)."""
        self.q_mu = q_tr.mean()
        self.b_q = self.R @ ((y_tr - self.y_mu) - (q_tr - self.q_mu))

    def predict_quad(self, xh, c, q):
        return self.design(xh, c) @ self.b_q + (q - self.q_mu) + self.y_mu


class QOnly(torch.nn.Module):
    """Q = M diag(s) M^T. Symmetric, rank<=r, indefinite allowed."""

    def __init__(self, d, r, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.r = r
        self.M = torch.nn.Parameter(torch.randn(d, r, generator=g) / np.sqrt(d))
        self.s = torch.nn.Parameter(0.1 * torch.randn(r, generator=g))

    def qfeat(self, xh):
        z = xh @ self.M
        return (z * z) @ self.s

    def q_frob_sq(self):
        G = self.M.T @ self.M
        return ((self.s[:, None] * self.s[None, :]) * G * G).sum()

    def Q(self):
        M = self.M.detach().double().numpy()
        s = self.s.detach().double().numpy()
        return (M * s) @ M.T


def fit_q(lin, xh_tr, y_tr, r, lam_q, seed=0, steps=STEPS, lr=LR):
    """Fit Q against the DEFLATED target (I-H)y. Returns the QOnly module (or None if r==0)."""
    if r == 0:
        return None
    d = xh_tr.shape[1]
    yc = y_tr - lin.y_mu
    yt = lin.deflate(yc)
    scale = yt.std() + 1e-12                      # scale target for stable optimization
    Xt = torch.tensor(xh_tr, dtype=torch.float32)
    Yt = torch.tensor(yt / scale, dtype=torch.float32)
    Ac = torch.tensor(lin.Ac, dtype=torch.float32)
    Rt = torch.tensor(lin.R, dtype=torch.float32)

    m = QOnly(d, r, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        q = m.qfeat(Xt)
        q = q - q.mean()
        qt = q - Ac @ (Rt @ q)                    # deflate the quad feature the same way
        loss = ((Yt - qt / scale) ** 2).mean() + lam_q * m.q_frob_sq()
        loss.backward()
        opt.step()
        sch.step()
    return m


def _qfeat_np(m, xh):
    if m is None:
        return np.zeros(len(xh))
    with torch.no_grad():
        return m.qfeat(torch.tensor(xh, dtype=torch.float32)).double().numpy()


def _fold_scores(X, y, tr, te, r, lam, seed, use_abundance):
    xh_tr, c_tr, xh_te, c_te = preprocess(X[tr], X[te], use_abundance)
    lin = LinearPart(xh_tr, c_tr, y[tr])
    p_lin = lin.predict_linear(xh_te, c_te)
    m = fit_q(lin, xh_tr, y[tr], r, lam, seed=seed)
    q_tr, q_te = _qfeat_np(m, xh_tr), _qfeat_np(m, xh_te)
    lin.refit_with_quad(y[tr], q_tr)
    p_quad = lin.predict_quad(xh_te, c_te, q_te)
    return p_lin, p_quad, m, lin, (xh_tr, c_tr, xh_te, c_te)


def _inner_select(X, y, tr, ranks, lams, seed, use_abundance):
    """Nested selection of (r, lam) on an inner 80/20 split of the TRAIN fold only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    cut = int(0.8 * len(tr))
    i_tr, i_va = tr[perm[:cut]], tr[perm[cut:]]
    best, best_cfg = -np.inf, (ranks[0], lams[0])
    for r in ranks:
        for lam in lams:
            _, p_q, _, _, _ = _fold_scores(X, y, i_tr, i_va, r, lam, seed, use_abundance)
            sc = r2_score(y[i_va], p_q)
            if sc > best:
                best, best_cfg = sc, (r, lam)
    return best_cfg


def outer_cv(X, y, ranks=RANKS, lams=LAMBDAS, seed=SEED, use_abundance=True, return_models=False):
    """Prereg §1/§3. Out-of-fold predictions for linear baseline and low-rank-Q model."""
    n = len(y)
    p_lin, p_quad = np.zeros(n), np.zeros(n)
    cfgs, models = [], []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
        r, lam = _inner_select(X, y, tr, ranks, lams, seed + k, use_abundance)
        pl, pq, m, lin, pre = _fold_scores(X, y, tr, te, r, lam, seed + k, use_abundance)
        p_lin[te], p_quad[te] = pl, pq
        cfgs.append((int(r), float(lam)))
        if return_models:
            models.append(dict(fold=k, Q=m.Q(), r=int(r), lam=float(lam),
                               train_idx=tr, test_idx=te))
    out = dict(r2_lin=float(r2_score(y, p_lin)), r2_quad=float(r2_score(y, p_quad)),
               cfgs=cfgs, p_lin=p_lin, p_quad=p_quad)
    out["delta_q"] = out["r2_quad"] - out["r2_lin"]
    if return_models:
        out["models"] = models
    return out


def bootstrap_delta(y, p_lin, p_quad, n_boot=2000, seed=SEED):
    """Bootstrap 95% CI of delta_q over cells (prereg §4)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ds[i] = r2_score(y[idx], p_quad[idx]) - r2_score(y[idx], p_lin[idx])
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def null_shuffle(X, y, n_rep=20, seed=SEED, **kw):
    """Prereg §5.1 -- permute y, refit the entire nested pipeline."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_rep):
        ys = y[rng.permutation(len(y))]
        out.append(outer_cv(X, ys, ranks=RANKS_NULL, lams=LAMBDAS_NULL,
                            seed=seed + 100 * i, **kw)["delta_q"])
    return np.array(out)


def null_linear_synthetic(X, y, p_lin, n_rep=20, seed=SEED, **kw):
    """Prereg §5.2 -- THE decisive control (the one that caught route_lineage's LEACE artifact).

    y_synth = out-of-fold linear prediction of the real y + Gaussian noise, scaled so the linear-decode
    R^2 of y_synth matches that of the real y. y_synth is a LINEAR function of x plus noise by
    construction, so a correct estimator must return delta_q ~ 0.
    """
    rng = np.random.default_rng(seed)
    r2 = float(np.clip(r2_score(y, p_lin), 1e-3, 0.999))
    sigma = np.sqrt(np.var(p_lin) * (1.0 - r2) / r2)
    out = []
    for i in range(n_rep):
        ysyn = p_lin + rng.normal(0, sigma, len(y))
        out.append(outer_cv(X, ysyn, ranks=RANKS_NULL, lams=LAMBDAS_NULL,
                            seed=seed + 100 * i, **kw)["delta_q"])
    return np.array(out)
