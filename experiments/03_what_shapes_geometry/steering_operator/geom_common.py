"""Shared steering-ladder helpers for the route_geometry_hypotheses gates.

Everything here is a thin re-use of route_steering primitives so the numbers are directly comparable to
LOCAL_STEERING_RESULTS.md. The only new thing is `run_ladder`, which factors out "given a whitened point
cloud Xz and a pseudotime y, evaluate the steering rules and return constrained advance", so a caller can
feed it EITHER a model embedding OR raw-counts PCA (Gate 0), a within-branch subset (Gate 1), or an extended
rule set with a retraction baseline (Gate 2).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

RS = f"{_DATA}/codebase/route_steering"
sys.path.insert(0, RS)
from tangent_diagnostic import (prep, zscore, unit, steer, oracle_field,  # noqa: E402
                                local_tangent_alignment, CAP, SEED, BUDGETS, T_STAR)
from local_steering import (project_constant, local_field, constrained_advance,  # noqa: E402
                            K_LOCAL, M_TANGENT, TAUS)

DATA = f"{_DATA}/branchpoint"
MAXA = f"{_DATA}/maxtoki_acts"


def npz_path(model, sfx="setty"):
    a = os.path.join(DATA, f"{model}_{sfx}.npz")
    b = os.path.join(MAXA, f"{model}_{sfx}.npz")
    return a if os.path.exists(a) else b


def load(model, sfx="setty", cap=CAP, seed=SEED):
    """Returns emb, pseudotime, cell_idx, clusters after finite-y filter and CAP subsample (fixed seed)."""
    z = np.load(npz_path(model, sfx), allow_pickle=True)
    X = z["emb"].astype(np.float64)
    y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int) if "cell_idx" in z.files else np.arange(len(y))
    clu = z["clusters"] if "clusters" in z.files else np.array(["?"] * len(y))
    ok = np.isfinite(y)
    X, y, ci, clu = X[ok], y[ok], ci[ok], clu[ok]
    if len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        X, y, ci, clu = X[idx], y[idx], ci[idx], clu[idx]
    return X, y, ci, clu


def retract_field(Xtr, w, k=30):
    """Retraction baseline: constant direction w for the STEP, then the integrator snaps toward the mean of the
    k nearest real cells. We fold the retraction into the returned direction so it slots into the same `steer`
    loop: dir = unit(w) + (mean_kNN(x) - x)/h_scale. To keep it a fair 'graph walk', the pull is scaled to the
    typical step so a single unit step lands ~on the local data mean while still advancing along w."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x):
        _, idx = nn.kneighbors(x)
        pull = np.array([Xtr[idx[i]].mean(0) for i in range(len(x))]) - x
        step = np.tile(unit(w), (len(x), 1)) + pull            # advance + snap-to-data
        return step
    return f


def local_retract_field(Xtr, ytr, k=K_LOCAL, kret=30):
    """Local pseudotime-gradient direction + retraction toward local data mean (Gate 2's honest-direction arm)."""
    lf = local_field(Xtr, ytr, k=k)
    nn = NearestNeighbors(n_neighbors=kret).fit(Xtr)

    def f(x):
        g = lf(x)
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        _, idx = nn.kneighbors(x)
        pull = np.array([Xtr[idx[i]].mean(0) for i in range(len(x))]) - x
        return g + pull
    return f


DEFAULT_RULES = ("linear", "linear_proj", "local", "local_proj", "oracle_knn")


def run_ladder(Xz, y, seed=SEED, rules=DEFAULT_RULES, n_src=300):
    """Evaluate the steering ladder in a given whitened space. Xz already whitened (use prep()); y raw pt.
    Returns dict rule -> {off_ratio@T, advance@T, align_T8, ca@tau}. Faithful to local_steering.run."""
    y = zscore(y)
    perm = np.random.default_rng(seed).permutation(len(y))
    tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    Xtr, ytr, Xte, yte = Xz[tr], y[tr], Xz[te], y[te]

    w_lin = unit(Ridge(alpha=10.0).fit(Xtr, ytr).coef_)
    judge = KNeighborsRegressor(15).fit(Xtr, ytr)
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]))
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    h = 0.25 * d0
    src = Xte[yte <= np.quantile(yte, 1.0 / 3.0)]
    if len(src) > n_src:
        src = src[np.random.default_rng(seed).choice(len(src), n_src, replace=False)]

    field_bank = {
        "linear": lambda x: np.tile(w_lin, (len(x), 1)),
        "linear_proj": project_constant(Xtr, w_lin),
        "local": local_field(Xtr, ytr),
        "local_proj": local_field(Xtr, ytr, project_m=M_TANGENT),
        "oracle_knn": oracle_field(Xtr, ytr),
        "linear_retract": retract_field(Xtr, w_lin),
        "local_retract": local_retract_field(Xtr, ytr),
    }
    res = {}
    for name in rules:
        fn = field_bank[name]
        pts = steer(src, fn, h, BUDGETS)
        base = judge.predict(pts[0])
        rows = {}
        for T in BUDGETS:
            dnn = nn1.kneighbors(pts[T])[0][:, 0]
            rows[T] = dict(off_manifold_ratio=float(dnn.mean() / d0),
                           advance=float((judge.predict(pts[T]) - base).mean()))
        rows["align_T8"] = local_tangent_alignment(Xtr, pts[T_STAR], fn(pts[T_STAR]))
        rows["ca"] = {f"{t}": constrained_advance(rows, t) for t in TAUS}
        res[name] = rows
    res["_meta"] = dict(d0=d0, n_train=len(tr), n_src=len(src))
    return res


def ca13(res, rule):
    return res[rule]["ca"]["1.3"]
