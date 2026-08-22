"""Can we steer along the trajectory if we DON'T assume one smooth bend?

tangent_diagnostic.py showed two things: (1) the local pseudotime direction rotates 48-113 deg along the
trajectory, but disordered -- a branching tree of locally-linear segments, not one arc; (2) a GLOBAL steering
rule (constant linear direction, or the single quadratic 2Qx+b) drifts off the real-cell manifold, while a
label-using kNN oracle that re-picks its direction locally stays on it. So a LOCAL field can follow the
manifold in principle. This script asks whether an HONEST local field (fit on train cells, applied to held-out
cells with NO query labels) recovers that, and how much locality it takes.

Ladder of steering rules, increasing locality (all: unit step of h=0.25*d0, equal path length, fit on train,
applied to held-out early cells; advance judged by a kNN regressor on train, never by the steering rule):

  linear         constant w = argmax linear pseudotime probe                       (global, 0 bends)
  quadratic      2Qx + b, one global symmetric Q                                   (global, 1 bend)
  linear_proj    constant w, but each STEP projected onto the local manifold tangent (top-m PCs of nearby
                 train cells) -- isolates 'does projection alone fix drift?'
  piecewise_K    K pseudotime segments, each with its own local direction; at position x snap to the nearest
                 segment CENTROID (no query label) and use its direction                (K-1 bends)
  local          locally-weighted linear regression of pseudotime on position over the k nearest train cells,
                 re-fit at every step -> a position-dependent gradient field           (fully local, honest)
  local_proj     `local`, each step projected onto the local tangent                   (local + on-manifold)
  oracle_knn     step toward the mean of higher-pseudotime train neighbours -- REFERENCE CEILING, uses labels

Metric = the tradeoff, not raw advance. A fixed step budget must be split between advancing pseudotime and
staying on the curved manifold. `constrained_advance(tau)` = max over budgets of pseudotime advance subject to
off_manifold_ratio <= tau. Reported at tau = 1.2 / 1.3 / 1.5 (ratio = mean dist to nearest real train cell / d0).

PRE-STATED success criterion (before running): an honest local rule SOLVES steering if, in >=3/4 models on
Setty, its constrained_advance at tau=1.3 exceeds `linear`'s by >= 0.25 (SD units) AND reaches >= 70% of the
oracle's. If local rules only match `linear`, the manifold is not honestly followable and the oracle's success
was label leakage. If they reach oracle level, steering works -- it just needs locality, not higher order.

Run:  ../../.venv/bin/python local_steering.py [setty|lung|gut|pancreas] [model ...]
Out:  results/local_steering_<dataset>.json  (+ printed tables)
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tangent_diagnostic import (path, unit, prep, zscore, cv_linear_r2, fit_quadratic, steer,  # noqa: E402
                                oracle_field, local_tangent_alignment, CAP, SEED, BUDGETS, T_STAR)

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MODELS = ["scgpt", "geneformer", "state", "maxtoki"]
K_LOCAL = 100        # neighbourhood for the locally-weighted field
M_TANGENT = 10       # local tangent dimension for projection
PIECES = (5, 10)     # segment counts for piecewise-linear
TAUS = (1.2, 1.3, 1.5)


def local_field(Xtr, ytr, k=K_LOCAL, project_m=None):
    """Position-dependent gradient field: at x, weighted linear regression pseudotime~position over the k
    nearest TRAIN cells (gaussian weights by relative distance), gradient = coef (points to increasing time).
    Uses train labels to FIT; never uses the query's label. project_m -> also project onto the top-m weighted
    local PCs (the local tangent), forcing the step to stay in the data's local subspace."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x):
        dist, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            nb = idx[i]; Xn = Xtr[nb]; yn = ytr[nb]
            w = np.exp(-(dist[i] / (dist[i][-1] + 1e-9)) ** 2)
            mu = np.average(Xn, 0, weights=w)
            Xc = Xn - mu
            yc = yn - np.average(yn, weights=w)
            Wr = np.sqrt(w)[:, None]
            A = (Xc * w[:, None]).T @ Xc + 1e-3 * np.eye(Xc.shape[1])
            g = np.linalg.solve(A, (Xc * w[:, None]).T @ yc)
            if project_m:
                V = np.linalg.svd(Xc * Wr, full_matrices=False)[2][:project_m]  # top-m weighted local PCs
                g = V.T @ (V @ g)
            out[i] = g
        return out
    return f


def project_constant(Xtr, w, k=K_LOCAL, m=M_TANGENT):
    """Constant direction w, but projected onto the local tangent at each query point (isolates projection)."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x):
        _, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            Xc = Xtr[idx[i]] - Xtr[idx[i]].mean(0)
            V = np.linalg.svd(Xc, full_matrices=False)[2][:m]
            out[i] = V.T @ (V @ w)
        return out
    return f


def piecewise_field(Xtr, ytr, K):
    """K pseudotime segments; per segment fit a local linear pseudotime direction and store its centroid.
    At x: snap to the nearest segment centroid (Euclidean, NO query label) and return that direction."""
    q = np.quantile(ytr, np.linspace(0, 1, K + 1))
    cents, dirs = [], []
    for b in range(K):
        m = (ytr >= q[b]) & ((ytr <= q[b + 1]) if b == K - 1 else (ytr < q[b + 1]))
        cents.append(Xtr[m].mean(0))
        dirs.append(unit(Ridge(alpha=10.0).fit(Xtr[m], ytr[m]).coef_))
    cents = np.array(cents); dirs = np.array(dirs)

    def f(x):
        j = np.argmin(((x[:, None, :] - cents[None]) ** 2).sum(-1), axis=1)
        return dirs[j]
    return f


def constrained_advance(rows, tau):
    """Max pseudotime advance over budgets subject to off_manifold_ratio <= tau (0 if never satisfied at T>0)."""
    ok = [rows[T]["advance"] for T in BUDGETS if T > 0 and rows[T]["off_manifold_ratio"] <= tau]
    return float(max(ok)) if ok else 0.0


def run(model, sfx):
    p = path(model, sfx)
    if not os.path.exists(p):
        print(f"[skip] {model}/{sfx}"); return None
    z = np.load(p, allow_pickle=True)
    X = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y); X, y = X[ok], y[ok]
    if len(y) > CAP:
        idx = np.random.default_rng(SEED).choice(len(y), CAP, replace=False)
        X, y = X[idx], y[idx]
    Xz = prep(X); y = zscore(y)

    perm = np.random.default_rng(SEED).permutation(len(y))
    tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    Xtr, ytr, Xte, yte = Xz[tr], y[tr], Xz[te], y[te]

    Q, b, quad_r2, _ = fit_quadratic(Xtr, ytr)
    w_lin = unit(Ridge(alpha=10.0).fit(Xtr, ytr).coef_)
    judge = KNeighborsRegressor(15).fit(Xtr, ytr)
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]))
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    h = 0.25 * d0
    src = Xte[yte <= np.quantile(yte, 1.0 / 3.0)]
    if len(src) > 300:
        src = src[np.random.default_rng(SEED).choice(len(src), 300, replace=False)]

    fields = {
        "linear": lambda x: np.tile(w_lin, (len(x), 1)),
        "quadratic": lambda x: 2.0 * (x @ Q) + b,
        "linear_proj": project_constant(Xtr, w_lin),
        f"piecewise_{PIECES[0]}": piecewise_field(Xtr, ytr, PIECES[0]),
        f"piecewise_{PIECES[1]}": piecewise_field(Xtr, ytr, PIECES[1]),
        "local": local_field(Xtr, ytr),
        "local_proj": local_field(Xtr, ytr, project_m=M_TANGENT),
        "oracle_knn": oracle_field(Xtr, ytr),
    }

    res = {}
    for name, fn in fields.items():
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

    out = dict(model=model, dataset=sfx, n=int(len(y)), d0=d0, quad_r2=quad_r2, n_src=len(src), curves=res)
    ca13 = {k: res[k]["ca"]["1.3"] for k in fields}
    print(f"\n===== {model} / {sfx}  (n={len(y)}, d0={d0:.3f}, src={len(src)}) =====")
    print(f"  {'rule':<14} | {'off-manifold ratio @ T=2/4/8':^30} | {'advance @ T=2/4/8':^24}"
          f" | {'constr.adv tau=1.2/1.3/1.5':^26} | align")
    for name in fields:
        c = res[name]
        om = "/".join(f"{c[T]['off_manifold_ratio']:.2f}" for T in (2, 4, 8))
        ad = "/".join(f"{c[T]['advance']:.2f}" for T in (2, 4, 8))
        ca = "/".join(f"{c['ca'][f'{t}']:.2f}" for t in TAUS)
        print(f"  {name:<14} | {om:^30} | {ad:^24} | {ca:^26} | {c['align_T8']:.2f}")
    return out, ca13


def main():
    sfx = sys.argv[1] if len(sys.argv) > 1 else "setty"
    models = sys.argv[2:] or MODELS
    out, summary = {}, {}
    for m in models:
        r = run(m, sfx)
        if r:
            out[m], summary[m] = r
    json.dump(out, open(os.path.join(RESULTS, f"local_steering_{sfx}.json"), "w"), indent=1)

    print(f"\n================ VERDICT ({sfx}) — constrained_advance at tau=1.3, pre-stated bar ================")
    print(f"  {'model':<12} {'linear':>8} {'quad':>7} {'lin_proj':>9} {'piece5':>7} {'piece10':>8}"
          f" {'local':>7} {'loc_proj':>9} {'oracle':>7} | best-honest>lin+.25 & >=70%oracle?")
    winners = []
    HONEST = ["linear_proj", f"piecewise_{PIECES[0]}", f"piecewise_{PIECES[1]}", "local", "local_proj"]
    for m, s in summary.items():
        best = max(s[k] for k in HONEST)
        passes = (best >= s["linear"] + 0.25) and (s["oracle_knn"] <= 0 or best >= 0.70 * s["oracle_knn"])
        winners.append(passes)
        row = [s[k] for k in ["linear", "quadratic", "linear_proj", f"piecewise_{PIECES[0]}",
                              f"piecewise_{PIECES[1]}", "local", "local_proj", "oracle_knn"]]
        print(f"  {m:<12} " + " ".join(f"{v:>7.2f}" for v in row) + f" | {'YES' if passes else 'no'}")
    n = sum(winners)
    print(f"\n  Honest local rule beats linear by >=0.25 AND reaches >=70% oracle: {n}/{len(summary)} models"
          f" -> {'STEERING FOLLOWABLE' if n >= 3 else 'NOT cleanly followable'}")
    print(f"[done] -> results/local_steering_{sfx}.json")


if __name__ == "__main__":
    main()
