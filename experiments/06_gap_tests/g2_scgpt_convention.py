"""G2 - how much does the scGPT input-convention bug actually change the answers?

The programme's stated outstanding integrity issue: `route_branchpoint/extract_scgpt.py` fed RAW
COUNTS to a checkpoint configured with `input_style: "binned"`, `n_bins: 51`. Only the Setty (blood)
substrate was ever re-extracted; nine other routes and `CROSS_MODEL_GEOMETRY_RESULTS.md` still rest
on the off-distribution embeddings, and none of them mentions it.

Re-extracting nine routes needs the scGPT model and a lot of time. But the question that matters --
"does the convention change the verdict?" -- is answerable right now, because BOTH versions are
already cached for three tissues:

    scgpt_{gut,lung,pancreas}.npz      raw counts fed to a binned checkpoint  (wrong)
    scgptbin_{gut,lung,pancreas}.npz   binned input                            (right)

Same cells, same layer, same everything else. So we can measure the size of the error directly.

Three readouts, chosen because they are what the affected routes actually report:
  pseudotime   cross-validated Spearman of a ridge probe (what route_utility / route_branchpoint use)
  lineage      cross-validated balanced accuracy of terminal cluster (what route_lineage uses)
  geometry     local tangent rotation between first and last pseudotime quintile, and the
               linear-vs-kNN decodability gap (what the curvature routes use)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

BT = f"{_DATA}"
BP, OUT = f"{BT}/data/branchpoint", f"{BT}/manifolds/gaps/results"
TISSUES = ["gut", "lung", "pancreas"]
K, SEED = 20, 0


def load(tag, tissue):
    z = np.load(f"{BP}/{tag}_{tissue}.npz", allow_pickle=True)
    y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y)
    return (z["emb"].astype(np.float64)[ok], y[ok],
            np.array([str(c) for c in z["clusters"]])[ok], z["cell_idx"][ok])


def prep(X, k=K):
    return PCA(n_components=min(k, X.shape[1], len(X) - 1), whiten=True,
               random_state=SEED).fit_transform(X)


def pseudotime_rho(Z, y):
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 15)), Z, y,
                          cv=KFold(5, shuffle=True, random_state=SEED))
    return float(stats.spearmanr(p, y).statistic)


def lineage_acc(Z, cl):
    keep = np.isin(cl, [c for c in set(cl) if (cl == c).sum() >= 25])
    if keep.sum() < 100 or len(set(cl[keep])) < 2:
        return float("nan")
    Zk, ck = Z[keep], cl[keep]
    p = cross_val_predict(LogisticRegression(max_iter=2000), Zk, ck,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return float(np.mean([np.mean(p[ck == c] == c) for c in np.unique(ck)]))


def curvature_gap(Z, y):
    """kNN minus linear cross-validated R2 -- the decodability gap the curvature routes report."""
    cv = KFold(5, shuffle=True, random_state=SEED)
    lin = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 15)), Z, y, cv=cv)
    knn = cross_val_predict(KNeighborsRegressor(15), Z, y, cv=cv)
    r2 = lambda p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2(knn) - r2(lin)), float(r2(lin))


def tangent_rotation(Z, y, n_q=5, k=100):
    """Mean angle between local pseudotime-gradient directions in the first vs last quintile."""
    q = np.quantile(y, np.linspace(0, 1, n_q + 1))
    def grad(mask):
        Zi, yi = Z[mask], y[mask]
        if len(Zi) < k:
            return None
        c = Zi - Zi.mean(0)
        g = np.linalg.lstsq(c, yi - yi.mean(), rcond=None)[0]
        n = np.linalg.norm(g)
        return g / n if n > 1e-12 else None
    a = grad((y >= q[0]) & (y <= q[1]))
    b = grad((y >= q[-2]) & (y <= q[-1]))
    if a is None or b is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"tissues": TISSUES, "k": K, "rows": []}
    print("scGPT input convention: RAW COUNTS (wrong) vs BINNED (right), same cells\n")
    print(f"{'tissue':9s} {'metric':14s} {'wrong':>8s} {'right':>8s} {'delta':>8s}")
    print("-" * 52)
    for t in TISSUES:
        Xw, yw, cw, iw = load("scgpt", t)
        Xr, yr, cr, ir = load("scgptbin", t)
        # same cells only
        common = np.intersect1d(iw, ir)
        Xw, yw, cw = [a[np.isin(iw, common)] for a in (Xw, yw, cw)]
        Xr, yr, cr = [a[np.isin(ir, common)] for a in (Xr, yr, cr)]
        Zw, Zr = prep(Xw), prep(Xr)
        gw, lw = curvature_gap(Zw, yw)
        gr, lr = curvature_gap(Zr, yr)
        row = {"tissue": t, "n": int(len(common)),
               "pseudotime_rho": [pseudotime_rho(Zw, yw), pseudotime_rho(Zr, yr)],
               "lineage_acc": [lineage_acc(Zw, cw), lineage_acc(Zr, cr)],
               "linear_r2": [lw, lr],
               "curvature_gap": [gw, gr],
               "tangent_rotation_deg": [tangent_rotation(Zw, yw), tangent_rotation(Zr, yr)]}
        out["rows"].append(row)
        for m in ("pseudotime_rho", "lineage_acc", "linear_r2", "curvature_gap",
                  "tangent_rotation_deg"):
            a, b = row[m]
            print(f"{t:9s} {m:14s} {a:8.3f} {b:8.3f} {b - a:+8.3f}")
        print()

    print("=== summary: mean |change| caused by the wrong convention ===")
    for m in ("pseudotime_rho", "lineage_acc", "linear_r2", "curvature_gap",
              "tangent_rotation_deg"):
        d = np.array([r[m][1] - r[m][0] for r in out["rows"]])
        d = d[np.isfinite(d)]
        print(f"  {m:22s} mean {d.mean():+.3f}   mean abs {np.abs(d).mean():.3f}   "
              f"max abs {np.abs(d).max():.3f}   sign flips: "
              f"{sum(np.sign(r[m][0]) != np.sign(r[m][1]) for r in out['rows'] if np.isfinite(r[m][0]))}")
    json.dump(out, open(f"{OUT}/g2_scgpt_convention.json", "w"), indent=1)
    print(f"\nwrote {OUT}/g2_scgpt_convention.json")


if __name__ == "__main__":
    main()
