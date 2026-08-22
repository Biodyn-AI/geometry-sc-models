"""Experiment 2 — manifold structure of the UCE cell-token space.

Question: does UCE's cell embedding (CLS / X_uce) contain CURVED cell-state manifold structure that linear
methods miss, or is it — like the gene-mean-pool and every other model in the program — linearly accessible?
And what is the manifold's shape (intrinsic dimension, trajectory curvature)?

Runs three representations side by side: emb (gene-mean-pool, the cross-model default), cls (layer-2 CLS
residual), xuce (UCE's true final output). For each dataset:
  - GLOBAL SHAPE: TwoNN intrinsic dim, PCA participation ratio, PCA dim@90% var.
  - TRAJECTORY (setty/lung/gut/pancreas): linear-vs-bilinear-vs-kNN pseudotime decode (curvature) with the
    mandatory synthetic-linear control; + TANGENT ROTATION (angle the pseudotime-increase direction turns
    from the first to the last pseudotime quintile — the sharp "is the trajectory a curved arc or
    piecewise-linear" probe, echoing route_steering).
  - CELL-TYPE (immune): kNN recovery + classification curvature + synthetic-linear control.

Reuses route_branchpoint / route_lineage / route_state_geometry machinery read-only.

Run:  ../../.venv_state/bin/python manifold_celltoken.py [--rep cls|emb|xuce|all] [--src default|xuce]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_branchpoint"))
sys.path.insert(0, os.path.join(HERE, "..", "route_lineage"))
sys.path.insert(0, os.path.join(HERE, "..", "route_state_geometry"))
from branchpoint_geometry import decode_pseudotime, synth_linear_trajectory, twonn_id, pca_reduce  # noqa
from lineage_manifold import curvature  # noqa
from celltype_state import knn_recovery, synthetic_linear_control  # noqa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA = f"{_DATA}"
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
TRAJ = ["setty", "lung", "gut", "pancreas"]


def paths(src):
    base = f"{DATA}/xuce" if src == "xuce" else None
    P = {}
    for t in TRAJ:
        P[t] = (f"{base}/uce_{t}.npz" if base else f"{DATA}/branchpoint/uce_{t}.npz", "pseudotime", "traj")
    P["cellcycle"] = (f"{base}/uce_k562.npz" if base else f"{DATA}/cellcycle/uce_k562.npz", None, "geom")
    P["immune"] = (f"{base}/uce_ts_immune.npz" if base else f"{DATA}/celltype/uce_ts_immune.npz", "cell_type", "celltype")
    return P


def global_shape(X):
    Xo = X.astype(np.float64)
    idd = twonn_id(pca_reduce(Xo[:min(1500, len(Xo))], 50))
    Xc = Xo - Xo.mean(0)
    ev = PCA(min(200, Xc.shape[1], Xc.shape[0] - 1), random_state=0).fit(Xc).explained_variance_ratio_
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())            # participation ratio
    d90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    return dict(twonn_id=float(idd), participation_ratio=pr, pca_dim_90=d90)


def tangent_rotation(X, y, d=20):
    """Angle (deg) the local pseudotime-increase direction turns from the 1st to the 5th pseudotime quintile.
    Large => the trajectory manifold is a curved arc; ~0 => globally-linear direction."""
    Xr = StandardScaler().fit_transform(pca_reduce(X.astype(np.float64), d))
    q = np.quantile(y, [0, .2, .4, .6, .8, 1.0])
    def grad(m):
        w = Ridge(1.0).fit(Xr[m], y[m]).coef_
        return w / (np.linalg.norm(w) + 1e-12)
    e = (y >= q[0]) & (y < q[1]); l = (y >= q[4]) & (y <= q[5])
    if e.sum() < 20 or l.sum() < 20:
        return None
    return float(np.degrees(np.arccos(np.clip(grad(e) @ grad(l), -1, 1))))


def run(rep, src):
    P = paths(src)
    out = {}
    for name, (path, tgt, kind) in P.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing"); continue
        z = np.load(path, allow_pickle=True)
        if rep not in z.files:
            print(f"[skip] {name}: rep '{rep}' not in {os.path.basename(path)}"); continue
        X = z[rep].astype(np.float64)
        rec = dict(n=int(len(X)), **global_shape(X))
        if kind == "traj":
            y = z[tgt].astype(float); ok = np.isfinite(y); Xo, yo = X[ok], y[ok]
            dp = decode_pseudotime(Xo, yo, 20); sc = synth_linear_trajectory(Xo, yo, 20)
            rec.update(linear_r2=dp["linear_r2"], knn_r2=dp["knn_r2"], curvature=dp["curvature"],
                       bilinear_gain=dp["bilinear_gain"], synth_curvature=sc["curvature"],
                       curv_over_control=dp["curvature"] - sc["curvature"],
                       tangent_rotation_deg=tangent_rotation(Xo, yo))
        elif kind == "celltype":
            ct = z[tgt]
            kr = knn_recovery(X, ct)
            y = np.asarray(ct); keep = np.array([np.sum(y == v) >= 25 for v in y])
            yk = np.unique(y[keep], return_inverse=True)[1]
            cur = curvature(X[keep], yk, d=20); syn = synthetic_linear_control(X[keep], yk, d=20)
            rec.update(knn_bal_acc=kr["bal_acc"], knn_null=kr["null"], linear_acc=cur["linear_acc"],
                       knn_acc=cur["knn_acc"], curvature_acc=cur["curvature_acc"],
                       synth_curvature=syn["curvature_acc"])
        out[name] = rec
        s = f"  [{name:9s} {rep:4s}] TwoNN={rec['twonn_id']:5.1f} PR={rec['participation_ratio']:5.1f} d90={rec['pca_dim_90']:3d}"
        if kind == "traj":
            s += f" | curv={rec['curvature']:+.3f} (ctrl {rec['synth_curvature']:+.3f}) tangent_rot={rec['tangent_rotation_deg']:.0f}deg"
        elif kind == "celltype":
            s += f" | recovery={rec['knn_bal_acc']:.3f}(null {rec['knn_null']:.3f}) curv={rec['curvature_acc']:+.3f}"
        print(s, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="all")
    ap.add_argument("--src", default="default", choices=["default", "xuce"])
    a = ap.parse_args()
    reps = ["emb", "cls", "xuce"] if a.rep == "all" else [a.rep]
    allout = {}
    for rep in reps:
        print(f"\n===== representation: {rep} (src={a.src}) =====")
        r = run(rep, a.src)
        if r:
            allout[rep] = r
    tag = f"_{a.src}" if a.src == "xuce" else ""
    json.dump(allout, open(os.path.join(RESULTS, f"manifold_celltoken{tag}.json"), "w"), indent=1)
    print(f"\n[done] -> results/manifold_celltoken{tag}.json")


if __name__ == "__main__":
    main()
