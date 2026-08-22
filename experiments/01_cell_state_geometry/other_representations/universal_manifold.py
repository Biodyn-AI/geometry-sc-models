"""Experiment 3 — the "universal manifold" test (the genuinely UCE-specific one).

UCE's headline claim is a SINGLE cross-species / cross-dataset cell space. No other model in the program is
trained for this. Pool the cell-token embeddings across datasets and ask:

  A. INTEGRATION vs SILOING. In the pooled space, is a cell's neighborhood dominated by its own dataset
     (siloed / batch-structured) or mixed (integrated)? Metric: mean fraction of a cell's k=30 nearest
     neighbors that come from the SAME dataset (1.0 = fully siloed; chance = each dataset's share).

  B. A SHARED DIFFERENTIATION AXIS? Pool the 4 branching trajectories (blood/lung/gut human, pancreas
     mouse). Leave-one-tissue-out: train a pseudotime decoder on 3 tissues, predict the held-out tissue's
     pseudotime. High transfer R^2 => UCE places different lineages on a COMMON differentiation geometry
     (universal); low => each trajectory has its own. Linear vs kNN (nonlinear) transfer; raw and
     per-dataset-centered (isolates the within-tissue direction from the tissue offset). Within-tissue R^2
     = the upper bound.

  C. CROSS-SPECIES. Does mouse pancreas integrate with the human trajectories, or sit apart? (its own
     kNN dataset purity, and whether a human-trained pseudotime decoder transfers to it.)

Run:  ../../.venv_state/bin/python universal_manifold.py [--rep cls|emb|xuce] [--src default|xuce]
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
from branchpoint_geometry import pca_reduce  # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import r2_score

DATA = f"{_DATA}"
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
TRAJ = ["setty", "lung", "gut", "pancreas"]
SPECIES = {"setty": "human", "lung": "human", "gut": "human", "pancreas": "mouse"}
ALL_DS = TRAJ + ["cellcycle", "immune"]


def npz_path(name, src):
    base = f"{DATA}/xuce" if src == "xuce" else None
    if name in TRAJ:
        return f"{base}/uce_{name}.npz" if base else f"{DATA}/branchpoint/uce_{name}.npz"
    if name == "cellcycle":
        return f"{base}/uce_k562.npz" if base else f"{DATA}/cellcycle/uce_k562.npz"
    if name == "immune":
        return f"{base}/uce_ts_immune.npz" if base else f"{DATA}/celltype/uce_ts_immune.npz"


def load(name, rep, src, cap=3000):
    p = npz_path(name, src)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    if rep not in z.files:
        return None
    X = z[rep].astype(np.float64)
    y = z["pseudotime"].astype(float) if "pseudotime" in z.files else None
    if y is not None:
        ok = np.isfinite(y); X, y = X[ok], y[ok]
    if len(X) > cap:
        idx = np.random.default_rng(0).choice(len(X), cap, replace=False)
        X = X[idx]; y = y[idx] if y is not None else None
    return X, y


def integration(rep, src, k=30):
    """kNN dataset purity in the pooled space (all datasets)."""
    Xs, labs = [], []
    for name in ALL_DS:
        d = load(name, rep, src)
        if d is None:
            continue
        Xs.append(d[0]); labs += [name] * len(d[0])
    X = np.vstack(Xs); labs = np.array(labs)
    Xz = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xz)
    _, ind = nn.kneighbors(Xz)
    same = np.array([(labs[ind[i, 1:]] == labs[i]).mean() for i in range(len(X))])
    shares = {n: float((labs == n).mean()) for n in np.unique(labs)}
    per_ds = {n: float(same[labs == n].mean()) for n in np.unique(labs)}
    # expected same-fraction if perfectly mixed = each dataset's own share
    excess = {n: per_ds[n] - shares[n] for n in per_ds}
    return dict(k=k, n=int(len(X)), datasets=list(shares.keys()), share=shares,
                knn_same_dataset_frac=per_ds, excess_over_chance=excess,
                mean_purity=float(same.mean()))


def shared_axis(rep, src, d=30):
    """Is there a universal differentiation axis? Pool the 4 trajectories in ONE PCA/scaler space.
    For each tissue report: within-tissue 5-fold CV R^2 (upper bound — each is individually decodable);
    leave-one-tissue-out transfer R^2 in the POOLED geometry (offsets retained); and transfer after
    removing each tissue's mean (tests whether the within-tissue direction itself is shared)."""
    from sklearn.model_selection import KFold
    D = {}
    for t in TRAJ:
        r = load(t, rep, src)
        if r is not None and r[1] is not None:
            D[t] = r
    tissues = list(D.keys())
    Xall = np.vstack([D[t][0] for t in tissues])
    Xr_all = StandardScaler().fit_transform(pca_reduce(Xall, d))   # one pooled scaler
    offs, i = {}, 0
    for t in tissues:
        n = len(D[t][0]); offs[t] = (i, i + n); i += n
    def blk(t, center):
        a, b = offs[t]; Xt = Xr_all[a:b].copy()
        if center:
            Xt = Xt - Xt.mean(0)
        return Xt, D[t][1]
    rows = {}
    for held in tissues:
        Xh, yh = blk(held, False)
        # within-tissue CV (upper bound)
        pr = np.zeros_like(yh)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xh):
            pr[te] = Ridge(10.0).fit(Xh[tr], yh[tr]).predict(Xh[te])
        within = float(r2_score(yh, pr))
        # transfer in pooled geometry (offsets retained)
        Xtr = np.vstack([blk(t, False)[0] for t in tissues if t != held])
        ytr = np.concatenate([blk(t, False)[1] for t in tissues if t != held])
        lin = Ridge(10.0).fit(Xtr, ytr).predict(Xh)
        knn = KNeighborsRegressor(15).fit(Xtr, ytr).predict(Xh)
        # transfer after per-tissue mean removal (direction-only)
        Xtr_c = np.vstack([blk(t, True)[0] for t in tissues if t != held])
        Xh_c, _ = blk(held, True)
        lin_c = Ridge(10.0).fit(Xtr_c, ytr).predict(Xh_c)
        rows[held] = dict(species=SPECIES[held], n=int(len(yh)),
                          within_tissue_r2=within,
                          transfer_linear_r2=float(r2_score(yh, lin)),
                          transfer_knn_r2=float(r2_score(yh, knn)),
                          transfer_linear_centered_r2=float(r2_score(yh, lin_c)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="cls")
    ap.add_argument("--src", default="default", choices=["default", "xuce"])
    a = ap.parse_args()
    print(f"===== universal-manifold  rep={a.rep} src={a.src} =====")
    out = {"rep": a.rep, "src": a.src}

    integ = integration(a.rep, a.src)
    out["integration"] = integ
    print("\n[A] kNN dataset purity (frac of 30 NN from same dataset; excess over the dataset's own share):")
    for n in integ["datasets"]:
        print(f"    {n:9s} purity={integ['knn_same_dataset_frac'][n]:.3f}  share={integ['share'][n]:.3f}"
              f"  excess=+{integ['excess_over_chance'][n]:.3f}")
    print(f"    mean purity={integ['mean_purity']:.3f}")

    sa = shared_axis(a.rep, a.src)
    out["shared_axis"] = sa
    print("\n[B/C] universal differentiation axis? within-tissue R2 (upper bound) vs leave-one-out transfer:")
    for t, r in sa.items():
        print(f"    {t:9s} ({r['species']:5s}) within={r['within_tissue_r2']:+.3f} | "
              f"transfer lin={r['transfer_linear_r2']:+.3f} knn={r['transfer_knn_r2']:+.3f} "
              f"centered={r['transfer_linear_centered_r2']:+.3f}")

    json.dump(out, open(os.path.join(RESULTS, f"universal_manifold_{a.rep}{('_'+a.src) if a.src=='xuce' else ''}.json"), "w"), indent=1)
    print(f"\n[done] -> results/universal_manifold_{a.rep}.json")


if __name__ == "__main__":
    main()
