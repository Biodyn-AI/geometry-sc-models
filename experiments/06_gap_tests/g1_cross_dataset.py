"""G1 - does a pretrained representation transfer across tissues better than expression?

The programme names this its single largest untested gap: every model-vs-expression verdict in the
corpus is WITHIN one dataset, and transfer is the one place a pretrained representation has a
structural advantage that a nearest-neighbour fit on expression cannot copy
(`route_utility/RESULTS.md:137-146`).

Design. Four developmental substrates that share a gene schema: blood (Setty CD34+), fetal gut,
lung airway, mouse pancreas. Fit a pseudotime probe on ONE tissue and apply it unchanged to ANOTHER.
Twelve ordered pairs.

Arms: scGPT (binned), Geneformer, STATE-SE, UCE, and raw expression on the 10,529 genes shared by
all four tissues. MaxToki is excluded: no blood extraction exists at the same layer as its
gut/lung/pancreas files, and guessing the layer would not be a matched comparison.

Controls that make it a real test:
  * The PCA is fitted on the TRAIN tissue and applied to the test tissue. Refitting on the target
    would leak the target's geometry and is not transfer.
  * Every arm is reduced to the same number of components, so no arm wins on width.
  * Within-tissue 5-fold CV is reported alongside, as the ceiling each arm is transferring from.
  * Expression uses the same cells as the model arms (joined on cell_idx).

Score = Spearman correlation between predicted and true pseudotime on the target tissue. Spearman is
scale-free, which matters because pseudotime is scaled differently per tissue.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os, sys, itertools
import numpy as np
import h5py
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

BT = f"{_DATA}"
BP = f"{BT}/data/branchpoint"
OUT = f"{BT}/manifolds/gaps/results"
K, SEED = 50, 0

H5 = {"blood": f"{BT}/data/hematopoiesis/setty19_cd34_bm.h5ad",
      "gut": f"{BT}/data/pancreas/gut_setty_schema.h5ad",
      "lung": f"{BT}/data/pancreas/lung_airway_setty_schema.h5ad",
      "pancreas": f"{BT}/data/pancreas/pancreas_setty_schema.h5ad"}
TISSUES = list(H5)
MODELS = ["scgptbin", "geneformer", "state", "uce"]


def emb_file(model, tissue):
    return f"{BP}/{model}_{'setty' if tissue == 'blood' else tissue}.npz"


def load_model(model, tissue):
    p = emb_file(model, tissue)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y)
    return z["emb"].astype(np.float64)[ok], y[ok], z["cell_idx"][ok]


def var_names(f):
    v = f["var"]["index"][:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def load_expression(tissue, keep_genes, cell_idx):
    """log1p CP10k on the shared gene set, for exactly the cells the model arms used."""
    with h5py.File(H5[tissue], "r") as f:
        names = var_names(f)
        gi = {g: i for i, g in enumerate(names)}
        cols = np.array([gi[g] for g in keep_genes])
        X = f["X"]
        n_obs = f["obs"]["index"].shape[0]
        M = sparse.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                              shape=(n_obs, len(names)))
    M = M[cell_idx][:, cols].toarray().astype(np.float64)
    M = M / np.clip(M.sum(1, keepdims=True), 1, None) * 1e4
    return np.log1p(M)


def fit_apply(Xtr, ytr, Xte, k=K):
    """PCA + ridge fitted on train only, applied to test. Returns Spearman on test."""
    p = PCA(n_components=min(k, Xtr.shape[1], len(Xtr) - 1), random_state=SEED).fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(p.transform(Xtr), ytr)
    return p, m


def score(p, m, Xte, yte):
    pred = m.predict(p.transform(Xte))
    return float(stats.spearmanr(pred, yte).statistic)


def within(X, y, k=K):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        p, m = fit_apply(X[tr], y[tr], X[te], k)
        pred[te] = m.predict(p.transform(X[te]))
    return float(stats.spearmanr(pred, y).statistic)


def main():
    os.makedirs(OUT, exist_ok=True)

    # shared gene space across all four tissues
    vs = {}
    for t in TISSUES:
        with h5py.File(H5[t], "r") as f:
            vs[t] = set(var_names(f))
    keep_genes = sorted(set.intersection(*vs.values()))
    print(f"shared genes across {len(TISSUES)} tissues: {len(keep_genes)}\n")

    # assemble every arm on the same cells per tissue
    arms = {}
    for model in MODELS:
        d = {}
        for t in TISSUES:
            r = load_model(model, t)
            if r is None:
                print(f"  MISSING {model}/{t} - arm dropped")
                d = None
                break
            d[t] = r
        if d:
            arms[model] = d

    expr = {}
    for t in TISSUES:
        ref = arms[MODELS[0]][t]
        expr[t] = (load_expression(t, keep_genes, ref[2]), ref[1], ref[2])
        print(f"  expression/{t:9s} {expr[t][0].shape}")
    arms["expression"] = expr
    print()

    out = {"k": K, "n_shared_genes": len(keep_genes), "tissues": TISSUES,
           "within": {}, "cross": {}}

    print("WITHIN-TISSUE (5-fold CV) - the ceiling each arm transfers from")
    for name, d in arms.items():
        row = {t: within(d[t][0], d[t][1]) for t in TISSUES}
        out["within"][name] = row
        print(f"  {name:11s} " + "  ".join(f"{t} {row[t]:+.3f}" for t in TISSUES))

    print("\nCROSS-TISSUE (fit on A, apply unchanged to B), Spearman on B")
    for name, d in arms.items():
        vals = {}
        for a, b in itertools.permutations(TISSUES, 2):
            Xa, ya, _ = d[a]
            Xb, yb, _ = d[b]
            if Xa.shape[1] != Xb.shape[1]:
                continue
            p, m = fit_apply(Xa, ya, Xb)
            vals[f"{a}->{b}"] = score(p, m, Xb, yb)
        out["cross"][name] = vals
        v = np.array(list(vals.values()))
        print(f"  {name:11s} mean {v.mean():+.3f}  median {np.median(v):+.3f}  "
              f"positive {int((v > 0).sum())}/{len(v)}  " +
              " ".join(f"{k.split('->')[0][:2]}>{k.split('->')[1][:2]}:{x:+.2f}"
                       for k, x in list(vals.items())[:4]))

    print("\n=== MODEL minus EXPRESSION, cross-tissue ===")
    e = out["cross"]["expression"]
    for name in MODELS:
        if name not in out["cross"]:
            continue
        d = np.array([out["cross"][name][k] - e[k] for k in e if k in out["cross"][name]])
        w = np.array([out["within"][name][t] - out["within"]["expression"][t] for t in TISSUES])
        print(f"  {name:11s} cross {d.mean():+.3f} ({int((d > 0).sum())}/{len(d)} pairs) | "
              f"within {w.mean():+.3f} ({int((w > 0).sum())}/{len(w)} tissues)")
        out.setdefault("model_minus_expression", {})[name] = {
            "cross_mean": float(d.mean()), "cross_wins": int((d > 0).sum()), "cross_n": len(d),
            "within_mean": float(w.mean()), "within_wins": int((w > 0).sum())}

    json.dump(out, open(f"{OUT}/g1_cross_dataset.json", "w"), indent=1)
    print(f"\nwrote {OUT}/g1_cross_dataset.json")


if __name__ == "__main__":
    main()
