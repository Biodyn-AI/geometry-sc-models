"""cc_benchmark_c2s — put C2S-Scale into the EXISTING cell-cycle benchmark, apples-to-apples.

WHY. `biotensor/codebase/route_cellcycle` already ran the cell-cycle manifold on scGPT, Geneformer, MaxToki,
STATE-SE and UCE and found the models beat an expression baseline in **1 of 45** (model x k x task) cells --
"the data carries everything". None of the C2S manifold work in this folder used that protocol, so C2S has never
been placed on the same ruler. This script does exactly that, on the IDENTICAL substrate (the canonical 3,000
Replogle K562 non-targeting controls; verified cell-for-cell against k562_cc_substrate.npz).

PROTOCOL (copied from route_cellcycle/cc_benchmark.py so the numbers are comparable):
  reduce to matched dimensionality PCA(k), k in {10,20,50}, standardize, small probe, held out
    (A) PHASE ORDERING   circular R^2 predicting phi        (ridge on cos/sin, 5-fold)
    (B) PHASE CLASSIFY   balanced accuracy on G1/S/G2M      (logistic, 50/50 split)
    (C) S-vs-G2M         AUROC                              (logistic, 50/50 split)
  compared against the EXPRESSION baseline on the SAME cells.

CIRCULARITY, stated as the original does: phi and the G1/S/G2M call are derived from marker expression, so the
expression baseline is circularly ADVANTAGED. A model that beats it anyway is meaningfully better; an expression
win is partly definitional.

Out: results/cc_benchmark_c2s.json
"""
from __future__ import annotations
import os, sys, json, glob, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

SEED = 0
KS = [10, 20, 50]


def circ_mean(a):
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def pca_k(X, k):
    return StandardScaler().fit_transform(
        PCA(min(k, X.shape[1], X.shape[0] - 1), random_state=SEED).fit_transform(X - X.mean(0)))


def order_circ_r2(Xk, phi):
    Y = np.column_stack([np.cos(phi), np.sin(phi)]); P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xk):
        P[te] = Ridge(alpha=10.0).fit(Xk[tr], Y[tr]).predict(Xk[te])
    err = 1 - np.cos(np.arctan2(P[:, 1], P[:, 0]) - phi)
    base = np.mean(1 - np.cos(phi - circ_mean(phi)))
    return float(1 - err.mean() / (base + 1e-12))


def classify_bal(Xk, lab):
    rng = np.random.default_rng(SEED); p = rng.permutation(len(lab))
    tr, te = p[:len(lab) // 2], p[len(lab) // 2:]
    m = LogisticRegression(max_iter=2000).fit(Xk[tr], lab[tr])
    return float(balanced_accuracy_score(lab[te], m.predict(Xk[te])))


def svg2m_auroc(Xk, lab):
    m_ = lab != "G1"; X2, y = Xk[m_], (lab[m_] == "G2M").astype(int)
    rng = np.random.default_rng(SEED); p = rng.permutation(len(y))
    tr, te = p[:len(y) // 2], p[len(y) // 2:]
    m = LogisticRegression(max_iter=2000).fit(X2[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X2[te])[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, help="C2S cell-summary states dir")
    ap.add_argument("--h5ad", required=True, help="the substrate h5ad (== k562_cc_substrate cells)")
    ap.add_argument("--substrate-npz", default=None, help="k562_cc_substrate.npz for canonical phi/cc_phase")
    ap.add_argument("--tag", default="C2S-2B")
    ap.add_argument("--out", default="results/cc_benchmark_c2s.json")
    a = ap.parse_args()
    import anndata, scipy.sparse as sp

    ad = anndata.read_h5ad(a.h5ad)
    # canonical phi / labels from the program's own substrate if available, else recompute
    if a.substrate_npz and os.path.exists(a.substrate_npz):
        z = np.load(a.substrate_npz, allow_pickle=True)
        phi = z["phi"].astype(float); cc = z["cc_phase"].astype(str)
        assert (cc == np.asarray(ad.obs["clusters"]).astype(str)).all(), "substrate cell order mismatch"
        src = "canonical k562_cc_substrate.npz"
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cc_phase import phase_angle
        phi, _, _, _ = phase_angle(ad); cc = np.asarray(ad.obs["clusters"]).astype(str)
        src = "recomputed (cc_phase.py)"
    print(f"phi/labels from: {src} | n={len(phi)}", flush=True)

    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    layers = sorted(int(os.path.basename(p).split("_")[1])
                    for p in glob.glob(os.path.join(a.states, "layer_*_activations.npy")))

    res = {"source": src, "expression": {}, "model": {}}
    for k in KS:
        Ex = pca_k(X, k)
        res["expression"][f"k{k}"] = dict(order=order_circ_r2(Ex, phi), classify=classify_bal(Ex, cc),
                                          auroc=svg2m_auroc(Ex, cc))
    e = res["expression"]
    print(f"\n  {'representation':<16}" + "".join(f"{'k='+str(k):>26}" for k in KS), flush=True)
    print(f"  {'EXPRESSION':<16}" + "".join(
        f"{e[f'k{k}']['order']:>8.3f}/{e[f'k{k}']['classify']:.3f}/{e[f'k{k}']['auroc']:.3f}" .rjust(26)
        for k in KS), flush=True)

    wins = 0; cells = 0
    for L in layers:
        H = np.load(os.path.join(a.states, f"layer_{L:02d}_activations.npy")).astype(np.float64)
        n = min(len(H), len(phi))
        Hs = H[:n]; ph = phi[ids[:n]] if len(ids) >= n and ids.max() < len(phi) else phi[:n]
        lab = cc[ids[:n]] if len(ids) >= n and ids.max() < len(cc) else cc[:n]
        rec = {}
        for k in KS:
            Ek = pca_k(Hs, k)
            rec[f"k{k}"] = dict(order=order_circ_r2(Ek, ph), classify=classify_bal(Ek, lab),
                                auroc=svg2m_auroc(Ek, lab))
            for t in ["order", "classify", "auroc"]:
                cells += 1
                if rec[f"k{k}"][t] > res["expression"][f"k{k}"][t]:
                    wins += 1
        res["model"][f"L{L:02d}"] = rec
        print(f"  {a.tag+' L'+str(L):<16}" + "".join(
            f"{rec[f'k{k}']['order']:>8.3f}/{rec[f'k{k}']['classify']:.3f}/{rec[f'k{k}']['auroc']:.3f}".rjust(26)
            for k in KS), flush=True)
    res["win_count"] = dict(model_wins=wins, total_cells=cells)
    print(f"\n  C2S beats the expression baseline in {wins} of {cells} (layer x k x task) cells", flush=True)
    print("  [route_cellcycle reference: scGPT/Geneformer/MaxToki/STATE/UCE won 1 of 45]", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
