"""IS POSITION A LINEAR DIRECTION? non-linear? curved? — concrete mechanistic characterisation (Ihor).

For MaxToki lm_head, within each chromosome, all leakage-clean (near-duplicate genes removed by embedding
cosine, random 5-fold out-of-fold), we ask:

  (1) THE DIRECTION. The linear ridge probe IS a weight vector w with <w, embedding> ~ position. We report its
      dimensionality (OOF rho using only the top-k principal components of the within-chromosome, identity-
      removed embedding) and its alignment with those PCs (|cos(w, PC_k)|). This says whether position is one
      direction or spread over several, and which axes carry it.

  (2) LINEAR vs NON-LINEAR. Compare OOF rho of a LINEAR ridge to two flexible non-linear probes on the same
      folds: a small MLP and gradient boosting. If they do not beat linear, position is genuinely a linear
      read-out (a direction); if they do, it is non-linear.

  (3) CURVED? Two senses, both tested.
      (a) RESPONSE curvature: is position a curved (but monotone) function of the linear projection z=<w,x>?
          Detect by Pearson R^2(linear z->pos) vs a spline of z, and Pearson(z,pos) vs Spearman(z,pos).
      (b) MANIFOLD curvature: do the genes lie on a CURVED 1-D manifold, so that a non-linear 1-D coordinate
          (Isomap) tracks position better than the best straight line? Compare rho(Isomap-1D, pos) to the
          linear rho. A 2-D PCA scatter coloured by position (figure) shows a straight gradient vs a curve.

Out: results/genome_position_direction.json  (+ figures/fig5_direction.pdf)
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from genome_position_geometry import dedup                      # embedding-cosine near-duplicate removal
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.manifold import Isomap
from scipy.stats import spearmanr, pearsonr, rankdata

ALPHAS = np.logspace(0, 5, 12)
MINCHR = 200
SEED = 0
BASIS = "maxtoki_lmhead"


def folds(n, k=5):
    return list(KFold(min(k, n), shuffle=True, random_state=SEED).split(np.arange(n)))


def oof(model_fn, X, y, fl):
    P = np.zeros(len(y))
    for tr, te in fl:
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        P[te] = m.predict(sc.transform(X[te]))
    return P


def sp(a, b):
    r = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def main():
    C = coords(); M, syms = G.basis(BASIS); pos_i = {s: i for i, s in enumerate(syms)}
    lin, mlp, gbr, iso = [], [], [], []
    pc_rho = {k: [] for k in [1, 2, 3, 5, 10, 20, 50]}
    align = []                 # |cos(w, PC_k)|, k=1..8
    resp_lin_r2, resp_spline_r2, pear, spear = [], [], [], []
    big_for_fig = []

    for c in AUTOSOMES:
        g = [s for s in C.index[C.chromosome == c] if s in pos_i]
        if len(g) < MINCHR:
            continue
        Xf = M[[pos_i[s] for s in g]]; start = C.loc[g, "start"].values.astype(float)
        keep = dedup(Xf, start)
        if keep.sum() < 120:
            continue
        X = Xf[keep]; y = start[keep]; n = len(y); fl = folds(n)

        # (2) linear vs non-linear, same folds
        lin.append(sp(oof(lambda: RidgeCV(alphas=ALPHAS), X, y, fl), y))
        mlp.append(sp(oof(lambda: MLPRegressor(hidden_layer_sizes=(64, 16), alpha=3.0, max_iter=600,
                                               early_stopping=True, random_state=SEED), X, y, fl), y))
        gbr.append(sp(oof(lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                                                learning_rate=0.05, l2_regularization=1.0,
                                                                random_state=SEED), X, y, fl), y))

        # (1) direction: fit on full, dimensionality via PCs, PC-alignment
        Xc = X - X.mean(0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        w = Ridge(alpha=100.0).fit(StandardScaler().fit_transform(X), y).coef_
        wv = w / (np.linalg.norm(w) + 1e-12)
        align.append([abs(float(wv @ (Vt[k] / (np.linalg.norm(Vt[k]) + 1e-12)))) for k in range(min(8, len(Vt)))])
        for k in pc_rho:
            if k <= Xc.shape[1]:
                Xk = Xc @ Vt[:k].T
                pc_rho[k].append(abs(sp(oof(lambda: RidgeCV(alphas=np.logspace(-2, 4, 8)), Xk, y, fl), y)))

        # (3a) response curvature: linear projection z, spline vs linear R^2, Pearson vs Spearman
        z = oof(lambda: RidgeCV(alphas=ALPHAS), X, y, fl)        # OOF linear projection ~ position
        pear.append(abs(float(pearsonr(z, y)[0]))); spear.append(abs(sp(z, y)))
        # isotonic-free spline: fit y ~ poly3(z) in-sample R^2 vs linear
        from numpy.polynomial import polynomial as PP
        zr = (z - z.mean()) / (z.std() + 1e-9)
        r2 = lambda p: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        clin = np.polyfit(zr, y, 1); resp_lin_r2.append(float(r2(np.polyval(clin, zr))))
        cpol = np.polyfit(zr, y, 3); resp_spline_r2.append(float(r2(np.polyval(cpol, zr))))

        # (3b) manifold curvature: Isomap-1D unsupervised coordinate vs position
        try:
            iy = Isomap(n_neighbors=12, n_components=1).fit_transform(StandardScaler().fit_transform(X))[:, 0]
            iso.append(abs(sp(iy, y)))
        except Exception:
            iso.append(np.nan)

        if len(g) >= 700 and len(big_for_fig) < 2:
            big_for_fig.append((c, X.copy(), y.copy()))

    res = dict(
        n_chr=len(lin),
        linear=float(np.mean(lin)), mlp=float(np.mean(mlp)), gbr=float(np.mean(gbr)),
        isomap1d=float(np.nanmean(iso)),
        dim_pcs=list(pc_rho), dim_rho=[float(np.mean(pc_rho[k])) for k in pc_rho],
        pc_alignment=[float(np.mean([a[k] for a in align if len(a) > k])) for k in range(8)],
        response_pearson=float(np.mean(pear)), response_spearman=float(np.mean(spear)),
        response_linear_r2=float(np.mean(resp_lin_r2)), response_cubic_r2=float(np.mean(resp_spline_r2)),
    )
    print(f"(2) LINEAR vs NON-LINEAR (OOF rho, mean over {res['n_chr']} chr):")
    print(f"    linear ridge {res['linear']:+.3f}   MLP {res['mlp']:+.3f}   gradient-boost {res['gbr']:+.3f}")
    print(f"    -> non-linear gain: MLP {res['mlp']-res['linear']:+.3f}, GBR {res['gbr']-res['linear']:+.3f}")
    print(f"(1) DIMENSIONALITY (OOF |rho| by #PCs): " + "  ".join(f"{k}:{v:.2f}" for k, v in zip(res['dim_pcs'], res['dim_rho'])))
    print(f"    PC-alignment |cos(w,PC_k)| k=1..8: " + " ".join(f"{a:.2f}" for a in res['pc_alignment']))
    print(f"(3a) RESPONSE curvature: Pearson {res['response_pearson']:.3f} vs Spearman {res['response_spearman']:.3f}"
          f" | linear R^2 {res['response_linear_r2']:.3f} vs cubic R^2 {res['response_cubic_r2']:.3f}")
    print(f"(3b) MANIFOLD curvature: Isomap-1D rho {res['isomap1d']:.3f} vs linear {res['linear']:.3f}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_position_direction.json"), "w"), indent=1)

    # ---- figure: 2-D PCA of a big chromosome coloured by position (straight gradient vs curve) ----
    if big_for_fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, axes = plt.subplots(1, len(big_for_fig) + 1, figsize=(3.0 * (len(big_for_fig) + 1), 3.0))
        for ax, (c, X, y) in zip(axes, big_for_fig):
            Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
            sctr = ax.scatter(P[:, 0], P[:, 1], c=rankdata(y) / len(y), cmap="viridis", s=8, linewidths=0)
            ax.set_title(f"chr{c}  (n={len(y)})", fontsize=9); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        # panel: linear vs nonlinear bars
        ax = axes[-1]
        ax.bar([0, 1, 2, 3], [res["linear"], res["mlp"], res["gbr"], res["isomap1d"]],
               color=["#2166AC", "#8A6FB0", "#D6743C", "#4A9B8E"], width=0.7, edgecolor="white")
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["linear", "MLP", "boost", "Isomap\n1-D"], fontsize=8)
        ax.set_ylabel("position ρ"); ax.set_ylim(0, max(0.5, res["mlp"] + 0.05))
        ax.set_title("linear ≈ non-linear", fontsize=9)
        cb = fig.colorbar(sctr, ax=axes[:len(big_for_fig)].tolist(), fraction=0.025, pad=0.02)
        cb.set_label("genomic position (percentile)", fontsize=8)
        fig.savefig(os.path.join(HERE, "figures", "fig5_direction.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(HERE, "figures", "fig5_direction.png"), bbox_inches="tight", dpi=200)
        print("\n[fig] figures/fig5_direction.pdf")
    print("[done] -> results/genome_position_direction.json")


if __name__ == "__main__":
    main()
