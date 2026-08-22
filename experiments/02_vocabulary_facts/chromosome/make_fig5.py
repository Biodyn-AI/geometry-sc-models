"""Clean 3-panel Fig 5 (position is a linear direction, orthogonal to top variance) from saved results."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from genome_position_geometry import dedup
from scipy.stats import rankdata

r = json.load(open(os.path.join(HERE, "results", "genome_position_direction.json")))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})
CM = "#2166AC"

# recompute chr1 dedup'd PCA for the scatter (cheap, one chromosome)
C = coords(); M, syms = G.basis("maxtoki_lmhead"); pos_i = {s: i for i, s in enumerate(syms)}
g = [s for s in C.index[C.chromosome == "1"] if s in pos_i]
Xf = M[[pos_i[s] for s in g]]; start = C.loc[g, "start"].values.astype(float)
keep = dedup(Xf, start); X = Xf[keep] - Xf[keep].mean(0); y = start[keep]
U, s, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:2].T

fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.6, 2.7), gridspec_kw={"wspace": 0.72})
# (a) position is not in the top-variance plane
sc = a.scatter(P[:, 0], P[:, 1], c=rankdata(y) / len(y), cmap="viridis", s=7, linewidths=0)
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_title("a  Top-variance plane (chr1)", loc="left", fontweight="bold")
# dedicated colorbar axis hugging panel a's right edge, short label, so it never reaches panel b's y-axis
cax = inset_axes(a, width="4.5%", height="100%", loc="lower left",
                 bbox_to_anchor=(1.03, 0.0, 1, 1), bbox_transform=a.transAxes, borderpad=0)
cb = fig.colorbar(sc, cax=cax); cb.set_label("percentile", fontsize=7.5, labelpad=2)
cb.ax.tick_params(labelsize=7)
# (b) dimensionality: position needs many low-variance directions
a2 = b
a2.plot(r["dim_pcs"], r["dim_rho"], "-o", color=CM, lw=1.8, ms=4)
a2.axhline(r["linear"], ls="--", lw=1, color="#6b7079"); a2.text(r["dim_pcs"][-1], r["linear"]+0.01, "full space", ha="right", fontsize=7.2, color="#6b7079")
a2.set_xscale("log"); a2.set_xlabel("# top principal components used"); a2.set_ylabel("position ρ")
a2.set_ylim(0, 0.46); a2.set_title("b  Spread over many dims", loc="left", fontweight="bold")
# (c) linear beats non-linear
labs, vals, cols = ["linear", "gradient\nboost", "Isomap\n1-D"], [r["linear"], r["gbr"], r["isomap1d"]], [CM, "#D6743C", "#4A9B8E"]
c.bar(range(3), vals, color=cols, width=0.68, edgecolor="white")
for xi, v in zip(range(3), vals): c.text(xi, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
c.set_xticks(range(3)); c.set_xticklabels(labs, fontsize=8); c.set_ylabel("position ρ"); c.set_ylim(0, 0.48)
c.set_title("c  Linear is best", loc="left", fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"fig5_direction.{ext}"), bbox_inches="tight", dpi=200)
print("wrote figures/fig5_direction.pdf")
