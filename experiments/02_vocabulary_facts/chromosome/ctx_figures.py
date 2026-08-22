"""Publication figures for the contextual-gene-representation paper. Reads results/ctx_*.json (+ the cross-model
and coexpr-null recomputation). Matplotlib, theme-neutral, saved to figures/ctx_fig{1..6}.pdf.

F1 contextualisation by layer (EXCESS, same vs diff-gene null, L0=0 sanity)
F2 rank control (stable vs moving vs residualised)
F3 co-expression null — the crux: functional axes placed on the co-expression-coherence power curve
F4 causal steering dose-response (signed swing vs random, by alpha)
F5 cross-model + scaling + random-weights control (EXCESS and FUNC-Z bars)
F6 Level-2 negatives (effect vs matched-null, the ceiling)
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#95a5a6", "#27ae60"
def L(f): return json.load(open(os.path.join(RES, f)))


def fig1():
    d = L("ctx_polysemy.json")["taps"]
    layers = sorted(int(k[1:]) for k in d)
    ex = [d[f"L{l:02d}"]["excess"] for l in layers]
    diff = [d[f"L{l:02d}"]["diff"] for l in layers]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(layers, ex, "-o", color=BLUE, label="same gene, independent cells")
    ax.plot(layers, diff, "-s", color=GREY, label="different gene (null)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("layer"); ax.set_ylabel("context-shift agreement (cosine)")
    ax.set_title("Gene-specific context response by layer", fontsize=9)
    ax.annotate("L0 = embedding\n(context-free) = 0", (0, 0), (1.5, 0.18), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(FIG, "ctx_fig1.pdf")); plt.close(fig); print("F1")


def fig2():
    d = L("ctx_position_confound.json")["taps"]
    layers = sorted(int(k[1:]) for k in d)
    st = [d[f"L{l:02d}"]["excess_rank_stable"] for l in layers]
    mv = [d[f"L{l:02d}"]["excess_rank_moving"] for l in layers]
    rs = [d[f"L{l:02d}"]["excess_residualised"] for l in layers]
    x = np.arange(len(layers)); w = 0.26
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.bar(x - w, st, w, color=BLUE, label="rank-stable genes")
    ax.bar(x, mv, w, color=RED, label="rank-moving genes")
    ax.bar(x + w, rs, w, color=GREEN, label="rank-residualised")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("gene-specific context response"); ax.set_ylim(0, 0.85)
    ax.set_title("The effect is not the gene's rank position", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(FIG, "ctx_fig2.pdf")); plt.close(fig); print("F2")


def fig4():
    d = L("ctx_causal.json")
    al = d["alphas_xResidNorm"]
    fs = [d["signed"][f"alpha_{a}"]["func_swing"] for a in al]
    se = [d["signed"][f"alpha_{a}"]["func_swing_sem"] for a in al]
    rs = [d["signed"][f"alpha_{a}"]["rand_swing"] for a in al]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(al, fs, yerr=se, fmt="-o", color=BLUE, label="functional push (signed swing)", capsize=3)
    ax.plot(al, rs, "-s", color=GREY, label="norm-matched random push")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("steering strength (× residual norm)")
    ax.set_ylabel("logit swing at untouched genes\n(+u minus −u, specific)")
    ax.set_title("The functional-context direction is causally used", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(FIG, "ctx_fig4.pdf")); plt.close(fig); print("F4")


def fig6():
    # Level-2 negatives: functional-axis modulation IS above random (Level 1) but NOT above co-expression (ceiling)
    cn = L("ctx_coexpr_null.json")["taps"]["L04"]["axes"]
    fa = L("ctx_functional_axes.json")["taps"]["L04"]
    axes = ["nuclear_vs_surface", "mito_vs_cytoskeleton", "transcription_vs_transport"]
    labels = ["nuclear/\nsurface", "mito/\ncytoskel", "transcr/\ntransport"]
    z_rand = [fa[a]["z_rank_controlled"] for a in axes]
    z_coexpr = [cn[a]["z_above_coexpr"] for a in axes]
    x = np.arange(len(axes)); w = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.bar(x - w / 2, z_rand, w, color=BLUE, label="vs random-axis null (Level 1)")
    ax.bar(x + w / 2, z_coexpr, w, color=RED, label="vs co-expression null (the test)")
    ax.axhline(3, color="k", ls="--", lw=0.7)
    ax.set_ylim(0, max(z_rand) * 1.32)                        # headroom so the legend clears the tallest bar
    ax.text(len(axes) - 0.5, 3.4, "z = 3 (significance)", fontsize=6, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("modulation z-score")
    ax.set_title("Functional organisation is co-expression", fontsize=9, pad=8)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.62, 1.0))
    fig.savefig(os.path.join(FIG, "ctx_fig6.pdf")); plt.close(fig); print("F6")


def fig5():
    d = L("ctx_cross_model.json")
    order = ["scGPT", "STATE-SE", "MaxToki-217M", "MaxToki-1B"]
    rand = d.get("MaxToki-217M-random")   # populated by ctx_cross_model once ctxrand exists
    rows = [(m, d[m]) for m in order if m in d and "error" not in d[m]]
    if rand and "error" not in rand:
        rows.append(("MaxToki-217M\n(random weights)", rand))
    names = [r[0] for r in rows]
    ex = [r[1]["excess"] for r in rows]
    fz = [r[1]["func_z"]["nuclear_vs_surface"]["z"] if r[1]["func_z"].get("nuclear_vs_surface") else 0 for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    cols = [BLUE if "random" not in n else GREY for n in names]
    a1.barh(names[::-1], ex[::-1], color=cols[::-1]); a1.set_xlabel("EXCESS (contextualisation)")
    a1.set_title("Contextualisation strength", fontsize=9)
    a1.tick_params(labelsize=8)
    a2.barh(range(len(names)), fz[::-1], color=cols[::-1]); a2.set_xlabel("functional-z")
    a2.set_yticks(range(len(names))); a2.set_yticklabels([""] * len(names))   # labels only on the left panel
    a2.axvline(3, color="k", ls="--", lw=0.7); a2.text(3, len(names) - 0.4, " z=3", fontsize=6, va="top")
    a2.set_title("Functional organisation", fontsize=9)
    fig.suptitle("Architecture, scale, and the learned vs. architectural split", fontsize=10, y=1.06)
    fig.subplots_adjust(top=0.84, wspace=0.12)
    fig.savefig(os.path.join(FIG, "ctx_fig5.pdf")); plt.close(fig); print("F5")


def fig3():
    """The crux figure (from ctx_coexpr_null_v2.py): per axis, the functional axis's context-modulation power
    (red dot) against the SIZE-MATCHED random-axis and co-expression-module null distributions, with the
    empirical p vs co-expression annotated. The functional dot sits at the top edge of the co-expression null,
    i.e. does not robustly exceed it."""
    d = L("ctx_coexpr_null_v2.json")["axes"]
    nice = {"nuclear_vs_surface": "nuclear /\nsurface", "mito_vs_cytoskeleton": "mito /\ncytoskel",
            "transcription_vs_transport": "transcr /\ntransport"}
    order = ["nuclear_vs_surface", "mito_vs_cytoskeleton", "transcription_vs_transport"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, name in enumerate(order):
        r = d[name]
        xr, xc = i - 0.18, i + 0.18
        rand = np.array(r["null_random"]); coex = np.array(r["null_indep_coexpr"])
        bp = ax.boxplot([rand, coex], positions=[xr, xc], widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k"))
        for patch, c in zip(bp["boxes"], [GREY, "#7fb0d8"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.scatter([i], [r["power"]], s=90, color=RED, zorder=6, edgecolor="k", lw=0.6)
        ax.annotate(f"p={r['indep_coexpr_p']:.2f}", (i, r["power"]), fontsize=7, xytext=(6, 0),
                    textcoords="offset points", va="center")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([nice[n] for n in order], fontsize=7.5)
    ax.set_ylabel("context-modulation power (rank-controlled)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(facecolor=GREY, alpha=0.7, label="random-axis null"),
                       Patch(facecolor="#7fb0d8", alpha=0.7, label="co-expression-module null (size-matched)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=8,
                              label="functional axis")],
              fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title("Functional organisation does not robustly exceed co-expression\n"
                 "(functional axis at the top edge of the size-matched co-expression null; p shown)",
                 fontsize=8.5)
    fig.savefig(os.path.join(FIG, "ctx_fig3.pdf")); plt.close(fig); print("F3")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}
    for k in which:
        try: fns[k]()
        except Exception as e: print(f"F{k} ERR {repr(e)[:120]}")
    print("[done] -> figures/")
