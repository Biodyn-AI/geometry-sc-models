"""Publication figures for PAPER_chromosome_variable, from results/*.json. Vector PDF, colourblind-safe palette.
Run: ../../.venv/bin/python -u make_figures.py   ->  figures/fig{1..4}.pdf (+ .png previews)
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
J = lambda n: json.load(open(os.path.join(R, n)))

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
})
# palette: model=blue, geneformer=teal, scGPT=grey, sequence(ESM2)=orange, data(coexpr)=green
CM, CG, CS, CSEQ, CDATA, CHANCE = "#2166AC", "#4A9B8E", "#8A8F98", "#D6743C", "#4C9F4C", "#B0B4BA"
CHANCE22 = 1/22


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("  wrote", name)


# ---------------- FIG 1: genome-wide chromosome decoding + the group-split control ----------------
def fig1():
    gs = J("genome_groupsplit.json")   # species_chrom, all 6 bases, random + 10-Mb group split
    order = [("maxtoki_lmhead", "MaxToki", CM), ("geneformer_we", "Geneformer", CG), ("scgpt_we", "scGPT", CS),
             ("esm2", "ESM2 (sequence)", CSEQ), ("coexpr_devel", "co-expression (data)", CDATA)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.1), gridspec_kw={"width_ratios": [1.3, 1], "wspace":0.32})
    # (a) 22-class balanced accuracy (random-split decoding)
    names = [o[1] for o in order]; vals = [gs[o[0]]["random"] for o in order]; cols = [o[2] for o in order]
    y = np.arange(len(names))[::-1]
    ax1.barh(y, vals, color=cols, height=0.68, edgecolor="white", linewidth=0.6)
    ax1.axvline(CHANCE22, ls="--", lw=1, color=CHANCE); ax1.text(CHANCE22, len(names)-0.3, " chance", color="#6b7079", fontsize=7.5, va="top")
    for yi, v in zip(y, vals): ax1.text(v+0.006, yi, f"{v:.2f}", va="center", fontsize=7.8)
    # group the bars: the three single-cell models (evaluated) vs the two baselines (controls)
    ax1.axhline(1.5, color="#C4C8CE", lw=0.9, ls=(0, (4, 2)), zorder=0)
    ax1.text(0.515, 3.0, "single-cell\nmodels", fontsize=6.6, color="#6b7079", va="center", ha="right", style="italic")
    ax1.text(0.515, 0.5, "baselines", fontsize=6.6, color="#6b7079", va="center", ha="right", style="italic")
    ax1.set_yticks(y); ax1.set_yticklabels(names); ax1.set_xlim(0, 0.52); ax1.set_xlabel("balanced accuracy (22 chromosomes)")
    ax1.set_title("a  Chromosome decoded", loc="left", fontweight="bold")
    # (b) random vs 10-Mb group split
    tri = [("maxtoki_lmhead", "MaxToki", CM), ("esm2", "ESM2 (sequence)", CSEQ), ("coexpr_devel", "co-expression", CDATA)]
    for key, lab, col in tri:
        r, g = gs[key]["random"], gs[key]["group"]
        ax2.plot([0, 1], [r, g], "-o", color=col, lw=2, ms=6, label=lab)
        ax2.text(1.03, g, lab, color=col, fontsize=7.6, va="center")
    ax2.axhline(CHANCE22, ls="--", lw=1, color=CHANCE); ax2.text(0.02, CHANCE22+0.006, "chance", color="#6b7079", fontsize=7.5)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["random\nsplit", "10-Mb\ngroup split"]); ax2.set_xlim(-0.15, 1.7)
    ax2.set_ylim(0, 0.55); ax2.set_ylabel("balanced accuracy")
    ax2.set_title("b  Neighbourhood held out", loc="left", fontweight="bold")
    save(fig, "fig1_decoding")


# ---------------- FIG 2: how position is encoded ----------------
def fig2():
    p2 = J("genome_position2.json"); geo = J("genome_position_geometry.json")
    fig = plt.figure(figsize=(7.4, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 0.9], wspace=0.55)
    # (a) null-corrected position excess per basis (beat-both)
    ax = fig.add_subplot(gs[0, 0])
    order = [("maxtoki_lmhead", "MaxToki", CM), ("geneformer_we", "Geneformer", CG), ("scgpt_we", "scGPT", CS),
             ("esm2", "ESM2", CSEQ), ("coexpr_devel", "co-expr", CDATA)]
    names = [o[1] for o in order]; vals = [p2[o[0]]["mean_excess"] for o in order]; cols = [o[2] for o in order]
    x = np.arange(len(names))
    ax.bar(x, vals, color=cols, width=0.7, edgecolor="white", linewidth=0.6)
    for xi, v in zip(x, vals): ax.text(xi, v+0.008, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.6)
    ax.set_ylabel("position signal (null-corrected)"); ax.set_ylim(0, 0.46)
    ax.set_title("a  Position per model", loc="left", fontweight="bold")
    # (b) predicted vs true position (percentile), MaxToki, leakage-clean
    ax = fig.add_subplot(gs[0, 1])
    t = np.array(geo["maxtoki_lmhead"]["scatter_true"]); p = np.array(geo["maxtoki_lmhead"]["scatter_pred"])
    hb = ax.hexbin(t, p, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#6b7079")
    rho = geo["maxtoki_lmhead"]["mean_rho"]
    ax.text(0.04, 0.93, f"ρ = {rho:+.2f}\n(n={len(t):,} genes)", fontsize=7.8, va="top")
    ax.set_xlabel("true position (percentile)"); ax.set_ylabel("predicted position")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("b  Predicted vs. true", loc="left", fontweight="bold")
    # (c) within-chromosome vs shared-axis transfer
    ax = fig.add_subplot(gs[0, 2])
    within = geo["maxtoki_lmhead"]["mean_rho"]; shared = geo["shared_axis"]["mean_rho"]
    ax.bar([0, 1], [within, shared], color=[CM, "#B9BEC6"], width=0.62, edgecolor="white")
    for xi, v in zip([0, 1], [within, shared]): ax.text(xi, v+0.012, f"{v:+.2f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["within-\nchromosome", "cross-\nchromosome"], fontsize=7.6)
    ax.set_ylabel("position ρ"); ax.set_ylim(0, 0.5)
    ax.set_title("c  Chromosome-specific", loc="left", fontweight="bold")
    save(fig, "fig2_position")


# ---------------- FIG 3: causal steering ----------------
def fig3():
    sw = J("genome_causal_sweep.json"); gc = J("genome_causal.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"width_ratios": [1, 1.2], "wspace":0.28})
    # (a) dose-response: specific effect vs alpha, 3 seeds
    alphas = sw["alphas"]; seeds = sw["seeds"]
    seedcol = ["#2166AC", "#4A9B8E", "#8A6FB0"]
    for si, s in enumerate(seeds):
        ys = [sw["cells"][f"seed{s}_alpha{a}"]["specific_mean"] for a in alphas]
        los = [sw["cells"][f"seed{s}_alpha{a}"]["ci"][0] for a in alphas]
        his = [sw["cells"][f"seed{s}_alpha{a}"]["ci"][1] for a in alphas]
        ax1.plot(alphas, ys, "-o", color=seedcol[si], lw=1.6, ms=4, label=f"seed {s}")
        ax1.fill_between(alphas, los, his, color=seedcol[si], alpha=0.12, linewidth=0)
    ax1.axhline(0, ls="--", lw=1, color=CHANCE)
    ax1.text(alphas[-1], 0.004, "random push ≈ 0", ha="right", fontsize=7.2, color="#6b7079")
    ax1.set_xscale("log", base=2); ax1.set_xticks(alphas); ax1.set_xticklabels([str(a) for a in alphas])
    ax1.set_xlabel("steering strength  α  (× token norm)"); ax1.set_ylabel("chr-C mass gain at unsteered genes")
    ax1.legend(fontsize=7.5, loc="upper left"); ax1.set_ylim(-0.005, 0.09)
    ax1.set_title("a  Dose–response (217M)", loc="left", fontweight="bold")
    # (b) per-chromosome specific effect, MaxToki-1B (averaged over the two split-half seeds at alpha=1)
    acc = {}
    for sd in (0, 1):
        d = J(f"steer_propagation_chromosome_1b_seed{sd}.json")
        row = next(s for s in d["sweep"] if abs(s["alpha"] - 1.0) < 1e-9)
        for e in row["per_cat"]:
            acc.setdefault(e["cat"], []).append(e["specific"])
    items = sorted(((c, float(np.mean(v))) for c, v in acc.items()), key=lambda kv: kv[1])
    labs = [f"chr{c}" for c, _ in items]; vals = [v for _, v in items]
    cols = [CM if v > 0 else CSEQ for v in vals]
    yy = np.arange(len(items))
    ax2.barh(yy, vals, color=cols, height=0.72, edgecolor="white", linewidth=0.4)
    ax2.axvline(0, lw=0.8, color="#4b5059")
    ax2.set_yticks(yy); ax2.set_yticklabels(labs, fontsize=6.6)
    ax2.set_xlabel("chr-C mass gain (steer − random)")
    ax2.set_title("b  Per chromosome (1B): all 22 positive", loc="left", fontweight="bold")
    save(fig, "fig3_causal")


# ---------------- FIG 4: two-tissue rotation (HOXB is used in both) ----------------
def fig4():
    fg = J("hox_causal_locus.json")["per_cluster_beta"]; bm = J("hox_causal_locus_agingmds.json")["per_cluster_beta"]
    clusters = ["A", "B", "C", "D"]
    def getb(d, c): return d[c]["beta"] if d.get(c) and d[c]["beta"] is not None else np.nan
    fgv = [getb(fg, c) for c in clusters]; bmv = [getb(bm, c) for c in clusters]
    x = np.arange(len(clusters)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    b1 = ax.bar(x - w/2, fgv, w, color="#4C9F4C", label="fetal gut (HOXB-dominant)", edgecolor="white")
    b2 = ax.bar(x + w/2, bmv, w, color="#8A6FB0", label="bone marrow (HOXA-dominant)", edgecolor="white")
    ax.axhline(0, lw=0.8, color="#4b5059")
    for bars, vals in [(b1, fgv), (b2, bmv)]:
        for bar, v in zip(bars, vals):
            if not np.isnan(v): ax.text(bar.get_x()+bar.get_width()/2, v+(0.008 if v >= 0 else -0.02), f"{v:+.2f}", ha="center", fontsize=7.4)
    ax.set_xticks(x); ax.set_xticklabels([f"HOX{c}" for c in clusters])
    ax.set_ylabel("causal use  β  (per cluster)"); ax.set_ylim(-0.08, 0.32)
    ax.legend(fontsize=7.8, loc="upper center")
    ax.set_title("Rotating the tissue does not move the used cluster", loc="left", fontweight="bold", fontsize=9.5)
    save(fig, "fig4_rotation")


if __name__ == "__main__":
    print("generating figures ->", FIG)
    fig1(); fig2(); fig3(); fig4()
    print("done.")
