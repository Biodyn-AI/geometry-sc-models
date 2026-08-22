"""Figures for the main paper. Numbers come from results/*.json where a run produced them,
and from the verified tables in the results documents otherwise (marked VERIFIED inline)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE = f"{_DATA}/manifolds"
SYN, GAP = f"{BASE}/synthetic/results", f"{BASE}/gaps/results"
OUT = f"{BASE}/paper/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": 0.045, "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.wspace": 0.06,
})
C = {"model": "#2b6cb0", "data": "#c05621", "null": "#a0aec0",
     "hi": "#2f855a", "lo": "#c53030", "mid": "#6b46c1"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1: two regimes
def fig1():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.25),
                           gridspec_kw={"width_ratios": [1.22, 1.02, 1.0]})

    # (a) information decomposition -- VERIFIED, C2S cell cycle
    steps = ["full\nexpression", "tokenised\ninput", "the\nmodel"]
    vals = [0.929, 0.882, 0.875]
    BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88)
    ax[0].plot(range(3), vals, "o-", color=C["model"], lw=1.6, ms=5, zorder=2)
    offs = [(-1, 11, "center"), (0, -13, "center"), (0, -13, "center")]
    for i, (v, (dx, dy, ha)) in enumerate(zip(vals, offs)):
        ax[0].annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                       xytext=(dx, dy), ha=ha, va="center", fontsize=7,
                       bbox=BOX, zorder=4)
    ax[0].annotate("", xy=(1, 0.882), xytext=(0, 0.929),
                   arrowprops=dict(arrowstyle="-", color=C["lo"], lw=3, alpha=.35))
    ax[0].text(0.34, 0.893, "-0.047\ntokeniser", ha="right", va="center",
               fontsize=6.5, color=C["lo"], bbox=BOX, zorder=4)
    ax[0].text(1.5, 0.8925, "-0.007\nthe model", ha="center", va="center",
               fontsize=6.5, color=C["hi"], bbox=BOX, zorder=4)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(steps, fontsize=6.0)
    ax[0].set_xlim(-0.5, 2.5)
    ax[0].set_ylim(0.848, 0.952); ax[0].set_ylabel("phase ordering (circ-$R^2$)")
    ax[0].set_title("(a) where information is lost")

    # (b) model vs matched expression, task by task -- VERIFIED (Supplement B.2)
    tasks = ["phase ordering", "phase class.", "S vs G2M", "phase ord. (C2S)",
             "pseudotime blood", "pseudotime lung", "pseudotime gut", "pseudotime panc."]
    expr  = [0.932, 0.879, 0.992, 0.929, 0.977, 0.983, 0.921, 0.987]
    modl  = [0.924, 0.896, 0.991, 0.876, 0.951, 0.970, 0.900, 0.974]
    y = np.arange(len(tasks))[::-1]
    for yy, e, m in zip(y, expr, modl):
        ax[1].plot([m, e], [yy, yy], "-", color="#cbd5e0", lw=1.6, zorder=1)
    ax[1].scatter(modl, y, s=17, color=C["model"], zorder=3, label="best model")
    ax[1].scatter(expr, y, s=17, color=C["data"], zorder=3, label="matched expression")
    ax[1].set_yticks(y); ax[1].set_yticklabels(tasks, fontsize=6.2)
    ax[1].set_xlim(0.855, 1.005); ax[1].set_xticks([0.88, 0.92, 0.96, 1.00])
    ax[1].set_ylim(-0.9, len(tasks) - 0.3)
    ax[1].set_xlabel("score on the task's own metric", fontsize=7)
    ax[1].legend(frameon=False, fontsize=6, loc="lower center", ncol=2,
                 bbox_to_anchor=(0.5, 1.0), handletextpad=0.25, columnspacing=1.0)
    ax[1].set_title("(b) cell level: model vs expression", pad=15)

    # (c) gene level -- VERIFIED chromosome + gene-name
    names = ["MaxToki-1B\ngene table", "best training-free\nfactorisation",
             "MaxToki-217M", "raw expression\nprofile"]
    v = [0.880, 0.720, 0.506, 0.044]
    col = [C["hi"], C["data"], C["model"], C["null"]]
    ax[2].barh(range(4)[::-1], v, color=col, height=.62)
    for i, x in enumerate(v):
        ax[2].text(x + .015, 3 - i, f"{x:.3f}", va="center", fontsize=7)
    ax[2].axvline(1 / 22, ls=":", color="k", lw=.9)
    ax[2].set_ylim(-0.65, 3.95)
    ax[2].text(1 / 22 + .015, 3.72, "chance 0.045", fontsize=6, va="center", ha="left")
    ax[2].set_yticks(range(4)[::-1]); ax[2].set_yticklabels(names, fontsize=6.5)
    ax[2].set_xlim(0, 1.02); ax[2].set_xlabel("chromosome, balanced accuracy\n(10-Mb holdout)")
    ax[2].set_title("(c) gene level: chromosome")
    save(fig, "fig1_two_regimes")


# ---------------------------------------------------------------- Fig 2: anatomy
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.25))

    # (a) disk vs ring calibration -- VERIFIED
    arms = ["synthetic\nRING", "C2S-2B", "C2S-27B", "raw\nexpression", "synthetic\nDISK"]
    cv = [0.15, 0.38, 0.38, 0.42, 0.35]
    inside = [0.000, 0.108, 0.110, 0.122, 0.126]
    col = ["#718096", C["model"], C["model"], C["data"], "#718096"]
    ax[0].scatter(cv, inside, c=col, s=48, zorder=3)
    tags = [("synthetic RING", (0, 9), "center"), (None, None, None), ("C2S-2B / 27B", (0, -14), "center"),
            ("raw expression", (7, 3), "left"), ("synthetic DISK", (-7, 2), "right")]
    for (x, y, (l, off, ha)) in zip(cv, inside, tags):
        if l:
            ax[0].annotate(l, (x, y), textcoords="offset points", xytext=off,
                           ha=ha, fontsize=6)
    ax[0].axhspan(0.09, 0.14, color=C["hi"], alpha=.08)
    ax[0].set_xlim(0.09, 0.52)
    ax[0].text(0.098, 0.0955, "disk regime", fontsize=6, color=C["hi"])
    ax[0].set_xlabel("radius CV")
    ax[0].set_ylabel("fraction inside half\nthe median radius", fontsize=7)
    ax[0].set_ylim(-0.02, 0.165); ax[0].set_title("(a) a filled disk, not a ring")

    # (b) angle vs radius decomposition -- VERIFIED
    cov = ["G2M score", "S score", "mitotic\nspindle", "replication",
           "stress/IEG", "MYC growth", "ribosome", "cycling\nstrength"]
    ang = [0.683, 0.609, 0.603, 0.529, 0.036, 0.025, 0.046, 0.040]
    rad = [0.018, 0.040, 0.020, 0.056, 0.001, 0.027, 0.012, 0.414]
    y = np.arange(len(cov))
    ax[1].barh(y - .2, ang, height=.38, color=C["model"], label="angle")
    ax[1].barh(y + .2, rad, height=.38, color=C["data"], label="radius")
    ax[1].set_yticks(y); ax[1].set_yticklabels(cov, fontsize=6.3)
    ax[1].invert_yaxis(); ax[1].set_xlabel("cross-validated $R^2$")
    ax[1].set_xlim(0, 0.80)
    ax[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.42),
                 frameon=False, fontsize=6.5, handlelength=1.2, handletextpad=0.4)
    ax[1].set_title("(b) angle vs radius")

    # (c) continuity -- VERIFIED
    tests = ["behavioural\ninterpolation", "metric\nprofile", "occupancy"]
    sc = [0.17, 0.10, 0.00]
    ax[2].bar(range(3), sc, color=C["model"], width=.55)
    ax[2].axhline(1.0, ls="--", color=C["lo"], lw=1)
    ax[2].text(1, 1.02, "pure discrete attractors", ha="center", fontsize=6, color=C["lo"])
    ax[2].axhline(0.0, color="k", lw=.8)
    for i, v in enumerate(sc):
        ax[2].text(i, v + .03, f"{v:.2f}", ha="center", fontsize=7)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(tests, fontsize=6.3)
    ax[2].set_ylim(-0.05, 1.15); ax[2].set_ylabel("discreteness score")
    ax[2].set_title("(c) continuous, not discrete")
    save(fig, "fig2_anatomy")


# ---------------------------------------------------------------- Fig 3: chromosome
def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3),
                           gridspec_kw={"width_ratios": [1.25, 1.0, 0.95]})

    # (a) 10-Mb holdout retention -- VERIFIED
    bases = ["MaxToki $lm\\_head$", "MaxToki $W_E$", "Geneformer", "scGPT", "ESM-2 (sequence)"]
    rnd = [0.433, 0.433, 0.174, 0.090, 0.190]
    grp = [0.347, 0.373, 0.111, 0.056, 0.074]
    y = np.arange(len(bases))[::-1]
    ax[0].barh(y + .19, rnd, height=.36, color=C["null"], label="random split")
    ax[0].barh(y - .19, grp, height=.36, color=C["model"], label="10-Mb holdout")
    ax[0].barh(y[-1] - .19, grp[-1], height=.36, color=C["lo"])
    ax[0].axvline(1 / 22, ls=":", color="k", lw=.9)
    ax[0].set_yticks(y); ax[0].set_yticklabels(bases, fontsize=6)
    ax[0].set_xlim(0, 0.50)
    ax[0].set_xlabel("balanced accuracy", fontsize=7)
    ax[0].legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.2,
                 handletextpad=0.4, borderaxespad=0.3)
    ax[0].set_title("(a) sequence collapses")

    # (b) depth profile -- VERIFIED (217M L0-L11 and 1B matched taps k=256)
    l217 = [0.453, 0.212, 0.185, 0.173, 0.168, 0.158, 0.146, 0.139, 0.122, 0.115, 0.098, 0.088]
    ax[1].plot(range(12), l217, "o-", color=C["model"], ms=3, lw=1.4, label="217M, L0-L11")
    ax[1].plot([0, 2, 4, 8], [0.813, 0.220, 0.091, 0.066], "s--", color=C["hi"],
               ms=4, lw=1.4, label="1B, matched width")
    ax[1].axhline(0.046, ls=":", color="k", lw=.9)
    ax[1].set_xticks([0, 2, 4, 6, 8, 10])
    ax[1].text(9.2, 0.075, "null", fontsize=6)
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("chromosome accuracy", fontsize=7)
    ax[1].legend(frameon=False); ax[1].set_title("(b) in the table, gone by depth")

    # (c) causal -- VERIFIED
    ax[2].bar([0, 1], [116 / 132, 132 / 132], color=[C["model"], C["hi"]], width=.5)
    ax[2].bar([2], [0.5], color=C["null"], width=.5)
    for i, t in enumerate(["116/132", "132/132", "flat"]):
        ax[2].text(i, [116 / 132, 1.0, 0.5][i] + .03, t, ha="center", fontsize=7)
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["217M", "1B", "random\npush"], fontsize=6.5)
    ax[2].set_ylim(0, 1.15)
    ax[2].set_ylabel("fraction of (chr. $\\times$ strength\n$\\times$ seed) cells positive", fontsize=6.6)
    ax[2].set_title("(c) the model acts on it")
    save(fig, "fig3_chromosome")


# ---------------------------------------------------------------- Fig 4: surface forms
def fig4():
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    labs = ["human $CCNB1$", "mouse $Ccnb1$", "lowercase", "non-symbol $CQNB1$",
            "anagram $NBCC1$"]
    v = [0.989, 0.782, 0.767, 0.685, 0.511]
    col = [C["hi"], C["model"], C["model"], C["model"], C["lo"]]
    y = np.arange(5)[::-1]
    ax.barh(y, v, color=col, height=.62)
    for yy, x in zip(y, v):
        ax.text(x + .012, yy, f"{x:.3f}", va="center", fontsize=6.5)
    ax.axvline(0.5, ls=":", color="k", lw=.9)
    ax.text(0.5 + .012, -0.85, "chance", fontsize=6, ha="left")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.5)
    ax.set_xlim(0, 1.09); ax.set_ylim(-1.1, 4.6)
    ax.set_xlabel("AUROC, phase pole", fontsize=7)
    ax.set_title("One gene symbol, no other phase information", fontsize=8)
    save(fig, "fig4_surface_forms")


# ---------------------------------------------------------------- Fig 5: warp law
def fig5():
    JIT = [0.0, 0.11, -0.11, 0.055, -0.055, 0.0, 0.09, -0.09]
    p = f"{SYN}/s3_metric_warp_vanilla_all8seeds.json"
    rows = json.load(open(p))
    arms = ["uniform", "sharp", "occupancy", "noisy"]
    lab = ["control\n(nothing changed)", "output changes\nfastest here",
           "3x more cells\nhere", "harder to\npredict here"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    for i, a in enumerate(arms):
        v = np.array([r["model_minus_data"]["stretch"] for r in rows if r["arm"] == a])
        c = C["hi"] if a == "sharp" else (C["lo"] if a == "noisy" else C["null"])
        ax[0].bar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                  color=c, width=.6, capsize=3)
        ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.16, .16, len(v)),
                      v, s=9, color="k", alpha=.55, zorder=3)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab, fontsize=6.2)
    ax[0].set_ylabel("model $-$ data stretch")
    ax[0].set_title("(a) what makes a model warp its metric (8 seeds)")
    ax[0].set_ylim(-0.33, 0.40)
    ax[0].text(1, 0.325, "p = 0.004", ha="center", fontsize=7, color=C["hi"],
               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    # (b) localisation
    bins = {"sharp": [5, 5, 6, 6, 5], "uniform": [1, 10, 5, 4, 2]}
    for k, (nm, b) in enumerate(bins.items()):
        c = C["hi"] if nm == "sharp" else C["null"]
        # spread coincident seeds so all five are countable
        seen, xs, ys = {}, [], []
        for v in b:
            n = seen.get(v, 0); seen[v] = n + 1
            xs.append(v + (n - (b.count(v) - 1) / 2) * 0.20)
            ys.append(k + JIT[n] * 1.4)
        ax[1].scatter(xs, ys, s=30, color=c, zorder=3,
                      edgecolors="white", linewidths=0.6)
    ax[1].axvspan(4.5, 7.5, color=C["hi"], alpha=.10)
    ax[1].text(6, 1.42, "manipulated arc", ha="center", fontsize=6.5, color=C["hi"])
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["output-change\narm", "control"], fontsize=6.5)
    ax[1].set_xlabel("bin holding the model's largest gap"); ax[1].set_xlim(-0.5, 11.5)
    ax[1].set_ylim(-0.6, 1.6)
    ax[1].set_title("(b) and it warps in the right place (5/5)")
    save(fig, "fig5_warp_law")


# ---------------------------------------------------------------- Fig 6: stall theorem
def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.25))
    th = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(np.cos(th), np.sin(th), color=C["null"], lw=1.2)
    w = np.array([1.0, 0.0])
    ax[0].arrow(0, 0, .78, 0, head_width=.07, color=C["lo"], lw=1.6, length_includes_head=True)
    ax[0].text(.40, -.17, r"fixed $\mathbf{w}$", color=C["lo"], fontsize=7, ha="center")
    for a, m in [(0, "start"), (np.pi / 2, "stall")]:
        ax[0].scatter([np.cos(a)], [np.sin(a)], s=40,
                      color=C["hi"] if a == 0 else C["lo"], zorder=4)
        ax[0].annotate(m, (np.cos(a), np.sin(a)), textcoords="offset points",
                       xytext=(4, 8) if m == "stall" else (7, 2), fontsize=7,
                       ha="left")
    ax[0].annotate("", xy=(0, 1.0), xytext=(1.0, 0),
                   arrowprops=dict(arrowstyle="->", color=C["model"],
                                   connectionstyle="arc3,rad=-.35", lw=1.5))
    ax[0].text(1.12, .62, r"$\oint \mathbf{w}\cdot d\mathbf{x}=0$", fontsize=8, ha="left")
    ax[0].text(0, -1.42, "predicted stall $\\pi/2 = 1.571$\nmeasured 1.49 rad",
               ha="center", va="top", fontsize=7)
    ax[0].set_xlim(-1.35, 2.05); ax[0].set_ylim(-1.75, 1.30)
    ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[0].set_title("(a) a fixed direction cannot lap")

    reps = ["scGPT", "Geneformer", "MaxToki", "STATE-SE", "raw expr.", "C2S-2B"]
    fixed = [0.34, 0.14, 0.01, 0.31, 0.36, 0.22]
    local = [4.53, 5.57, 6.03, 4.70, 5.08, 3.45]
    x = np.arange(len(reps))
    ax[1].bar(x - .19, fixed, width=.36, color=C["lo"], label="fixed direction")
    ax[1].bar(x + .19, local, width=.36, color=C["hi"], label="local + retraction")
    for xi, f in zip(x, fixed):          # tiny bars are values, not missing data
        ax[1].text(xi - .19, f + .12, f"{f:.2f}", ha="center", fontsize=5.6, color=C["lo"])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(reps, fontsize=6.2, rotation=18, ha="right",
                          rotation_mode="anchor")
    ax[1].set_ylabel("laps completed"); ax[1].legend(frameon=False)
    ax[1].set_title("(b) including with no model at all")
    save(fig, "fig6_stall_theorem")


# ---------------------------------------------------------------- Fig 7: curvature
def fig7():
    fig, ax = plt.subplots(1, 2, figsize=(5.98, 2.15))
    scg = [0.0480, 0.0430, 0.0380, 0.0340, 0.0300, 0.0270, 0.0235, 0.0210,
           0.0195, 0.0175, 0.0155, 0.0134]
    ax[0].plot(np.linspace(0, 100, 12), scg, "o-", color=C["model"], ms=3, label="scGPT")
    ax[0].scatter([0], [0.0552], color=C["hi"], s=30, zorder=4, label="MaxToki-1B peak")
    ax[0].scatter([9], [0.0598], color=C["data"], s=30, zorder=4, label="MaxToki-217M peak")
    ax[0].set_xlim(-5, 105)
    ax[0].set_xlabel("relative depth (%)"); ax[0].set_ylabel(r"angular curvature $\Delta_Q$")
    ax[0].legend(frameon=False, fontsize=6.3)
    ax[0].set_title("(a) curvature peaks at the input")

    mods = ["scGPT", "MaxToki-217M", "MaxToki-1B"]
    ang = [0.0480, 0.0598, 0.0552]
    rad = [0.0209, 0.0252, 0.0050]
    x = np.arange(3)
    ax[1].bar(x - .19, ang, width=.36, color=C["model"], label="angular (shape)")
    ax[1].bar(x + .19, rad, width=.36, color=C["data"], label="radial (size)")
    ax[1].set_ylim(0, 0.070)
    for i, (a, r) in enumerate(zip(ang, rad)):
        ax[1].text(i, max(a, r) + .0045, f"{r/a:.2f}", ha="center", fontsize=6.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(mods, fontsize=6.3)
    ax[1].set_ylabel("curvature at peak layer")
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower center", ncol=2,
                 bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.0)
    ax[1].set_title("(b) radial/angular ratio", pad=15)
    save(fig, "fig7_curvature")


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
    f()
print("\nall figures ->", OUT)
