"""Graphical abstract for PAPER_chromosome_variable — a standalone composite figure that tells the story
(setup -> encoded -> causally used) and embeds the two headline graphs from results/*.json.
Run: ../../.venv/bin/python -u make_graphical_abstract.py  ->  figures/graphical_abstract.{pdf,png}
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
J = lambda n: json.load(open(os.path.join(R, n)))

CM, CBASE, CSEQ, CHANCE = "#2166AC", "#4C9F4C", "#D6743C", "#AEB4BB"
INK, MUT, CBG, EDGE, HILITE = "#1e2227", "#6b7079", "#F5F7F9", "#E1E5EA", "#EAF2FB"
plt.rcParams.update({"font.family": "DejaVu Serif", "axes.spines.top": False, "axes.spines.right": False})

# ---------------- data ----------------
# (1) encoding headline: matched genes, best probe, 10-Mb neighbourhood-holdout (= the abstract's numbers)
CHANCE22 = 1 / 22
enc = [("MaxToki-1B", 0.880, CM), ("co-occurrence\nbaseline", 0.720, CBASE), ("chance", CHANCE22, CHANCE)]
# (2) causal: MaxToki-1B per-chromosome specific effect, averaged over the two split-half seeds at alpha=1
acc = {}
for sd in (0, 1):
    d = J(f"steer_propagation_chromosome_1b_seed{sd}.json")
    row = next(s for s in d["sweep"] if abs(s["alpha"] - 1.0) < 1e-9)
    for e in row["per_cat"]:
        acc.setdefault(e["cat"], []).append(e["specific"])
perc = sorted(float(np.mean(v)) for v in acc.values())
mean_specific = float(np.mean(perc))

# ---------------- canvas ----------------
fig = plt.figure(figsize=(12.2, 6.7))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
bg.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec="none", zorder=-10))


def card(x, y, w, h, fc=CBG):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.014",
                                fc=fc, ec=EDGE, lw=1.1, zorder=0))


def badge(x, y, n, col):
    bg.add_patch(Circle((x, y), 0.016, fc=col, ec="none", zorder=3))
    bg.text(x, y - 0.0015, str(n), ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=4)


def arrow(x0, y0, x1, y1, col=MUT, lw=2.6, ms=9):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=col, shrinkA=0, shrinkB=0, zorder=5))


# ---------------- title ----------------
bg.text(0.5, 0.958, "Chromosome identity: an emergent, causally-used variable in a single-cell foundation model",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)
bg.text(0.5, 0.912, "Trained only to predict gene expression, the model encodes — and causally uses — which "
        "chromosome each gene is on", ha="center", va="center", fontsize=10.8, style="italic", color=MUT)

# ---------------- three story cards ----------------
Ax, Aw = 0.014, 0.284
Bx, Bw = 0.322, 0.300
Cx, Cw = 0.642, 0.344
top, bot = 0.858, 0.238
for x, w in [(Ax, Aw), (Bx, Bw), (Cx, Cw)]:
    card(x, bot, w, top - bot)
arrow(Ax + Aw + 0.004, 0.55, Bx - 0.004, 0.55)
arrow(Bx + Bw + 0.004, 0.55, Cx - 0.004, 0.55)

# ============ CARD A — the setup ============
acx = Ax + Aw / 2
badge(Ax + 0.028, 0.822, 1, MUT)
bg.text(Ax + 0.052, 0.822, "The setup", ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
# gene tokens
gcols = [CM, CSEQ, CBASE, "#8A6FB0", CM, "#4A9B8E"]
gx0, gw, gy = Ax + 0.045, 0.028, 0.712
for i, c in enumerate(gcols):
    bg.add_patch(FancyBboxPatch((gx0 + i * (gw + 0.006), gy), gw, 0.036,
                 boxstyle="round,pad=0.001,rounding_size=0.004", fc=c, ec="white", lw=0.8, alpha=0.9))
bg.text(acx, 0.762, "one cell = a sequence of its genes (tokens)", ha="center", va="center", fontsize=8.6, color=MUT)
arrow(acx, 0.700, acx, 0.662, col="#9aa0a8", lw=1.8, ms=8)
bg.add_patch(FancyBboxPatch((Ax + 0.052, 0.598), Aw - 0.104, 0.052,
             boxstyle="round,pad=0.004,rounding_size=0.010", fc="white", ec=CM, lw=1.4))
bg.text(acx, 0.624, "Transformer  (MaxToki)", ha="center", va="center", fontsize=10, fontweight="bold", color=CM)
arrow(acx, 0.594, acx, 0.556, col="#9aa0a8", lw=1.8, ms=8)
bg.text(acx, 0.536, "predict the next gene", ha="center", va="center", fontsize=9.6, color=INK)
bg.text(acx, 0.470, "Never shown DNA, chromosomes,\nor where genes lie in the genome.",
        ha="center", va="center", fontsize=9.2, style="italic", color=MUT)
bg.add_patch(FancyBboxPatch((Ax + 0.028, 0.300), Aw - 0.056, 0.072,
             boxstyle="round,pad=0.006,rounding_size=0.012", fc=HILITE, ec=CM, lw=1.3))
bg.text(acx, 0.336, "Does it know which chromosome\neach gene is on?", ha="center", va="center",
        fontsize=10.6, fontweight="bold", color=INK)

# ============ CARD B — it's encoded ============
bcx = Bx + Bw / 2
badge(Bx + 0.028, 0.822, 2, CM)
bg.text(Bx + 0.052, 0.822, "It is encoded", ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
bg.text(bcx, 0.778, "Yes — genome-wide, from expression alone", ha="center", va="center",
        fontsize=9.6, color=INK)
axB = fig.add_axes([Bx + 0.045, 0.470, Bw - 0.085, 0.230])
axB.barh([1, 0], [0.880, 0.720], color=[CM, CBASE], height=0.52, edgecolor="white")
axB.axvline(CHANCE22, ls="--", lw=1.2, color=CHANCE)
axB.text(CHANCE22 + 0.015, 1.74, "chance", fontsize=7.6, color=MUT, va="center")
# model names placed ABOVE each bar, inside the axes, so no label spills past the card
axB.text(0.015, 1.42, "MaxToki-1B", fontsize=9.2, color=CM, fontweight="bold", va="center")
axB.text(0.015, 0.42, "co-occurrence baseline", fontsize=9.2, color="#2E7D32", fontweight="bold", va="center")
axB.text(0.880 - 0.02, 1, "0.88", va="center", ha="right", fontsize=10.5, color="white", fontweight="bold")
axB.text(0.720 - 0.02, 0, "0.72", va="center", ha="right", fontsize=10.5, color="white", fontweight="bold")
axB.set_yticks([]); axB.set_xlim(0, 1.0); axB.set_ylim(-0.55, 2.0)
axB.set_xlabel("balanced accuracy  (chance = 1/22)", fontsize=8.2); axB.tick_params(labelsize=7.6)
bg.text(bcx, 0.352, "The strongest gene→chromosome classifier from\nexpression we know of — beating a genuine\n"
        "co-occurrence baseline under a neighbourhood-holdout.", ha="center", va="center", fontsize=8.5, color=MUT)

# ============ CARD C — it's causally used ============
ccx = Cx + Cw / 2
badge(Cx + 0.028, 0.822, 3, CM)
bg.text(Cx + 0.052, 0.822, "It is causally used", ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
# mini steering schematic (compact, contained within the card)
sy = 0.745
sx0 = Cx + 0.052
for i in range(7):
    pushed = i < 3
    fc = CM if pushed else "#DCE6F2"
    bg.add_patch(FancyBboxPatch((sx0 + i * 0.030, sy), 0.024, 0.032,
                 boxstyle="round,pad=0.001,rounding_size=0.003", fc=fc, ec="white", lw=0.7))
    if pushed:
        bg.text(sx0 + i * 0.030 + 0.012, sy + 0.016, "+", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
bg.text(sx0 + 0.037, sy + 0.048, "push → chr-C", ha="center", va="center", fontsize=8.0, color=CM)
bg.text(sx0 + 5 * 0.030 + 0.012, sy + 0.048, "read here", ha="center", va="center", fontsize=8.0, color=INK)
bg.text(ccx, sy - 0.028, "→ more chr-C genes at the untouched half,\ncarried across genes by attention",
        ha="center", va="center", fontsize=8.0, style="italic", color=MUT)
# per-chromosome specific effect: all 22 positive
axC = fig.add_axes([Cx + 0.055, 0.470, Cw - 0.105, 0.215])
yy2 = np.arange(len(perc))
axC.barh(yy2, perc, color=CM, height=0.85, edgecolor="white", linewidth=0.3)
axC.axvline(0, lw=1.0, color="#4b5059")
axC.set_yticks([]); axC.set_ylim(-0.6, len(perc) - 0.4); axC.set_xlim(0, 0.42)
axC.set_xlabel("chr-C mass gain,  steer − random", fontsize=8.0); axC.tick_params(labelsize=7.4)
axC.text(0.97, 0.40, "all 22\nchromosomes\npositive", transform=axC.transAxes, ha="right", va="center",
         fontsize=8.4, color=CM, fontweight="bold")
bg.text(ccx, 0.352, "Steer a cell toward a chromosome and it expects\nmore of that chromosome's genes elsewhere —\n"
        "for all 22 chromosomes (mean +0.17); random ≈ 0.", ha="center", va="center", fontsize=8.5, color=MUT)

# ---------------- bottom takeaway ----------------
by, bh = 0.028, 0.176
card(0.014, by, 0.638, bh, fc=HILITE)
card(0.664, by, 0.322, bh, fc="#FBF3EC")
bg.text(0.028, by + bh - 0.028, "Why it matters", ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
bg.text(0.028, by + 0.058,
        "Decodability is not use — we show both. This is a new kind of latent variable: absent from any single\n"
        "input and orthogonal to the training objective, unlike the input-determined world variables of prior work\n"
        "(a game board, or a language model's “space and time”). It is assembled purely from cross-cell co-occurrence.",
        ha="left", va="center", fontsize=9.3, color=INK)
bg.text(0.676, by + bh - 0.028, "External check", ha="left", va="center", fontsize=11, fontweight="bold", color=CSEQ)
bg.text(0.825, by + 0.066, "r = +0.43", ha="center", va="center", fontsize=20, fontweight="bold", color=CSEQ)
bg.text(0.825, by + 0.028, "the same directions reproduce a real\nleukaemia karyotype (aneuploid K562)",
        ha="center", va="center", fontsize=8.4, color=MUT)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"graphical_abstract.{ext}"), dpi=200, bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figures/graphical_abstract.pdf  and  .png")
