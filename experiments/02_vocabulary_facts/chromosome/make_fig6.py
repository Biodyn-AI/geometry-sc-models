"""Fig 6, redrawn to separate the two OBJECT TYPES clearly:
   - contextual hidden states (the curve, layers 0..N)  -- these decay monotonically
   - static weight tables (embed_tokens, lm_head)       -- reference lines; NOT layers, not points on the curve
Reads results/maxtoki_layers.json (no model rerun needed).
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results", "maxtoki_layers.json")))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
CURVE, TIN, TOUT, CHANCE = "#2166AC", "#4A9B8E", "#D6743C", "#B0B4BA"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"wspace": 0.3})
xs = list(range(r["n_hidden"]))

def panel(ax, ys, t_in, t_out, ylab, title, chance=None):
    # static tables first, as clearly-labelled reference bands
    ax.axhline(t_out, ls=(0, (4, 2)), lw=1.4, color=TOUT)
    ax.axhline(t_in, ls=(0, (4, 2)), lw=1.4, color=TIN)
    ax.text(len(xs) - 1, t_out + 0.012, "static OUTPUT table (lm_head)", ha="right", fontsize=7.3, color=TOUT)
    ax.text(len(xs) - 1, t_in + 0.012, "static INPUT table (embed_tokens)", ha="right", fontsize=7.3, color=TIN)
    # the contextual trajectory
    ax.plot(xs, ys, "-o", color=CURVE, lw=1.9, ms=4.5, label="contextual hidden states", zorder=3)
    if chance is not None:
        ax.axhline(chance, ls=":", lw=1, color=CHANCE)
        ax.text(0, chance + 0.008, "chance", fontsize=7.2, color="#6b7079")
    ax.set_xlabel("transformer layer  (0 = input embedding)"); ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold")

panel(a1, r["layer_bal"], r["static_embed"]["bal"], r["static_lmhead"]["bal"],
      "chromosome balanced acc", "a  Chromosome", chance=r["chance_bal"])
a1.set_ylim(0, 0.60)
panel(a2, r["layer_pos"], r["static_embed"]["pos"], r["static_lmhead"]["pos"],
      "within-chromosome position ρ", "b  Position")
a2.set_ylim(-0.06, 0.36)
a1.legend(fontsize=7.6, loc="center right")
fig.text(0.5, -0.06, "Dashed lines are STATIC weight tables (one vector per gene, context-free) — they are not "
                     "layers and not points on the curve.", ha="center", fontsize=7.6, style="italic")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"fig6_layers.{ext}"), bbox_inches="tight", dpi=200)
print("wrote figures/fig6_layers.pdf")
