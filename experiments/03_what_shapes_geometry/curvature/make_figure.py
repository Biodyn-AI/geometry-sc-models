"""Route Q — summary figure.

Panels:
  A  rank sweep: out-of-fold delta_Q vs rank of a directly-fit Q (is it low-rank?)
  B  nulls: real delta_Q vs the label-shuffle and linear-synthetic null distributions
  C  the saddle: delta_Q for full Q, positive arm only, negative arm only, rank-1
  D  redundancy discriminator: R^2_x vs R^2_SAE(h_bar) per latent, with prereg thresholds

Usage: ../../.venv/bin/python make_figure.py
Out:   figures/route_q_summary.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# Okabe-Ito (project convention)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#999999")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main(model="scgpt", dataset="setty"):
    prim = load(f"q_{model}_{dataset}.json")
    nul = load(f"nulls_{model}_{dataset}.json")
    arm = load(f"arms_{model}_{dataset}.json")
    disc = load(f"discriminator_{model}_{dataset}.json")

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    # --- A: rank sweep ---
    a = ax[0]
    if prim and prim.get("rank_sweep"):
        rs = prim["rank_sweep"]
        rr = sorted(int(k) for k in rs)
        vv = [rs[str(r)] for r in rr]
        a.plot(rr, vv, "o-", color=OI["blue"], lw=2)
        a.axhline(0, color="k", lw=0.8)
        a.set_xscale("log", base=2); a.set_xticks(rr); a.set_xticklabels(rr)
        a.set_xlabel("rank r of directly-fit Q"); a.set_ylabel(r"held-out $\Delta_Q$")
        a.set_title("A. The nonlinearity is low-rank\n(r=1 recovers ~40%; saturates by r=8)", fontsize=9)

    # --- B: nulls ---
    b = ax[1]
    if nul:
        real = nul["delta_q_real"]
        sh = np.array(nul["shuffle"]["values"])
        sy = np.array(nul["linear_synthetic"]["values"])
        parts = b.violinplot([sh, sy], positions=[0, 1], showmeans=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(OI["grey"]); pc.set_alpha(0.6)
        b.axhline(real, color=OI["red"], lw=2, label=f"real $\\Delta_Q$={real:+.4f}")
        b.set_xticks([0, 1]); b.set_xticklabels(["label\nshuffle", "linear-\nsynthetic"])
        b.set_ylabel(r"$\Delta_Q$"); b.legend(fontsize=8)
        b.set_title("B. Both pre-registered nulls\n(the decisive controls)", fontsize=9)

    # --- C: the arm split is NOT identified (lambda control) ---
    c = ax[2]
    ident = load("arms_identifiability.json")
    if ident:
        labs, full, pos, neg = [], [], [], []
        for r in ident:
            labs.append(f"{r['model'][:4]}\nr={r['r']}\n$\\lambda$={r['lam']:g}")
            full.append(r["full"]); pos.append(r["pos"]); neg.append(r["neg"])
        x = np.arange(len(labs)); w = 0.27
        c.bar(x - w, full, w, color=OI["green"], label="full Q")
        c.bar(x, pos, w, color=OI["orange"], label="+ arm only")
        c.bar(x + w, neg, w, color=OI["purple"], label="− arm only")
        c.axhline(0, color="k", lw=0.8)
        c.set_xticks(x); c.set_xticklabels(labs, fontsize=6.5)
        c.set_ylabel(r"held-out $\Delta_Q$"); c.legend(fontsize=6.5)
        c.set_title("C. The arm split is NOT identified:\narms flip sign with $\\lambda$, not with model", fontsize=9)
    elif arm:
        d = arm["delta_q"]
        keys = ["full", "pos_only", "neg_only", "rank1"]
        lab = ["full Q", "+ arm", "− arm", "rank-1"]
        c.bar(range(4), [d[k] for k in keys], color=[OI["green"], OI["red"], OI["red"], OI["blue"]])
        c.axhline(0, color="k", lw=0.8)
        c.set_xticks(range(4)); c.set_xticklabels(lab, fontsize=8)
        c.set_ylabel(r"held-out $\Delta_Q$")
        c.set_title("C. Arm split (single fit)", fontsize=9)

    # --- D: discriminator ---
    dax = ax[3]
    if disc:
        for r in disc["latents"]:
            nm = r["latent"]
            is_q = nm.startswith("q_saddle")
            is_r1 = nm.startswith("rank1")
            col = OI["green"] if is_q else (OI["blue"] if is_r1 else OI["grey"])
            dax.scatter(r["r2_x"], r["r2_sae_hbar"], s=90 if (is_q or is_r1) else 40,
                        color=col, zorder=3,
                        label=("saddle q(x)" if is_q else ("rank-1 latent" if is_r1 else None)))
        dax.axvline(0.25, ls="--", color="k", lw=1)
        dax.axhline(0.50, ls="--", color="k", lw=1)
        dax.set_xlabel(r"$R^2_x$  (linear in raw $x$)")
        dax.set_ylabel(r"$R^2_{SAE}$  (linear in atlas $\bar{h}$)")
        dax.set_xlim(-0.05, 1.05); dax.set_ylim(-0.05, 1.05)
        dax.text(0.02, 0.06, "headline\nquadrant", fontsize=7, color=OI["grey"])
        dax.legend(fontsize=7, loc="lower right")
        dax.set_title("D. Redundancy discriminator\n(pre-registered thresholds)", fontsize=9)

    fig.suptitle("Route Q — the developmental-trajectory nonlinearity as a supervised low-rank interaction "
                 f"({model} / {dataset})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "route_q_summary.png")
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
