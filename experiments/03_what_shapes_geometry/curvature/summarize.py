"""Route Q — assemble the generality table (4 models x 4 datasets) as markdown.

Cross-checks each Delta_Q against the published full-space kernel curvature in
../route_branchpoint/results/full_space_curvature.json (curv_poly = poly2_full - linear_full).

Usage: ../../.venv/bin/python summarize.py
Out:   GENERALITY.md
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KERN = os.path.join(HERE, "..", "route_branchpoint", "results", "full_space_curvature.json")

MODELS = ["scgpt", "geneformer", "state", "maxtoki"]
DATASETS = [("setty", "blood (human)"), ("lung", "lung airway (human)"),
            ("gut", "fetal gut (human)"), ("pancreas", "pancreas (MOUSE — neg. control)")]


def main():
    kern = json.load(open(KERN)) if os.path.exists(KERN) else {}
    rows, missing = [], []
    for ds, dsname in DATASETS:
        for m in MODELS:
            p = os.path.join(RES, f"q_{m}_{ds}.json")
            if not os.path.exists(p):
                missing.append(f"{m}/{ds}"); continue
            d = json.load(open(p))
            k = kern.get(f"{m}_{ds}", {}).get("full", {}).get("curv_poly")
            rows.append(dict(model=m, dataset=ds, dsname=dsname, n=d["n"], dd=d["d"],
                             r2_lin=d["r2_lin"], dq=d["delta_q"], ci=d["delta_q_ci"],
                             modal_r=d["modal_r"], kern=k,
                             dq_noab=d.get("delta_q_no_abundance")))

    lines = ["# Route Q — generality across models and lineages", "",
             "Out-of-fold `Δ_Q` from the nested-CV pipeline (`run_substrate.py`). `Δ_kernel` is the published",
             "full-space degree-2 kernel curvature from `route_branchpoint/results/full_space_curvature.json`",
             "(`poly2_full − linear_full`) — an independent, full-rank estimate of the same quantity.", "",
             "A 95% CI excluding 0 means an explicit low-rank Q reproduces the curvature on that substrate.", ""]

    for ds, dsname in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        lines += [f"## {ds} — {dsname}", "",
                  "| model | n | d | R²_lin | **Δ_Q** | 95% CI | CI excl. 0 | Δ_Q / (1−R²_lin) | modal rank | Δ_kernel (ref) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            excl = "yes" if (r["ci"][0] > 0 or r["ci"][1] < 0) else "**no**"
            kk = f"{r['kern']:+.4f}" if r["kern"] is not None else "—"
            norm = r["dq"] / max(1e-9, 1.0 - r["r2_lin"])
            r["norm"] = norm
            lines.append(f"| {r['model']} | {r['n']} | {r['dd']} | {r['r2_lin']:.4f} | "
                         f"**{r['dq']:+.4f}** | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {excl} | "
                         f"{100*norm:.1f}% | {r['modal_r']} | {kk} |")
        lines.append("")
    lines += ["> `Δ_Q / (1−R²_lin)` = the share of the variance the *linear* decoder leaves on the table that",
              "> the quadratic term recovers. Absolute Δ_Q is not comparable across substrates whose linear",
              "> ceilings differ by 0.79–0.97; this normalisation is.", ""]

    # abundance guard, where run in full mode
    ab = [r for r in rows if r["dq_noab"] is not None]
    if ab:
        lines += ["## Abundance guard (full-mode substrates only)", "",
                  "| model / dataset | Δ_Q (abundance covariate on) | Δ_Q (abundance removed) |",
                  "|---|---|---|"]
        for r in ab:
            lines.append(f"| {r['model']} / {r['dataset']} | {r['dq']:+.4f} | {r['dq_noab']:+.4f} |")
        lines.append("")

    pan = [r for r in rows if r["dataset"] == "pancreas"]
    hum = [r for r in rows if r["dataset"] != "pancreas"]
    if pan:
        pos = [r for r in pan if r["ci"][0] > 0]
        pn = np.mean([r["norm"] for r in pan]) if pan else float("nan")
        hn = np.mean([r["norm"] for r in hum]) if hum else float("nan")
        nul = os.path.join(RES, "nulls_scgpt_pancreas.json")
        nulltxt = ""
        if os.path.exists(nul):
            d = json.load(open(nul))
            nulltxt = (f"\n`scgpt/pancreas` vs its OWN nulls (prereg §5.3, the actual criterion): "
                       f"real Δ_Q = {d['delta_q_real']:+.4f}; shuffle p95 = {d['shuffle']['p95']:+.4f} "
                       f"(pass={d['shuffle']['passes']}); linear-synthetic p95 = "
                       f"{d['linear_synthetic']['p95']:+.4f} (pass={d['linear_synthetic']['passes']}). "
                       f"**Both nulls pass = {d['both_nulls_pass']}**.\n")

        lines += ["## Mouse-pancreas negative control (prereg §5.3) — PARTIAL FAILURE", "",
                  f"Substrates with a CI strictly above 0: **{len(pos)}/{len(pan)}** "
                  f"({', '.join(r['model'] for r in pos) if pos else 'none'}).", "",
                  "Prereg §5.3 required Δ_Q ≈ 0 here, because `CROSS_MODEL_GEOMETRY_RESULTS.md` concluded the",
                  "mouse-pancreas curvature is absent (a species/ortholog artifact, not lineage). **It is not**",
                  "**zero.** The effect is smaller in absolute terms than in the human lineages, but the mouse",
                  "linear ceiling is also much higher (R²_lin ≈ 0.94–0.97 vs 0.79–0.95), so absolute Δ_Q is the",
                  "wrong comparison:", "",
                  f"- mean `Δ_Q / (1−R²_lin)` across the **human** substrates: **{100*hn:.1f}%**",
                  f"- mean `Δ_Q / (1−R²_lin)` across the **mouse-pancreas** substrates: **{100*pn:.1f}%**", "",
                  nulltxt,
                  "**Reading.** Two things are true and must be reported together. (i) Our estimator does *not*",
                  "manufacture curvature from nothing — the label-shuffle and linear-synthetic nulls on pancreas",
                  "are the correct test for that, and they are the ones that matter. (ii) But the pre-registered",
                  "expectation of a null result on mouse pancreas is **not met**: an explicit low-rank Q finds a",
                  "small, abundance-independent, CI-excluding-zero quadratic gain there too. Once normalised by",
                  "linear headroom the mouse effect is of *comparable relative size* to the weaker human",
                  "substrates (e.g. scgpt/gut). This **weakens the species/ortholog interpretation** in",
                  "`CROSS_MODEL_GEOMETRY_RESULTS.md`, which was based on a full-rank kernel decoder near its",
                  "ceiling. We flag this as an open discrepancy rather than resolving it here.", ""]

    if missing:
        lines += ["## Missing substrates", "", ", ".join(missing), ""]

    open(os.path.join(HERE, "GENERALITY.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
