"""Aggregate every model's cell-cycle geometry + steering results into one cross-model table.

Reads whatever results/cc_{geometry,steering}_<model>.json exist and prints the comparison. Regenerate after
adding a model. No computation here -- pure assembly, so it is always safe to re-run.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TWO_PI = 2.0 * np.pi
MODELS = ["expr", "scgptcc", "geneformer", "maxtoki", "statecc"]
LABEL = {"expr": "RAW EXPRESSION", "scgptcc": "scGPT-L11", "geneformer": "Geneformer-L11",
         "maxtoki": "MaxToki-L8", "statecc": "STATE-SE-L11"}
ARCH = {"expr": "no model at all", "scgptcc": "encoder (MVC)", "geneformer": "BERT / ranks",
        "maxtoki": "Llama / ranks", "statecc": "bidir. encoder"}


def load(kind, model):
    p = os.path.join(RESULTS, f"cc_{kind}_{model}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    rows = []
    for m in MODELS:
        g, s = load("geometry", m), load("steering", m)
        if g is None and s is None:
            continue
        row = dict(model=m, label=LABEL.get(m, m), arch=ARCH.get(m, "?"))
        if g:
            row["lin_r2"] = g["linear_circ_r2"]
            row["gap"] = g["decodability_gap"]
            row["Hflat_rejected"] = g["H_flat_rejected"]
            row["theta_far"] = g["real"]["theta_far_broken"]
            row["theta_far_null"] = g["flat_null"]["theta_far_mean"]
        if s:
            A = s["arms"]
            row["fixed_proj_laps"] = A["fixed_proj"]["constrained_advance"] / TWO_PI
            row["fixed_rate_dies"] = A["fixed_proj"]["phase_rate_first_nonpositive_T"]
            row["fixedR_laps"] = A["fixed_proj_retract"]["constrained_advance"] / TWO_PI
            row["localR_laps"] = A["local_proj_retract"]["constrained_advance"] / TWO_PI
            row["reverse_rad"] = A["local_proj_retract"]["advance_reverse_min"]
            row["gain_local"] = s["verdict"]["gain_from_locality"]
            row["gain_proj"] = s["verdict"]["gain_from_projection"]
        rows.append(row)

    if not rows:
        print("no results yet"); return

    print("=" * 100)
    print("CROSS-MODEL CELL-CYCLE STEERING  (each model on its own K562 cycling cells)")
    print("=" * 100)
    print("\n-- GEOMETRY: is the loop flat, and does theta_far mislead? --")
    print(f"  {'model':<16}{'arch':<16}{'lin circ-R2':>12}{'decod. gap':>12}{'H_flat':>14}"
          f"{'theta_far':>11}{'(flat null)':>12}")
    for r in rows:
        if "lin_r2" not in r:
            continue
        hf = "REJECTED" if r["Hflat_rejected"] else "not rej. (flat)"
        print(f"  {r['label']:<16}{r['arch']:<16}{r['lin_r2']:>12.3f}{r['gap']:>+12.3f}{hf:>14}"
              f"{r['theta_far']:>10.1f}d{r['theta_far_null']:>11.1f}d")

    print("\n-- STEERING: fixed direction stalls; only a local (rotating) direction traverses --")
    print(f"  {'model':<16}{'fixed_proj':>12}{'rate dies':>11}{'fixed+retr':>12}{'local+retr':>12}"
          f"{'reverse':>10}{'loc>proj?':>11}")
    for r in rows:
        if "fixed_proj_laps" not in r:
            continue
        rd = f"T{r['fixed_rate_dies']}" if r["fixed_rate_dies"] is not None else "never"
        locwins = "YES" if r["gain_local"] > r["gain_proj"] else "no"
        print(f"  {r['label']:<16}{r['fixed_proj_laps']:>10.2f}L{rd:>11}{r['fixedR_laps']:>10.2f}L"
              f"{r['localR_laps']:>10.2f}L{r['reverse_rad']:>9.1f}{locwins:>11}")

    done = [r["label"] for r in rows if "fixed_proj_laps" in r]
    print(f"\n  {len(done)}/{len(MODELS)} models: {', '.join(done)}")
    print("  Pattern to check across all: loop FLAT (H_flat not rejected), theta_far < its own flat-circle")
    print("  null (meaningless), fixed direction stalls on-manifold, retraction does NOT rescue it,")
    print("  local+retraction winds multiple laps, fully reversible, locality beats projection.")

    json.dump(rows, open(os.path.join(RESULTS, "cc_summary.json"), "w"), indent=1)
    print(f"\n[done] -> {os.path.join(RESULTS, 'cc_summary.json')}")


if __name__ == "__main__":
    main()
