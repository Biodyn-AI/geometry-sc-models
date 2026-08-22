"""GATE 2 — is local-tangent projection more than graph-walking?

LOCAL_STEERING already ruled out snap-to-CENTROID (piecewise). The stronger baseline is a RETRACTION: take the
step along the constant direction, then pull toward the mean of the k nearest real cells (predictor-corrector on
the data graph). If retraction matches linear_proj, the contribution is "use the data graph", not the smooth
local-linear tangent. If tangent projection still wins, the tangent estimation is doing real work.

Runs the ladder with linear / linear_proj / local_proj / linear_retract / local_retract / oracle_knn, in the
MODEL space for all four models and additionally in the COUNTS space for scgptbin (so the comparison is anchored
to Gate 0's finding that counts is where the steering signal lives).

Run:  ../../.venv/bin/python gate2_retraction.py [model ...]
Out:  results/gate2_retraction.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import load, run_ladder, ca13, prep, RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from gene_decode import load_expression  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MODELS = ["scgptbin", "geneformer", "state", "maxtoki"]
RULES = ("linear", "linear_proj", "local_proj", "linear_retract", "local_retract", "oracle_knn")


def summarize(r):
    return {k: {"ca13": ca13(r, k), "off_T8": r[k][8]["off_manifold_ratio"],
                "adv_T8": r[k][8]["advance"], "align_T8": r[k]["align_T8"]} for k in RULES}


def report(tag, r):
    print(f"\n  --- {tag} ---")
    print(f"  {'rule':<16} | {'ca13':>6} | {'off@T8':>7} | {'adv@T8':>7} | {'align@T8':>8}")
    for k in RULES:
        print(f"  {k:<16} | {ca13(r,k):>6.2f} | {r[k][8]['off_manifold_ratio']:>7.2f} | "
              f"{r[k][8]['advance']:>7.2f} | {r[k]['align_T8']:>8.2f}")
    print(f"  projection vs retraction: linear_proj {ca13(r,'linear_proj'):.2f} "
          f"vs linear_retract {ca13(r,'linear_retract'):.2f}  (proj-retract {ca13(r,'linear_proj')-ca13(r,'linear_retract'):+.2f})")


def run_model(model, do_counts=False):
    X, y, ci, clu = load(model)
    r_model = run_ladder(prep(X), y, seed=SEED, rules=RULES)
    rec = dict(model=model, model_space=summarize(r_model),
               proj_minus_retract_model=float(ca13(r_model, "linear_proj") - ca13(r_model, "linear_retract")))
    print(f"\n===== {model} =====")
    report("MODEL space", r_model)
    if do_counts:
        r_counts = run_ladder(prep(load_expression(ci)[0]), y, seed=SEED, rules=RULES)
        rec["counts_space"] = summarize(r_counts)
        rec["proj_minus_retract_counts"] = float(ca13(r_counts, "linear_proj") - ca13(r_counts, "linear_retract"))
        report("COUNTS space", r_counts)
    return rec


def main():
    models = sys.argv[1:] or MODELS
    out = {}
    for m in models:
        try:
            out[m] = run_model(m, do_counts=(m == "scgptbin"))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[skip] {m}: {e}")
    json.dump(out, open(os.path.join(RESULTS, "gate2_retraction.json"), "w"), indent=1)
    print("\n================ GATE 2 VERDICT ================")
    print(f"  {'model':<12} {'linear_proj':>12} {'linear_retract':>15} {'local_proj':>11} {'proj-retract':>13}")
    for m, r in out.items():
        ms = r["model_space"]
        print(f"  {m:<12} {ms['linear_proj']['ca13']:>12.2f} {ms['linear_retract']['ca13']:>15.2f} "
              f"{ms['local_proj']['ca13']:>11.2f} {r['proj_minus_retract_model']:>13.2f}")
    print("  If proj-retract ~ 0, projection is just graph-walking. If > 0, the smooth tangent does real work.")
    print("[done] -> results/gate2_retraction.json")


if __name__ == "__main__":
    main()

