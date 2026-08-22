"""GATE 0 — does the foundation model matter, or is the steering win pure data-manifold geometry?

The tangent-projection steer estimates its tangent by local PCA of the point cloud in a 20-PC whitened space.
That space is the MODEL's embedding. Question: if we run the identical ladder in the 20-PC whitened space of
raw log-normalized COUNTS (no model at all), on the SAME cells, does tangent projection still deliver the
0.46 -> ~1.24 jump in constrained advance?

  - counts >= model on the projection win  => the algorithm needs no SCFM; it is manifold learning on scRNA-seq.
  - model >> counts                          => pretraining buys geometry (the mech-interp claim).

Matched design: same cells, same CAP subsample, same seed/split. Counts space = prep(E) where E is log1p CP10k
over the model's own cell_idx, so the ONLY thing that differs between the two ladders is model-emb vs counts.

Run:  ../../.venv/bin/python gate0_counts_vs_model.py [model ...]   (default: all four)
Out:  results/gate0_counts_vs_model.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import load, run_ladder, ca13, prep, RS, CAP, SEED  # noqa: E402
sys.path.insert(0, RS)
from gene_decode import load_expression  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MODELS = ["scgptbin", "geneformer", "state", "maxtoki"]
RULES = ("linear", "linear_proj", "local_proj", "oracle_knn")


def counts_space(ci):
    """log1p CP10k over the given h5ad rows, then project-standard 20-PC whitened space (no model)."""
    E, genes, clusters = load_expression(ci)
    return prep(E), E.shape[1]


def run_model(model):
    X, y, ci, clu = load(model)
    Xz_model = prep(X)
    Xz_counts, n_genes = counts_space(ci)

    r_model = run_ladder(Xz_model, y, seed=SEED, rules=RULES)
    r_counts = run_ladder(Xz_counts, y, seed=SEED, rules=RULES)

    def summ(r):
        return {k: {"ca13": ca13(r, k), "align_T8": r[k]["align_T8"],
                    "off_T8": r[k][8]["off_manifold_ratio"], "adv_T8": r[k][8]["advance"]} for k in RULES}

    proj_gain_model = ca13(r_model, "linear_proj") - ca13(r_model, "linear")
    proj_gain_counts = ca13(r_counts, "linear_proj") - ca13(r_counts, "linear")
    rec = dict(model=model, n_cells=int(len(y)), n_genes=int(n_genes),
               model_space=summ(r_model), counts_space=summ(r_counts),
               proj_gain_model=float(proj_gain_model), proj_gain_counts=float(proj_gain_counts),
               model_minus_counts_projgain=float(proj_gain_model - proj_gain_counts))
    print(f"\n===== {model}  (n={len(y)}, genes={n_genes}) =====")
    print(f"  {'rule':<14} | {'MODEL space ca13':>16} | {'COUNTS space ca13':>17}")
    for k in RULES:
        print(f"  {k:<14} | {ca13(r_model,k):>16.2f} | {ca13(r_counts,k):>17.2f}")
    print(f"  projection gain (linear_proj - linear):  MODEL {proj_gain_model:+.2f}   COUNTS {proj_gain_counts:+.2f}"
          f"   => model-counts {proj_gain_model-proj_gain_counts:+.2f}")
    return rec


def main():
    models = sys.argv[1:] or MODELS
    out = {}
    for m in models:
        try:
            out[m] = run_model(m)
        except Exception as e:
            print(f"[skip] {m}: {e}")
    json.dump(out, open(os.path.join(RESULTS, "gate0_counts_vs_model.json"), "w"), indent=1)

    print("\n================ GATE 0 VERDICT ================")
    print(f"  {'model':<12} {'proj_gain MODEL':>16} {'proj_gain COUNTS':>17} {'model-counts':>13}")
    for m, r in out.items():
        print(f"  {m:<12} {r['proj_gain_model']:>16.2f} {r['proj_gain_counts']:>17.2f} "
              f"{r['model_minus_counts_projgain']:>13.2f}")
    gm = np.mean([r["proj_gain_model"] for r in out.values()]) if out else float("nan")
    gc = np.mean([r["proj_gain_counts"] for r in out.values()]) if out else float("nan")
    print(f"  mean projection gain: MODEL {gm:+.2f} vs COUNTS {gc:+.2f}")
    print("  Interpretation: if COUNTS gain ~ MODEL gain, the projection win is data-manifold geometry, not the")
    print("  foundation model. If MODEL >> COUNTS, pretraining buys the geometry.")
    print(f"[done] -> results/gate0_counts_vs_model.json")


if __name__ == "__main__":
    main()
