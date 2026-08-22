"""Route Q — the two pre-registered null distributions (prereg §5.1, §5.2).

Both refit the ENTIRE nested pipeline (rank + lambda selection included) on a surrogate target, so a
gain manufactured by Q's extra capacity or by SNR headroom shows up in the null.

  1. label-shuffle      : permute y across cells.
  2. linear-synthetic   : y_synth = out-of-fold linear prediction of the real y + Gaussian noise scaled
                          to match the real linear-decode R^2. y_synth is LINEAR in x by construction,
                          so a correct estimator must return delta_q ~ 0. This is the control that
                          caught route_lineage's LEACE false positive.

Usage:  ../../.venv/bin/python run_nulls.py scgpt setty [n_rep]
Out:    results/nulls_<model>_<dataset>.json
"""
import os, sys, json, time
import numpy as np
import torch

import qfit

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main(model, dataset, n_rep=20):
    t0 = time.time()
    X, y = qfit.load(model, dataset)
    if X is None:
        print(f"[skip] {model}/{dataset}"); return

    prim = json.load(open(os.path.join(RES, f"q_{model}_{dataset}.json")))
    d_real = prim["delta_q"]
    p_lin = np.load(os.path.join(RES, f"q_{model}_{dataset}.npz"))["p_lin"]

    print(f"[{model}/{dataset}] delta_Q(real)={d_real:+.4f}; running {n_rep} reps x 2 nulls", flush=True)

    d_shuf = qfit.null_shuffle(X, y, n_rep=n_rep)
    print(f"  shuffle null:  mean={d_shuf.mean():+.4f} p95={np.percentile(d_shuf,95):+.4f} "
          f"max={d_shuf.max():+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    d_syn = qfit.null_linear_synthetic(X, y, p_lin, n_rep=n_rep)
    print(f"  lin-synth null: mean={d_syn.mean():+.4f} p95={np.percentile(d_syn,95):+.4f} "
          f"max={d_syn.max():+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    def emp_p(nulls):
        # one-sided: P(null >= real), with the +1/(n+1) correction
        return float((np.sum(np.asarray(nulls) >= d_real) + 1) / (len(nulls) + 1))

    out = dict(model=model, dataset=dataset, n_rep=n_rep, delta_q_real=d_real,
               shuffle=dict(values=d_shuf.tolist(), mean=float(d_shuf.mean()),
                            p95=float(np.percentile(d_shuf, 95)), max=float(d_shuf.max()),
                            p_emp=emp_p(d_shuf), passes=bool(d_real > np.percentile(d_shuf, 95))),
               linear_synthetic=dict(values=d_syn.tolist(), mean=float(d_syn.mean()),
                                     p95=float(np.percentile(d_syn, 95)), max=float(d_syn.max()),
                                     p_emp=emp_p(d_syn), passes=bool(d_real > np.percentile(d_syn, 95))),
               seconds=time.time() - t0)
    out["both_nulls_pass"] = out["shuffle"]["passes"] and out["linear_synthetic"]["passes"]
    json.dump(out, open(os.path.join(RES, f"nulls_{model}_{dataset}.json"), "w"), indent=1)
    print(f"  BOTH NULLS PASS = {out['both_nulls_pass']}  -> results/nulls_{model}_{dataset}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
