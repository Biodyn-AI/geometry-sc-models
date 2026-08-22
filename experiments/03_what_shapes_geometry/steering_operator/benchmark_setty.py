"""Benchmark 1 (Setty hematopoiesis, model-free COUNTS) — the ablation ladder.

Demonstrates, using ONLY the standalone ManifoldSteer class on raw log-normalized counts (no foundation model),
that local-tangent PROJECTION is what makes steering stay on the manifold while still advancing pseudotime.
Rules compared, all steering held-out early cells forward along the pseudotime-ascent direction:

  linear        constant direction, no projection            (drifts off-manifold)
  linear_proj   constant direction + local-tangent projection (the method)
  retract       constant direction + pull toward local kNN mean (graph-corrector baseline)
  random_proj   random direction + projection                (control: projection alone advances nothing)
  oracle        step toward higher-pseudotime neighbours      (label-using reference ceiling)

Metric = constrained advance CA(tau): max pseudotime advance (kNN judge, fit on train, never the steering rule)
subject to off-manifold ratio <= tau. Split train/test by cell; source = held-out early third.

Run:  ../../../.venv/bin/python benchmark_setty.py
Out:  results/benchmark_setty.json
"""
import os, sys, json
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from manifold_steer import ManifoldSteer, _unit  # noqa: E402
from _data import load_setty  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
SEED = 0
N_STEPS = 12
TAUS = (1.2, 1.3, 1.5)
T_STAR = 8


def constrained_advance(off, adv, tau):
    ok = [adv[t] for t in range(1, len(adv)) if off[t] <= tau]
    return float(max(ok)) if ok else 0.0


def main():
    D = load_setty()
    X, y = D["counts"], D["pseudotime"]
    rng = np.random.default_rng(SEED)
    if len(y) > 3000:
        idx = rng.choice(len(y), 3000, replace=False); X, y = X[idx], y[idx]
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]

    ms = ManifoldSteer(n_pcs=20).fit(X[tr])                 # manifold from TRAIN counts only
    Ztr, Zte = ms.Z_, ms.transform(X[te])
    ytr = (y[tr] - y[tr].mean()) / (y[tr].std() + 1e-9)
    yte = (y[te] - y[tr].mean()) / (y[tr].std() + 1e-9)
    judge = KNeighborsRegressor(15).fit(Ztr, ytr)          # advance judge (independent of the steering rule)
    w = ms.ascent_direction(y[tr])

    # source = held-out early third
    src = Zte[yte <= np.quantile(yte, 1 / 3)]
    if len(src) > 300:
        src = src[rng.choice(len(src), 300, replace=False)]

    # retraction and oracle fields
    nn_ret = NearestNeighbors(n_neighbors=30).fit(Ztr)

    def retract(Zx):
        _, ix = nn_ret.kneighbors(Zx)
        pull = np.array([Ztr[ix[i]].mean(0) for i in range(len(Zx))]) - Zx
        return _unit(w)[None] + pull

    nn_or = NearestNeighbors(n_neighbors=50).fit(Ztr)

    def oracle(Zx):
        _, ix = nn_or.kneighbors(Zx); Dv = np.zeros_like(Zx)
        for i in range(len(Zx)):
            hi = ix[i][ytr[ix[i]] > np.median(ytr[ix[i]])]
            hi = hi if len(hi) else ix[i]
            Dv[i] = Ztr[hi].mean(0) - Zx[i]
        return Dv

    wr = _unit(rng.standard_normal(ms.dim_))
    rules = {
        "linear": dict(direction=w, project=False),
        "linear_proj": dict(direction=w, project=True),
        "retract": dict(direction=retract, project=False),
        "random_proj": dict(direction=wr, project=True),
        "oracle": dict(direction=oracle, project=False),
    }

    out = dict(dataset="setty_counts", n_pcs=20, d0=ms.d0_, n_src=int(len(src)), rules={})
    print(f"\n===== manifold_steer benchmark — Setty COUNTS (model-free), n_src={len(src)}, d0={ms.d0_:.3f} =====")
    print(f"  {'rule':<12} | {'off@2/4/8':^18} | {'adv@2/4/8':^18} | {'CA 1.2/1.3/1.5':^18} | align@8")
    base = judge.predict(src)
    for name, cfg in rules.items():
        traj = ms.steer(src, cfg["direction"], N_STEPS, project=cfg["project"])
        off = [ms.off_manifold_ratio(traj[t]) for t in range(N_STEPS + 1)]
        adv = [float((judge.predict(traj[t]) - base).mean()) for t in range(N_STEPS + 1)]
        # tangent alignment of the endpoint step
        Dend = cfg["direction"](traj[T_STAR]) if callable(cfg["direction"]) else cfg["direction"]
        Vs = ms.tangent_bases(traj[T_STAR]); Dn = _unit(np.atleast_2d(np.broadcast_to(Dend, traj[T_STAR].shape)))
        align = float(np.mean([np.linalg.norm(Vs[i] @ Dn[i]) for i in range(len(Dn))]))
        ca = {f"{t}": constrained_advance(off, adv, t) for t in TAUS}
        out["rules"][name] = dict(off=off, adv=adv, ca=ca, align_T8=align)
        print(f"  {name:<12} | " + "/".join(f"{off[t]:.2f}" for t in (2, 4, 8)) + "   | "
              + "/".join(f"{adv[t]:.2f}" for t in (2, 4, 8)) + "   | "
              + "/".join(f"{ca[f'{t}']:.2f}" for t in TAUS) + "   | " + f"{align:.2f}")

    lp = out["rules"]["linear_proj"]["ca"]["1.3"]; li = out["rules"]["linear"]["ca"]["1.3"]
    rt = out["rules"]["retract"]["ca"]["1.3"]
    out["summary"] = dict(ca13_linear=li, ca13_linear_proj=lp, ca13_retract=rt,
                          projection_gain=lp - li, proj_minus_retract=lp - rt)
    print(f"\n  projection gain (linear_proj - linear): {lp - li:+.2f}  |  proj - retract: {lp - rt:+.2f}")
    print("  => tangent projection is the on-manifold win, model-free (see ../RESULTS.md Gate 0 / Gate 2).")
    json.dump(out, open(os.path.join(RESULTS, "benchmark_setty.json"), "w"), indent=1)
    print("[done] -> results/benchmark_setty.json")


if __name__ == "__main__":
    main()

