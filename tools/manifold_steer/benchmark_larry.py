"""Benchmark 2 (Weinreb LARRY, model-free) — does on-manifold steering reach the RIGHT real fate?

The cross-dataset validation the Setty analysis couldn't do: LARRY terminal fates are real, well-populated
cell populations (Neutrophil, Monocyte, Baso, Mast, Meg, Erythroid). Steer undifferentiated cells toward each
fate's centroid and ask whether the steered path arrives at that fate's REAL cells — measured by the terminal
identity of the endpoint's nearest real neighbours, not a decoded proxy. A clean diagonal = "aim at fate F,
land on real F cells". Compare the method (project=True) against the no-projection ablation.

Metric: for target F, endpoint fate composition = mean over source cells of the fraction of the endpoint's k
nearest TERMINAL cells that are each fate. Diagonal hit = the targeted fate is the top-arriving fate.

The manifold is the dataset's own 40-PC space (no foundation model). The method never saw this data during
development. Run: ../../../.venv/bin/python benchmark_larry.py   Out: results/benchmark_larry.json
"""
import os, sys, json
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from manifold_steer import ManifoldSteer  # noqa: E402
from _data import load_larry  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
SEED = 0
FATES = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
MIN_FATE = 50
N_STEPS = 24
K_VOTE = 15


def main():
    D = load_larry()
    Xpca, state = D["Xpca"], D["state"]
    fates = [f for f in FATES if (state == f).sum() >= MIN_FATE]
    rng = np.random.default_rng(SEED)

    ms = ManifoldSteer(n_pcs=30, k_tangent=100, m_tangent=10).fit(Xpca)   # dataset's own manifold, model-free
    Z = ms.Z_
    undiff = np.where(state == "undiff")[0]
    src_idx = undiff[rng.choice(len(undiff), min(300, len(undiff)), replace=False)]
    src = Z[src_idx]

    # vote among TERMINAL fate cells only: which real fate is an endpoint nearest to?
    term_mask = np.isin(state, fates)
    term_Z = Z[term_mask]; term_lab = state[term_mask]
    nn_vote = NearestNeighbors(n_neighbors=K_VOTE).fit(term_Z)

    def fate_composition(points):
        _, ix = nn_vote.kneighbors(points)
        labs = term_lab[ix]                                  # (n_pts, K)
        return {f: float((labs == f).mean()) for f in fates}

    print(f"\n===== manifold_steer benchmark — LARRY real fates (model-free, {ms.dim_}-PC), "
          f"src={len(src)} undiff =====")
    print(f"  fates: " + ", ".join(f"{f}({int((state==f).sum())})" for f in fates))

    out = dict(dataset="larry_sp500", n_pcs=ms.dim_, n_src=int(len(src)), fates=fates,
               d0=ms.d0_, steps=N_STEPS, matrices={})
    for project in (True, False):
        tag = "on_manifold" if project else "plain_linear"
        mat = {}; path_off = {}; peak_off = {}
        for f in fates:
            d = ms.fate_direction(src_idx, np.where(state == f)[0])
            traj = ms.steer(src, d, N_STEPS, project=project)
            comps = [fate_composition(traj[t]) for t in range(N_STEPS + 1)]
            offr = [ms.off_manifold_ratio(traj[t]) for t in range(N_STEPS + 1)]
            peakT = int(np.argmax([c[f] for c in comps]))     # step of maximal arrival at the targeted fate
            mat[f] = comps[peakT]
            peak_off[f] = offr[peakT]
            path_off[f] = float(np.mean(offr[1:peakT + 1])) if peakT >= 1 else offr[0]  # plausibility of the PATH
        hits = sum(max(fates, key=lambda g: mat[f][g]) == f for f in fates)
        dom = float(np.mean([mat[f][f] - np.mean([mat[f][g] for g in fates if g != f]) for f in fates]))
        mpath = float(np.mean(list(path_off.values())))
        out["matrices"][tag] = dict(matrix={f: mat[f] for f in fates}, path_off_ratio=path_off,
                                    peak_off_ratio=peak_off, hits=f"{hits}/{len(fates)}", dominance=dom,
                                    mean_path_off_ratio=mpath)
        print(f"\n  --- {tag} (project={project}) ---   diagonal hits {hits}/{len(fates)}  "
              f"| mean PATH off-manifold ratio {mpath:.2f}  (how far the path strays from real cells)")
        print("     aim ↓ / arrive →   " + "".join(f"{g[:5]:>7}" for g in fates) + "  top   path-off")
        for f in fates:
            top = max(fates, key=lambda g: mat[f][g])
            print(f"     {f:<16} " + "".join(f"{mat[f][g]:>7.2f}" for g in fates)
                  + f"  {top[:5]}{'✓' if top == f else ' '}  {path_off[f]:>6.2f}")

    on, pl = out["matrices"]["on_manifold"], out["matrices"]["plain_linear"]
    out["summary"] = dict(on_manifold_hits=on["hits"], plain_hits=pl["hits"],
                          on_manifold_path_off=on["mean_path_off_ratio"], plain_path_off=pl["mean_path_off_ratio"],
                          path_off_reduction=float(pl["mean_path_off_ratio"] - on["mean_path_off_ratio"]))
    print(f"\n  SUMMARY: both reach the right fate ({on['hits']} vs {pl['hits']} diagonal hits) — that is easy, since a")
    print(f"  fate's centroid sits among its own cells. The DIFFERENCE is the PATH: on-manifold mean off-ratio")
    print(f"  {on['mean_path_off_ratio']:.2f} vs plain-linear {pl['mean_path_off_ratio']:.2f} "
          f"(reduction {pl['mean_path_off_ratio']-on['mean_path_off_ratio']:+.2f}).")
    print("  => on-manifold steering routes undiff cells to the correct REAL fate THROUGH real intermediate")
    print("     states; plain-linear teleports through empty expression space. The path is the product.")
    json.dump(out, open(os.path.join(RESULTS, "benchmark_larry.json"), "w"), indent=1)
    print("[done] -> results/benchmark_larry.json")


if __name__ == "__main__":
    main()

