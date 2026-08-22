"""H-A3 — does a steered path RECONSTRUCT a population deleted from the manifold (generate vs interpolate)?

Remove the erythroid INTERMEDIATE (Ery_1) from every fitted object — the manifold, the tangent, the decoder
reference. The erythroid lineage becomes progenitor -> [GAP] -> Ery_2. Steer held-out progenitors toward the
Ery_2 centroid on-manifold, decode the path with scGPT's own head, and ask: at intermediate steps, does the
decoded expression match the HELD-OUT Ery_1 transcriptome — i.e. does the path pass through the missing
intermediate it never saw? Specificity control: a myeloid-aimed steer must NOT reconstruct Ery_1.

  path transiently best-matches held-out Ery_1  => the geometry GENERATES an unobserved intermediate state
  path only ever matches retained clusters       => it INTERPOLATES observed cells; no generation

Metric: Pearson r over HVG between decoded path point and each cluster's real mean expression, per step T.
Report, for the erythroid steer, whether Ery_1 is the top-matching cluster at any T, and the peak-r step; same
for the myeloid steer (should not peak on Ery_1).

scGPT native head (circularity-free). Run: ../../.venv/bin/python ha3_heldout_population.py
Out: results/ha3_heldout_population.json
"""
import os, sys, json
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from tangent_diagnostic import path as npz_path, unit, steer, DIM, BUDGETS, T_STAR  # noqa: E402
from native_decode import build_native_head, build_spaces  # noqa: E402
from local_steering import project_constant  # noqa: E402
from gene_decode import load_expression, N_HVG  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
HELDOUT = "Ery_1"                        # the intermediate we delete from the manifold
TARGETS = {"ery": ["Ery_2"], "myeloid": ["Mono_1", "Mono_2"]}   # erythroid (test) + myeloid (specificity)


def run():
    z = np.load(npz_path("scgptbin", "setty"), allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int); clu = z["clusters"]
    ok = np.isfinite(y); emb, y, ci, clu = emb[ok], y[ok], ci[ok], clu[ok]
    E, genes, clusters = load_expression(ci)
    M, keep = build_native_head(genes); genes_k = genes[keep]; E = E[:, keep]
    hvg = np.argsort(-E.var(0))[:N_HVG]

    # cluster mean expression (real), for EVERY cluster incl. the held-out one -- computed from ALL cells
    clusters_all = np.unique(clu)
    cl_mean = {c: E[clu == c].mean(0) for c in clusters_all}

    # DELETE the intermediate from every fitted object
    train = clu != HELDOUT
    Xz, to_raw = build_spaces(emb[train], dim=DIM)
    native = lambda pz: to_raw(pz) @ M.T                     # noqa: E731  decode a whitened point -> expression
    ytr = y[train]; clu_tr = clu[train]
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz).kneighbors(Xz)[0][:, 1]))

    # source = progenitor cells (low pseudotime), from the retained manifold
    prog = Xz[(ytr <= np.quantile(ytr, 1 / 3)) & np.isin(clu_tr, ["HSC_1", "HSC_2", "Precursors"])]
    rng = np.random.default_rng(SEED)
    if len(prog) > 200:
        prog = prog[rng.choice(len(prog), 200, replace=False)]
    src_c = prog.mean(0)
    print(f"held-out {HELDOUT}: {int((~train).sum())} cells removed from manifold | source progenitors: {len(prog)}")
    print(f"clusters with real means: {list(clusters_all)}")

    out = dict(heldout=HELDOUT, n_heldout=int((~train).sum()), n_src=int(len(prog)), steers={})
    for name, cls in TARGETS.items():
        tmask = np.isin(clu_tr, cls)
        w = unit(Xz[tmask].mean(0) - src_c)
        pth = steer(prog, project_constant(Xz, w), 0.25 * d0, BUDGETS)
        # per-step correlation of the decoded path (mean over source cells) with each cluster's real mean
        traj = {c: [] for c in clusters_all}
        for T in BUDGETS:
            dec = native(pth[T]).mean(0)                    # mean decoded expression at step T
            for c in clusters_all:
                traj[c].append(float(np.corrcoef(dec[hvg], cl_mean[c][hvg])[0, 1]))
        # for the held-out cluster: peak correlation and whether it is ever the TOP cluster
        heldout_r = np.array(traj[HELDOUT])
        top_at = [max(clusters_all, key=lambda c: traj[c][T]) for T in range(len(BUDGETS))]
        ho_is_top = [c == HELDOUT for c in top_at]
        peakT = int(np.argmax(heldout_r))
        # rank of held-out among all clusters at its peak step (1 = best match)
        rank_at_peak = int(sorted(clusters_all, key=lambda c: -traj[c][peakT]).index(HELDOUT) + 1)
        rec = dict(target=cls, heldout_r_by_T=[round(v, 3) for v in heldout_r.tolist()],
                   heldout_peak_r=float(heldout_r.max()), heldout_peak_T=peakT,
                   heldout_is_top_any_T=bool(any(ho_is_top)),
                   heldout_top_steps=[BUDGETS[t] for t, b in enumerate(ho_is_top) if b],
                   heldout_rank_at_peak=rank_at_peak,
                   top_cluster_by_T={str(BUDGETS[t]): top_at[t] for t in range(len(BUDGETS))})
        out["steers"][name] = rec
        print(f"\n  [{name} steer -> {cls}]")
        print(f"    held-out {HELDOUT} corr by T: " + " ".join(f"{v:+.2f}" for v in heldout_r))
        print(f"    peak r={heldout_r.max():+.3f} @T{peakT} (rank {rank_at_peak}/{len(clusters_all)});"
              f" {HELDOUT} is top cluster at steps {rec['heldout_top_steps'] or 'NONE'}")
        print(f"    top cluster by T: " + " ".join(f"{BUDGETS[t]}:{top_at[t]}" for t in (0, 2, 4, 6, 8, 12)))

    ery, mye = out["steers"]["ery"], out["steers"]["myeloid"]
    out["verdict"] = dict(
        ery_reconstructs_heldout=bool(ery["heldout_is_top_any_T"] or ery["heldout_rank_at_peak"] <= 2),
        specific=bool((ery["heldout_peak_r"] - mye["heldout_peak_r"]) > 0.05),
        ery_peak_r=ery["heldout_peak_r"], mye_peak_r=mye["heldout_peak_r"])
    print(f"\n================ H-A3 VERDICT ================")
    print(f"  erythroid steer: held-out {HELDOUT} peak r={ery['heldout_peak_r']:+.3f} rank {ery['heldout_rank_at_peak']}"
          f" (top at steps {ery['heldout_top_steps'] or 'none'})")
    print(f"  myeloid steer  : held-out {HELDOUT} peak r={mye['heldout_peak_r']:+.3f} (specificity control)")
    print(f"  reconstructs held-out intermediate: {out['verdict']['ery_reconstructs_heldout']} | "
          f"specific to erythroid: {out['verdict']['specific']}")
    json.dump(out, open(os.path.join(RESULTS, "ha3_heldout_population.json"), "w"), indent=1)
    print("[done] -> results/ha3_heldout_population.json")
    return out


if __name__ == "__main__":
    run()

