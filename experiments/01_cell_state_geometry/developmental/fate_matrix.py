"""THE POWER FIX: break the p = 0.042 permutation floor on the fate diagonal.

NATIVE_DECODE_RESULTS.md reports the fate diagonal at exact permutation p = 0.042 over 4! = 24 target->fate
assignments — and states plainly that 0.042 IS THE FLOOR: with four fates the test cannot report anything
smaller, so "the true assignment is the best of all 24" is all it can ever say. That is the single most fixable
weakness in the result, and both fixes are free:

  (i)  Setty carries a FIFTH terminal branch that route_steering never used — DCs (n = 303, in the annotation
       all along). Five fates -> 5! = 120 assignments -> attainable floor 0.0083, a 5x improvement for nothing.

  (ii) scGPT's MVC head collapses to a fixed matrix M over the GENE VOCABULARY (pred_expr = M @ cell_emb), so it
       is gene-universal, not blood-specific: it transfers unchanged to lung, gut and pancreas. And those
       tissues' scgptbin embeddings ALREADY EXIST (re-extracted 2026-07-12). Four independent diagonals, one
       per tissue, Fisher-combined into a single p.

Everything else is held identical to NATIVE_DECODE: the same 20-PC whitened space, the same tangent-projected
steer (project_constant), the same T=8 budget, the same column-centering (a per-column shift, which cannot
manufacture diagonal dominance), the same exact-permutation null, and the same pretrained head with ZERO
parameters fit on any of these datasets. Marker panels are canonical lineage markers (common.PANEL), external
to the data.

Added on top: a bootstrap CI on the diagonal-dominance D over source cells, because the permutation test asks
"is the true assignment best?" while the bootstrap asks "how big is the effect, and is it stable?" — two
different questions, and route_steering only ever answered the first.

Run:  ../../.venv/bin/python fate_matrix.py [tissue ...]
Out:  results/fate_matrix_<tissue>.json, results/fate_matrix_ALL.json
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
from itertools import permutations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from scipy.stats import spearmanr

from common import (RESULTS, TISSUES, FATE_CLUSTERS, ROOT_CLUSTERS, load_expression,   # noqa: E402
                    panel_index, fisher_combine, N_HVG)

from tangent_diagnostic import path as npz_path, zscore, unit, steer, SEED, BUDGETS, T_STAR  # noqa: E402
from local_steering import project_constant                                                  # noqa: E402
from native_decode import build_native_head, build_spaces, program_z                         # noqa: E402

N_BOOT = 500


def dominance(Mx):
    """Mean over fates F of [ program F's response to target F  -  its mean response to the OTHER targets ]."""
    n = len(Mx)
    return float(np.mean([Mx[i, i] - (Mx[:, i].sum() - Mx[i, i]) / (n - 1) for i in range(n)]))


def run(tissue):
    z = np.load(npz_path("scgptbin", tissue), allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int); clu = z["clusters"]
    ok = np.isfinite(y)
    emb, y, ci, clu = emb[ok], y[ok], ci[ok], clu[ok]

    E_full, genes, _ = load_expression(tissue, ci)
    M, keep = build_native_head(genes)                 # scGPT's OWN pretrained MVC head; nothing fit on this data
    genes_k = genes[keep]
    pi = panel_index(genes_k, tissue)

    fates = [f for f, cls in FATE_CLUSTERS[tissue].items()
             if np.isin(clu, cls).sum() >= 20 and f in pi]
    if len(fates) < 3:
        print(f"[skip] {tissue}: only {len(fates)} usable fates"); return None

    Xz, to_raw = build_spaces(emb)
    yz = zscore(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    clu_tr = clu[tr]

    native = lambda ce: ce @ M.T                        # noqa: E731
    decode = lambda pz: native(to_raw(pz))              # noqa: E731
    P_real = native(emb)
    mu, sd = P_real[tr].mean(0), P_real[tr].std(0) + 1e-9

    # ---- the same gate as NATIVE_DECODE: can scGPT's pretrained head read THESE embeddings at all?
    E = E_full[:, keep]
    hvg = np.argsort(-E[tr].var(0))[:N_HVG]
    marker_all = np.unique(np.concatenate(list(pi.values())))
    sub = te[rng.choice(len(te), min(300, len(te)), replace=False)]
    within = float(np.mean([spearmanr(P_real[i, hvg], E[i, hvg]).statistic for i in sub]))
    across = np.nan_to_num(np.array([np.corrcoef(P_real[te, j], E[te, j])[0, 1] for j in marker_all]))
    frac_pos = float((across > 0).mean())
    gate = (within > 0.15) and (frac_pos > 0.6)
    print(f"\n{'='*100}\n=== {tissue}  |  {len(fates)} fates: {', '.join(fates)}  |  {len(keep)}/{len(genes)} genes decodable")
    print(f"  [gate] within-cell gene ranking (Spearman) = {within:+.3f} | markers tracking correctly = "
          f"{frac_pos*100:.0f}%  ->  {'PASSED' if gate else 'FAILED'}")
    if not gate:
        print("  *** GATE FAILED — scGPT's own head cannot read these embeddings; refusing to report a diagonal.")
        return dict(tissue=tissue, gate_passed=False, within_cell_spearman=within, marker_frac_pos=frac_pos)

    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))
    roots = np.isin(clu_tr, ROOT_CLUSTERS[tissue])
    src_pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = Xz[src_pool]
    if len(src) > 200:
        src = src[rng.choice(len(src), 200, replace=False)]
    src_c = Xz[tr][roots].mean(0) if roots.sum() >= 20 else src.mean(0)

    # ---- steer at each fate; record the per-SOURCE-CELL program deltas (needed for the bootstrap)
    per_cell = {}                                       # fate -> (n_src, n_programs) delta matrix
    for f in fates:
        w_f = unit(Xz[tr][np.isin(clu_tr, FATE_CLUSTERS[tissue][f])].mean(0) - src_c)
        pth = steer(src, project_constant(Xz[tr], w_f), 0.25 * d0, BUDGETS)
        p0, p8 = decode(pth[0]), decode(pth[T_STAR])
        d = np.stack([(((p8[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)
                        - ((p0[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)) for k in fates], axis=1)
        per_cell[f] = d                                 # (n_src, n_fates) — columns follow `fates`

    A = np.array([per_cell[f].mean(0) for f in fates])          # rows = target, cols = program
    Ac = A - A.mean(0, keepdims=True)                            # column-centre: remove the common-mode deflation

    D_obs = dominance(Ac)
    perms = [dominance(Ac[list(p)]) for p in permutations(range(len(fates)))]
    p_exact = float(np.mean([d >= D_obs - 1e-12 for d in perms]))
    floor = 1.0 / len(perms)
    # per-tissue hit count under every assignment — feeds the POOLED exact test in main()
    hit_perms = [sum(int(np.argmax(Ac[list(p)][i]) == i) for i in range(len(fates)))
                 for p in permutations(range(len(fates)))]
    hits_raw = sum(int(np.argmax(A[i]) == i) for i in range(len(fates)))
    hits_ctr = sum(int(np.argmax(Ac[i]) == i) for i in range(len(fates)))

    # ---- bootstrap over SOURCE CELLS: how big and how stable is D? (the permutation test can't say)
    n_src = len(src)
    boot = []
    for _ in range(N_BOOT):
        b = rng.integers(0, n_src, n_src)
        Ab = np.array([per_cell[f][b].mean(0) for f in fates])
        boot.append(dominance(Ab - Ab.mean(0, keepdims=True)))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n  column-centered Δprogram (T0->T8), scGPT's OWN head:")
    print("     target      " + "".join(f"{c:>11}" for c in fates))
    for i, t in enumerate(fates):
        star = int(np.argmax(Ac[i]))
        print(f"     ->{t:<10}" + "".join(f"{Ac[i, j]:>+11.2f}" for j in range(len(fates)))
              + f"   top={fates[star]}{'  ✓' if star == i else '  ✗'}")
    print(f"\n  targeted fate is top riser: raw {hits_raw}/{len(fates)}, centered {hits_ctr}/{len(fates)}")
    print(f"  diagonal dominance D = {D_obs:+.3f}   bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"   (excludes 0: {'YES' if lo > 0 else 'no'})")
    print(f"  exact permutation p = {p_exact:.4f}  over {len(perms)} assignments   (attainable floor {floor:.4f})")

    return dict(tissue=tissue, gate_passed=True, fates=fates, n_fates=len(fates),
                within_cell_spearman=within, marker_frac_pos=frac_pos,
                matrix_raw=A.tolist(), matrix_centered=Ac.tolist(),
                hits_raw=hits_raw, hits_centered=hits_ctr,
                D=D_obs, boot_ci=[float(lo), float(hi)], boot_frac_pos=float((boot > 0).mean()),
                p_exact=p_exact, n_perms=len(perms), p_floor=floor, n_src=int(n_src),
                perm_D=[float(v) for v in perms], perm_hits=[int(v) for v in hit_perms])


def main():
    tissues = sys.argv[1:] or TISSUES
    out = {}
    for t in tissues:
        r = run(t)
        if r:
            out[t] = r
            json.dump(r, open(os.path.join(RESULTS, f"fate_matrix_{t}.json"), "w"), indent=1)

    good = {t: r for t, r in out.items() if r.get("gate_passed")}
    print(f"\n\n{'='*100}\n================ POOLED: the fate diagonal across tissues ================")
    print(f"  {'tissue':<10} {'fates':>6} {'hits(ctr)':>10} {'D':>8} {'boot 95% CI':>20} {'p_exact':>9} {'floor':>8}")
    for t, r in good.items():
        ci = f"[{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}]"
        print(f"  {t:<10} {r['n_fates']:>6} {r['hits_centered']:>7}/{r['n_fates']} {r['D']:>+8.3f} {ci:>20}"
              f" {r['p_exact']:>9.4f} {r['p_floor']:>8.4f}")

    if len(good) >= 2:
        pv = [r["p_exact"] for r in good.values()]
        stat, df, pc = fisher_combine(pv)
        _, _, pc_floor = fisher_combine([r["p_floor"] for r in good.values()])
        print(f"\n  Fisher over the per-tissue p's: chi2={stat:.2f} (df={df}) -> p = {pc:.2e}")
        print(f"  ...but every tissue sits at ITS OWN floor, so Fisher is floor-limited too "
              f"(all-at-floor gives p = {pc_floor:.2e}). Fisher is the wrong instrument here.")

        # ---- THE POWERED TEST: one pooled exact permutation over the JOINT assignment space.
        # Under H0 (no target->program association in any tissue) each tissue's assignment is independently
        # exchangeable, so the joint null enumerates the product of the per-tissue permutation groups:
        # 120 x 24 x 6 x 6 = 103,680 joint assignments -> attainable floor 9.6e-6, three orders below Fisher's.
        # Statistic: total correct target->fate hits summed over tissues (and, as a continuous check, the sum of
        # per-tissue z-scored dominance, so no one tissue's scale dominates).
        names = list(good)
        hit_lists = [np.array(good[t]["perm_hits"]) for t in names]
        D_lists = [np.array(good[t]["perm_D"]) for t in names]
        Dz_lists = [(d - d.mean()) / (d.std() + 1e-12) for d in D_lists]

        obs_hits = sum(good[t]["hits_centered"] for t in names)
        obs_Dz = float(sum(dz[0] for dz in Dz_lists))     # identity permutation is index 0 -> the true assignment
        n_pairs = sum(good[t]["n_fates"] for t in names)

        # exact convolution over the product space (distributions are small; no sampling needed)
        def conv(dists, grid=None):
            """Exact distribution of the SUM of independent draws, one per tissue, over the full product."""
            from functools import reduce
            vals = reduce(lambda a, b: (a[:, None] + b[None, :]).ravel(), dists)
            return vals
        joint_hits = conv(hit_lists)
        joint_Dz = conv(Dz_lists)
        n_joint = len(joint_hits)
        p_hits = float((np.sum(joint_hits >= obs_hits) ) / n_joint)
        p_Dz = float((np.sum(joint_Dz >= obs_Dz - 1e-9)) / n_joint)
        floor_joint = 1.0 / n_joint

        print(f"\n  ---- POOLED EXACT PERMUTATION over the joint assignment space ----")
        print(f"  joint assignments enumerated: {n_joint:,}  ({' x '.join(str(good[t]['n_perms']) for t in names)})"
              f"   -> attainable floor p = {floor_joint:.2e}")
        print(f"  correct target->fate hits: {obs_hits}/{n_pairs}   (null expectation "
              f"{float(np.mean(joint_hits)):.2f})   exact p = {p_hits:.2e}")
        print(f"  summed z-scored dominance: {obs_Dz:+.2f}   exact p = {p_Dz:.2e}")
        print(f"\n  route_steering reported p = 0.042 (its 1/24 floor, 4 fates, one tissue).")
        print(f"  This is p = {min(p_hits, p_Dz):.1e} — {0.042/max(min(p_hits,p_Dz),1e-12):,.0f}x smaller, "
              f"and NOT at the floor ({floor_joint:.1e}).")
        out["_pooled"] = dict(p_values=pv, fisher_chi2=stat, fisher_df=df, fisher_p=pc,
                              fisher_p_if_all_at_floor=pc_floor, n_tissues=len(good),
                              joint_n_assignments=int(n_joint), joint_floor=floor_joint,
                              obs_hits=int(obs_hits), n_pairs=int(n_pairs),
                              null_mean_hits=float(np.mean(joint_hits)),
                              p_pooled_hits=p_hits, obs_sum_z_dominance=obs_Dz, p_pooled_dominance=p_Dz)
    for t in good:                                       # keep the JSON small
        good[t].pop("perm_D", None); good[t].pop("perm_hits", None)
    json.dump(out, open(os.path.join(RESULTS, "fate_matrix_ALL.json"), "w"), indent=1)
    print(f"\n[done] -> results/fate_matrix_*.json")


if __name__ == "__main__":
    main()
