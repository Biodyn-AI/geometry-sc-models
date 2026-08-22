"""H-A2 — read in-silico perturbation shifts ON-MANIFOLD; does it rescue the driver-nomination negative?

PERTURB_INVERT is a negative: cos(perturbation shift, on-manifold fate direction) nominates known drivers no
better than differential expression (nothing survives BH). But the fate direction is ALREADY on-manifold, so
the raw cosine equals the tangent overlap DILUTED by the shift's off-manifold magnitude:

    cos(dz, d_f) = [ tangent(dz).d_f + normal(dz).d_f ] / |dz|      and normal(dz).d_f ~ 0  (d_f is on-manifold)
                 = ( |tangent(dz)| / |dz| ) * cos(tangent(dz), d_f)

so a perturbation with a big OFF-manifold (biologically nonsensical) shift plus a small on-manifold fate-aligned
component is penalised by the |dz| denominator. Two on-manifold readings:

  TANGENT   score = cos(tangent(dz), d_f)   -- remove the off-manifold dilution (normalise by tangent magnitude)
  PLAUSIBLE normal_frac = |normal(dz)| / |dz|  -- a real driver should move the cell ALONG the manifold (low
            normal_frac); a nonsense perturbation knocks it off (high). Use -normal_frac as a driver score and
            as a gate on the tangent score.

If TANGENT (or a normal-gated tangent) lifts driver AUROC past DE and survives residualisation, the negative is
rescued and the fix generalises to any perturbation-shift method. If not, the negative hardens: the model does
not know drivers even when its shifts are read purely on-manifold.

Cached shifts only (maxtoki, state); scGPT shifts are computed live in perturb_invert.py (recompute optional).

Run:  ../../.venv/bin/python ha2_onmanifold_perturb.py [model ...]   (default: maxtoki state)
Out:  results/ha2_onmanifold_perturb.json
"""
import os, sys, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from tangent_diagnostic import unit, DIM  # noqa: E402
from local_steering import project_constant, K_LOCAL, M_TANGENT  # noqa: E402
from perturb_invert import FATE_CLUSTERS, DRIVERS, auroc, auroc_perm_p  # noqa: E402
from score_shifts import bh, load_expr, EMB, SHIFTS  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)


def local_tangent_bases(Xtr, pts, k=K_LOCAL, m=M_TANGENT):
    """Top-m local PCs (the tangent basis) at each query point, from its k nearest train cells."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    _, idx = nn.kneighbors(pts)
    Vs = []
    for i in range(len(pts)):
        Xc = Xtr[idx[i]] - Xtr[idx[i]].mean(0)
        Vs.append(np.linalg.svd(Xc, full_matrices=False)[2][:m])       # (m, d)
    return Vs


def run(model):
    z = np.load(EMB[model], allow_pickle=True)
    s = np.load(SHIFTS[model], allow_pickle=True)
    y = z["pseudotime"].astype(np.float64); ok = np.isfinite(y)
    emb = z["emb"].astype(np.float64)[ok]; ci = z["cell_idx"].astype(int)[ok]; clusters = z["clusters"][ok]
    tfs = [str(t) for t in s["tfs"]]
    src, tr = s["src_rows"], s["tr_rows"]
    base, pert = s["base"].astype(np.float64), s["pert"].astype(np.float64)

    mean = emb.mean(0)
    pca = PCA(min(DIM, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    wsc = StandardScaler().fit(pca.transform(emb - mean))
    to_z = lambda E: wsc.transform(pca.transform(E - mean))       # noqa: E731
    Xz = to_z(emb); clu_tr = clusters[tr]; src_c = Xz[src].mean(0)

    # on-manifold fate directions per source cell (identical to score_shifts)
    dirs, fl0 = {}, []
    for f, cls in FATE_CLUSTERS.items():
        m = np.isin(clu_tr, cls)
        if m.sum() < 20:
            continue
        w = unit(Xz[tr][m].mean(0) - src_c)
        D = project_constant(Xz[tr], w)(Xz[src]); dirs[f] = D / (np.linalg.norm(D, 1, keepdims=True) + 1e-12)
        fl0.append(f)
    Dmean = np.mean([dirs[f] for f in fl0], axis=0)
    spec = {f: (dirs[f] - Dmean) / (np.linalg.norm(dirs[f] - Dmean, axis=1, keepdims=True) + 1e-12) for f in fl0}

    # local tangent basis at each source cell (from train cells)
    Vs = local_tangent_bases(Xz[tr], Xz[src])

    bz = to_z(base)
    raw = {f: np.zeros(len(tfs)) for f in fl0}
    tan = {f: np.zeros(len(tfs)) for f in fl0}
    normfrac = np.zeros(len(tfs))                        # mean over source cells of |normal|/|dz|
    for k in range(len(tfs)):
        dz = to_z(pert[k]) - bz                          # (n_src, d)
        nf = np.zeros(len(src))
        dz_tan = np.zeros_like(dz)
        for i in range(len(src)):
            t = Vs[i].T @ (Vs[i] @ dz[i])                # tangent component
            dz_tan[i] = t
            nf[i] = np.linalg.norm(dz[i] - t) / (np.linalg.norm(dz[i]) + 1e-12)
        normfrac[k] = float(nf.mean())
        dz_u = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
        dzt_u = dz_tan / (np.linalg.norm(dz_tan, axis=1, keepdims=True) + 1e-12)
        for f in fl0:
            raw[f][k] = float((dz_u * spec[f]).sum(1).mean())
            tan[f][k] = float((dzt_u * spec[f]).sum(1).mean())

    # DE baseline (same common-mode removal as score_shifts)
    E, genes = load_expr(ci); gsym = {g: i for i, g in enumerate(genes)}
    src_mean = E[src].mean(0)
    de0 = {f: np.array([E[np.isin(clusters, FATE_CLUSTERS[f])][:, gsym[t]].mean() - src_mean[gsym[t]]
                        for t in tfs]) for f in fl0}
    dm = np.mean([de0[f] for f in fl0], axis=0); de = {f: de0[f] - dm for f in fl0}
    drivers = {f: [d for d in DRIVERS[f] if d in tfs] for f in fl0}

    out = dict(model=model, n_src=int(len(src)), n_tfs=len(tfs), drivers=drivers,
               normfrac_summary=dict(mean=float(normfrac.mean()), min=float(normfrac.min()),
                                     max=float(normfrac.max())), fates={})
    print(f"\n===== {model.upper()}  on-manifold perturbation rescore (n_src={len(src)}, {len(tfs)} TFs) =====")
    print(f"  shift normal-fraction |normal|/|dz|: mean={normfrac.mean():.2f} "
          f"[{normfrac.min():.2f},{normfrac.max():.2f}]  (higher = more off-manifold)")
    print(f"  {'fate':<9} {'raw':>6} {'TAN':>6} {'-nf':>6} {'gate':>6} | {'DE':>6} | {'TAN|DE':>7} {'q':>6} | {'DE+TAN':>7}")
    rows = []
    for f in fl0:
        pos = np.array([t in drivers[f] for t in tfs])
        a_raw = auroc(raw[f], pos)
        a_tan, p_tan = auroc_perm_p(tan[f], pos)
        a_nf = auroc(-normfrac, pos)                                  # low normal-fraction = plausible driver
        # normal-gated tangent: tangent alignment weighted toward on-manifold shifts
        gate = tan[f] * (1.0 - normfrac)
        a_gate = auroc(gate, pos)
        a_de = auroc(de[f], pos)
        res = tan[f] - LinearRegression().fit(de[f][:, None], tan[f]).predict(de[f][:, None])
        a_rs, p_rs = auroc_perm_p(res, pos)
        zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)             # noqa: E731
        a_cb = auroc(zs(de[f]) + zs(tan[f]), pos)
        r_rt = float(np.corrcoef(raw[f], tan[f])[0, 1])
        out["fates"][f] = dict(n_drivers=int(pos.sum()), auroc_raw=a_raw, auroc_tangent=a_tan, p_tangent=p_tan,
                               auroc_neg_normfrac=a_nf, auroc_gate=a_gate, auroc_de=a_de,
                               auroc_tangent_resid_de=a_rs, p_tangent_resid_de=p_rs,
                               auroc_de_plus_tangent=a_cb, r_raw_tangent=r_rt)
        rows.append((f, a_raw, a_tan, a_nf, a_gate, a_de, a_rs, p_rs, a_cb))
    q_tan = bh([out["fates"][f]["p_tangent"] for f in fl0])
    q_rs = bh([out["fates"][f]["p_tangent_resid_de"] for f in fl0])
    for i, (f, a_raw, a_tan, a_nf, a_gate, a_de, a_rs, p_rs, a_cb) in enumerate(rows):
        out["fates"][f]["q_tangent"] = float(q_tan[i]); out["fates"][f]["q_tangent_resid_de"] = float(q_rs[i])
        print(f"  {f:<9} {a_raw:>6.3f} {a_tan:>6.3f} {a_nf:>6.3f} {a_gate:>6.3f} | {a_de:>6.3f} | "
              f"{a_rs:>7.3f} {q_rs[i]:>6.3f} | {a_cb:>7.3f}")

    beats_de = [f for f in fl0 if out["fates"][f]["auroc_tangent"] > out["fates"][f]["auroc_de"]]
    tan_gt_raw = [f for f in fl0 if out["fates"][f]["auroc_tangent"] > out["fates"][f]["auroc_raw"] + 0.02]
    sig_resid = [f for f in fl0 if out["fates"][f]["q_tangent_resid_de"] < 0.05]
    out["verdict"] = dict(fates_tangent_beats_DE=beats_de, fates_tangent_gt_raw=tan_gt_raw,
                          fates_sig_after_resid_bh=sig_resid,
                          mean_auroc_raw=float(np.mean([out["fates"][f]["auroc_raw"] for f in fl0])),
                          mean_auroc_tangent=float(np.mean([out["fates"][f]["auroc_tangent"] for f in fl0])),
                          mean_auroc_de=float(np.mean([out["fates"][f]["auroc_de"] for f in fl0])),
                          mean_auroc_neg_normfrac=float(np.mean([out["fates"][f]["auroc_neg_normfrac"] for f in fl0])))
    v = out["verdict"]
    print(f"  mean AUROC: raw={v['mean_auroc_raw']:.3f}  tangent={v['mean_auroc_tangent']:.3f}  "
          f"DE={v['mean_auroc_de']:.3f}  -normfrac={v['mean_auroc_neg_normfrac']:.3f}")
    print(f"  tangent beats DE in {len(beats_de)}/{len(fl0)}; tangent>raw in {len(tan_gt_raw)}/{len(fl0)}; "
          f"survives resid|DE+BH in {len(sig_resid)}/{len(fl0)}")
    return out


def main():
    models = sys.argv[1:] or ["maxtoki", "state"]
    out = {}
    for m in models:
        try:
            out[m] = run(m)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[skip] {m}: {e}")
    json.dump(out, open(os.path.join(RESULTS, "ha2_onmanifold_perturb.json"), "w"), indent=1)
    print("\n================ H-A2 VERDICT ================")
    print("  On-manifold (tangent) reading of perturbation shifts vs the raw negative and the DE baseline:")
    for m, r in out.items():
        v = r["verdict"]
        print(f"  {m:<9} raw={v['mean_auroc_raw']:.3f} tangent={v['mean_auroc_tangent']:.3f} "
              f"DE={v['mean_auroc_de']:.3f} | tangent>DE in {len(v['fates_tangent_beats_DE'])}, "
              f"resid-sig in {len(v['fates_sig_after_resid_bh'])}")
    print("[done] -> results/ha2_onmanifold_perturb.json")


if __name__ == "__main__":
    main()

