"""Score in-silico TF-activation shifts against known lineage drivers — ONE scorer, all models.

Consumes the per-model dumps (dump_shifts_{maxtoki,state}.py, and scGPT via perturb_invert.py) and applies the
IDENTICAL statistics to each, so the cross-model comparison is apples-to-apples. Everything here is the analysis
that perturb_invert.py already validated on scGPT:

  - fate directions in the model's OWN 20-PC whitened embedding space (progenitor -> fate centroid, on-manifold
    via the local-tangent projection), then made FATE-SPECIFIC by orthogonalising each against the mean of the
    four (the bifurcation contrast) -- without this the four directions share a huge "generic differentiation"
    axis and every fate nominates the same TFs.
  - score(g, f) = mean over source cells of cos(whitened shift from forcing g high, fate-specific direction f)
  - DE baseline de(g, f) = mean log-expr in fate cluster - mean in source progenitors, with the SAME
    common-mode removal, so model and baseline are judged on the same footing.
  - AUROC of the pre-registered driver set, permutation p, BH-FDR across the 4 fates, DE baseline,
    residual-on-DE, DE+model, and a random-direction control.

Usage: ../../.venv/bin/python score_shifts.py <model>
       model in {maxtoki, state}  (scGPT is scored in-place by perturb_invert.py)
Out:   results/perturb_invert_<model>_setty.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tangent_diagnostic import unit, SEED, DIM  # noqa: E402
from local_steering import project_constant  # noqa: E402
from perturb_invert import FATE_CLUSTERS, DRIVERS, auroc, auroc_perm_p  # noqa: E402

PROJ = f"{_DATA}"
SETTY = f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad"
RESULTS = os.path.join(HERE, "results")
EMB = {"maxtoki": f"{PROJ}/data/maxtoki_acts/maxtoki_setty.npz",
       "state": f"{PROJ}/data/branchpoint/state_setty.npz"}
SHIFTS = {m: f"{PROJ}/data/branchpoint/shifts_{m}_setty.npz" for m in EMB}


def bh(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); n = len(ps); q = np.empty(n); prev = 1.0
    for r in range(n - 1, -1, -1):
        i = o[r]; prev = min(prev, ps[i] * n / (r + 1)); q[i] = prev
    return q


def load_expr(cell_idx):
    with h5py.File(SETTY, "r") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"]["index"][:]])
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        G = len(genes)
        E = np.zeros((len(cell_idx), G), np.float32)
        for r, i in enumerate(cell_idx):
            a, b = int(ip[i]), int(ip[i + 1])
            E[r, ind[a:b]] = dat[a:b]
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    return np.log1p(E / tot * 1e4), genes


def run(model):
    z = np.load(EMB[model], allow_pickle=True)
    s = np.load(SHIFTS[model], allow_pickle=True)
    y = z["pseudotime"].astype(np.float64)
    ok = np.isfinite(y)
    emb = z["emb"].astype(np.float64)[ok]; ci = z["cell_idx"].astype(int)[ok]; clusters = z["clusters"][ok]
    y = y[ok]
    tfs = [str(t) for t in s["tfs"]]
    src, tr = s["src_rows"], s["tr_rows"]
    base, pert = s["base"].astype(np.float64), s["pert"].astype(np.float64)

    # whitened 20-PC space (identical construction to perturb_invert.py)
    mean = emb.mean(0)
    pca = PCA(min(DIM, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    wsc = StandardScaler().fit(pca.transform(emb - mean))
    to_z = lambda E: wsc.transform(pca.transform(E - mean))       # noqa: E731
    Xz = to_z(emb)
    clu_tr = clusters[tr]
    src_c = Xz[src].mean(0)

    dirs, fl0 = {}, []
    for f, cls in FATE_CLUSTERS.items():
        m = np.isin(clu_tr, cls)
        if m.sum() < 20:
            print(f"  [skip fate {f}: only {m.sum()} train cells]")
            continue
        w = unit(Xz[tr][m].mean(0) - src_c)
        D = project_constant(Xz[tr], w)(Xz[src])
        dirs[f] = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        fl0.append(f)
    Dmean = np.mean([dirs[f] for f in fl0], axis=0)
    spec = {}
    for f in fl0:
        S = dirs[f] - Dmean
        spec[f] = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(SEED)
    Dr = project_constant(Xz[tr], unit(rng.standard_normal(Xz.shape[1])))(Xz[src])
    Dr = Dr / (np.linalg.norm(Dr, axis=1, keepdims=True) + 1e-12)

    # alignment scores
    bz = to_z(base)
    align = {f: np.zeros(len(tfs)) for f in list(spec) + ["_random"]}
    for k in range(len(tfs)):
        dz = to_z(pert[k]) - bz
        dz = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
        for f in spec:
            align[f][k] = float((dz * spec[f]).sum(1).mean())
        align["_random"][k] = float((dz * Dr).sum(1).mean())

    # DE baseline, same common-mode removal
    E, genes = load_expr(ci)
    gsym = {g: i for i, g in enumerate(genes)}
    src_mean = E[src].mean(0)
    de0 = {f: np.array([E[np.isin(clusters, FATE_CLUSTERS[f])][:, gsym[t]].mean() - src_mean[gsym[t]]
                        for t in tfs]) for f in fl0}
    dm = np.mean([de0[f] for f in fl0], axis=0)
    de = {f: de0[f] - dm for f in fl0}

    drivers = {f: [d for d in DRIVERS[f] if d in tfs] for f in fl0}
    out = dict(model=model, n_src=int(len(src)), n_tfs=len(tfs), drivers=drivers, fates={})
    rows = []
    for f in fl0:
        pos = np.array([t in drivers[f] for t in tfs])
        a_al, p_al = auroc_perm_p(align[f], pos)
        a_de = auroc(de[f], pos)
        res = align[f] - LinearRegression().fit(de[f][:, None], align[f]).predict(de[f][:, None])
        a_rs, p_rs = auroc_perm_p(res, pos)
        zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)          # noqa: E731
        a_cb = auroc(zs(de[f]) + zs(align[f]), pos)
        a_rd = auroc(align["_random"], pos)
        rr = float(np.corrcoef(align[f], de[f])[0, 1])
        order = np.argsort(-align[f])
        out["fates"][f] = dict(n_drivers=len(drivers[f]), auroc_align=a_al, p_align=p_al,
                               auroc_de=a_de, auroc_align_resid_de=a_rs, p_align_resid_de=p_rs,
                               auroc_de_plus_align=a_cb, auroc_random_dir=a_rd, r_align_de=rr,
                               top10=[tfs[i] for i in order[:10]],
                               driver_ranks={t: int(np.where(order == tfs.index(t))[0][0]) + 1
                                             for t in drivers[f]})
        rows.append((f, a_al, p_al, a_rs, p_rs, a_de, a_cb, a_rd, rr))

    q_al = bh([r[2] for r in rows]); q_rs = bh([r[4] for r in rows])
    for i, f in enumerate(fl0):
        out["fates"][f]["q_align"] = float(q_al[i])
        out["fates"][f]["q_align_resid_de"] = float(q_rs[i])

    print(f"\n===== {model.upper()}  in-silico TF activation -> driver recovery "
          f"(n_src={len(src)}, {len(tfs)} TFs) =====")
    print(f"  {'fate':<9} {'AUROC':>7} {'p':>6} {'q(BH)':>7} | {'resid|DE':>9} {'q(BH)':>7} | "
          f"{'AUROC de':>9} | {'DE+model':>9} | {'rand':>6} | {'r(al,de)':>9}")
    for i, (f, a_al, p_al, a_rs, p_rs, a_de, a_cb, a_rd, rr) in enumerate(rows):
        print(f"  {f:<9} {a_al:>7.3f} {p_al:>6.3f} {q_al[i]:>7.3f} | {a_rs:>9.3f} {q_rs[i]:>7.3f} | "
              f"{a_de:>9.3f} | {a_cb:>9.3f} | {a_rd:>6.3f} | {rr:>9.3f}")
    for f in fl0:
        r = out["fates"][f]
        print(f"    ->{f:<9} top10: " + ", ".join(f"**{t}**" if t in drivers[f] else t for t in r["top10"]))
        print("       drivers at ranks: "
              + ", ".join(f"{t}={rk}" for t, rk in sorted(r["driver_ranks"].items(), key=lambda kv: kv[1])))

    A = np.array([[auroc(align[f], np.array([t in drivers[g] for t in tfs])) for g in fl0] for f in fl0])
    hits = sum(fl0[int(np.argmax(A[i]))] == fl0[i] for i in range(len(fl0)))
    out["diagonal"] = dict(matrix=A.tolist(), fates=fl0, hits=f"{hits}/{len(fl0)}")
    print("\n  fate-direction x driver-set AUROC (row=aimed at, col=whose drivers):")
    print("     dir       " + "".join(f"{g:>10}" for g in fl0))
    for i, f in enumerate(fl0):
        star = fl0[int(np.argmax(A[i]))]
        print(f"     ->{f:<8} " + "".join(f"{A[i, j]:>10.3f}" for j in range(len(fl0)))
              + f"   top={star}{'  ✓' if star == f else ''}")
    print(f"     own drivers score highest in {hits}/{len(fl0)} fates")

    sig = [f for f in fl0 if out["fates"][f]["q_align"] < 0.05]
    beats = [f for f in fl0 if out["fates"][f]["auroc_align"] > out["fates"][f]["auroc_de"]]
    out["verdict"] = dict(n_fates_sig_after_bh=len(sig), fates_beating_DE=beats)
    print(f"\n  VERDICT: {len(sig)}/{len(fl0)} fates significant after BH; "
          f"{len(beats)}/{len(fl0)} fates where the model BEATS the DE baseline.")

    json.dump(out, open(os.path.join(RESULTS, f"perturb_invert_{model}_setty.json"), "w"), indent=1)
    print(f"[done] -> results/perturb_invert_{model}_setty.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "maxtoki")
