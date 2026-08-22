"""Does the on-manifold steered path decode to REAL, ORDERED gene biology -- or to a smear?

This is the make-or-break test for the steering result. tangent_diagnostic.py + local_steering.py showed that
a constant linear direction + local-tangent projection walks a progenitor's embedding forward along the
trajectory while staying on the real-cell manifold. But "advance" was judged by a kNN in embedding space; it
could be a geometric triviality (drifting toward the centroid of later cells) with no coherent gene logic.

Here we walk held-out progenitor cells forward, decode each step back to gene expression (a linear decoder fit
on TRAIN real cells), and ask whether the swept gene program matches known hematopoiesis:

  1. MARKER PROGRAMS   stem (CD34/HLF/AVP/...) should FALL; a committed program (erythroid / myeloid / ...)
                       should RISE. Which one rises tells us the fate the model defaults to under a forward push.
  2. ORDERING FIDELITY cosine between the steered early->late decoded gene-change and the REAL early->late
                       change (mean late minus mean early real cells), over highly-variable genes.
  3. ON- vs OFF-MANIFOLD  decode the path point, compare to the mean expression of REAL cells at its decoded
                       pseudotime. On-manifold (linear_proj) should track real cells; the off-manifold control
                       (plain linear) should diverge in gene space even though it also "advances".
  4. ENDPOINT IDENTITY the steered endpoint's nearest real cells -- a coherent committed cluster (a real cell
                       type) or a void / mixture (a hallucinated cell)?

Honest caveat: the decoder is fit on real cells, so decoded expression near a real cell ~ that cell's
expression. The non-trivial content is therefore (i) the ORDER of the gene program along a geometry-generated
path, and (ii) the on- vs off-manifold CONTRAST (same decoder; only the path differs). Not the endpoint values.

Run:  ../../.venv/bin/python gene_decode.py [scgpt|geneformer] [dataset=setty]
Out:  results/gene_decode_<model>_<dataset>.json  (+ printed report)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
import h5py
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tangent_diagnostic import path as npz_path, zscore, unit, steer, SEED, DIM, BUDGETS, T_STAR  # noqa: E402
from local_steering import project_constant  # noqa: E402

ROOT = f"{_DATA}"
H5AD = os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad")
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

PANEL = {
    "stem": ["CD34", "AVP", "HLF", "CRHBP", "MEIS1", "HOXA9", "MLLT3", "SPINK2", "PRSS57"],
    "ery": ["GATA1", "KLF1", "GYPA", "HBB", "HBD", "TFRC", "CD36", "GATA2", "APOC1"],
    "myeloid": ["SPI1", "CEBPA", "MPO", "ELANE", "LYZ", "CD14", "CSF1R", "AZU1"],
    "lymphoid": ["IL7R", "JCHAIN", "EBF1", "VPREB1"],
    "mega": ["PF4", "PPBP", "ITGA2B", "VWF"],
}
N_HVG = 2000


def load_expression(cell_idx):
    """CP10k + log1p expression for the given h5ad rows; returns (E[n,g], gene_symbols, clusters)."""
    with h5py.File(H5AD, "r") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"]["index"][:]])
        cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["__categories"]["clusters"][:]]
        clu_codes = f["obs"]["clusters"][:]
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        g = len(genes)
        E = np.zeros((len(cell_idx), g), dtype=np.float32)
        for r, i in enumerate(cell_idx):
            s, e = ip[i], ip[i + 1]
            E[r, ind[s:e]] = dat[s:e]
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    E = np.log1p(E / tot * 1e4)
    clusters = np.array([cats[c] for c in clu_codes])[cell_idx]
    return E, genes, clusters


def program_score(E, gene_idx, mu, sd):
    """Mean z-scored (against real-cell stats) expression over a program's genes."""
    return ((E[:, gene_idx] - mu[gene_idx]) / sd[gene_idx]).mean(1)


def build_spaces(emb, dim=DIM):
    """Steering space (dim-PC whitened) plus the maps needed to DECODE from the full embedding.
    Steering runs in Xz; a path point pz is reconstructed to full standardized-embedding space (where the
    higher-fidelity gene decoder lives) via inverse-whiten -> inverse-PCA -> standardize."""
    mean = emb.mean(0)
    pca = PCA(min(dim, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    Xr = pca.transform(emb - mean)
    wsc = StandardScaler().fit(Xr)
    Xz = wsc.transform(Xr)
    esc = StandardScaler().fit(emb)                       # decoder is fit on full standardized embedding
    E_std = esc.transform(emb)

    def to_full_std(pz):                                  # whitened dim-PC point -> full standardized emb
        return esc.transform(pca.inverse_transform(wsc.inverse_transform(pz)) + mean)
    return Xz, E_std, to_full_std


def run(model, sfx="setty"):
    p = npz_path(model, sfx)
    z = np.load(p, allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int)
    ok = np.isfinite(y); emb, y, ci = emb[ok], y[ok], ci[ok]
    E, genes, clusters = load_expression(ci)
    gsym = {s: i for i, s in enumerate(genes)}
    panel_idx = {k: np.array([gsym[s] for s in v if s in gsym]) for k, v in PANEL.items()}

    Xz, E_std, to_full_std = build_spaces(emb)       # steer in 20-PC whitened; decode from full embedding
    yz = zscore(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]

    # decoder: FULL standardized embedding -> log-normalized expression, fit on TRAIN cells.
    # alpha CV-picked on a held-out slice of train (d can exceed n_tr for Geneformer -> needs heavy shrinkage).
    marker_all = np.unique(np.concatenate(list(panel_idx.values())))
    va = tr[: len(tr) // 5]; fit = tr[len(tr) // 5:]
    best_a, best = 10.0, -np.inf
    for a in (10.0, 100.0, 1000.0, 1e4, 1e5):
        m = Ridge(alpha=a).fit(E_std[fit], E[fit]).predict(E_std[va])
        s = np.median([r2_score(E[va, j], m[:, j]) for j in marker_all])
        if s > best:
            best, best_a = s, a
    dec = Ridge(alpha=best_a).fit(E_std[tr], E[tr])
    decode = lambda pz: dec.predict(to_full_std(pz))     # noqa: E731  (whitened path point -> expression)
    Epred_te = dec.predict(E_std[te])
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6
    var = E[tr].var(0)
    hvg = np.argsort(-var)[:N_HVG]
    decode_r2_markers = float(np.median([r2_score(E[te, j], Epred_te[:, j]) for j in marker_all]))
    decode_r2_hvg = float(np.median([r2_score(E[te, j], Epred_te[:, j]) for j in hvg]))

    # real early->late direction (actual expression), and real mean-expr as a function of pseudotime
    lo, hi = np.quantile(yz[tr], 1 / 3), np.quantile(yz[tr], 2 / 3)
    real_dir = E[tr][yz[tr] >= hi].mean(0) - E[tr][yz[tr] <= lo].mean(0)
    order = np.argsort(yz[tr]); pts_sorted = yz[tr][order]; E_sorted = E[tr][order]
    def real_expr_at(pt):                            # nearest-in-pseudotime real mean (window of 60 cells)
        j = np.searchsorted(pts_sorted, pt); a, b = max(0, j - 30), min(len(order), j + 30)
        return E_sorted[a:b].mean(0)

    judge = KNeighborsRegressor(15).fit(Xz[tr], yz[tr])
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xz[tr])
    nn_id = NearestNeighbors(n_neighbors=15).fit(Xz[tr])
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))
    w_lin = unit(Ridge(alpha=10.0).fit(Xz[tr], yz[tr]).coef_)
    clu_tr = clusters[tr]

    src_mask = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = Xz[src_mask]
    if len(src) > 200:
        src = src[rng.choice(len(src), 200, replace=False)]

    rdir = unit(rng.standard_normal(Xz.shape[1]))                    # fixed random direction (neg. control)
    fields = {"on_manifold": project_constant(Xz[tr], w_lin),        # linear_proj (the method)
              "off_manifold": lambda x: np.tile(w_lin, (len(x), 1)),  # plain linear (drifts off-manifold)
              "random_dir": lambda x: np.tile(rdir, (len(x), 1))}     # negative control: no developmental info

    out = dict(model=model, dataset=sfx, n=int(len(y)), n_src=int(len(src)),
               decode_r2_markers=decode_r2_markers, decode_r2_hvg=decode_r2_hvg, rules={})
    print(f"\n===== {model} / {sfx}  gene decode  (n={len(y)}, src={len(src)}, "
          f"decode R2: markers={decode_r2_markers:.2f} HVG={decode_r2_hvg:.2f}) =====")

    for name, fn in fields.items():
        pth = steer(src, fn, 0.25 * d0, BUDGETS)
        rec = dict(programs={k: [] for k in PANEL}, decoded_pt=[], gene_traj_err=[],
                   real_fidelity=[], off_ratio=[])
        e0 = decode(pth[0])
        for T in BUDGETS:
            eT = decode(pth[T])
            for k, idx in panel_idx.items():
                rec["programs"][k].append(float(program_score(eT, idx, mu, sd).mean()))
            pT = judge.predict(pth[T])
            rec["decoded_pt"].append(float(pT.mean()))
            r_at = np.array([real_expr_at(v) for v in pT])                      # real expr at matched pt
            num = np.linalg.norm((eT - r_at)[:, hvg], axis=1)
            den = np.linalg.norm((r_at - e0)[:, hvg], axis=1) + 1e-9
            rec["gene_traj_err"].append(float(np.median(num / den)))
            # gene-space FIDELITY: does the decoded steered cell look like a REAL cell of its decoded age?
            fid = [np.corrcoef(eT[i, hvg], r_at[i, hvg])[0, 1] for i in range(len(eT))]
            rec["real_fidelity"].append(float(np.nanmean(fid)))
            rec["off_ratio"].append(float(nn1.kneighbors(pth[T])[0][:, 0].mean() / d0))
        # alignment with the GLOBAL developmental axis (both real rules should be high; ~0 = no dev signal)
        eTs = decode(pth[T_STAR])
        chg = (eTs - e0)[:, hvg]; rd = real_dir[hvg]
        cos = float(np.mean([(c @ rd) / (np.linalg.norm(c) * np.linalg.norm(rd) + 1e-9) for c in chg]))
        # endpoint identity: nearest real cells' dominant cluster + purity
        idx_id = nn_id.kneighbors(pth[T_STAR])[1]
        doms, purs = [], []
        for row in idx_id:
            labs, cnts = np.unique(clu_tr[row], return_counts=True)
            doms.append(labs[cnts.argmax()]); purs.append(cnts.max() / cnts.sum())
        dl, dc = np.unique(doms, return_counts=True)
        rec.update(dev_axis_cosine=cos, endpoint_purity=float(np.mean(purs)),
                   endpoint_clusters={l: int(c) for l, c in zip(dl, dc)},
                   off_ratio_T8=rec["off_ratio"][T_STAR], gene_traj_err_T8=rec["gene_traj_err"][T_STAR],
                   real_fidelity_T8=rec["real_fidelity"][T_STAR])
        out["rules"][name] = rec

        P = rec["programs"]
        print(f"\n  [{name}]  dev-axis cosine={cos:+.2f} | real-cell fidelity @T8={rec['real_fidelity_T8']:+.2f}"
              f" | gene-traj err @T8={rec['gene_traj_err_T8']:.2f} | off-manifold ratio @T8={rec['off_ratio_T8']:.2f}")
        print(f"     program score  T0 -> T8 :  " + "  ".join(
            f"{k}={P[k][0]:+.2f}->{P[k][T_STAR]:+.2f}" for k in PANEL))
        top = sorted(rec["endpoint_clusters"].items(), key=lambda kv: -kv[1])[:3]
        print(f"     endpoint nearest-real clusters: {top}  (mean purity {rec['endpoint_purity']:.2f})")

    # ---------------- fate-biased steering: aim at each terminal branch, read the decoded program ----------------
    # target direction = toward the (on-manifold-projected) centroid of a terminal cluster; does the TARGETED
    # fate's marker program rise the most? A clean diagonal = the model can be steered to a CHOSEN fate.
    FATE_CLUSTERS = {"ery": ["Ery_1", "Ery_2"], "myeloid": ["Mono_1", "Mono_2"],
                     "lymphoid": ["CLP"], "mega": ["Mega"]}
    fate_mat = {}
    src_c = src.mean(0)
    for fate, cls in FATE_CLUSTERS.items():
        tgt_mask = np.isin(clu_tr, cls)
        if tgt_mask.sum() < 20:
            continue
        w_f = unit(Xz[tr][tgt_mask].mean(0) - src_c)                # progenitor -> fate centroid
        pth = steer(src, project_constant(Xz[tr], w_f), 0.25 * d0, BUDGETS)   # on-manifold projected
        e0f, eTf = decode(pth[0]), decode(pth[T_STAR])
        deltas = {k: float(program_score(eTf, panel_idx[k], mu, sd).mean()
                           - program_score(e0f, panel_idx[k], mu, sd).mean()) for k in PANEL}
        fate_mat[fate] = deltas
    out["fate_steering"] = fate_mat
    if fate_mat:
        print(f"\n  ---- fate-biased steering (Δprogram score T0->T8; row=target fate, want diagonal) ----")
        cols = list(PANEL.keys())
        print("     target    " + "".join(f"{c:>10}" for c in cols))
        hits = 0
        for fate, d in fate_mat.items():
            star = max(d, key=d.get)
            hits += (star == fate)
            print(f"     ->{fate:<8} " + "".join(f"{d[c]:>+10.2f}" for c in cols)
                  + f"   top={star}{'  ✓' if star == fate else ''}")
        print(f"     targeted fate is the top riser in {hits}/{len(fate_mat)} branches")
        out["fate_steering_hits"] = f"{hits}/{len(fate_mat)}"

    json.dump(out, open(os.path.join(RESULTS, f"gene_decode_{model}_{sfx}.json"), "w"), indent=1)
    # verdict
    onm, off, rnd = out["rules"]["on_manifold"], out["rules"]["off_manifold"], out["rules"]["random_dir"]
    print(f"\n  ---- summary ({model}/{sfx}) ----")
    st = onm['programs']['stem']
    print(f"    stem program (on-manifold): {st[0]:+.2f} -> {st[T_STAR]:+.2f}  "
          f"(random-dir control: {rnd['programs']['stem'][0]:+.2f} -> {rnd['programs']['stem'][T_STAR]:+.2f})")
    lineages = {k: onm['programs'][k][T_STAR] - onm['programs'][k][0] for k in ['ery', 'myeloid', 'lymphoid', 'mega']}
    fate = max(lineages, key=lineages.get)
    print(f"    default fate under forward push: {fate} (+{lineages[fate]:.2f})  | all: "
          + ", ".join(f"{k}{v:+.2f}" for k, v in lineages.items()))
    print(f"    dev-axis cosine   on={onm['dev_axis_cosine']:+.2f}  off={off['dev_axis_cosine']:+.2f}  rand={rnd['dev_axis_cosine']:+.2f}")
    print(f"    real-cell fidelity@T8  on={onm['real_fidelity_T8']:+.2f}  off={off['real_fidelity_T8']:+.2f}  rand={rnd['real_fidelity_T8']:+.2f}")
    print(f"    endpoint purity   on={onm['endpoint_purity']:.2f}  off={off['endpoint_purity']:.2f}  rand={rnd['endpoint_purity']:.2f}")
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "scgpt"
    sfx = sys.argv[2] if len(sys.argv) > 2 else "setty"
    run(model, sfx)
    print(f"\n[done] -> results/gene_decode_{model}_{sfx}.json")


if __name__ == "__main__":
    main()
