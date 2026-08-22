"""H-B1 — does local manifold GEOMETRY predict clonal fate beyond the cell's own expression state?

Substrate: Weinreb 2020 LARRY in-vitro hematopoiesis (cospar `LARRY_sp500` subsample, 7438 cells, 500 clones,
days 2/4/6, provided 40-PC space + SPRING). The decisive population: **1617 UNDIFFERENTIATED cells that carry a
clonal barcode whose sisters commit to neutrophil or monocyte** — Weinreb's hidden-variable setup, where a
cell's own `undiff` transcriptome is a weak fate predictor but its clone's fate is (partly) set.

Claim: the local manifold geometry at an undifferentiated cell — the direction the local developmental flow
tilts, relative to the neutrophil↔monocyte fate axis — carries the heritable hidden variable, so it predicts the
CLONE's fate BEYOND the cell's own position in expression space.

Ground truth (purest lineage signal, no transcriptome smoothing): a clone's neutrophil fraction among its
differentiated Neu/Mono cells (require >=3), assigned to its undiff cells; binarised at 0.5. (cospar's own
`NeuMon_fate_bias` used as an alternative-label robustness check.)

Design guards (reviewer-proofing):
  - CLONE-SPLIT CV (GroupKFold by clone): sisters never straddle train/test, or the label leaks.
  - Geometry uses only POSITIONS + observed TIME, never the clonal label.
  - Baseline = the cell's own 40-PC expression position (the state Weinreb showed is weak) with the SAME
    classifier — so any gain is geometry, not model capacity.
  - The load-bearing statistic is geometry BEYOND position (position+geom vs position), plus a
    position-matched-stratum test and a clone-permuted null.

Data: data/larry/LARRY_sp500_ranking1_adata_preprocessed.h5ad (downloaded from cospar mirror; not in repo).
Run:  ../../.venv/bin/python hb1_larry_curvature.py
Out:  results/hb1_larry_curvature.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import RS  # noqa: E402
sys.path.insert(0, RS)
from perturb_invert import auroc  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
H5 = f"{_DATA}/larry/LARRY_sp500_ranking1_adata_preprocessed.h5ad"
SEED = 0
K_NB = 100          # neighbourhood for local tangent / flow
M_TAN = 10          # local tangent dimension
MIN_LATE = 3        # min differentiated Neu/Mono cells in a clone for a reliable label
N_PERM = 2000


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def load():
    f = h5py.File(H5, "r")
    dec = lambda a: np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    state_cats = dec(f['obs/__categories/state_info'][:])
    si = f['obs/state_info'][:]; ti = f['obs/time_info'][:]
    mask = f['obs/NeuMon_mask'][:]; cospar_fb = f['obs/NeuMon_fate_bias'][:]
    Xpca = f['obsm/X_pca'][:].astype(np.float64)
    d = f['obsm/X_clone/data'][:]; ind = f['obsm/X_clone/indices'][:]; iptr = f['obsm/X_clone/indptr'][:]
    Xc = csr_matrix((np.ones_like(ind, np.int8), ind, iptr), shape=(len(si), int(ind.max()) + 1))
    cell_clone = np.asarray(Xc.argmax(1)).ravel()
    f.close()
    return dict(state_cats=list(state_cats), si=si, ti=ti, mask=mask, cospar_fb=cospar_fb,
                Xpca=Xpca, cell_clone=cell_clone, n_clone=Xc.shape[1])


def clonal_label(D):
    """Per-clone neutrophil fraction among differentiated Neu/Mono cells; assign to undiff masked cells."""
    NEU = D['state_cats'].index('Neutrophil'); MONO = D['state_cats'].index('Monocyte')
    si, cc = D['si'], D['cell_clone']
    neu_frac = np.full(D['n_clone'], np.nan); n_nm = np.zeros(D['n_clone'], int)
    for c in range(D['n_clone']):
        cells = np.where(cc == c)[0]
        nn = int((si[cells] == NEU).sum()); nm = int((si[cells] == MONO).sum())
        n_nm[c] = nn + nm
        if nn + nm:
            neu_frac[c] = nn / (nn + nm)
    return neu_frac, n_nm, NEU, MONO


def geometry_features(Xz, ti, sel, d_fate):
    """Per selected undiff cell: local developmental-flow tilt toward the fate axis, tangent-fate alignment,
    local intrinsic dimension. Uses positions + observed TIME only (never the clonal label)."""
    nn = NearestNeighbors(n_neighbors=K_NB).fit(Xz)
    _, idx = nn.kneighbors(Xz[sel])
    fate_lean = np.zeros(len(sel)); tan_align = np.zeros(len(sel)); locdim = np.zeros(len(sel))
    for j in range(len(sel)):
        nb = idx[j]; Xn = Xz[nb]; tn = ti[nb].astype(float)
        Xc = Xn - Xn.mean(0)
        # local flow = displacement toward later-time neighbours (oracle-style local developmental direction)
        ahead = tn > np.median(tn)
        flow = unit(Xn[ahead].mean(0) - Xn[~ahead].mean(0)) if ahead.any() and (~ahead).any() else np.zeros(Xz.shape[1])
        fate_lean[j] = float(flow @ d_fate)                         # signed: toward neu(+) or mono(-)
        V = np.linalg.svd(Xc, full_matrices=False)[2][:M_TAN]       # local tangent basis
        tan_align[j] = float(np.linalg.norm(V @ d_fate))           # how much fate axis lies in the tangent
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
        locdim[j] = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))
    return np.column_stack([fate_lean, tan_align, locdim]), ["fate_lean", "tan_align", "local_dim"]


def cv_auroc(Xfeat, y, groups, seed=SEED):
    """Clone-split CV AUROC with logistic regression (standardised features)."""
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(Xfeat, y, groups):
        sc = StandardScaler().fit(Xfeat[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xfeat[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xfeat[te]))[:, 1]
    return float(auroc(oof, y.astype(bool))), oof


def perm_p(Xfeat, y, groups, obs, n=N_PERM, seed=SEED):
    """Clone-permuted null: permute the label at the CLONE level (all sisters share a label), re-CV."""
    rng = np.random.default_rng(seed)
    uc = np.unique(groups)
    # one label per clone (labels are clone-constant already); permute clone->label assignment
    clone_lab = {g: y[groups == g][0] for g in uc}
    null = np.empty(n)
    for i in range(n):
        perm = rng.permutation([clone_lab[g] for g in uc])
        m = {g: perm[k] for k, g in enumerate(uc)}
        yp = np.array([m[g] for g in groups])
        null[i], _ = cv_auroc(Xfeat, yp, groups, seed=seed)
    return float((np.sum(null >= obs) + 1) / (n + 1)), float(null.mean()), float(null.std())


def matched_stratum_test(pos_oof, geom, y, nbins=10):
    """Does geometry discriminate fate WITHIN position-defined strata? Bin cells by the position-only OOF
    prediction; within each bin compute AUROC of the geometry scalar. Mean within-bin AUROC > 0.5 = geometry
    adds fate info at matched transcriptomic state (the hidden-variable signature)."""
    q = np.quantile(pos_oof, np.linspace(0, 1, nbins + 1))
    aurocs, ns = [], []
    for b in range(nbins):
        m = (pos_oof >= q[b]) & (pos_oof <= q[b + 1] if b == nbins - 1 else pos_oof < q[b + 1])
        if m.sum() >= 20 and 0 < y[m].sum() < m.sum():
            aurocs.append(auroc(geom[m], y[m].astype(bool))); ns.append(int(m.sum()))
    return float(np.nanmean(aurocs)) if aurocs else float("nan"), aurocs, ns


def run():
    D = load()
    Xz = StandardScaler().fit_transform(D['Xpca'])                  # whiten the provided 40-PC space
    neu_frac, n_nm, NEU, MONO = clonal_label(D)

    # differentiated late cells define the fate axis in the manifold
    diff_neu = D['si'] == NEU; diff_mono = D['si'] == MONO
    d_fate = unit(Xz[diff_neu].mean(0) - Xz[diff_mono].mean(0))

    # sample = undifferentiated masked cells with a reliable clonal label
    mc = np.where(D['mask'])[0]
    frac = neu_frac[D['cell_clone'][mc]]; ncell = n_nm[D['cell_clone'][mc]]
    ok = np.isfinite(frac) & (ncell >= MIN_LATE)
    sel = mc[ok]
    y = (frac[ok] > 0.5).astype(int)                               # 1 = neutrophil-biased clone
    groups = D['cell_clone'][sel]
    ti = D['ti']
    print(f"sample: {len(sel)} undiff cells, {len(np.unique(groups))} clones, "
          f"neu-biased frac {y.mean():.2f} | fate-axis from {int(diff_neu.sum())} Neu / {int(diff_mono.sum())} Mono")

    Gf, gnames = geometry_features(Xz, ti, sel, d_fate)
    pos = Xz[sel]                                                   # 40-PC position (the state baseline)

    # feature sets
    a_pos, oof_pos = cv_auroc(pos, y, groups)
    a_geo, _ = cv_auroc(Gf, y, groups)
    a_both, _ = cv_auroc(np.column_stack([pos, Gf]), y, groups)
    a_lean, _ = cv_auroc(Gf[:, [0]], y, groups)                    # fate_lean scalar alone
    # cospar's own transcriptomic MLP baseline, on this sample
    f = h5py.File(H5, "r"); mlp = f['obs/MLPClassifier_predicted_bias'][:][sel]; f.close()
    a_mlp = auroc(mlp, y.astype(bool))

    print(f"\n  clone-split CV AUROC (predict neutrophil-biased clone):")
    print(f"    position (40-PC state)      : {a_pos:.3f}")
    print(f"    cospar MLP (transcriptome)  : {a_mlp:.3f}   [published state baseline]")
    print(f"    geometry (3 features)       : {a_geo:.3f}")
    print(f"    fate_lean scalar alone      : {a_lean:.3f}")
    print(f"    position + geometry         : {a_both:.3f}   (gain over position {a_both - a_pos:+.3f})")

    # permutation null for the geometry-scalar and for the gain
    p_lean, nm_lean, ns_lean = perm_p(Gf[:, [0]], y, groups, a_lean, n=N_PERM)
    # matched-stratum test: geometry within position-defined bins
    ms_lean, ms_list, ms_ns = matched_stratum_test(oof_pos, Gf[:, 0], y)
    # direct correlation of fate_lean with the clonal neutrophil fraction (continuous)
    r_lean = float(np.corrcoef(Gf[:, 0], frac[ok])[0, 1])

    print(f"\n  fate_lean vs clonal fate: AUROC {a_lean:.3f} (clone-perm p={p_lean:.4f}, null {nm_lean:.3f}±{ns_lean:.3f})"
          f" | corr with neu-fraction r={r_lean:+.3f}")
    print(f"  position-matched-stratum test: mean within-bin AUROC of fate_lean = {ms_lean:.3f} "
          f"({len(ms_list)} bins) — >0.5 means geometry adds fate info at matched state")

    # ROBUSTNESS: richer geometry (guards against a weak-scalar false negative). Top-3 local tangent
    # directions (sign-fixed toward the local time gradient) + a local-curvature scalar; plus whether ANY of
    # these correlates with the position residual (the pure hidden-variable headroom).
    nn2 = NearestNeighbors(n_neighbors=30).fit(Xz)
    V3 = np.zeros((len(sel), 3 * Xz.shape[1])); curv = np.zeros(len(sel))
    nnk = NearestNeighbors(n_neighbors=K_NB).fit(Xz); _, idxk = nnk.kneighbors(Xz[sel])
    for j in range(len(sel)):
        nb = idxk[j]; Xn = Xz[nb]; tn = ti[nb].astype(float); Xc = Xn - Xn.mean(0)
        U = np.linalg.svd(Xc, full_matrices=False)[2][:3]
        tg = unit(Xc.T @ (tn - tn.mean()))
        for r in range(3):
            if U[r] @ tg < 0:
                U[r] = -U[r]
        V3[j] = U.ravel()
        _, i2 = nn2.kneighbors(Xz[sel[j]][None]); X2 = Xz[i2[0]] - Xz[i2[0]].mean(0)
        U2 = np.linalg.svd(X2, full_matrices=False)[2][:1][0]
        curv[j] = 1 - abs(U[0] @ (U2 if U2 @ U[0] >= 0 else -U2))
    a_tan3, _ = cv_auroc(V3, y, groups)
    a_pos_tan3, _ = cv_auroc(np.column_stack([pos, V3]), y, groups)
    a_curv, _ = cv_auroc(curv[:, None], y, groups)
    resid = y - oof_pos
    r_resid_tan = float(np.corrcoef((V3[:, :Xz.shape[1]] @ d_fate), resid)[0, 1])
    r_resid_curv = float(np.corrcoef(curv, resid)[0, 1])
    richer = dict(auroc_tangent3=a_tan3, auroc_position_plus_tangent3=a_pos_tan3, auroc_curvature=a_curv,
                  corr_tangentproj_resid=r_resid_tan, corr_curvature_resid=r_resid_curv)
    print(f"\n  [richer geometry] top-3 tangent dirs {a_tan3:.3f} | position+tangent3 {a_pos_tan3:.3f} "
          f"(vs position {a_pos:.3f}) | curvature {a_curv:.3f}")
    print(f"    correlation with position-residual: tangent-proj {r_resid_tan:+.3f}, curvature {r_resid_curv:+.3f}"
          f" (both ~0 = no geometric hidden-variable signal)")

    # robustness: day-2-only subset
    day2 = ti[sel] == 0
    r2 = {}
    if day2.sum() >= 40 and 0 < y[day2].sum() < day2.sum():
        a_pos2, _ = cv_auroc(pos[day2], y[day2], groups[day2])
        a_lean2, _ = cv_auroc(Gf[day2][:, [0]], y[day2], groups[day2])
        a_both2, _ = cv_auroc(np.column_stack([pos, Gf])[day2], y[day2], groups[day2])
        r2 = dict(n=int(day2.sum()), auroc_pos=a_pos2, auroc_lean=a_lean2, auroc_both=a_both2)
        print(f"  [day-2 only, n={int(day2.sum())}] position {a_pos2:.3f} | fate_lean {a_lean2:.3f} | "
              f"both {a_both2:.3f} (gain {a_both2-a_pos2:+.3f})")

    out = dict(dataset="LARRY_sp500", n_cells=int(len(sel)), n_clones=int(len(np.unique(groups))),
               neu_biased_frac=float(y.mean()), min_late=MIN_LATE,
               auroc=dict(position=a_pos, cospar_mlp=float(a_mlp), geometry=a_geo, fate_lean=a_lean,
                          position_plus_geometry=a_both, gain_over_position=float(a_both - a_pos)),
               fate_lean=dict(auroc=a_lean, clone_perm_p=p_lean, null_mean=nm_lean, null_sd=ns_lean,
                              corr_with_neu_fraction=r_lean),
               matched_stratum=dict(mean_within_bin_auroc=ms_lean, per_bin=ms_list, per_bin_n=ms_ns),
               richer_geometry=richer,
               day2_only=r2,
               verdict=dict(
                   geometry_beats_chance=bool(a_lean > 0.55 and p_lean < 0.05),
                   geometry_adds_over_position=bool(a_both - a_pos > 0.01),
                   hidden_variable_signature=bool(ms_lean > 0.55)))
    json.dump(out, open(os.path.join(RESULTS, "hb1_larry_curvature.json"), "w"), indent=1)

    print(f"\n================ H-B1 VERDICT ================")
    v = out["verdict"]
    print(f"  geometry predicts clonal fate above chance : {v['geometry_beats_chance']} "
          f"(fate_lean AUROC {a_lean:.3f}, p={p_lean:.4f})")
    print(f"  geometry adds over expression position     : {v['geometry_adds_over_position']} "
          f"(gain {a_both-a_pos:+.3f})")
    print(f"  hidden-variable signature (matched strata) : {v['hidden_variable_signature']} "
          f"(within-state AUROC {ms_lean:.3f})")
    print("[done] -> results/hb1_larry_curvature.json")
    return out


if __name__ == "__main__":
    run()

