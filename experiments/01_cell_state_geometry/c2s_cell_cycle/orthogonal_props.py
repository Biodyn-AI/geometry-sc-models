"""orthogonal_props — which cell properties live ON the cell-cycle circle, and which are ORTHOGONAL to it?

The days-of-week analogy: day-of-week is a circle, but *parts of day* (morning/evening) are encoded on a
DIFFERENT, orthogonal direction. Same question here — is everything cell-cycle-related carried by the phase
angle, or does the model keep some cycle properties off the circle?

DECOMPOSITION. Fit the phase plane on real cells (ridge -> cos/sin of phase); its two weight vectors span a
2-D "phase subspace" in activation space. For each cell-level covariate y, compare cross-validated R^2 from:
    R2_circle  y ~ [cos(phi), sin(phi)]        the ANGLE only (position on the circle)
    R2_radius  y ~ r                            the RADIUS only (distance from the centre)
    R2_full    y ~ Z                            the whole activation (upper bound on what is encoded)
    R2_orth    y ~ Z_perp                       activation with the phase subspace PROJECTED OUT
Then
    circle_fraction = R2_circle / R2_full       ~1 -> the property IS the circle coordinate
    orth_fraction   = R2_orth   / R2_full       ~1 -> the property is carried OFF the circle
(These need not sum to 1: a property can be partly redundant across both.)

Out: results/orthogonal_props.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle, score_genes, S_GENES, G2M_GENES

# cell-cycle-related and unrelated covariates
PANELS = {
    "S_score":        S_GENES,
    "G2M_score":      G2M_GENES,
    "histone_S":      ["HIST1H1B", "HIST1H1E", "HIST1H4C", "HIST2H2AC", "HIST1H2BK", "HIST1H1C", "H2AFZ"],
    "mitotic_spindle": ["TPX2", "AURKA", "KIF11", "KIF23", "ANLN", "ECT2", "PLK1", "CENPE", "CENPF"],
    "replication":    ["PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CDC45", "GINS2", "CLSPN"],
    "cdk_inhibitors": ["CDKN1A", "CDKN1B", "CDKN2A", "GADD45A", "GADD45B"],   # arrest / checkpoint
    "stress_IEG":     ["JUN", "JUNB", "FOS", "FOSB", "EGR1", "ATF3", "IER2"],
    "myc_growth":     ["MYC", "NPM1", "NCL", "EIF4A1", "SRM", "ODC1"],
    "ribosome":       ["RPL5", "RPL10", "RPS3", "RPS6", "RPL13", "RPS4X", "RPL11", "RPS8"],
    "oxphos":         ["NDUFA1", "NDUFB2", "COX7C", "COX5B", "ATP5F1E", "UQCRB", "ATP5MC2"],
}


def cv_r2(X, y, folds=5, alpha=1e3, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    pred = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return max(0.0, 1 - ss_res / (ss_tot + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="results/orthogonal_props.json")
    a = ap.parse_args()
    import anndata, scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    theta, s_sc, g_sc, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sub = ids[:len(H)]

    # phase subspace in activation space
    sc = StandardScaler().fit(H); Z = sc.transform(H)
    m = Ridge(alpha=1e3).fit(Z, np.column_stack([np.cos(t), np.sin(t)]))
    W = m.coef_.T                                        # (d, 2) the two phase directions
    Q, _ = np.linalg.qr(W)                               # orthonormal basis of the phase subspace
    Z_perp = Z - (Z @ Q) @ Q.T                           # activation with phase projected OUT
    P = m.predict(Z); P = P - P.mean(0)
    phi = np.arctan2(P[:, 1], P[:, 0]); rad = np.linalg.norm(P, axis=1)
    circle_feats = np.column_stack([np.cos(phi), np.sin(phi)])

    # covariates
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    cov = {}
    for name, genes in PANELS.items():
        v, idx = score_genes(ad, genes)
        if len(idx) >= 3:
            cov[name] = v[sub]
    cov["library_size"] = X.sum(1)[sub]
    cov["n_genes"] = (X > 0).sum(1)[sub].astype(float)
    cov["cycling_strength"] = np.sqrt(s_sc ** 2 + g_sc ** 2)[sub]     # how strongly cycling, phase-independent
    cov["true_phase_cos"] = np.cos(theta)[sub]                        # positive control: IS the circle

    res = {}
    print(f"L{a.layer:02d}  n={len(H)} cells | phase subspace = 2 dims of {Z.shape[1]}\n", flush=True)
    print(f"  {'covariate':<18}{'R2_full':>9}{'R2_circle':>11}{'R2_radius':>11}{'R2_orth':>9}"
          f"{'circle_frac':>13}{'orth_frac':>11}", flush=True)
    for name, y in cov.items():
        y = np.asarray(y, float)
        if np.std(y) < 1e-9:
            continue
        r2_full = cv_r2(Z, y)
        r2_circ = cv_r2(circle_feats, y)
        r2_rad = cv_r2(rad.reshape(-1, 1), y)
        r2_orth = cv_r2(Z_perp, y)
        cf = r2_circ / (r2_full + 1e-12); of = r2_orth / (r2_full + 1e-12)
        res[name] = dict(r2_full=r2_full, r2_circle=r2_circ, r2_radius=r2_rad, r2_orth=r2_orth,
                         circle_fraction=float(cf), orth_fraction=float(of))
        print(f"  {name:<18}{r2_full:>9.3f}{r2_circ:>11.3f}{r2_rad:>11.3f}{r2_orth:>9.3f}"
              f"{cf:>13.2f}{of:>11.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(layer=a.layer, n=len(H), props=res), open(a.out, "w"), indent=1)
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
