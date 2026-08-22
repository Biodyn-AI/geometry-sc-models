"""cc_phase — continuous CELL-CYCLE PHASE ANGLE (the cyclic ordering variable for manifold steering).

The cell cycle is the biological analogue of Goodfire's days-of-week CIRCLE: G1 -> S -> G2M -> G1. To place
cells on that circle we use the tricycle/Revelio construction: PCA on cell-cycle marker genes only; the leading
two PCs span an ellipse and the ANGLE around it is the continuous cell-cycle position.

  theta(cell) = atan2(PC2, PC1)   over the cell-cycle-gene expression submatrix

Validation (all in expression space, no model needed):
  * the discrete G1/S/G2M labels must occupy CONTIGUOUS ARCS of theta (circular ordering is real),
  * mean phase of S must lie between G1 and G2M going around the circle,
  * S-score and G2M-score must peak ~90 deg apart (quadrature) -- the signature of a genuine cycle.

Also exposes marker sets for the model-readout (logit mass on phase-specific genes).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import numpy as np

# Tirosh/Scanpy canonical sets (the standard ones; intersected with the panel at runtime)
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
           "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1",
           "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11",
             "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C",
             "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1",
             "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


def _dense(X):
    import scipy.sparse as sp
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def score_genes(adata, genes, n_bins=25, n_ctrl=50, seed=0):
    """Scanpy-style module score: mean(set) - mean(expression-binned control set)."""
    rng = np.random.default_rng(seed)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    pos = {s: i for i, s in enumerate(var)}
    idx = np.array([pos[g] for g in genes if g in pos])
    if len(idx) == 0:
        return np.zeros(adata.n_obs), idx
    X = _dense(adata.X)
    mean_all = X.mean(0)
    order = np.argsort(mean_all)
    ranks = np.empty(len(mean_all), int); ranks[order] = np.arange(len(mean_all))
    bins = (ranks / len(ranks) * n_bins).astype(int)
    ctrl = []
    for b in np.unique(bins[idx]):
        pool = np.where(bins == b)[0]
        k = int((bins[idx] == b).sum()) * n_ctrl
        ctrl.append(rng.choice(pool, min(k, len(pool)), replace=False))
    ctrl = np.concatenate(ctrl) if ctrl else np.array([], int)
    return X[:, idx].mean(1) - (X[:, ctrl].mean(1) if len(ctrl) else 0.0), idx


def phase_angle(adata, seed=0):
    """Continuous cell-cycle position: angle in the PC1-PC2 plane of cell-cycle-gene expression.
    Returns (theta in [0,2pi), s_score, g2m_score, info)."""
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    pos = {s: i for i, s in enumerate(var)}
    cc = [g for g in (S_GENES + G2M_GENES) if g in pos]
    idx = np.array([pos[g] for g in cc])
    X = _dense(adata.X)[:, idx].astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = U[:, :2] * S[:2]
    theta = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
    s_score, _ = score_genes(adata, S_GENES, seed=seed)
    g2m_score, _ = score_genes(adata, G2M_GENES, seed=seed)
    var_ratio = float((S[:2] ** 2).sum() / (S ** 2).sum())
    return theta, s_score, g2m_score, dict(n_cc_genes=len(cc), pc12_var_ratio=var_ratio)


def circ_mean(a):
    return float(np.mod(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))), 2 * np.pi))


def phase_angle_oriented(adata, seed=0, s_at_deg=45.0, top=200):
    """phase_angle with a CANONICAL orientation, so that theta is comparable ACROSS datasets.

    WHY THIS IS NEEDED. phase_angle takes theta from the SVD of cell-cycle-gene expression, so both its
    handedness and its origin are arbitrary: the sign of a singular vector is not determined, and the origin is
    wherever PC1 happens to point. Within one dataset that is harmless -- every metric we compute there
    (circular correlation, ring score, decodability) is invariant to rotation, and reflection only flips a sign
    we never compare across datasets. ACROSS datasets it is fatal: two datasets can carry the same biology and
    still produce theta fields that are mirror images, so a phase readout fit on one and applied to the other
    scores strongly NEGATIVE while nothing is actually wrong. Any cross-dataset transfer number computed from
    the unoriented phase_angle is uninterpretable.

    THE FIX uses the marker biology, which is dataset-independent:
      handedness  S phase precedes G2/M within the same half-turn, so the displacement from the S peak to the
                  G2/M peak taken in the +theta direction must lie in (0, 180). If it does not, theta -> -theta.
      origin      rotate so the S peak sits at `s_at_deg`. With the default 45 deg this puts the G1/S boundary
                  near 0 and G2/M near 150-200, i.e. approximately Whitfield's G1/S-origin convention.

    Returns (theta, s_score, g2m_score, info) with info["flipped"] and info["rotation_deg"] recording exactly
    what was applied, plus the oriented S/G2M peaks so the caller can verify the two datasets agree.
    """
    theta, s, g2m, info = phase_angle(adata, seed=seed)
    mu_s = circ_mean(theta[np.argsort(-s)[:top]])
    mu_g = circ_mean(theta[np.argsort(-g2m)[:top]])
    flipped = bool(np.mod(mu_g - mu_s, 2 * np.pi) > np.pi)
    if flipped:
        theta = np.mod(-theta, 2 * np.pi)
        mu_s, mu_g = np.mod(-mu_s, 2 * np.pi), np.mod(-mu_g, 2 * np.pi)
    rot = np.deg2rad(s_at_deg) - mu_s
    theta = np.mod(theta + rot, 2 * np.pi)
    info = dict(info, flipped=flipped, rotation_deg=round(float(np.rad2deg(np.mod(rot, 2 * np.pi))), 1),
                s_peak_deg=round(float(np.rad2deg(np.mod(mu_s + rot, 2 * np.pi))), 1),
                g2m_peak_deg=round(float(np.rad2deg(np.mod(mu_g + rot, 2 * np.pi))), 1))
    d = abs(info["s_peak_deg"] - info["g2m_peak_deg"]) % 360
    info["s_g2m_separation_deg"] = round(min(d, 360 - d), 1)
    return theta, s, g2m, info


def validate(adata, label_key="clusters", seed=0):
    """Is the recovered theta a genuine cycle? Report per-label circular means/spreads + quadrature."""
    theta, s, g2m, info = phase_angle(adata, seed)
    out = dict(**info)
    if label_key in adata.obs:
        lab = np.asarray(adata.obs[label_key]).astype(str)
        out["labels"] = {}
        for L in sorted(set(lab)):
            m = lab == L
            R = float(np.abs(np.mean(np.exp(1j * theta[m]))))     # concentration: 1 = tight arc, 0 = uniform
            out["labels"][L] = dict(n=int(m.sum()), mean_deg=round(np.rad2deg(circ_mean(theta[m])), 1),
                                    concentration=round(R, 3))
    # quadrature: the angular separation between the S and G2M score peaks should be ~90 deg for a real cycle
    out["s_peak_deg"] = round(np.rad2deg(circ_mean(theta[np.argsort(-s)[:200]])), 1)
    out["g2m_peak_deg"] = round(np.rad2deg(circ_mean(theta[np.argsort(-g2m)[:200]])), 1)
    d = abs(out["s_peak_deg"] - out["g2m_peak_deg"]) % 360
    out["s_g2m_separation_deg"] = round(min(d, 360 - d), 1)
    return theta, s, g2m, out


if __name__ == "__main__":
    import sys, json, anndata
    p = sys.argv[1] if len(sys.argv) > 1 else \
        f"{_DATA}/cellcycle/k562_substrate_for_state.h5ad"
    a = anndata.read_h5ad(p)
    theta, s, g2m, info = validate(a)
    print(json.dumps(info, indent=1))
