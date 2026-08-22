"""Shared substrate + circular machinery for the cell-cycle (closed-loop) steering tests.

WHY THIS ROUTE EXISTS
---------------------
route_steering established, on four BRANCHING developmental trajectories, that a constant direction plus a
local-tangent projection (`linear_proj`) walks a progenitor forward while staying on the real-cell manifold,
and that a *cleverer direction* (`local`) buys nothing -- the projection is the whole effect.

That conclusion is regime-specific, and the cell cycle is the regime that breaks it. Two facts:

  (1) For ANY closed curve and ANY fixed vector w,      |  ∮ w · dx  =  w · ∮ dx  =  0
      A constant steering direction does exactly zero net work around a cycle. This is vector calculus, not
      an empirical claim: it holds for every model, every layer, curved or flat.

  (2) steer() unit-normalises each step, so a projected constant direction becomes sign(w·t̂)·t̂ -- it advances
      phase while w·t̂ > 0 and RETREATS once w·t̂ < 0, converging to a stable stall where w ⊥ t̂ (about a
      quarter-turn from the start) while remaining ON the manifold.

So `linear_proj` -- the program's headline operator -- must fail here, in a specific, pre-stated way:
ON-MANIFOLD BUT PHASE-STALLED. That failure is not a defect of the models; it is a defect of the operator,
and it is the reason the cell cycle is worth running.

Why the branching trajectories never exposed this: in tangent_diagnostic's own `angle_to_global` arrays, the
tangent fan spans 48-113 deg end-to-end but the global direction sits in the MIDDLE of the fan, so 82/95
pseudotime bins keep the local tangent < 90 deg from the global w -- a fixed direction retains positive forward
overlap almost everywhere. The fan never closes. On a loop the fan spans a full 360 deg and no fixed direction
can stay within 90 deg of every tangent. Half the loop necessarily pushes backwards.

WHAT IS *NOT* CLAIMED HERE. On a loop, "the tangent rotates" is not evidence of nonlinearity -- a perfectly
flat circle in a linear 2-plane turns its tangent a full 360 deg. Rotation is topologically mandatory. The
existing cell-cycle verdict ("flat, linearly-embedded loop", circ-R^2 0.91-0.93) is therefore expected to
SURVIVE, and cc_geometry.py tests it against the only null that means anything on a loop -- "the loop lies in a
fixed 2-plane" -- rather than re-running the discredited decodability gap. The steering claim does not depend
on the loop being curved. That is the point: decodability and steerability are independent, and a flat,
perfectly linearly-decodable loop still defeats a fixed direction.

SUBSTRATE. Replogle K562 CRISPRi, non-targeting controls only (the project's canonical cell-cycle substrate,
build_rulers.py:77-104). Cells are ~71-81% cycling with near-uniform phase occupancy -- far better conditioned
for a loop than any developmental trajectory in the program, where progenitors are rare.

PHASE COORDINATE. Tirosh S/G2M marker z-scores -> PCA(2) -> phi = atan2 (the same recipe as
route_topology/cellcycle_loop.py:36-54), then ORIENTED so phi increases G1 -> S -> G2M and rotated to put G1 at
phi = 0. Note the standing caveat, inherited: phi is (up to radial normalisation) a linear projection of
z-scored marker expression, so it structurally advantages linear probes. This is why phi is used ONLY as
ground truth and as the judge -- never as the thing being optimised -- and why the load-bearing readout in
cc_decode.py goes through scGPT's own pretrained head instead.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os
import numpy as np

ROOT = f"{_DATA}"
REPLOGLE = (f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/"
            "crispri_validation/data/replogle_concat.h5ad")
DATA = os.path.join(ROOT, "data", "cellcycle")
os.makedirs(DATA, exist_ok=True)

N_CELLS, SEED = 3000, 42
CAP, DIM = 3000, 20

# Tirosh et al. 2016, verbatim from route_geometry/build_rulers.py:30-38 (same substrate, same ruler).
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP",
           "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN",
           "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1",
             "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C",
             "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
             "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]

# Phase-resolved modules for the ORDERED marker-wave readout (cc_decode.py). Peaks should come round the loop
# in this cyclic order. Kept deliberately small and canonical; every gene is a textbook phase marker.
WAVE = {
    "G1S": ["CCNE1", "CCNE2", "PCNA", "CDC6", "CDT1", "E2F1", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"],
    "S":   ["RRM2", "TYMS", "TK1", "DHFR", "GINS2", "CLSPN", "RRM1", "FEN1", "CDC45", "POLA1"],
    "G2":  ["CCNA2", "TOP2A", "CDK1", "BUB1", "CCNF", "NUSAP1", "SMC4", "KIF11", "CENPF"],
    "G2M": ["CCNB1", "CCNB2", "PLK1", "AURKA", "AURKB", "CDC20", "UBE2C", "PTTG1", "CDC25C", "CENPA", "TTK"],
}
WAVE_ORDER = ["G1S", "S", "G2", "G2M"]


# ----------------------------------------------------------------------------- circular primitives
def wrap(a):
    """Map angles into (-pi, pi]. The single most important function in this file: every seam bug in a
    circular pipeline is a missing wrap()."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def circ_mean(a):
    a = np.asarray(a)
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def circ_corr(a, b):
    """Jammalamadaka circular correlation. Bounded [-1, 1]; 0 under independence."""
    a, b = np.asarray(a), np.asarray(b)
    sa, sb = np.sin(a - circ_mean(a)), np.sin(b - circ_mean(b))
    return float((sa * sb).sum() / (np.sqrt((sa ** 2).sum() * (sb ** 2).sum()) + 1e-12))


def circ_corr_timeshift_p(A, B, n=2000, seed=0):
    """p-value for circ_corr between two PATH series, under a random shift along the BUDGET axis.

    A, B: (T+1, n_cells) phase along a steered path. Statistic = |circ_corr| over all pooled (cell, budget).

    CAREFUL -- the obvious null is wrong. Jammalamadaka's circ_corr uses sin(a - circmean(a)), so adding a
    constant ANGLE to one series shifts its circular mean by the same constant and leaves the statistic exactly
    unchanged. An "angle rotation null" is therefore a no-op: it returns null_r == real_r identically. (We hit
    this; it is a silent failure that looks like p ~ 0.9 rather than like a crash.)

    The null that actually bites for a path is a shift along TIME: roll B's budget axis by a random offset.
    That preserves B's marginal AND its serial autocorrelation, while destroying its alignment with A. It stays
    continuous -- unlike the fate experiment's 4! = 24 assignment permutation, whose p FLOOR is 0.042, p here is
    limited only by n.
    """
    rng = np.random.default_rng(seed)
    T = A.shape[0]
    real = abs(circ_corr(np.asarray(A).ravel(), np.asarray(B).ravel()))
    null = np.array([abs(circ_corr(A.ravel(), np.roll(B, int(rng.integers(1, T)), axis=0).ravel()))
                     for _ in range(n)])
    return float(real), float((np.sum(null >= real) + 1) / (n + 1)), float(null.mean())


def unwrap_path(phis):
    """Cumulative SIGNED phase travelled along a steered path (radians, unbounded -- can exceed 2pi).

    phis: (T+1, n_cells) predicted phase at each budget. Returns (T+1, n_cells) cumulative advance from T=0.
    Summing wrapped increments is what lets a winding number exceed 1; a naive difference of angles cannot.
    """
    phis = np.asarray(phis)
    d = wrap(np.diff(phis, axis=0))
    return np.vstack([np.zeros((1, phis.shape[1])), np.cumsum(d, axis=0)])


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ang(u, v):
    """Unsigned angle in degrees. No abs on the dot product: a reversal must read as 180, not 0."""
    return float(np.degrees(np.arccos(np.clip(unit(u) @ unit(v), -1.0, 1.0))))


# ----------------------------------------------------------------------------- substrate
def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _load_categorical(grp, name):
    """AnnData columns come in three shapes here: a categorical Group, a legacy `__categories` pair, or a
    plain string dataset (replogle_concat's var/gene_name_index). Handle all three (cf. build_rulers.py:41)."""
    import h5py
    col = grp[name]
    if isinstance(col, h5py.Group):
        return _dec(col["categories"][:])[col["codes"][:]]
    if "__categories" in grp and name in grp["__categories"]:
        return _dec(grp["__categories"][name][:])[col[:]]
    data = col[:]
    return _dec(data) if data.dtype.kind in ("O", "S") else data


def build_substrate(n_cells=N_CELLS, seed=SEED, force=False):
    """Select non-targeting K562 controls, load counts, and compute the cell-cycle ruler.

    Selection mirrors build_rulers.py:83-85 (the program's canonical recipe) so this substrate is the same
    population the existing cell-cycle numbers were computed on -- only larger (3000 vs 2000, the project CAP).
    Cached to data/cellcycle/k562_cc_substrate.npz.
    """
    import h5py
    f_out = os.path.join(DATA, "k562_cc_substrate.npz")
    if os.path.exists(f_out) and not force:
        return dict(np.load(f_out, allow_pickle=True))

    with h5py.File(REPLOGLE, "r") as f:
        cell_gene = _load_categorical(f["obs"], "gene")
        genes = _load_categorical(f["var"], "gene_name_index")
        ctrl = np.where(np.isin(cell_gene, ["non-targeting", "Non-targeting", "non_targeting"]))[0]
        rs = np.random.RandomState(seed)                      # RandomState, matching build_rulers.py's np.random.seed
        sel = np.sort(rs.choice(ctrl, min(n_cells, len(ctrl)), replace=False))
        X = np.empty((len(sel), f["X"].shape[1]), dtype=np.float32)
        for i, r in enumerate(sel):
            X[i] = f["X"][int(r), :]

    Xlog = to_lognorm(X)
    gu = np.char.upper(genes.astype(str))
    s_score, ns = _score(Xlog, gu, S_GENES)
    g2m_score, ng = _score(Xlog, gu, G2M_GENES)
    cc_phase = np.where(np.maximum(s_score, g2m_score) < 0, "G1",
                        np.where(s_score >= g2m_score, "S", "G2M"))
    phi = phase_angle(Xlog, gu, cc_phase)

    out = dict(cell_idx=sel, phi=phi, s_score=s_score, g2m_score=g2m_score, cc_phase=cc_phase,
               genes=genes.astype(str), total_counts=X.sum(1), n_s_markers=ns, n_g2m_markers=ng)
    np.savez(f_out, **out)
    print(f"[substrate] {len(sel)} non-targeting K562 controls | S markers {ns}/{len(S_GENES)}, "
          f"G2M {ng}/{len(G2M_GENES)}")
    from collections import Counter
    print(f"[substrate] discrete phase: {dict(Counter(cc_phase))}")
    return out


def to_lognorm(X):
    """Return log1p(CP10k) expression, WITHOUT double-normalising data that is already in that form.

    replogle_concat.h5ad stores X as float32 in [0, 5.88], non-integer, ~51% nonzero -- i.e. it is ALREADY
    log1p(CP10k), not raw counts. Applying log1p(X / rowsum * 1e4) on top of that is a second normalisation.

    Why it matters, and why it is easy to miss: a per-cell monotone map leaves WITHIN-CELL gene ranks untouched
    (we measured within-cell Spearman +0.315 either way), so anything rank-based inside a cell looks fine. What
    it corrupts is the per-gene VARIANCE across cells, and therefore the highly-variable-gene selection. Feeding
    the doubly-normalised matrix to scGPT's MVC gate picked a different top-1500-variance set and drove the
    within-cell HVG Spearman from +0.326 (passes) down to +0.094 (fails) -- a gate failure that looks exactly
    like "the pretrained head cannot read these embeddings" but is entirely a preprocessing artifact.

    NOTE: route_geometry/build_rulers.py:91-92 applies `np.log1p(X / rs * 1e4)` to this same file, so the
    program's existing K562 cell-cycle ruler (s_score / g2m_score / cc_phase, and every cell-cycle number
    downstream of it) carries this double normalisation. Flagged in RESULTS.md.
    """
    X = np.asarray(X, dtype=np.float64)
    already_log = (X.max() < 20.0) and not np.allclose(X, np.round(X))
    if already_log:
        return X
    tot = X.sum(1, keepdims=True); tot[tot == 0] = 1.0
    return np.log1p(X / tot * 1e4)


def _score(Xlog, gene_upper, gene_set):
    idx = [np.where(gene_upper == g)[0][0] for g in gene_set if g in set(gene_upper)]
    if not idx:
        return np.zeros(Xlog.shape[0]), 0
    sub = Xlog[:, idx]
    z = (sub - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-8)
    return z.mean(1), len(idx)


def phase_angle(Xlog, gene_upper, cc_phase):
    """phi in [-pi, pi): PCA(2) of z-scored S+G2M marker expression -> atan2, then ORIENTED and ROTATED.

    Orientation matters and is easy to get wrong. atan2 fixes neither the handedness nor the origin, so the
    raw angle can run G2M -> S -> G1 (backwards) on one dataset and forwards on another. We pin it by the
    discrete Tirosh call: require the cyclic order G1 -> S -> G2M to be the direction of INCREASING phi, then
    rotate G1 to phi = 0. Without this, "forward" steering is a coin flip.
    """
    idx = [np.where(gene_upper == g)[0][0] for g in (S_GENES + G2M_GENES) if g in set(gene_upper)]
    Z = Xlog[:, idx]
    Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)
    from sklearn.decomposition import PCA
    P = PCA(2, random_state=0).fit_transform(Z - Z.mean(0))
    phi = np.arctan2(P[:, 1], P[:, 0])

    m = {p: circ_mean(phi[cc_phase == p]) for p in ("G1", "S", "G2M") if (cc_phase == p).sum() > 10}
    if {"G1", "S", "G2M"} <= set(m):
        fwd = wrap(m["S"] - m["G1"]) > 0 and wrap(m["G2M"] - m["S"]) > 0
        if not fwd:
            phi = -phi
            m = {p: circ_mean(phi[cc_phase == p]) for p in ("G1", "S", "G2M")}
        phi = wrap(phi - m["G1"])                              # G1 at the origin
    return phi


def prep(X, d=DIM, seed=0):
    """Project-standard steering space: center -> PCA(d) -> whiten. Identical to tangent_diagnostic.prep."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xc = X - X.mean(0)
    Xr = PCA(min(d, Xc.shape[1], Xc.shape[0] - 1), random_state=seed).fit_transform(Xc)
    return StandardScaler().fit_transform(Xr)


def emb_path(model):
    return os.path.join(DATA, f"{model}_k562.npz")


if __name__ == "__main__":
    s = build_substrate(force=True)
    phi, cc = s["phi"], s["cc_phase"]
    print(f"[substrate] phi range [{phi.min():+.2f}, {phi.max():+.2f}]")
    for p in ("G1", "S", "G2M"):
        sel = cc == p
        print(f"  {p:>4}: n={sel.sum():4d}  circ-mean phi = {np.degrees(circ_mean(phi[sel])):+7.1f} deg")
    print("  (phi must increase G1 -> S -> G2M; G1 pinned to 0)")
