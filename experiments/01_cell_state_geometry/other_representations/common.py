"""Shared helpers for STATE-SE geometry/curvature tests.

Imports analysis methods READ-ONLY from route_topology/cellcycle_loop.py (decode_phase,
decode_phase_knn, loop_alignment, circular_corr) and route_geometry (S_GENES/G2M_GENES).
cellcycle_loop.py imports `ripser` at module top; ripser is not installed in .venv_state, so we
inject a stub into sys.modules *before* importing it. We never call the ripser-dependent function
(h1_with_null / top_h1_persistence) — only the pure-numpy/sklearn curvature + alignment functions.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, types
import numpy as np

BIO = f"{_DATA}"
CODEBASE = os.path.join(BIO, "codebase")
CACHE = os.path.join(BIO, "data", "state_geometry_cache")
os.makedirs(CACHE, exist_ok=True)

# --- inject a ripser stub so cellcycle_loop imports cleanly (we don't use topology here) ---
if "ripser" not in sys.modules:
    _stub = types.ModuleType("ripser")
    def _ripser_unavailable(*a, **k):
        raise RuntimeError("ripser stub: topology (H1) not run in STATE geometry tests")
    _stub.ripser = _ripser_unavailable
    sys.modules["ripser"] = _stub

sys.path.insert(0, os.path.join(CODEBASE, "route_topology"))
sys.path.insert(0, os.path.join(CODEBASE, "route_geometry"))
import cellcycle_loop as CCL           # READ-ONLY reuse  # noqa: E402
from build_rulers import S_GENES, G2M_GENES  # noqa: E402

# re-export the reused methods (so callers cite the source clearly)
decode_phase = CCL.decode_phase
decode_phase_knn = CCL.decode_phase_knn
loop_alignment = CCL.loop_alignment
circular_corr = CCL.circular_corr
_circ_mean = CCL._circ_mean


def phase_from_expression(X, var_names):
    """Tirosh S/G2M cell-cycle phase from raw counts X (cells x genes), matching cellcycle_loop's recipe:
    log1p CP10k, z-score cell-cycle genes, PCA-2, atan2."""
    vg = np.char.upper(np.asarray(var_names).astype(str))
    idx = [np.where(vg == g)[0][0] for g in (S_GENES + G2M_GENES) if g in set(vg)]
    from sklearn.decomposition import PCA
    Xc = np.asarray(X[:, idx], dtype=np.float32)
    rs = Xc.sum(1, keepdims=True); rs[rs == 0] = 1
    Xl = np.log1p(Xc / rs * 1e4)
    Z = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-8)
    pcs = PCA(2, random_state=0).fit_transform(Z)
    return np.arctan2(pcs[:, 1], pcs[:, 0]), len(idx)


def curvature_report(E, phi):
    """Full linear-vs-bilinear-vs-kNN curvature report on embedding E for target angle phi.
    curvature = kNN(20D) - linear(20D); >0 => curved/linearly-inaccessible, <=0 => flat/linearly accessible."""
    E = np.asarray(E, dtype=float)
    lin = decode_phase(E, phi, degree=1)
    bil = decode_phase(E, phi, degree=2)
    ss = decode_phase_knn(E, phi)
    align = loop_alignment(E, phi)
    return dict(
        dim=int(E.shape[1]), n=int(len(phi)),
        linear_full_circ_r2=lin["circ_r2"],
        bilinear_circ_r2=bil["circ_r2"],
        bilinear_gain=bil["circ_r2"] - lin["circ_r2"],
        linear_20d=ss["linear_samespace"],
        knn_20d=ss["knn"],
        curvature=ss["knn"] - ss["linear_samespace"],
        loop_align_rho=align,
    )


def synthetic_linear_control(phi, d, noise_target_r2, seed=0):
    """DECISIVE CONTROL (route_lineage discipline): phase LINEARLY EMBEDDED by construction.
    E_syn = [cos phi, sin phi] @ A(2xd random) + Gaussian noise; noise scaled so the *linear* decode
    circ-R2 roughly matches the real embedding's, making the comparison fair. A curvature signal here
    would be an artifact; the control must return curvature ~ 0 for the real signal to count."""
    rng = np.random.default_rng(seed)
    base = np.column_stack([np.cos(phi), np.sin(phi)])          # (n,2) — the ONLY structure, purely linear
    A = rng.standard_normal((2, d))
    signal = base @ A                                            # (n,d) linear image of the circle
    signal /= signal.std()
    # tune noise sigma so linear-20D decode ~= noise_target_r2
    lo, hi = 0.0, 8.0
    for _ in range(18):
        sig = 0.5 * (lo + hi)
        E = signal + sig * rng.standard_normal(signal.shape)
        r2 = decode_phase_knn(E, phi)["linear_samespace"]
        if r2 > noise_target_r2:
            lo = sig
        else:
            hi = sig
    return E
