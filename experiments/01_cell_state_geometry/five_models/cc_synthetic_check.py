"""Ground-truth check: a PERFECTLY FLAT circle already defeats the program's headline steering operator.

Two jobs, and the second one matters more than the first.

(1) IMPLEMENTATION TEST. Every circular pipeline dies of a missing wrap() somewhere. Here the answer is known
    in closed form, so a sign error, a seam bug, or a mis-oriented phase shows up immediately.

(2) THE ARGUMENT, ISOLATED. The data is a circle lying EXACTLY in a linear 2-plane of a 60-D space plus
    isotropic noise -- H_flat is true by construction, there is no curvature to find, and phi is as linearly
    decodable as it gets. If `fixed_proj` (= local_steering.py's `linear_proj`, the operator behind the
    program's 16/16 branching result) still stalls here, then its failure has NOTHING to do with curvature and
    everything to do with the tangent being position-dependent. That decouples decodability from steerability
    in the cleanest possible setting -- no model, no biology, no confound to argue about.

Predicted from ∮ w·dx = w·∮dx = 0, before running: a fixed direction advances only while w·t̂ > 0 and converges
to a stall where w ⊥ t̂ -- a QUARTER TURN, pi/2 = 1.571 rad. The projection keeps it on the manifold but cannot
fix its orientation.

Observed (results_synth.txt):

    arm            peak adv   laps   @T  off@peak  constr.adv  reverse
    fixed             +1.50  +0.24   46     11.08       +1.15    -1.67   <- stalls AND flies off the manifold
    fixed_proj        +1.49  +0.24    8      1.08       +1.49    -1.70   <- ON-MANIFOLD BUT STALLED. pi/2.
    local            +10.32  +1.64   46      1.76       +4.96   -10.30
    local_proj       +10.76  +1.71   46      1.65       +6.49   -10.84   <- traverses the loop
    transport         +2.45  +0.39   46      1.44       +2.29    -2.12
    oracle_phase     +24.05  +3.83   46      0.64      +24.05    -0.94   <- ceiling

fixed_proj stalls at 1.49 rad against a predicted 1.571, at off-manifold ratio 1.08 (real cells sit at ~1.0).
It is not lost, not drifting, not off-distribution -- it is sitting on the manifold, going nowhere. That is the
signature, and it appears on an object with zero curvature.

Run: ../../.venv/bin/python cc_synthetic_check.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import sys, os
sys.path.insert(0, f"{_DATA}/codebase/route_cellcycle")
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cc_common import wrap, unit, prep
from cc_steering import (CircJudge, local_phase_field, project_constant, transport_field, retraction,
                         oracle_phase_field, steer_path, cumulative_advance, loop_circumference,
                         M_TANGENT, TAU)

rng = np.random.default_rng(0)
n, d = 3000, 60
phi = rng.uniform(-np.pi, np.pi, n)
# a FLAT circle in a random 2-plane of a 60-D space, plus isotropic noise (H_flat is TRUE by construction)
A = rng.standard_normal((2, d))
X = np.column_stack([np.cos(phi), np.sin(phi)]) @ A + 0.35 * rng.standard_normal((n, d))
Xz = prep(X, 20)

perm = rng.permutation(n); tr, te = perm[:n//2], perm[n//2:]
Xtr, phitr = Xz[tr], phi[tr]
judge = CircJudge(Xtr, phitr)
d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:,1]))
h = 0.25*d0
circ = loop_circumference(Xz, phi)
T = int(np.clip(np.ceil(2.5*circ/h), 40, 400))
nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)

src_m = te[np.abs(wrap(phi[te])) < np.pi/4]
src = Xz[src_m][:200]
w0 = unit(local_phase_field(Xtr, phitr)(src.mean(0)[None,:])[0])

print(f"SYNTHETIC FLAT CIRCLE  n={n} d={d}  d0={d0:.2f} h={h:.2f} circumference={circ:.1f} T_max={T}")
print("H_flat is TRUE by construction: the loop lies exactly in a linear 2-plane.\n")
retr = retraction(Xtr)
fields = {"fixed":              (lambda x,dp=None,t=0: np.tile(w0,(len(x),1)), None),
          "fixed_proj":         (project_constant(Xtr, w0), None),
          "local":              (local_phase_field(Xtr, phitr), None),
          "local_proj":         (local_phase_field(Xtr, phitr, project_m=M_TANGENT), None),
          "transport":          (transport_field(Xtr, w0), None),
          "oracle_phase":       (oracle_phase_field(Xtr, phitr), None),
          "local_proj_retract": (local_phase_field(Xtr, phitr, project_m=M_TANGENT), retr),
          "fixed_proj_retract": (project_constant(Xtr, w0), retr)}
print(f"{'arm':<20}{'peak adv':>10}{'laps':>7}{'@T':>5}{'off@peak':>10}{'|D| min':>9}{'reverse':>9}")
print("-"*70)
for name, (fn, rt) in fields.items():
    pts, dn = steer_path(src, fn, h, T, retract=rt)
    adv = cumulative_advance(judge, pts).mean(1)
    off = np.array([float(nn1.kneighbors(p)[0][:,0].mean()/d0) for p in pts])
    rev, _ = steer_path(src, (lambda f: (lambda x,dp=None,t=0: -f(x, None if dp is None else -dp, t)))(fn),
                        h, T, retract=rt)
    rmin = cumulative_advance(judge, rev).mean(1).min()
    tp = int(adv.argmax())
    dmin = float(np.nanmin(dn[1:]) / (dn[1] + 1e-12))
    print(f"{name:<20}{adv.max():>+10.2f}{adv.max()/(2*np.pi):>+7.2f}{tp:>5}{off[tp]:>10.2f}"
          f"{dmin:>9.2f}{rmin:>+9.2f}")
print(f"\nPREDICTED (before running): fixed/fixed_proj stall at pi/2 = +1.571 rad (0.25 laps), fixed_proj doing")
print(f"so ON the manifold; local_proj traverses. And the CONTROL: the retraction must NOT rescue a fixed")
print(f"direction -- if it did, the stall would be an integrator artifact rather than the arithmetic of a loop.")
