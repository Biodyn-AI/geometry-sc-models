"""Minimal, fully standalone example — no project data, no foundation model.

Builds a synthetic curved 2-branch "trajectory" (a bent manifold with two arms), then shows that steering a
progenitor toward a branch tip WITH local-tangent projection follows the curve and stays on the data, while
plain-linear steering cuts the corner and leaves it. Run: ../../../.venv/bin/python example.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifold_steer import ManifoldSteer

rng = np.random.default_rng(0)

# synthetic curved manifold: a shared stem that bends, then splits into two arms (like a branch point)
def arm(n, bend, spread=0.05):
    t = np.linspace(0, 1, n)
    x = t
    y = bend * t ** 2                      # curvature: the arm bends as it progresses
    pts = np.stack([x, y], 1) + spread * rng.standard_normal((n, 2))
    return pts, t

stem, ts = arm(400, bend=0.0)             # straight early stem near y=0
armA, ta = arm(400, bend=1.2)             # bends up
armB, tb = arm(400, bend=-1.2)            # bends down
X = np.vstack([stem, armA, armB])         # (1200, 2) the "expression" space
pt = np.concatenate([ts, 1 + ta, 1 + tb])  # a pseudotime

ms = ManifoldSteer(n_pcs=2, k_tangent=60, m_tangent=1, step_frac=0.3).fit(X)

# steer the earliest cells toward arm A's tip, with and without projection
early = ms.transform(X[pt < 0.2])
tipA = ms.transform(armA[-20:]).mean(0)
direction = tipA - early.mean(0)

for project in (True, False):
    traj = ms.steer(early, direction, n_steps=20, project=project)
    off = [ms.off_manifold_ratio(traj[t]) for t in (0, 5, 10, 20)]
    tag = "on-manifold (projected)" if project else "plain linear     "
    print(f"  {tag}: off-manifold ratio @steps 0/5/10/20 = " + " / ".join(f"{o:.2f}" for o in off))

print("\n  The projected path bends with the arm (off-ratio stays ~1); the plain path cuts the corner off the")
print("  manifold (off-ratio climbs). Swap X for your own log-normalized counts and pt for a pseudotime.")

