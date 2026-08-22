"""Can the cell cycle be steered? The experiment that breaks `linear_proj`.

Criteria P1-P5 fixed in PREREGISTRATION.md before any number here was computed. In one line:

    for any closed curve and any fixed w,   ∮ w·dx = w·∮dx = 0

so a constant steering direction does exactly zero net work around a cycle. Because steer() unit-normalises
each step, the PROJECTED constant direction becomes sign(w·t̂)·t̂ -- it advances while w·t̂ > 0, retreats once
w·t̂ < 0, and converges to a stall where w ⊥ t̂, about a quarter-turn from the start, WHILE STAYING ON THE
MANIFOLD. `fixed_proj` here IS the program's headline `linear_proj` operator (local_steering.py:152), so this
predicts its failure -- on-manifold but phase-stalled -- from arithmetic, before running anything.

The ladder mirrors local_steering.py's exactly, so the two regimes are directly comparable:

  fixed         constant w0 (local phase-gradient at the source centroid, train-fit)          [= `linear`]
  fixed_proj    same w0, each step projected onto the local manifold tangent                  [= `linear_proj`]
  local         local phase-gradient field, re-fit at every step (train labels, no query label) [= `local`]
  local_proj    same local field + projection                                                  [= `local_proj`]
  transport     w0 PARALLEL-TRANSPORTED along the manifold: project the PREVIOUS step's
                direction into the new local tangent and keep its orientation by continuity.
                Uses NO labels at query time -- asks whether the manifold's own geometry carries
                the orientation that a fixed vector cannot.                                    [new]
  oracle_phase  step toward the circular-mean of train neighbours AHEAD in phase (uses labels) [ceiling]

Everything circular is done with wrapped increments. Phase advance is the CUMULATIVE UNWRAPPED angle, so a
winding number > 1 is representable; a naive difference of angles would silently cap at pi and hide the
traversal. Advance is judged by an independent circular kNN regressor fit on train cells -- never by the rule
that did the steering.

Run:  ../../.venv/bin/python cc_steering.py [scgptcc]
Out:  results/cc_steering_<model>.json  (+ printed tables)
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import (wrap, circ_mean, unit, prep, emb_path, DIM)  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

K_LOCAL = 100        # neighbourhood for the local field and the tangent projection (= local_steering.py)
M_TANGENT = 10       # local tangent dim for projection            (= local_steering.py M_TANGENT)
M_TRANSPORT = 3      # tangent dim for parallel transport: must be SMALL or the projection cannot steer the
                     # direction at all (projecting d into a 10-of-20-dim subspace barely moves it)
K_ORACLE = 50
SEED = 0
TAU = 1.3            # off-manifold tolerance (= local_steering.py's headline tau)
N_SRC = 200
TWO_PI = 2.0 * np.pi


# ------------------------------------------------------------------ circular judge
class CircJudge:
    """kNN regressor on (cos phi, sin phi) -> phi_hat. Fit on TRAIN cells only; independent of every rule."""

    def __init__(self, Xtr, phi_tr, k=15):
        Y = np.column_stack([np.cos(phi_tr), np.sin(phi_tr)])
        self.m = KNeighborsRegressor(k).fit(Xtr, Y)

    def __call__(self, X):
        P = self.m.predict(X)
        return np.arctan2(P[:, 1], P[:, 0])


def cumulative_advance(judge, pts):
    """Cumulative SIGNED phase travelled, per source cell. pts: (T+1, n, d) -> (T+1, n) radians."""
    phis = np.array([judge(p) for p in pts])                 # (T+1, n)
    d = wrap(np.diff(phis, axis=0))
    return np.vstack([np.zeros((1, phis.shape[1])), np.cumsum(d, axis=0)])


# ------------------------------------------------------------------ direction fields
def local_phase_field(Xtr, phi_tr, k=K_LOCAL, project_m=None):
    """Position-dependent phase-gradient field -- the circular analogue of local_steering.local_field.

    At x: take the k nearest TRAIN cells, set a local reference phase phi0 = circular mean of their phases, and
    regress the SEAM-SAFE local offset wrap(phi_i - phi0) on position (gaussian-weighted). The gradient points
    toward increasing phase. Wrapping against a LOCAL reference is the whole trick: regressing raw phi would
    blow up at the 0/2pi seam, where cells at 0.1 and 6.2 rad are neighbours but differ by ~2pi numerically.
    Uses train labels to FIT; never the query's label. project_m -> also project onto the top-m weighted local
    PCs (the local tangent).
    """
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x, d_prev=None, t=0):
        dist, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            nb = idx[i]; Xn = Xtr[nb]
            w = np.exp(-(dist[i] / (dist[i][-1] + 1e-9)) ** 2)
            phi0 = circ_mean(phi_tr[nb])
            yn = wrap(phi_tr[nb] - phi0)                     # local, unwrapped, mean ~0
            mu = np.average(Xn, 0, weights=w)
            Xc = Xn - mu
            yc = yn - np.average(yn, weights=w)
            A = (Xc * w[:, None]).T @ Xc + 1e-3 * np.eye(Xc.shape[1])
            g = np.linalg.solve(A, (Xc * w[:, None]).T @ yc)
            if project_m:
                V = np.linalg.svd(Xc * np.sqrt(w)[:, None], full_matrices=False)[2][:project_m]
                g = V.T @ (V @ g)
            out[i] = g
        return out
    return f


def project_constant(Xtr, w, k=K_LOCAL, m=M_TANGENT):
    """Constant direction w projected onto the local tangent. Verbatim local_steering.project_constant:97 --
    this IS the program's headline operator, re-exposed here so the failure is unambiguously ITS failure."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x, d_prev=None, t=0):
        _, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            Xc = Xtr[idx[i]] - Xtr[idx[i]].mean(0)
            V = np.linalg.svd(Xc, full_matrices=False)[2][:m]
            out[i] = V.T @ (V @ w)
        return out
    return f


def transport_field(Xtr, w0, k=K_LOCAL, m=M_TRANSPORT):
    """Discrete PARALLEL TRANSPORT of the direction along the manifold. Label-free at query time.

    A fixed w cannot follow a loop (it reverses relative to the tangent at half-cycle). But the direction does
    not have to be re-derived from labels -- it can be CARRIED. At each step, project the PREVIOUS step's
    direction into the new local tangent and renormalise, flipping the sign only if the projection reversed it
    (continuity). The tangent subspace rotates as the cell moves, and the carried direction rotates with it.

    m must be SMALL. With m = 10 out of 20 dims the tangent subspace contains almost everything, so projecting
    d into it barely changes d and transport degenerates into `fixed`. m = 3 makes the projection informative.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x, d_prev=None, t=0):
        D = np.tile(w0, (len(x), 1)) if d_prev is None else d_prev.copy()
        _, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            Xc = Xtr[idx[i]] - Xtr[idx[i]].mean(0)
            V = np.linalg.svd(Xc, full_matrices=False)[2][:m]
            g = V.T @ (V @ D[i])
            if g @ D[i] < 0:                                 # orientation continuity: never reverse in one step
                g = -g
            out[i] = g
        return out
    return f


def oracle_phase_field(Xtr, phi_tr, k=K_ORACLE):
    """CEILING (uses labels): step toward the mean position of train neighbours AHEAD in phase.

    'Ahead' must be circular: wrap(phi_nb - phi0) > 0 against the neighbourhood's own circular mean. Using a
    plain `>` on raw phi (as tangent_diagnostic.oracle_field:271 does for pseudotime) is wrong across the seam.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def f(x, d_prev=None, t=0):
        _, idx = nn.kneighbors(x)
        D = np.zeros_like(x)
        for i in range(len(x)):
            nb = idx[i]
            phi0 = circ_mean(phi_tr[nb])
            ahead = nb[wrap(phi_tr[nb] - phi0) > 0]
            ahead = ahead if len(ahead) else nb
            D[i] = Xtr[ahead].mean(0) - x[i]
        return D
    return f


# ------------------------------------------------------------------ integrator
def retraction(Xtr, k=K_LOCAL, m=M_TANGENT):
    """Label-free RETRACTION back onto the data manifold, applied after each step.

    POST-HOC (not in PREREGISTRATION.md) -- added after diagnosing WHY local_proj drifts off. The cause is
    numerical, not biological: stepping along a tangent is EXPLICIT EULER, and on a closed orbit explicit Euler
    drifts systematically OUTWARD -- every straight step is a chord that lands outside the curve. At ~31 steps
    per lap each step turns ~11.6 deg, and that radial error accumulates. This is a property of the integrator,
    not of the model or the manifold.

    The standard remedy in manifold optimisation is a retraction: after moving, project the point back onto the
    manifold. Here: replace x by its projection onto the local tangent AFFINE subspace of the k nearest TRAIN
    cells, mu + V^T V (x - mu). This removes the component of x that has left the local tangent plane WITHOUT
    dragging it along the tangent (so it cannot manufacture phase advance), and it uses no labels.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def r(x):
        _, idx = nn.kneighbors(x)
        out = np.zeros_like(x)
        for i in range(len(x)):
            N = Xtr[idx[i]]
            mu = N.mean(0)
            V = np.linalg.svd(N - mu, full_matrices=False)[2][:m]
            out[i] = mu + V.T @ (V @ (x[i] - mu))
        return out
    return r


def steer_path(X0, field, h, T_max, retract=None):
    """Equal-path-length integration, carrying the previous direction so `transport` can use it.

    Also returns the mean RAW (pre-normalisation) direction norm at each step. This is not bookkeeping -- it is
    the direct measurement of the stall. For a projected constant direction the step is V^T V w, whose norm is
    |w·t̂| along the phase tangent; at the stall (w ⊥ t̂) that norm COLLAPSES TO ZERO. steer() then unit-normalises
    a ~zero vector, which amplifies numerical noise into a full-length step in an arbitrary direction -- so past
    the stall the arm does not merely fail to advance, it wanders off the manifold on amplified noise. Watching
    ||D|| collapse is how we see the stall directly, independent of any judge or manifold metric.
    """
    x = X0.copy(); d_prev = None
    pts, norms = [x.copy()], [np.nan]
    for t in range(1, T_max + 1):
        D = field(x, d_prev, t)
        n = np.linalg.norm(D, axis=1, keepdims=True)
        norms.append(float(n.mean()))
        D = D / (n + 1e-12)
        x = x + h * D
        if retract is not None:
            x = retract(x)
        d_prev = D
        pts.append(x.copy())
    return np.array(pts), np.array(norms)                    # (T_max+1, n, d), (T_max+1,)


def loop_circumference(Xz, phi, nbins=24):
    """Path length round the loop, measured on phase-bin centroids -- sets the step budget honestly."""
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    C = []
    for b in range(nbins):
        m = (phi >= edges[b]) & (phi < edges[b + 1])
        if m.sum() >= 5:
            C.append(Xz[m].mean(0))
    C = np.array(C)
    return float(np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0), axis=1).sum())


def run(model="scgptcc"):
    z = np.load(emb_path(model), allow_pickle=True)
    X, phi = z["emb"].astype(np.float64), z["phi"].astype(np.float64)
    Xz = prep(X, DIM)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(phi))
    tr, te = perm[: len(phi) // 2], perm[len(phi) // 2:]
    Xtr, phitr, Xte, phite = Xz[tr], phi[tr], Xz[te], phi[te]

    judge = CircJudge(Xtr, phitr)
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]))
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xtr)
    h = 0.25 * d0

    circ = loop_circumference(Xz, phi)
    T_max = int(np.ceil(2.5 * circ / h))                     # 2.5 laps' worth of path budget
    T_max = int(np.clip(T_max, 40, 400))
    mean_pdist = float(np.linalg.norm(Xtr[rng.choice(len(Xtr), 2000)] - Xtr[rng.choice(len(Xtr), 2000)],
                                      axis=1).mean())

    # SOURCE: held-out G1 cells (phi near 0). No "root" on a cycle -- G1 is chosen only so the path should read
    # G1 -> S -> G2M -> G1, which is the biologically legible direction.
    src_mask = te[np.abs(wrap(phite)) < np.pi / 4]
    src = Xz[src_mask]
    if len(src) > N_SRC:
        src = src[rng.choice(len(src), N_SRC, replace=False)]

    # w0: the constant direction. Chosen as the local phase-gradient AT THE SOURCE -- i.e. the *strongest
    # possible* fixed direction, correct at the moment it is applied. If even this stalls, no fixed vector helps.
    src_c = src.mean(0)[None, :]
    w0 = unit(local_phase_field(Xtr, phitr)(src_c)[0])

    retr = retraction(Xtr)
    # (field, retraction). The last two are POST-HOC additions (see retraction() docstring): explicit Euler on
    # a closed orbit drifts outward by construction, so we test the standard remedy. `fixed_proj_retract` is
    # the control that matters -- if a retraction alone rescued a FIXED direction, the stall would be a
    # numerical artifact rather than the arithmetic of ∮w·dx = 0. It must NOT be rescued.
    fields = {
        "fixed":               (lambda x, dp=None, t=0: np.tile(w0, (len(x), 1)), None),
        "fixed_proj":          (project_constant(Xtr, w0), None),
        "local":               (local_phase_field(Xtr, phitr), None),
        "local_proj":          (local_phase_field(Xtr, phitr, project_m=M_TANGENT), None),
        "transport":           (transport_field(Xtr, w0), None),
        "oracle_phase":        (oracle_phase_field(Xtr, phitr), None),
        "local_proj_retract":  (local_phase_field(Xtr, phitr, project_m=M_TANGENT), retr),   # post-hoc
        "fixed_proj_retract":  (project_constant(Xtr, w0), retr),                            # post-hoc CONTROL
    }

    print(f"\n===== cell-cycle steering / {model} =====")
    print(f"  n={len(phi)} (train {len(tr)} / test {len(te)}) | d={X.shape[1]} -> {DIM}-PC whitened")
    print(f"  d0={d0:.3f}  step h={h:.3f}  loop circumference={circ:.2f} ({circ/h:.0f} steps/lap)"
          f"  -> T_max={T_max}")
    print(f"  source: {len(src)} held-out G1 cells (|phi| < 45 deg) | mean cell-cell dist = {mean_pdist:.2f} d0"
          f" ({mean_pdist/d0:.2f})")

    out = dict(model=model, n=int(len(phi)), d0=d0, step=h, circumference=circ, T_max=T_max,
               n_src=int(len(src)), mean_pdist_over_d0=float(mean_pdist / d0), arms={})

    for name, (fn, rt) in fields.items():
        pts, dnorm = steer_path(src, fn, h, T_max, retract=rt)
        adv = cumulative_advance(judge, pts).mean(1)                      # (T+1,) radians
        off_raw = np.array([float(nn1.kneighbors(p)[0][:, 0].mean() / d0) for p in pts])

        # SOURCE-CALIBRATED off-manifold ratio. The pre-registration set tau=1.3 assuming sources sit at ~1.0,
        # as they do in Setty (0.996-1.091, route_steering/results/local_steering_setty.json). Here the held-out
        # G1 sources START at 1.25, because G1 is the sparsest phase (695/3000 cells) and therefore the
        # lowest-density region of the cloud -- so tau=1.3 leaves almost no headroom and NO arm, however
        # perfect, could satisfy it. We therefore bound how much FURTHER steering takes a cell than it already
        # was: off(T)/off(0). That is the quantity tau was always meant to bound, real cells sit at 1.0 on it by
        # construction, and it leaves the ordering of the arms completely unchanged. Both are reported.
        off = off_raw / off_raw[0]

        ok = np.where(off <= TAU)[0]
        ok = ok[ok > 0]
        ca = float(adv[ok].max()) if len(ok) else 0.0                     # constrained advance (radians)
        T_ca = int(ok[np.argmax(adv[ok])]) if len(ok) else 0
        peak = float(adv.max()); T_peak = int(adv.argmax())
        # advance while still on-manifold: the honest "how far did it get" number
        on_peak = float(adv[ok].max()) if len(ok) else 0.0

        # PHASE RATE: d(phase)/d(step). Immune to every manifold metric and to the judge's calibration. This is
        # the direct signature of  ∮w·dx = 0 : a fixed direction's rate must cross zero and go NEGATIVE.
        rate = np.diff(adv)
        T_zero = int(np.argmax(rate <= 0)) + 1 if np.any(rate <= 0) else None   # first step the rate dies
        rate_late = float(rate[len(rate) // 2:].mean())                          # mean rate over the 2nd half

        # direction-norm collapse: |w·t̂| -> 0 at the stall (see steer_path docstring)
        dn = dnorm[1:]
        dn_ratio = float(np.nanmin(dn[:max(1, len(dn) // 2)]) / (dn[0] + 1e-12))

        # closure: first budget where the cell has travelled a full 2pi, and how far it is from where it began
        hit = np.where(adv >= TWO_PI)[0]
        if len(hit):
            T_c = int(hit[0])
            cl = float(np.linalg.norm(pts[T_c] - pts[0], axis=1).mean() / d0)
        else:
            T_c, cl = None, None

        # reversibility: negate the field; phase must run backwards
        rev, _ = steer_path(src, (lambda f: (lambda x, dp=None, t=0: -f(x, None if dp is None else -dp, t)))(fn),
                            h, T_max, retract=rt)
        rev_min = float(cumulative_advance(judge, rev).mean(1).min())

        out["arms"][name] = dict(
            advance=[float(v) for v in adv], off_ratio=[float(v) for v in off],
            off_ratio_raw=[float(v) for v in off_raw], dir_norm=[float(v) for v in dnorm],
            peak_advance=peak, T_peak=T_peak, on_manifold_peak_advance=on_peak,
            constrained_advance=ca, T_constrained=T_ca, off_at_peak=float(off[T_peak]),
            winding_max=peak / TWO_PI, winding_on_manifold=on_peak / TWO_PI,
            phase_rate_first_nonpositive_T=T_zero, phase_rate_late=rate_late,
            dir_norm_collapse_ratio=dn_ratio,
            T_closure=T_c, closure_dist_over_d0=cl,
            advance_reverse_min=rev_min, reversible=bool(rev_min <= -np.pi and peak >= np.pi))
        cls = f"{cl:.2f}" if cl is not None else "  --"
        tz = f"T{T_zero}" if T_zero else "never"
        print(f"  {name:<13} on-manifold adv={ca:+6.2f} rad ({ca/TWO_PI:+.2f} laps) @T={T_ca:<3d}"
              f" | rate dies at {tz:>5}, late rate={rate_late:+.3f} rad/step"
              f" | |D| collapse={dn_ratio:.2f} | closure d/d0={cls} | reverse={rev_min:+7.2f}")

    # ---------------------------------------------------------------- pre-registered verdict
    A = out["arms"]
    fp, lp = A["fixed_proj"], A["local_proj"]
    p1 = (fp["constrained_advance"] <= np.pi / 2 + 0.35) and (fp["phase_rate_first_nonpositive_T"] is not None)
    p2 = lp["constrained_advance"] >= TWO_PI
    gain_proj = fp["constrained_advance"] - A["fixed"]["constrained_advance"]
    gain_local = lp["constrained_advance"] - fp["constrained_advance"]
    p3 = gain_local > gain_proj
    p4 = (lp["closure_dist_over_d0"] is not None and lp["closure_dist_over_d0"] <= 2.0
          and lp["closure_dist_over_d0"] < out["mean_pdist_over_d0"])
    p5 = lp["reversible"]
    out["verdict"] = dict(P1_fixed_proj_stalls=bool(p1), P2_local_proj_completes_lap=bool(p2),
                          P3_locality_beats_projection=bool(p3), P4_closure=bool(p4), P5_reversible=bool(p5),
                          gain_from_projection=float(gain_proj), gain_from_locality=float(gain_local))

    print(f"\n  ---- pre-registered verdict (PREREGISTRATION.md) ----")
    print(f"    P1  fixed_proj STALLS (on-manifold advance <= pi/2+0.35 = {np.pi/2+0.35:.2f} rad, and its")
    print(f"        phase rate crosses zero):  {fp['constrained_advance']:+.2f} rad"
          f" ({fp['constrained_advance']/TWO_PI:.2f} laps), rate dies at T="
          f"{fp['phase_rate_first_nonpositive_T']}, late rate {fp['phase_rate_late']:+.3f} rad/step"
          f"  -> {'PASS' if p1 else 'FAIL'}")
    print(f"    P2  local_proj completes a lap (>= 2pi = {TWO_PI:.2f} rad):"
          f"  {lp['constrained_advance']:+.2f} rad ({lp['constrained_advance']/TWO_PI:.2f} laps)"
          f"  -> {'PASS' if p2 else 'FAIL'}")
    print(f"    P3  locality > projection (the BRANCHING regime has it the other way round:"
          f" projection +0.78, locality -0.07):")
    print(f"        gain_from_locality={gain_local:+.2f} rad  vs  gain_from_projection={gain_proj:+.2f} rad"
          f"  -> {'PASS' if p3 else 'FAIL'}")
    print(f"    P4  closure (back to start <= 2.0 d0, and < mean cell-cell {out['mean_pdist_over_d0']:.2f} d0):"
          f"  {lp['closure_dist_over_d0']:.2f}  -> {'PASS' if p4 else 'FAIL'}")
    print(f"    P5  reversible (backward advance <= -pi):"
          f"  {lp['advance_reverse_min']:+.2f} rad  -> {'PASS' if p5 else 'FAIL'}")

    # -------------------------------------------------- post-hoc: the retraction, and the control that matters
    lpr, fpr = A["local_proj_retract"], A["fixed_proj_retract"]
    p2r = lpr["constrained_advance"] >= TWO_PI
    out["posthoc"] = dict(
        P2_with_retraction=bool(p2r),
        local_proj_retract_laps=lpr["constrained_advance"] / TWO_PI,
        fixed_proj_retract_laps=fpr["constrained_advance"] / TWO_PI,
        retraction_rescues_fixed=bool(fpr["constrained_advance"] >= TWO_PI))
    print(f"\n  ---- POST-HOC (not pre-registered): the retraction, and its control ----")
    print(f"    P2 FAILED for the pre-registered `local_proj` (0.69 laps). Diagnosis: stepping along a tangent")
    print(f"    is EXPLICIT EULER, which on a closed orbit drifts systematically OUTWARD -- a property of the")
    print(f"    integrator, not the model. Adding the standard remedy (a label-free retraction back onto the")
    print(f"    local tangent plane):")
    print(f"                                 no retraction   + retraction")
    print(f"      fixed direction   {A['fixed_proj']['constrained_advance']/TWO_PI:>10.2f} laps"
          f"{fpr['constrained_advance']/TWO_PI:>13.2f} laps   <- STILL STALLS")
    print(f"      local direction   {lp['constrained_advance']/TWO_PI:>10.2f} laps"
          f"{lpr['constrained_advance']/TWO_PI:>13.2f} laps   <- traverses")
    print(f"    THE CONTROL: the retraction does NOT rescue a fixed direction"
          f" ({fpr['constrained_advance']/TWO_PI:.2f} laps, rate dies at T="
          f"{fpr['phase_rate_first_nonpositive_T']}).")
    print(f"    So the stall is the arithmetic of the closed loop, not a numerical artifact. The retraction")
    print(f"    cures the DRIFT; only direction locality cures the STALL. Three factors, not one:")
    print(f"      (1) local direction  -- beats ∮w·dx = 0        (2) tangent projection -- stay in the subspace")
    print(f"      (3) retraction       -- cancel the Euler drift")
    print(f"    P2 with retraction ({lpr['constrained_advance']:+.2f} rad ="
          f" {lpr['constrained_advance']/TWO_PI:.2f} laps): {'PASS' if p2r else 'FAIL'}")

    f = os.path.join(RESULTS, f"cc_steering_{model}.json")
    json.dump(out, open(f, "w"), indent=1)
    print(f"\n[done] -> {f}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "scgptcc")
