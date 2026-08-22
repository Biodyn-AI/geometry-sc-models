"""Loop geometry, tested with the only null that means anything on a closed curve.

THE POINT OF THIS FILE IS MOSTLY NEGATIVE, AND THAT IS BY DESIGN.

The program's cell-cycle "curvature" verdicts (-0.133 Geneformer / -0.124 MaxToki / -0.085 STATE, "a FLAT,
linearly-embedded loop") came from the decodability gap `kNN R^2 - linear R^2`, which route_steering/RESULTS.md
:150 later showed is structurally headroom-limited: with linear circ-R^2 already at 0.905-0.929 there is almost
no variance left for a nonlinear kernel to claim, so the statistic goes silent whether or not the loop is
curved. It is the wrong instrument. But the RIGHT instrument is NOT `theta_far`:

  * `theta_far = theta[0, -1]` (tangent_diagnostic.py:217) is the angle between the FIRST and LAST bin. On a
    circle those bins are ADJACENT. A perfect circle scores theta_far ~ 0 -- the maximally-turning object reads
    as maximally flat. We compute it here anyway, purely to show that number and close the door on porting it.
  * `monotonicity` (Spearman of angle vs bin separation) fails the same way: on a loop the true angle rises to
    180 deg at half-cycle then FALLS, so rho ~ 0 for an ideal circle.
  * All three of its nulls collapse. NULL A's stated premise (:117) is that "a monotone reparametrisation of a
    linear score leaves the tangent constant" -- but NO monotone global scalar exists on a loop, so the null is
    not even definable.

Deeper still: on a loop, "the tangent rotates" is not evidence of anything. A perfectly flat circle sitting in
a linear 2-plane turns its tangent a full 360 deg. Rotation is topologically MANDATORY. So the correct null on
a loop is not "the tangent is constant" but:

        H_flat:  the loop lies in a fixed linear 2-plane,  phi = atan2(a·x, b·x)

and the statistic that can reject it is rotation OUT of the best-fit 2-plane, not rotation per se. We build
that null explicitly (a synthetic planar circle, noise-calibrated to the real embedding's own circular
decodability) and compare.

PRE-REGISTERED EXPECTATION (PREREGISTRATION.md): H_flat is NOT rejected. The existing "flat loop" verdict is
expected to SURVIVE. That is fine, and it is the whole rhetorical point of this route: the steering result in
cc_steering.py does not depend on the loop being curved. A perfectly flat, perfectly linearly-decodable loop
still defeats a fixed steering direction, because what defeats it is that the tangent is POSITION-DEPENDENT --
not that the manifold is bent.

Also implements the fix route_topology/RESULTS.md:27 wrote down and never built: "a cleaner topology probe
would run H1 inside the fitted cell-cycle 2-plane."

Run:  ../../.venv/bin/python cc_geometry.py [scgptcc]     (needs ripser -> .venv, not .venv_state)
Out:  results/cc_geometry_<model>.json
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import wrap, circ_mean, unit, ang, prep, emb_path, DIM  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
NBINS, SEED = 12, 0
N_NULL = 20


# ------------------------------------------------------------------ circular decodability
def circ_r2(Xz, phi, model="linear", seed=SEED):
    """Circular R^2 for predicting phi from Xz. 1 - E[1-cos(err)] / E[1-cos(phi - circmean)]."""
    Y = np.column_stack([np.cos(phi), np.sin(phi)])
    P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xz):
        m = Ridge(alpha=10.0) if model == "linear" else KNeighborsRegressor(15)
        P[te] = m.fit(Xz[tr], Y[tr]).predict(Xz[te])
    err = 1.0 - np.cos(np.arctan2(P[:, 1], P[:, 0]) - phi)
    base = float(np.mean(1.0 - np.cos(phi - circ_mean(phi))))
    return float(1.0 - err.mean() / (base + 1e-12))


# ------------------------------------------------------------------ tangents around the loop
def loop_tangents(Xz, phi, nbins=NBINS):
    """Local phase-gradient direction in each of `nbins` equal phase sectors.

    Within a sector, regress the SEAM-SAFE local offset wrap(phi - phi_centre) on position. Regressing raw phi
    would explode in the sector straddling the 0/2pi seam.
    """
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    dirs, ns = [], []
    for b in range(nbins):
        m = (phi >= edges[b]) & (phi < edges[b + 1])
        idx = np.where(m)[0]
        if len(idx) < 20:
            dirs.append(None); ns.append(len(idx)); continue
        c = 0.5 * (edges[b] + edges[b + 1])
        t = wrap(phi[idx] - c)
        dirs.append(unit(Ridge(alpha=10.0).fit(Xz[idx], t).coef_))
        ns.append(len(idx))
    return dirs, ns


def turning_stats(dirs):
    """Total turning round the loop + planarity of the tangent field.

    total_turning = sum of consecutive tangent-to-tangent angles all the way round (wrapping). For a faithfully
    embedded closed curve this approaches 360 deg -- a parameter-free prediction with no analogue in the
    branching case (the turning-tangent theorem).

    planarity = fraction of the tangent field's energy in its best-fit 2-plane. A planar circle gives 1.0.
    out_of_plane_deg = mean angle by which the local tangents leave that 2-plane -- THE statistic that can
    reject H_flat, and the only sense in which a loop can be "curved" beyond topological necessity.
    """
    D = [d for d in dirs if d is not None]
    if len(D) < 4:
        return dict(total_turning=float("nan"), planarity=float("nan"), out_of_plane_deg=float("nan"),
                    theta_far_broken=float("nan"), monotonicity_broken=float("nan"))
    D = np.array(D); n = len(D)
    total = float(sum(ang(D[i], D[(i + 1) % n]) for i in range(n)))

    U, S, Vt = np.linalg.svd(D - 0.0, full_matrices=False)    # tangent directions are already unit vectors
    planarity = float((S[:2] ** 2).sum() / (S ** 2).sum())
    P2 = Vt[:2]
    oop = float(np.mean([np.degrees(np.arccos(np.clip(np.linalg.norm(P2 @ d), 0, 1))) for d in D]))

    # the two statistics that a naive port of tangent_diagnostic would have used, shown to be meaningless here
    theta_far = ang(D[0], D[-1])                              # bins 0 and -1 are ADJACENT on a circle
    sep = np.array([abs(i - j) for i in range(n) for j in range(i + 1, n)])
    off = np.array([ang(D[i], D[j]) for i in range(n) for j in range(i + 1, n)])
    rho = float(spearmanr(sep, off).statistic)
    return dict(total_turning=total, planarity=planarity, out_of_plane_deg=oop,
                theta_far_broken=theta_far, monotonicity_broken=rho)


# ------------------------------------------------------------------ H_flat: the planar-circle null
def flat_null(phi, d, lin_r2, seed=0):
    """Synthetic embedding whose loop lies EXACTLY in a linear 2-plane, noise-calibrated to match the real
    embedding's circular decodability. This is H_flat made concrete: the honest statement of "the model encodes
    the cell cycle as a plain circle in a plane". Mirrors route_state_geometry/common.py:73-93."""
    rng = np.random.default_rng(seed)
    base = np.column_stack([np.cos(phi), np.sin(phi)])
    A = rng.standard_normal((2, d))
    sig = base @ A
    sig /= sig.std()
    lo, hi = 0.0, 20.0                                        # binary-search noise for matched linear circ-R^2
    for _ in range(18):
        s = 0.5 * (lo + hi)
        E = prep(sig + s * rng.standard_normal(sig.shape), DIM)
        if circ_r2(E, phi, "linear") > lin_r2:
            lo = s
        else:
            hi = s
    return prep(sig + 0.5 * (lo + hi) * rng.standard_normal(sig.shape), DIM)


# ------------------------------------------------------------------ H1 inside the fitted 2-plane
def h1_in_plane(Xz, phi, n_sub=800, seed=0):
    """The probe route_topology/RESULTS.md:27 asked for and never built.

    Global H1 on the raw residual stream is dominated by the ambient variance concentration (Geneformer scored
    z = -10, MaxToki +16.4 with the SAME code -- the sign flip is a representation artifact, not biology). The
    honest question is whether a loop exists in the plane the cell cycle actually occupies. So: fit the 2-plane
    that best carries (cos phi, sin phi), project cells into it, and run H1 there against a null that destroys
    the loop while preserving the 2-D covariance (phase-shuffled cells re-embedded in the same plane).
    """
    from ripser import ripser
    Y = np.column_stack([np.cos(phi), np.sin(phi)])
    B = Ridge(alpha=10.0).fit(Xz, Y).coef_                    # (2, d): the fitted cell-cycle 2-plane
    P = Xz @ B.T
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)

    rng = np.random.default_rng(seed)
    def top_h1(A):
        idx = rng.choice(len(A), min(n_sub, len(A)), replace=False)
        dg = ripser(A[idx], maxdim=1)["dgms"][1]
        if not len(dg):
            return 0.0
        return float(np.sort(dg[:, 1] - dg[:, 0])[::-1][0])

    real = top_h1(P)
    # null: same 2-D marginal covariance, no loop (gaussian matched to the projected cloud)
    cov = np.cov(P, rowvar=False)
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(2))
    nulls = np.array([top_h1(rng.standard_normal(P.shape) @ L.T) for _ in range(10)])
    return dict(h1_real=real, h1_null_mean=float(nulls.mean()), h1_null_sd=float(nulls.std()),
                h1_z=float((real - nulls.mean()) / (nulls.std() + 1e-9)),
                h1_p=float((np.sum(nulls >= real) + 1) / (len(nulls) + 1)))


def run(model="scgptcc"):
    z = np.load(emb_path(model), allow_pickle=True)
    X, phi = z["emb"].astype(np.float64), z["phi"].astype(np.float64)
    Xz = prep(X, DIM)

    lin = circ_r2(Xz, phi, "linear")
    knn = circ_r2(Xz, phi, "knn")
    dirs, ns = loop_tangents(Xz, phi)
    real = turning_stats(dirs)

    nulls = []
    for s in range(N_NULL):
        Xn = flat_null(phi, X.shape[1], lin, seed=100 + s)
        dn, _ = loop_tangents(Xn, phi)
        nulls.append(turning_stats(dn))
    def nstat(k):
        v = np.array([n[k] for n in nulls])
        return float(v.mean()), float(v.std())

    tt_m, tt_s = nstat("total_turning")
    pl_m, pl_s = nstat("planarity")
    oo_m, oo_s = nstat("out_of_plane_deg")
    tf_m, _ = nstat("theta_far_broken")

    # H_flat is rejected only if the real loop leaves its own best-fit 2-plane MORE than a planar circle does
    # at the same noise level. One-sided, 20 draws -> p floor 1/21 = 0.048.
    oop_null = np.array([n["out_of_plane_deg"] for n in nulls])
    p_oop = float((np.sum(oop_null >= real["out_of_plane_deg"]) + 1) / (len(oop_null) + 1))
    rejected = bool(real["out_of_plane_deg"] > np.percentile(oop_null, 95))

    h1 = h1_in_plane(Xz, phi)

    out = dict(model=model, n=int(len(phi)), dim=DIM,
               linear_circ_r2=lin, knn_circ_r2=knn, decodability_gap=knn - lin,
               real=real, bin_n=ns,
               flat_null=dict(total_turning_mean=tt_m, total_turning_sd=tt_s,
                              planarity_mean=pl_m, planarity_sd=pl_s,
                              out_of_plane_mean=oo_m, out_of_plane_sd=oo_s,
                              theta_far_mean=tf_m, n_draws=N_NULL),
               out_of_plane_excess=real["out_of_plane_deg"] - oo_m, p_out_of_plane=p_oop,
               H_flat_rejected=rejected, h1_in_plane=h1)

    print(f"\n===== cell-cycle loop geometry / {model}  (n={len(phi)}, {DIM}-PC whitened) =====")
    print(f"  circular decodability:  linear circ-R2 = {lin:.3f} | kNN = {knn:.3f}"
          f" | decodability gap = {knn-lin:+.3f}")
    print(f"     (the program's 'curvature' statistic. Geneformer -0.133 / MaxToki -0.124 / STATE -0.085.")
    print(f"      With linear R2 this high there is no headroom for a nonlinear kernel -- it cannot speak.)")
    print(f"\n  TOTAL TURNING round the loop = {real['total_turning']:.0f} deg"
          f"   (planar-circle null {tt_m:.0f}+-{tt_s:.0f}; a faithfully embedded closed curve -> 360)")
    print(f"  planarity of the tangent field = {real['planarity']:.3f}  (null {pl_m:.3f}+-{pl_s:.3f};"
          f" a flat circle -> 1.000)")
    print(f"  out-of-plane tangent rotation  = {real['out_of_plane_deg']:.1f} deg"
          f"  (null {oo_m:.1f}+-{oo_s:.1f})  excess = {real['out_of_plane_deg']-oo_m:+.1f} deg,"
          f" p = {p_oop:.3f}")
    print(f"  -> H_flat ('the loop lies in a fixed linear 2-plane'): "
          f"{'REJECTED' if rejected else 'NOT REJECTED'}"
          f"  {'(loop is genuinely curved)' if rejected else '(the loop is FLAT -- as pre-registered)'}")
    print(f"\n  [why theta_far cannot be ported] on this loop theta_far (bin 0 vs bin {NBINS-1}, which are")
    print(f"      ADJACENT on a circle) = {real['theta_far_broken']:.1f} deg, and the planar-circle null gives")
    print(f"      {tf_m:.1f} deg. The maximally-turning object reads as barely turning. monotonicity rho ="
          f" {real['monotonicity_broken']:+.2f}.")
    print(f"\n  H1 INSIDE the fitted cell-cycle 2-plane (the probe route_topology/RESULTS.md:27 asked for):")
    print(f"      persistence {h1['h1_real']:.3f} vs matched-covariance null {h1['h1_null_mean']:.3f}"
          f"+-{h1['h1_null_sd']:.3f}  -> z = {h1['h1_z']:+.1f}, p = {h1['h1_p']:.3f}")

    f = os.path.join(RESULTS, f"cc_geometry_{model}.json")
    json.dump(out, open(f, "w"), indent=1)
    print(f"\n[done] -> {f}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "scgptcc")
