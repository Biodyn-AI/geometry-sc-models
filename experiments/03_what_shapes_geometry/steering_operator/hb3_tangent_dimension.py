"""H-B3 — is local intrinsic (tangent) dimension a potency measure, and does it drop DISCONTINUOUSLY at commitment?

Local tangent dimension = participation ratio of the local covariance spectrum in a cell's k-NN neighbourhood,
PR = (sum lambda)^2 / sum lambda^2. A multipotent cell should sit where several fates are locally accessible
(higher-dim tangent); a committed cell has one direction left (lower-dim). Three tests:

  B3a POTENCY     Spearman rho(local_dim, pseudotime) < 0, with a BLOCKED-permutation p (pseudotime is
                  autocorrelated, so the null shuffles dim within contiguous pseudotime blocks).
  B3b GROUPS      local_dim higher in multipotent clusters (HSC/Precursors) than terminal clusters; Cohen's d.
  B3c STEP        the interesting claim: commitment is a discrete event, so local_dim vs pseudotime should be a
                  STEP, not a ramp. Fit a sigmoid (low, high, midpoint t*, width w) and a line; report the sigmoid
                  transition WIDTH as a fraction of the pseudotime range (narrow = step) and BIC(sigmoid)<BIC(line).

Run in BOTH the model's whitened space and the raw-counts whitened space (Gate 0 showed the geometry is the
data's, so a real potency signal should appear in counts too; the model may or may not sharpen it).

Run:  ../../.venv/bin/python hb3_tangent_dimension.py [model ...]
Out:  results/hb3_tangent_dimension.json
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import load, prep, RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from gene_decode import load_expression  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MODELS = ["scgptbin", "geneformer", "state", "maxtoki"]
MULTIPOTENT = ["HSC_1", "HSC_2", "Precursors"]
TERMINAL = ["Ery_1", "Ery_2", "Mono_1", "Mono_2", "CLP", "Mega"]
K_NB = 100


def local_dim(Xz, k=K_NB):
    """Participation ratio of the local covariance spectrum in each cell's k-NN neighbourhood."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xz)
    _, idx = nn.kneighbors(Xz)
    pr = np.zeros(len(Xz))
    for i in range(len(Xz)):
        Xc = Xz[idx[i]] - Xz[idx[i]].mean(0)
        lam = np.linalg.svd(Xc, compute_uv=False) ** 2
        pr[i] = (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)
    return pr


def blocked_perm_p(dim, y, rho_obs, n=2000, nblock=20, seed=SEED):
    """One-sided p for a NEGATIVE rho, shuffling dim within contiguous pseudotime blocks (autocorrelation-safe)."""
    order = np.argsort(y); rng = np.random.default_rng(seed)
    blocks = np.array_split(order, nblock)
    null = np.empty(n)
    for j in range(n):
        d2 = dim.copy()
        for b in blocks:
            d2[b] = dim[b][rng.permutation(len(b))]
        null[j] = spearmanr(d2, y).statistic
    return float((np.sum(null <= rho_obs) + 1) / (n + 1))


def sigmoid(t, lo, hi, t0, w):
    return lo + (hi - lo) / (1.0 + np.exp((t - t0) / (w + 1e-9)))


def step_test(dim, y):
    """Fit sigmoid vs line to dim~pseudotime; return transition width (frac of pt range) and BIC gap."""
    t = (y - y.min()) / (np.ptp(y) + 1e-9)
    d = (dim - dim.mean()) / (dim.std() + 1e-9)
    n = len(t)
    # line
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    sse_lin = float(((A @ coef - d) ** 2).sum())
    # sigmoid (dim usually falls with time -> hi at low t, lo at high t)
    try:
        p0 = [d[t > 0.7].mean(), d[t < 0.3].mean(), 0.5, 0.1]
        popt, _ = curve_fit(sigmoid, t, d, p0=p0, maxfev=20000)
        sse_sig = float(((sigmoid(t, *popt) - d) ** 2).sum())
        width = abs(popt[3]) * 4.4        # ~10-90% transition width in units of t (frac of range)
        t0 = float(popt[2])
    except Exception:
        sse_sig, width, t0 = sse_lin, float("nan"), float("nan")
    bic_lin = n * np.log(sse_lin / n + 1e-12) + 2 * np.log(n)
    bic_sig = n * np.log(sse_sig / n + 1e-12) + 4 * np.log(n)
    return dict(transition_width_frac=float(width), transition_midpoint=t0,
                r2_lin=float(1 - sse_lin / (n * d.var())), r2_sig=float(1 - sse_sig / (n * d.var())),
                bic_gap=float(bic_lin - bic_sig))       # >0 favours the step


def analyse(Xz, y, clu, tag):
    dim = local_dim(Xz)
    rho = spearmanr(dim, y).statistic
    p = blocked_perm_p(dim, y, rho)
    mm = np.isin(clu, MULTIPOTENT); tt = np.isin(clu, TERMINAL)
    d_hi, d_lo = dim[mm], dim[tt]
    pooled = np.sqrt((d_hi.var() + d_lo.var()) / 2) + 1e-9
    cohen = float((d_hi.mean() - d_lo.mean()) / pooled)
    st = step_test(dim, y)
    rec = dict(tag=tag, mean_dim=float(dim.mean()), rho_dim_pt=float(rho), p_blocked=float(p),
               dim_multipotent=float(d_hi.mean()), dim_terminal=float(d_lo.mean()), cohen_d=cohen, **st)
    print(f"  [{tag:6s}] mean_dim={dim.mean():4.1f} | rho(dim,pt)={rho:+.2f} (p={p:.3f}) | "
          f"MPP {d_hi.mean():.1f} vs terminal {d_lo.mean():.1f} (d={cohen:+.2f}) | "
          f"step: width={st['transition_width_frac']:.2f} t0={st['transition_midpoint']:.2f} "
          f"R2 lin/sig={st['r2_lin']:.2f}/{st['r2_sig']:.2f} BICgap={st['bic_gap']:+.0f}")
    return rec


def run_model(model):
    X, y, ci, clu = load(model)
    print(f"\n===== {model}  (n={len(y)}) =====")
    rec = dict(model=model, n=int(len(y)),
               model_space=analyse(prep(X), y, clu, "model"),
               counts_space=analyse(prep(load_expression(ci)[0]), y, clu, "counts"))
    return rec


def main():
    models = sys.argv[1:] or MODELS
    out = {}
    for m in models:
        try:
            out[m] = run_model(m)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[skip] {m}: {e}")
    json.dump(out, open(os.path.join(RESULTS, "hb3_tangent_dimension.json"), "w"), indent=1)
    print("\n================ H-B3 VERDICT ================")
    print(f"  {'model':<12} {'space':<7} {'rho(dim,pt)':>11} {'p':>7} {'cohen_d':>8} {'stepWidth':>10} {'BICgap':>8}")
    for m, r in out.items():
        for sp in ("model_space", "counts_space"):
            s = r[sp]
            print(f"  {m:<12} {sp.split('_')[0]:<7} {s['rho_dim_pt']:>+11.2f} {s['p_blocked']:>7.3f} "
                  f"{s['cohen_d']:>+8.2f} {s['transition_width_frac']:>10.2f} {s['bic_gap']:>+8.0f}")
    print("  Potency: rho<0 & d>0 = local dim falls as cells mature & is higher in multipotent cells.")
    print("  Step: small transition width + positive BIC gap = discontinuous drop at commitment.")
    print("[done] -> results/hb3_tangent_dimension.json")


if __name__ == "__main__":
    main()

