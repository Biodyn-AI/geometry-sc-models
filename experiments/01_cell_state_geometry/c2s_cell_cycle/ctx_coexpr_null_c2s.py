"""ctx_coexpr_null_c2s — HEADLINE #3 (decisive): is the functional-axis modulation BEYOND co-expression?
Port of ctx_coexpr_null.py.

Build the co-expression matrix C (gene x gene Pearson over the panel cells). Generate N_NULL synthetic
module-axes spanning coherence (looseness knob), record (coherence, rank-residualised along-axis power) ->
a power~coherence baseline CURVE (quadratic fit + residual scatter). Score each GO axis by its residual above
that curve: z_above = (power - curve_pred(coherence)) / resid_sd. Verdict BEYOND iff best z_above > 3.
Out: results/ctx_coexpr_null_c2s.json
"""
import os, sys, json, argparse, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_lib_c2s as L
from ctx_functional_axes_c2s import AXES, G2G_PATH, pole_rows

N_NULL = 400
IAN = IBN = 400
SEED = 0


def build_coexpr(panel_h5ad, genes):
    """(nG, nG) Pearson correlation over panel cells, aligned to `genes` (upper symbols)."""
    import anndata, scipy.sparse as sp
    a = anndata.read_h5ad(panel_h5ad)
    var_up = np.char.upper(np.asarray(a.var_names).astype(str))
    col = {s: i for i, s in enumerate(var_up)}
    cols = np.array([col.get(g, -1) for g in genes])
    ok = cols >= 0
    X = a.X[:, cols[ok]]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    tot = X.sum(1, keepdims=True); tot[tot == 0] = 1
    E = np.log1p(X / tot * 1e4)
    Z = (E - E.mean(0)) / (E.std(0) + 1e-8)
    C = np.full((len(genes), len(genes)), 0.0)
    Cok = (Z.T @ Z) / len(Z)
    ii = np.where(ok)[0]
    C[np.ix_(ii, ii)] = Cok
    return C, ok


def coherence(C, idx):
    idx = np.asarray(idx)
    if len(idx) < 2:
        return 0.0
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    return float(sub[iu].mean())


def for_tap(tap, bases, g2g, C, gene_ok):
    d = L.load(tap, bases)
    full = L.balanced(d["counts"], d["cap"])
    Mz, _ = L.zscore_dims(d["M"], full)
    A = L.a_space(Mz, full)
    rank = L.rank_mean(d["rank_tok"])
    ok = gene_ok & np.isfinite(A).all(1)
    pool = np.where(ok)[0]
    rng = np.random.default_rng(SEED)

    def axis(ia, ib):
        return L.axis_from_poles(A, None, np.asarray(ia), np.asarray(ib))

    def power(vec):
        return L.axis_power(L.along_axis_interaction(Mz, full, vec, rank))   # rank-residualised

    def module(size, loose):
        seed = rng.choice(pool)
        nb = pool[np.argsort(-C[seed, pool])]
        top = nb[:min(len(nb), int(size * loose))]
        return rng.choice(top, min(size, len(top)), replace=False)

    nx, ny = [], []
    for _ in range(N_NULL):
        loose = rng.choice([1, 1, 2, 3, 5, 8, 15, 40, 120])
        ga, gb = module(IAN, loose), module(IBN, loose)
        nx.append(0.5 * (coherence(C, ga) + coherence(C, gb))); ny.append(power(axis(ga, gb)))
    nx, ny = np.array(nx), np.array(ny)
    Amat = np.column_stack([np.ones_like(nx), nx, nx ** 2])
    coef, *_ = np.linalg.lstsq(Amat, ny, rcond=None)
    resid_sd = float((ny - Amat @ coef).std()) + 1e-9
    rho_cc = float(spearmanr(nx, ny).statistic)
    out = {"coexpr_power_coherence_rho": rho_cc, "null_coh_range": [float(nx.min()), float(nx.max())], "axes": {}}
    for name, (Ag, Bg) in AXES.items():
        ia, ib = pole_rows(d["genes"], g2g, Ag, Bg)
        ia = ia[np.isin(ia, pool)]; ib = ib[np.isin(ib, pool)]
        if len(ia) < 6 or len(ib) < 6:
            out["axes"][name] = dict(skipped=f"A={len(ia)} B={len(ib)}"); continue
        coh = 0.5 * (coherence(C, ia) + coherence(C, ib))
        pw = power(axis(ia, ib))
        pred = float(np.array([1, coh, coh ** 2]) @ coef)
        z = (pw - pred) / resid_sd
        out["axes"][name] = dict(coherence=coh, coherence_pctl=float((nx < coh).mean()),
                                 power=pw, curve_pred=pred, z_above_coexpr=float(z))
        print(f"  L{tap:02d} {name:<26} coh={coh:+.3f}(pctl{out['axes'][name]['coherence_pctl']:.2f}) "
              f"power={pw:+.3f} pred={pred:+.3f} -> z_above={z:+.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="ctx_bases")
    ap.add_argument("--panel", default="data/ts_panel_celltype.h5ad")
    ap.add_argument("--taps", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results/ctx_coexpr_null_c2s.json")
    a = ap.parse_args()
    import glob
    g2g = pickle.load(open(G2G_PATH, "rb"))
    taps = a.taps or sorted(int(os.path.basename(p).split("_L")[1].split(".")[0]) for p in glob.glob(os.path.join(a.bases, "ctx_c2s_L*.npz")))
    genes = L.load(taps[0], a.bases)["genes"]
    print("building co-expression matrix...", flush=True)
    C, gene_ok = build_coexpr(a.panel, genes)
    res = {"taps": {}}
    for t in taps:
        res["taps"][f"L{t:02d}"] = for_tap(t, a.bases, g2g, C, gene_ok)
    allz = [(tk, ax, dd["z_above_coexpr"]) for tk, tv in res["taps"].items()
            for ax, dd in tv["axes"].items() if "z_above_coexpr" in dd]
    if allz:
        best = max(allz, key=lambda x: x[2])
        res["verdict"] = (f"BEYOND co-expression ({best[1]} @ {best[0]}, {best[2]:+.1f} sigma above curve)"
                          if best[2] > 3 else
                          f"NOT beyond co-expression (best {best[1]} @ {best[0]} = {best[2]:+.1f} sigma)")
        print(f"\nVERDICT: {res['verdict']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
