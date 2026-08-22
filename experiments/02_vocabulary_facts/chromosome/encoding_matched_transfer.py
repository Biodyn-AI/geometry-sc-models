"""encoding_matched_transfer — is C2S better or worse than expression at carrying cell-cycle phase across
cell lines, once BOTH are given the same information?

WHY THIS EXISTS. transfer_diag.py compared the model (R_diff 0.789) against expression (0.878) and the gap read
as a clean model loss. It is not a like-for-like comparison. A C2S cell sentence is the **top-512 genes of that
cell in descending expression order** — magnitudes are discarded and ~92% of the panel is dropped — while the
expression baseline was handed all 6,544 shared genes with continuous magnitudes. The comparison therefore
charged the model for its tokenizer.

WHAT THIS RUNS. The identical frozen-transfer protocol (fit ridge(cos,sin) on K562, apply unchanged to RPE1)
across a ladder of expression encodings that ends at exactly what the model receives:

    expr_full      all shared genes, magnitudes        <- what was compared before
    expr_512_mag   top-512 per cell, magnitudes        <- isolates the truncation loss
    expr_512_rank  top-512 per cell, RANK ONLY         <- what the model actually sees
    model          C2S layer-21 cell-summary activations

and then a PAIRED bootstrap over target cells: the same resampled cell indices are used for every arm, so the
difference between two arms is measured on the same cells and its CI is not inflated by between-cell variance
that both arms share. An unpaired comparison of two 3,000-cell arms cannot resolve a 0.015 gap; a paired one can.

METRICS. R_diff = |mean(exp(i(pred-true)))|, rotation-invariant and fold-free, plus median angular error after
best rotation. Both against a measured constant-predictor floor (~86 deg here), never an assumed 90 deg.
The Jammalamadaka circ_corr is NOT used: the RPE1 phase is near-uniform (R = 0.043) while frozen predictions are
concentrated, which is exactly the regime where it returns the wrong sign.

Out: results/encoding_matched_transfer.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle_oriented


def R_diff(a, b):
    return float(np.abs(np.mean(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def med_err(a, b):
    off = np.angle(np.mean(np.exp(1j * (np.asarray(b) - np.asarray(a)))))
    d = np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b) + off)))
    return float(np.degrees(np.median(np.abs(d))))


def topk_mag(X, k):
    Y = np.zeros_like(X)
    for i in range(len(X)):
        idx = np.argpartition(-X[i], k)[:k]
        Y[i, idx] = X[i, idx]
    return Y


def topk_rank(X, k):
    """Membership + ordinal position only — the information content of a cell sentence."""
    Y = np.zeros_like(X)
    for i in range(len(X)):
        idx = np.argpartition(-X[i], k)[:k]
        Y[i, idx[np.argsort(-X[i][idx])]] = np.arange(k, 0, -1)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5ad", required=True)
    ap.add_argument("--tgt-h5ad", required=True)
    ap.add_argument("--src-states", required=True)
    ap.add_argument("--tgt-states", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--k", type=int, default=512, help="genes per cell sentence, must match extraction")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="results/encoding_matched_transfer.json")
    a = ap.parse_args()
    import anndata, scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    src = anndata.read_h5ad(a.src_h5ad); tgt = anndata.read_h5ad(a.tgt_h5ad)
    ts, _, _, si = phase_angle_oriented(src); tt, _, _, ti = phase_angle_oriented(tgt)
    si_ = np.load(os.path.join(a.src_states, "row_cell_ids.npy"))
    ti_ = np.load(os.path.join(a.tgt_states, "row_cell_ids.npy"))
    Hs = np.load(os.path.join(a.src_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    Ht = np.load(os.path.join(a.tgt_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    ts_m = ts[si_][:len(Hs)]; Hs = Hs[:len(ts_m)]
    tt_m = tt[ti_][:len(Ht)]; Ht = Ht[:len(tt_m)]

    sv = np.char.upper(np.asarray(src.var_names).astype(str))
    tv = np.char.upper(np.asarray(tgt.var_names).astype(str))
    shared = sorted(set(sv) & set(tv))
    sx = {g: i for i, g in enumerate(sv)}; tx = {g: i for i, g in enumerate(tv)}
    Xs = (src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X))[:, [sx[g] for g in shared]][si_][:len(ts_m)]
    Xt = (tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X))[:, [tx[g] for g in shared]][ti_][:len(tt_m)]
    print(f"source {Hs.shape} | target {Ht.shape} | shared genes {len(shared)} | k={a.k}", flush=True)

    def frozen(A, B):
        Y = np.column_stack([np.cos(ts_m), np.sin(ts_m)])
        s = StandardScaler().fit(A)
        P = Ridge(alpha=1e3).fit(s.transform(A), Y).predict(s.transform(B))
        return np.arctan2(P[:, 1], P[:, 0])

    preds = {"model": frozen(Hs, Ht),
             "expr_full": frozen(Xs, Xt),
             "expr_512_mag": frozen(topk_mag(Xs, a.k), topk_mag(Xt, a.k)),
             "expr_512_rank": frozen(topk_rank(Xs, a.k), topk_rank(Xt, a.k))}
    preds["constant"] = np.full_like(tt_m, np.angle(np.mean(np.exp(1j * tt_m))))
    preds["random"] = np.random.default_rng(0).uniform(0, 2 * np.pi, len(tt_m))

    order = ["expr_full", "expr_512_mag", "expr_512_rank", "model", "constant", "random"]
    print(f"\n{'arm':<16}{'R_diff':>9}{'med err':>10}", flush=True)
    res = {"n_target": int(len(tt_m)), "n_shared_genes": len(shared), "k": a.k, "layer": a.layer,
           "target_phase_R": float(np.abs(np.mean(np.exp(1j * tt_m)))), "point": {}}
    for nm in order:
        res["point"][nm] = dict(R_diff=R_diff(preds[nm], tt_m), median_err_deg=med_err(preds[nm], tt_m))
        print(f"{nm:<16}{res['point'][nm]['R_diff']:>9.3f}{res['point'][nm]['median_err_deg']:>9.0f}d", flush=True)

    # ---- PAIRED bootstrap: same resampled cells for every arm ----
    rng = np.random.default_rng(0); n = len(tt_m)
    boot = {nm: np.empty(a.n_boot) for nm in order}
    for b in range(a.n_boot):
        i = rng.integers(0, n, n)
        for nm in order:
            boot[nm][b] = R_diff(preds[nm][i], tt_m[i])
    print(f"\nPAIRED CONTRASTS ({a.n_boot} draws, same cells per draw)", flush=True)
    print(f"  {'contrast':<34}{'delta R_diff':>14}{'95% CI':>22}{'P(>0)':>9}", flush=True)
    res["contrasts"] = {}
    for x, y in [("model", "expr_512_rank"), ("model", "expr_full"),
                 ("expr_full", "expr_512_rank"), ("expr_full", "expr_512_mag"),
                 ("expr_512_mag", "expr_512_rank")]:
        d = boot[x] - boot[y]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(np.mean(d > 0))
        res["contrasts"][f"{x}_minus_{y}"] = dict(delta=float(d.mean()), ci=[float(lo), float(hi)], p_gt_0=p)
        print(f"  {x + ' - ' + y:<34}{d.mean():>+14.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]{p:>9.3f}", flush=True)

    dm = boot["model"] - boot["expr_512_rank"]
    lo, hi = np.percentile(dm, [2.5, 97.5])
    verdict = ("PARITY — the CI on model minus its own input encoding spans 0"
               if lo <= 0 <= hi else
               ("MODEL ADDS information beyond its input encoding" if lo > 0 else
                "MODEL LOSES information relative to its input encoding"))
    res["verdict"] = verdict
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"  tokenisation loss (expr_full - expr_512_rank) = "
          f"{res['contrasts']['expr_full_minus_expr_512_rank']['delta']:+.4f} "
          f"{res['contrasts']['expr_full_minus_expr_512_rank']['ci']}", flush=True)
    print(f"  model vs its own input               = {dm.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
