"""transfer_test — ZERO-SHOT CROSS-DATASET TRANSFER: does the model's cell-cycle circle generalise better
than expression's?

WHY THIS IS THE RIGHT EXPERIMENT. Everything so far was measured *within* one dataset, where the phase label is
marker co-expression and the expression baseline is handed the answer key — a degenerate comparison that C2S
loses (0 of 36) and that a reparametrisation account fully explains. Transfer breaks the degeneracy:

    fit the phase readout on K562  ->  apply it, frozen, to a different dataset

Expression readouts are notoriously batch/dataset-sensitive (different platform, tissue, depth, gene panel).
Robust transfer is the *claimed* strength of single-cell foundation models. So this asks the question the
within-dataset benchmark cannot: **does the model's representation carry cell-cycle structure that survives a
dataset change better than raw expression does?** A model win here is NOT explicable as reparametrising this
dataset's input, because the readout never sees the target dataset.

DESIGN
  SOURCE  K562 Replogle non-targeting controls (n = 3,000), the canonical substrate.
  TARGET  RPE1 Replogle non-targeting controls (n = 3,000).

CHOOSING A VALID TARGET was most of the work here, and two candidates were rejected on measured grounds:
    setty CD34+ bone marrow   only 18 of 94 cell-cycle markers present, and its phase field tracks LINEAGE
                              rather than cycle -- per-cell-type phase concentration 0.95-0.98 (HSC_1, CLP),
                              |Spearman(cos phi, pseudotime)| = 0.345. A "phase" readout there is a
                              differentiation readout.
    Tabula Sapiens cycling    full marker coverage but only 887 cycling cells (7.4%) spread over 20+ cell
                              types, mean per-cell-type concentration 0.66 -- again cell-type dominated.
  RPE1 passes where both failed: 93 of 94 markers present, ONE cell type (so phase cannot be confounded with
  identity), and phase concentration 0.267, i.e. cells genuinely spread around the circle. 6,544 of 6,546
  K562 genes are shared.

  HONEST LIMIT: RPE1 is the same lab and platform as K562, so this tests generalisation across BIOLOGY
  (aneuploid erythroleukemia -> near-diploid retinal epithelium, different karyotype and sex) and NOT across
  batch or platform. It is not the harder cross-platform test, and should not be described as one.

  Both readouts are fit ON SOURCE ONLY and applied frozen to TARGET:
    model       ridge(cos phi, sin phi) on C2S cell-state activations
    expression  ridge(cos phi, sin phi) on expression restricted to the SHARED gene space (fair to both)
  Ground truth in each dataset is its own Tirosh-marker phase (computed independently per dataset).

METRICS
  within  circular correlation on held-out SOURCE cells (the ceiling each readout achieves at home)
  transfer circular correlation on TARGET cells, readout frozen
  retention = transfer / within   <- the actual quantity of interest: who degrades less?

Out: results/transfer_test.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle, phase_angle_oriented


def circ_corr(a, b):
    """Jammalamadaka circular correlation. RETAINED FOR THE RECORD ONLY -- DO NOT JUDGE TRANSFER ON THIS.

    It scores sin(x - xbar), which needs a well-defined circular mean. A cell-cycle phase over an asynchronous
    population is near-uniform (R = 0.043 in RPE1, 0.079 in K562), so its mean direction is arbitrary noise;
    a frozen ridge readout under domain shift is comparatively concentrated (R = 0.341). With that mismatch the
    two mean estimates are inconsistent, sin() folds the circle, and the coefficient returns a large value OF
    THE WRONG SIGN. Measured here: circ_corr = -0.814 while R_diff = +0.789 and the median angular error was
    23 deg against 86 deg for a constant predictor. The readout was fine; this statistic was wrong.
    It is reliable only when both variables share a concentration -- e.g. the oracle refits, where it agrees
    with R_diff (+0.928 vs 0.909, +0.958 vs 0.952).
    """
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def circ_R_diff(a, b):
    """PRIMARY association measure: resultant length of the angular residuals, |mean(exp(i(a-b)))|.
    Rotation-invariant (a constant offset leaves it unchanged) and fold-free (no circular mean is estimated),
    so it stays valid however concentrated either variable is. 1 = perfect up to rotation, 0 = independent."""
    return float(np.abs(np.mean(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def med_err_rot(a, b):
    """Median angular error after the single best rigid rotation. Interpretable in degrees, and unlike the raw
    median error it does not punish a readout for a constant offset that carries no phase information."""
    off = np.angle(np.mean(np.exp(1j * (np.asarray(b) - np.asarray(a)))))
    return float(np.rad2deg(np.median(circ_dist(a + off, b)))), float(np.rad2deg(off) % 360)


def circ_R(a):
    return float(np.abs(np.mean(np.exp(1j * np.asarray(a)))))


def fit_apply(Xs, ts, Xt, tt, alpha=1e3, seed=0):
    """Fit ridge(cos,sin) on SOURCE, report within-source (held-out) and frozen TARGET transfer."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    Ys = np.column_stack([np.cos(ts), np.sin(ts)])
    # within-source, out-of-fold
    P = np.zeros_like(Ys)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
        sc = StandardScaler().fit(Xs[tr])
        P[te] = Ridge(alpha=alpha).fit(sc.transform(Xs[tr]), Ys[tr]).predict(sc.transform(Xs[te]))
    wh = np.arctan2(P[:, 1], P[:, 0])
    within = circ_R_diff(wh, ts)
    within_err, _ = med_err_rot(wh, ts)
    # frozen transfer
    sc = StandardScaler().fit(Xs)
    m = Ridge(alpha=alpha).fit(sc.transform(Xs), Ys)
    Pt = m.predict(sc.transform(Xt))
    that = np.arctan2(Pt[:, 1], Pt[:, 0])
    transfer = circ_R_diff(that, tt)
    transfer_err, offset = med_err_rot(that, tt)
    # TARGET-RECENTERED arm: refit only the feature scaler on the target, using NO target labels. This stays
    # genuine zero-shot transfer while removing the part of the domain shift that is a pure feature offset,
    # rather than penalising a readout for a batch effect unrelated to cell cycle.
    Pr = m.predict(StandardScaler().fit(Xt).transform(Xt))
    rhat = np.arctan2(Pr[:, 1], Pr[:, 0])
    rec_err, _ = med_err_rot(rhat, tt)
    return dict(within=within, within_median_err_deg=within_err,
                transfer=transfer, transfer_median_err_deg=transfer_err, transfer_offset_deg=offset,
                recentered=circ_R_diff(rhat, tt), recentered_median_err_deg=rec_err,
                retention=float(transfer / (within + 1e-12)),
                pred_R=circ_R(that), legacy_circ_corr=circ_corr(that, tt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-states", required=True)
    ap.add_argument("--tgt-states", required=True)
    ap.add_argument("--src-h5ad", required=True)
    ap.add_argument("--tgt-h5ad", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--out", default="results/transfer_test.json")
    a = ap.parse_args()
    import anndata, scipy.sparse as sp

    src = anndata.read_h5ad(a.src_h5ad); tgt = anndata.read_h5ad(a.tgt_h5ad)
    # CANONICALLY ORIENTED phase in both datasets. Mandatory here: the raw phase_angle takes theta from an SVD
    # whose sign and origin are arbitrary per dataset, so two datasets with identical biology can yield mirrored
    # theta fields and a perfectly good readout then scores strongly negative. The first run of this script did
    # exactly that. Orientation is fixed from marker biology alone (S precedes G2/M), never from the readout.
    ts_phase, _, _, si = phase_angle_oriented(src)
    tt_phase, _, _, ti = phase_angle_oriented(tgt)
    print(f"SOURCE {src.shape} ({si['n_cc_genes']} cc genes) | TARGET {tgt.shape} ({ti['n_cc_genes']} cc genes)",
          flush=True)
    for nm, i in [("SOURCE", si), ("TARGET", ti)]:
        print(f"  {nm} orientation: flipped={i['flipped']} rot={i['rotation_deg']}deg -> "
              f"S peak {i['s_peak_deg']}deg, G2M peak {i['g2m_peak_deg']}deg (sep {i['s_g2m_separation_deg']}deg)",
              flush=True)

    # ---- MODEL ----
    si_ = np.load(os.path.join(a.src_states, "row_cell_ids.npy"))
    ti_ = np.load(os.path.join(a.tgt_states, "row_cell_ids.npy"))
    Hs = np.load(os.path.join(a.src_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    Ht = np.load(os.path.join(a.tgt_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    ts_m = ts_phase[si_][:len(Hs)]; Hs = Hs[:len(ts_m)]
    tt_m = tt_phase[ti_][:len(Ht)]; Ht = Ht[:len(tt_m)]
    model = fit_apply(Hs, ts_m, Ht, tt_m)

    # ---- EXPRESSION, on the SHARED gene space (fair to both directions) ----
    sv = np.char.upper(np.asarray(src.var_names).astype(str))
    tv = np.char.upper(np.asarray(tgt.var_names).astype(str))
    shared = sorted(set(sv) & set(tv))
    sidx = {g: i for i, g in enumerate(sv)}; tidx = {g: i for i, g in enumerate(tv)}
    Xs = src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X)
    Xt = tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X)
    Xs = Xs[:, [sidx[g] for g in shared]][si_][:len(ts_m)]
    Xt = Xt[:, [tidx[g] for g in shared]][ti_][:len(tt_m)]
    print(f"shared gene space: {len(shared)} genes", flush=True)
    expr = fit_apply(Xs, ts_m, Xt, tt_m)

    # ---- EMPIRICAL CHANCE FLOOR, measured rather than assumed ----
    # A constant predictor emitting the target's mean phase, and a uniform random predictor. Without these two
    # numbers no median angular error can be read: whether 24 deg is good depends entirely on what chance is
    # for THIS target's phase distribution. Here the target is near-uniform (R = 0.043), so chance is ~86-90 deg.
    rng = np.random.default_rng(0)
    const = np.full_like(tt_m, np.angle(np.mean(np.exp(1j * tt_m))))
    rand = rng.uniform(0, 2 * np.pi, len(tt_m))
    floor = {}
    for nm, p in [("constant", const), ("random", rand)]:
        e, _ = med_err_rot(p, tt_m)
        floor[nm] = dict(R_diff=circ_R_diff(p, tt_m), median_err_deg=e)
    res = dict(n_shared_genes=len(shared), n_source=int(len(Hs)), n_target=int(len(Ht)),
               target_phase_R=circ_R(tt_m), source_phase_R=circ_R(ts_m), chance_floor=floor,
               model=model, expression=expr)
    print(f"\n  chance floor on this target (phase R={circ_R(tt_m):.3f}): "
          f"constant {floor['constant']['median_err_deg']:.0f}deg (R_diff {floor['constant']['R_diff']:.3f}) | "
          f"random {floor['random']['median_err_deg']:.0f}deg (R_diff {floor['random']['R_diff']:.3f})", flush=True)
    print(f"\n  {'readout':<12}{'within R_diff':>15}{'transfer R_diff':>17}{'err(rot)':>11}"
          f"{'recentered':>12}{'err':>7}{'retention':>11}", flush=True)
    for nm, r in [("MODEL", model), ("EXPRESSION", expr)]:
        print(f"  {nm:<12}{r['within']:>9.3f} ({r['within_median_err_deg']:>3.0f}d)"
              f"{r['transfer']:>12.3f}{r['transfer_median_err_deg']:>10.0f}d"
              f"{r['recentered']:>12.3f}{r['recentered_median_err_deg']:>6.0f}d{r['retention']:>11.2f}", flush=True)
    # VALIDITY GATE, now on R_diff against the MEASURED floor rather than on the invalid circ_corr and an
    # assumed 90 deg. A readout counts as transferring only if its residuals are appreciably more concentrated
    # than a constant predictor's and its rotation-aligned error beats the constant predictor's.
    thr = max(0.15, floor["constant"]["R_diff"] * 2)
    valid = lambda r: (r["transfer"] > thr) and (r["transfer_median_err_deg"] < floor["constant"]["median_err_deg"])
    res["model_valid"], res["expression_valid"] = valid(model), valid(expr)
    if not (res["model_valid"] or res["expression_valid"]):
        # Do NOT assert a cause here. An earlier version of this message declared "the target's phase label is
        # not cell cycle", which was right for the setty run but would have been wrong for RPE1, whose label was
        # verified against K562 gene by gene (circ_corr +0.983, median 15 deg). Report the failure, list the
        # candidate causes, and leave the diagnosis to transfer_diag.py.
        res["transfer_winner"] = "NEITHER — neither readout transferred; cause undetermined"
        print(f"\n  !! NO VALID TRANSFER: model R_diff {model['transfer']:.3f} "
              f"({model['transfer_median_err_deg']:.0f}deg), expression R_diff {expr['transfer']:.3f} "
              f"({expr['transfer_median_err_deg']:.0f}deg) vs a constant predictor's "
              f"{floor['constant']['R_diff']:.3f}/{floor['constant']['median_err_deg']:.0f}deg. NO WINNER.\n"
              f"     Candidate causes, in order of cost to check: (a) the target's phase label does not measure "
              f"cell cycle -- verify it per-gene against the source; (b) domain shift the recentered arm did not "
              f"absorb -- compare the recentered numbers above; (c) the target representation lacks the "
              f"information -- run the oracle refit in transfer_diag.py.", flush=True)
    else:
        win = "MODEL" if model["transfer"] > expr["transfer"] else "EXPRESSION"
        res["transfer_winner"] = win
        res["transfer_margin"] = float(model["transfer"] - expr["transfer"])
        print(f"\n  ZERO-SHOT TRANSFER WINNER: {win} (margin {res['transfer_margin']:+.3f})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
