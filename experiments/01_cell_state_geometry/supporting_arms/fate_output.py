"""Route Aging — does the model's HSC geometry predict a donor's lineage output, BETTER than markers?

See PREREGISTRATION.md (written before any embedding was extracted), incl. the documented pivot: the textbook
"aged HSCs are myeloid-biased" phenotype is ABSENT (indeed reversed) in this human dataset, so we do not
validate against it. Instead we test the capability that actually matters:

    Can the model's geometry on a donor's HSCs predict that donor's MEASURED committed-lineage output --
    better than the HSCs' own marker expression can?

  H1  bias(d,f) from the model's fate directions  correlates with truth(d,f) = donor d's committed composition
  H2  *** THE TEST ***  does it beat a MARKER-SCORE baseline (the same thing DE did to us 12/12 in the
      inversion)? A marker score gives a cell's POSITION; the manifold is supposed to give its DIRECTION.
      If the model does not beat markers, it adds nothing and we say so.
  H3  does it survive WITHIN the 5 young donors (i.e. is it more than a 2-group age/batch effect)?

LEAVE-ONE-DONOR-OUT: donor d's own committed cells never define the directions used to score d's HSCs
(otherwise a donor with many erythroid cells drags the erythroid centroid toward its own batch).

All statistics are PER DONOR (n=8). Never per cell -- 20k cells nested in 8 donors would manufacture
significance from pseudo-replication. This is a PILOT: Spearman on n=8 needs |rho| ~ 0.74 for p<0.05.

Run: ../../.venv/bin/python fate_output.py
Out: results/fate_output_scgptbin_aging.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_steering"))
from tangent_diagnostic import unit, SEED, DIM  # noqa: E402
from local_steering import project_constant  # noqa: E402
from gene_decode import PANEL  # noqa: E402

ROOT = f"{_DATA}"
H5AD = f"{ROOT}/data/aging/aging_setty_schema.h5ad"
EMB = f"{ROOT}/data/branchpoint/scgptbin_aging.npz"
GT = f"{ROOT}/data/aging/ground_truth.json"
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

LIN = {"MEP": "ery", "Erythroid_early": "ery", "Erythroid_late": "ery",
       "GMP": "myeloid", "GMP_Granulocytes": "myeloid", "Monocytes": "myeloid", "Basophils": "myeloid",
       "CLP": "lymphoid", "ProB": "lymphoid", "T_NK": "lymphoid", "pDC": "lymphoid",
       "Megakaryocytes": "mega"}
FATES = ["ery", "myeloid", "lymphoid", "mega"]


def load():
    z = np.load(EMB, allow_pickle=True)
    emb = z["emb"].astype(np.float64); ci = z["cell_idx"].astype(int)
    with h5py.File(H5AD, "r") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"]["index"][:]])
        cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["__categories"]["clusters"][:]]
        ct = np.array([cats[c] for c in f["obs"]["clusters"][:]])[ci]
        donor = np.array([d.decode() if isinstance(d, bytes) else d for d in f["obs"]["donor"][:]])[ci]
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        E = np.zeros((len(ci), len(genes)), np.float32)
        for r, i in enumerate(ci):
            a, b = int(ip[i]), int(ip[i + 1])
            E[r, ind[a:b]] = dat[a:b]
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    return emb, np.log1p(E / tot * 1e4), genes, ct, donor


def main():
    emb, E, genes, ct, donor = load()
    lin = np.array([LIN.get(c, "stem" if c in ("HSC", "LMPP") else "other") for c in ct])
    hsc = ct == "HSC"
    donors = sorted(set(donor))
    truth = json.load(open(GT))
    print(f"[data] {len(emb)} cells | {hsc.sum()} HSC | donors {donors}")

    # ---- marker baseline: each HSC's lineage-marker z-score (POSITION, from raw expression)
    gsym = {g: i for i, g in enumerate(genes)}
    mu, sd = E.mean(0), E.std(0) + 1e-6
    mk = {}
    for f in FATES:
        idx = np.array([gsym[g] for g in PANEL[f] if g in gsym])
        mk[f] = ((E[:, idx] - mu[idx]) / sd[idx]).mean(1)
    print(f"[markers] genes found per fate: " + ", ".join(f"{f}={sum(g in gsym for g in PANEL[f])}" for f in FATES))

    # ---- LEAVE-ONE-DONOR-OUT model readout
    bias = {d: {} for d in donors}          # READOUT A: position (projection onto the fate direction)
    acc_bias = {d: {} for d in donors}      # READOUT B: direction (tangent accessibility of the fate)
    for d in donors:
        out_d = donor != d                       # directions built from the OTHER 7 donors only
        tr = out_d & (lin != "other")            # cells used to build the manifold/centroids
        Xtr = emb[tr]
        mean = Xtr.mean(0)
        pca = PCA(min(DIM, Xtr.shape[1], len(Xtr) - 1), random_state=SEED).fit(Xtr - mean)
        wsc = StandardScaler().fit(pca.transform(Xtr - mean))
        to_z = lambda A: wsc.transform(pca.transform(A - mean))          # noqa: E731
        Ztr = to_z(Xtr)
        lin_tr = lin[tr]
        src_c = Ztr[lin_tr == "stem"].mean(0)                            # stem centroid (other donors)

        Zh = to_z(emb[hsc & (donor == d)])                               # THIS donor's HSCs
        raw, acc = {}, {}
        for f in FATES:
            m = lin_tr == f
            if m.sum() < 20:
                continue
            w = unit(Ztr[m].mean(0) - src_c)                             # stem -> fate centroid
            D = project_constant(Ztr, w)(Zh)                             # w projected onto each HSC's LOCAL TANGENT
            # READOUT B (DIRECTION): the NORM of that projection = how much of the fate direction lies along the
            # manifold FROM THIS CELL, i.e. how ACCESSIBLE fate f is from here. This is the direction-based
            # signal -- and it is exactly the quantity Readout A throws away by normalising. Needs no pseudotime.
            acc[f] = float(np.linalg.norm(D, axis=1).mean())
            raw[f] = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        # FATE-SPECIFIC: orthogonalise each against the mean of the four (the bifurcation contrast)
        Dm = np.mean([raw[f] for f in raw], axis=0)
        for f in raw:
            S = raw[f] - Dm
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
            # READOUT A (POSITION): projection of the HSC onto the fate-specific direction
            bias[d][f] = float((Zh * S).sum(1).mean())
        acc_bias[d] = acc

    # ---- assemble the 8 x 4 table
    rows = []
    for d in donors:
        for f in FATES:
            if f in bias[d]:
                rows.append(dict(donor=d, fate=f, model=bias[d][f], access=acc_bias[d][f],
                                 marker=float(mk[f][hsc & (donor == d)].mean()),
                                 truth=truth[d][f], young=d.startswith("young")))
    M = np.array([r["model"] for r in rows]); K = np.array([r["marker"] for r in rows])
    Aq = np.array([r["access"] for r in rows])
    T = np.array([r["truth"] for r in rows]); Y = np.array([r["young"] for r in rows])
    F = np.array([r["fate"] for r in rows])

    # within-fate z-scoring: compare donors WITHIN a lineage (levels differ hugely across lineages)
    def zf(v):
        o = np.zeros_like(v)
        for f in FATES:
            m = F == f
            if m.sum() > 1:
                o[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-12)
        return o
    mz, kz, tz, az = zf(M), zf(K), zf(T), zf(Aq)

    def rep(v, mask=None):
        m = np.ones(len(v), bool) if mask is None else mask
        r, p = spearmanr(v[m], tz[m])
        return float(r), float(p)

    out = dict(n_donors=len(donors), rows=rows)
    print(f"\n{'='*96}\n  H1/H2 — predicting each donor's committed-lineage output from its HSCs "
          f"(n={len(donors)} donors x {len(FATES)} lineages, within-lineage z)")
    r_m, p_m = rep(mz); r_k, p_k = rep(kz); r_a, p_a = rep(az)
    print(f"    [A] MODEL position  (projection onto fate direction) : rho = {r_m:+.3f}  p = {p_m:.4f}")
    print(f"    [B] MODEL DIRECTION (tangent accessibility of fate)   : rho = {r_a:+.3f}  p = {p_a:.4f}")
    print(f"    [-] MARKER baseline (expression)                      : rho = {r_k:+.3f}  p = {p_k:.4f}   <- the bar")

    # does either model readout add anything ON TOP of markers?
    def partial(v):
        res_v = v - LinearRegression().fit(kz[:, None], v).predict(kz[:, None])
        res_t = tz - LinearRegression().fit(kz[:, None], tz).predict(kz[:, None])
        r, p = spearmanr(res_v, res_t)
        return float(r), float(p)
    r_r, p_r = partial(mz); r_ar, p_ar = partial(az)
    print(f"    [A] position  residualised on MARKER : partial rho = {r_r:+.3f}  p = {p_r:.4f}")
    print(f"    [B] DIRECTION residualised on MARKER : partial rho = {r_ar:+.3f}  p = {p_ar:.4f}")
    out.update(rho_model=r_m, p_model=p_m, rho_marker=r_k, p_marker=p_k,
               rho_access=r_a, p_access=p_a,
               partial_rho_model_given_marker=float(r_r), partial_p=float(p_r),
               partial_rho_access_given_marker=float(r_ar), partial_p_access=float(p_ar))

    # ---- DONOR-SHUFFLE PERMUTATION — the only honest test here.
    # The 32 (donor x lineage) points are NOT independent: they are 8 donors x 4 lineages, so the p-values above
    # (which assume n=32) are optimistic. Permuting the donor->truth assignment respects that structure and
    # gives an exact-in-spirit p at the DONOR level (8! = 40320 assignments; we sample 10k).
    rng = np.random.default_rng(SEED)
    res_t = tz - LinearRegression().fit(kz[:, None], tz).predict(kz[:, None])   # truth, markers removed
    res_a = az - LinearRegression().fit(kz[:, None], az).predict(kz[:, None])   # direction, markers removed
    nullA, nullB, nullP = [], [], []
    for _ in range(10000):
        perm = {d: p for d, p in zip(donors, rng.permutation(donors))}
        tp = zf(np.array([truth[perm[r["donor"]]][r["fate"]] for r in rows]))
        nullA.append(spearmanr(mz, tp)[0])
        nullB.append(spearmanr(az, tp)[0])
        rt = tp - LinearRegression().fit(kz[:, None], tp).predict(kz[:, None])
        nullP.append(spearmanr(res_a, rt)[0])
    nullA, nullB, nullP = map(np.array, (nullA, nullB, nullP))
    pA = float((np.sum(nullA >= r_m) + 1) / 10001)
    pB = float((np.sum(nullB >= r_a) + 1) / 10001)
    pP = float((np.sum(nullP >= r_ar) + 1) / 10001)
    out.update(p_perm_position=pA, p_perm_direction=pB, p_perm_partial_direction=pP)
    print(f"\n    *** DONOR-SHUFFLE PERMUTATION (the honest test; n=8 donors, not n=32 points) ***")
    print(f"    [A] position                     : p = {pA:.4f}   (null mean {nullA.mean():+.3f})")
    print(f"    [B] DIRECTION                    : p = {pB:.4f}   (null mean {nullB.mean():+.3f})")
    print(f"    [B] DIRECTION | markers removed  : p = {pP:.4f}   (null mean {nullP.mean():+.3f})  <- the claim")

    print(f"\n  per-lineage Spearman across the {len(donors)} donors:")
    for f in FATES:
        m = F == f
        if m.sum() < 3:
            continue
        rm, pm = spearmanr(M[m], T[m]); rk, pk = spearmanr(K[m], T[m]); ra, pa = spearmanr(Aq[m], T[m])
        out.setdefault("per_fate", {})[f] = dict(rho_model=float(rm), p_model=float(pm),
                                                 rho_access=float(ra), p_access=float(pa),
                                                 rho_marker=float(rk), p_marker=float(pk))
        print(f"    {f:9s} position rho={rm:+.3f}  DIRECTION rho={ra:+.3f} (p={pa:.3f})  "
              f"marker rho={rk:+.3f} (p={pk:.3f})")

    # ---- H3: does it survive within the 5 YOUNG donors (beyond the age/batch 2-group split)?
    ym = Y
    if ym.sum() >= 8:
        ry, py = spearmanr(mz[ym], tz[ym]); rky, pky = spearmanr(kz[ym], tz[ym])
        ray, pay = spearmanr(az[ym], tz[ym])
        out.update(h3_young_rho_model=float(ry), h3_young_p=float(py),
                   h3_young_rho_access=float(ray), h3_young_p_access=float(pay),
                   h3_young_rho_marker=float(rky))
        print(f"\n  H3 (within the 5 YOUNG donors only — is it more than an age/batch 2-group effect?)")
        print(f"    position rho = {ry:+.3f} (p={py:.3f})   DIRECTION rho = {ray:+.3f} (p={pay:.3f})   "
              f"marker rho = {rky:+.3f}")

    beats = max(r_m, r_a) > r_k
    adds = (pP < 0.05) and (r_ar > 0)
    out["verdict"] = dict(position_beats_marker=bool(r_m > r_k), direction_beats_marker=bool(r_a > r_k),
                          any_beats_marker=bool(beats), direction_ADDS_to_marker=bool(adds))
    print(f"\n  {'='*92}\n  VERDICT")
    print(f"    head-to-head : marker {r_k:+.3f}  |  model-position {r_m:+.3f}  |  model-DIRECTION {r_a:+.3f}")
    print(f"    -> the model does {'' if beats else 'NOT '}beat markers head-to-head (prereg H2 as literally "
          f"stated: {'PASS' if beats else 'FAIL'}).")
    if adds:
        print(f"    BUT (also pre-registered, §5 H2): residualised on markers, the DIRECTION readout carries")
        print(f"    information the markers do NOT — partial rho {r_ar:+.3f}, donor-shuffle p = {pP:.4f}.")
        print(f"    => The model is COMPLEMENTARY to markers, not superior and not redundant.")
        print(f"    *** CAVEATS THAT GATE THIS CLAIM (see RESULTS.md) ***")
        print(f"      - The DIRECTION readout is a DEVIATION: the prereg operationalised POSITION, which failed.")
        print(f"        The rationale (direction, not position) predates the run, but this is a second look.")
        print(f"      - H3: it does NOT survive within the 5 young donors alone "
              f"(rho {out.get('h3_young_rho_access', float('nan')):+.3f}) -> may be an age/batch 2-group effect.")
        print(f"      - n = 8 donors. Signal concentrated in lymphoid. This is a LEAD requiring replication,")
        print(f"        not a result.")
    json.dump(out, open(f"{RESULTS}/fate_output_scgptbin_aging.json", "w"), indent=1)
    print(f"\n[done] -> results/fate_output_scgptbin_aging.json")


if __name__ == "__main__":
    main()
