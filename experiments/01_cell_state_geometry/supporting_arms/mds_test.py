"""Route Aging — MDS HELD-OUT TEST. Does the DIRECTION readout detect the lineage defect that defines MDS?

See MDS_PREREGISTRATION.md (written before a single MDS cell was embedded). This is the de-risking test for
RESULTS.md, whose headline rests on a documented deviation (the DIRECTION readout was defined after the
pre-registered POSITION readout failed). The 4 MDS patients have never been touched by this route.

GROUND TRUTH, checked first: the naive blockade framing FAILS (MDS immature fraction 0.469 vs healthy 0.413,
p=0.18 -- elderly are MORE immature than MDS). But LYMPHOID OUTPUT COLLAPSES in MDS: 0.035 vs 0.272 young /
0.111 elderly (16 CLP + 1 ProB + 14 T_NK across 41,724 MDS cells). And lymphoid is exactly the lineage where the
DIRECTION readout worked in the healthy cohort (rho=+0.714). Hence:

  H1  lymphoid tangent-ACCESSIBILITY of a donor's HSCs predicts that donor's lymphoid output, across ALL 12
      donors (5 young + 3 elderly + 4 MDS). LODO, donor-shuffle permutation.
  H2  does it beat / ADD TO the marker baseline?
  H3  MDS donors have lower lymphoid accessibility than healthy (Mann-Whitney 4 vs 8; min p = 1/495 = 0.002).
  H4  *** SPECIFICITY -- THE CONTROL WE MOST EXPECT TO FAIL ***  The drop must be LYMPHOID-SPECIFIC. If MDS HSC
      accessibility falls for EVERY fate, we have merely detected that MDS cells sit somewhere geometrically
      different -- not a lineage-specific defect. A global drop REFUTES the lineage interpretation.
  H5  OFF-MANIFOLD control. Are MDS HSCs simply farther from the healthy manifold? If so a global accessibility
      drop is a geometric artifact.

CONFOUND THAT GATES EVERYTHING: MDS is a separate batch (GSM6946065-68 submitted later than GSM5460406-13), so
disease is perfectly confounded with batch. Mitigations: H1 is a CONTINUOUS 12-donor correlation, not a 2-group
contrast; and H4 -- a batch effect should depress ALL accessibilities, not lymphoid selectively.

Run: ../../.venv/bin/python mds_test.py
Out: results/mds_test_scgptbin.json
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
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "route_steering"))
from tangent_diagnostic import unit, SEED, DIM  # noqa: E402
from local_steering import project_constant  # noqa: E402
from gene_decode import PANEL  # noqa: E402

ROOT = f"{_DATA}"
H5AD = f"{ROOT}/data/aging/agingmds_setty_schema.h5ad"
EMB = f"{ROOT}/data/branchpoint/scgptbin_agingmds.npz"
GT = f"{ROOT}/data/aging/ground_truth_mds.json"
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
    donors = sorted(set(donor), key=lambda d: (d[:3], d))
    truth = json.load(open(GT))
    grp = {d: ("mds" if d.startswith("mds") else "healthy") for d in donors}
    print(f"[data] {len(emb)} cells | {hsc.sum()} HSC | {len(donors)} donors "
          f"({sum(g=='mds' for g in grp.values())} MDS)")

    gsym = {g: i for i, g in enumerate(genes)}
    mu, sd = E.mean(0), E.std(0) + 1e-6
    mk = {f: ((E[:, np.array([gsym[g] for g in PANEL[f] if g in gsym])] -
               mu[np.array([gsym[g] for g in PANEL[f] if g in gsym])]) /
              sd[np.array([gsym[g] for g in PANEL[f] if g in gsym])]).mean(1) for f in FATES}

    acc = {d: {} for d in donors}          # DIRECTION: tangent accessibility
    pos = {d: {} for d in donors}          # POSITION (the readout that failed) — kept for completeness
    offm = {}                              # H5: distance of the donor's HSCs to the training manifold
    for d in donors:
        out_d = donor != d
        tr = out_d & (lin != "other")
        Xtr = emb[tr]
        mean = Xtr.mean(0)
        pca = PCA(min(DIM, Xtr.shape[1], len(Xtr) - 1), random_state=SEED).fit(Xtr - mean)
        wsc = StandardScaler().fit(pca.transform(Xtr - mean))
        to_z = lambda A: wsc.transform(pca.transform(A - mean))          # noqa: E731
        Ztr = to_z(Xtr); lin_tr = lin[tr]
        src_c = Ztr[lin_tr == "stem"].mean(0)
        Zh = to_z(emb[hsc & (donor == d)])

        nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
        d0 = np.median(NearestNeighbors(n_neighbors=2).fit(Ztr).kneighbors(Ztr)[0][:, 1])
        offm[d] = float(nn.kneighbors(Zh)[0][:, 0].mean() / d0)          # H5

        raw = {}
        for f in FATES:
            m = lin_tr == f
            if m.sum() < 20:
                continue
            w = unit(Ztr[m].mean(0) - src_c)
            D = project_constant(Ztr, w)(Zh)
            acc[d][f] = float(np.linalg.norm(D, axis=1).mean())          # DIRECTION readout
            raw[f] = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        Dm = np.mean([raw[f] for f in raw], axis=0)
        for f in raw:
            S = raw[f] - Dm
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
            pos[d][f] = float((Zh * S).sum(1).mean())

    rows = [dict(donor=d, fate=f, access=acc[d][f], position=pos[d][f],
                 marker=float(mk[f][hsc & (donor == d)].mean()),
                 truth=truth[d][f], mds=(grp[d] == "mds"))
            for d in donors for f in FATES if f in acc[d]]
    A = np.array([r["access"] for r in rows]); P = np.array([r["position"] for r in rows])
    K = np.array([r["marker"] for r in rows]); T = np.array([r["truth"] for r in rows])
    F = np.array([r["fate"] for r in rows]); DN = np.array([r["donor"] for r in rows])

    def zf(v):
        o = np.zeros_like(v, dtype=float)
        for f in FATES:
            m = F == f
            if m.sum() > 1:
                o[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-12)
        return o
    az, pz, kz, tz = zf(A), zf(P), zf(K), zf(T)
    out = dict(n_donors=len(donors), rows=rows)

    # ---------------- H1 / H2
    r_a, _ = spearmanr(az, tz); r_k, _ = spearmanr(kz, tz); r_p, _ = spearmanr(pz, tz)
    res_t = tz - LinearRegression().fit(kz[:, None], tz).predict(kz[:, None])
    res_a = az - LinearRegression().fit(kz[:, None], az).predict(kz[:, None])
    r_ar, _ = spearmanr(res_a, res_t)
    rng = np.random.default_rng(SEED)
    nB, nP = [], []
    for _ in range(10000):
        perm = {d: p for d, p in zip(donors, rng.permutation(donors))}
        tp = zf(np.array([truth[perm[r["donor"]]][r["fate"]] for r in rows]))
        nB.append(spearmanr(az, tp)[0])
        rt = tp - LinearRegression().fit(kz[:, None], tp).predict(kz[:, None])
        nP.append(spearmanr(res_a, rt)[0])
    pB = float((np.sum(np.array(nB) >= r_a) + 1) / 10001)
    pP = float((np.sum(np.array(nP) >= r_ar) + 1) / 10001)
    print(f"\n{'='*100}\n  H1/H2 — predicting lineage output from HSC geometry, ALL {len(donors)} DONORS")
    print(f"    [B] DIRECTION (tangent accessibility) : rho = {r_a:+.3f}   donor-shuffle p = {pB:.4f}")
    print(f"    [A] position                          : rho = {r_p:+.3f}")
    print(f"    [-] MARKER baseline                   : rho = {r_k:+.3f}   <- the bar")
    print(f"    [B] DIRECTION | markers removed       : partial rho = {r_ar:+.3f}   donor-shuffle p = {pP:.4f}")
    out.update(rho_direction=float(r_a), p_direction=pB, rho_position=float(r_p), rho_marker=float(r_k),
               partial_rho=float(r_ar), p_partial=pP)

    # replication check: the healthy 8 donors, re-embedded on the NEW (smaller) gene set
    hm = np.array([not r["mds"] for r in rows])
    rh, ph = spearmanr(az[hm], tz[hm])
    out["healthy_only_rho_direction"] = float(rh)
    print(f"    [replication] healthy 8 donors only, on this NEW gene set: rho = {rh:+.3f} (p={ph:.3f})")

    # ---------------- H3 / H4: is the MDS drop LYMPHOID-SPECIFIC?
    print(f"\n  H3/H4 — MDS vs healthy HSC accessibility, per fate  (H4: must be LYMPHOID-SPECIFIC)")
    print(f"    {'fate':<9} {'healthy':>9} {'MDS':>9} {'delta':>8} {'MWU p':>8}")
    h3 = {}
    for f in FATES:
        hv = np.array([acc[d][f] for d in donors if grp[d] == "healthy" and f in acc[d]])
        mv = np.array([acc[d][f] for d in donors if grp[d] == "mds" and f in acc[d]])
        u, p = mannwhitneyu(mv, hv, alternative="less")
        h3[f] = dict(healthy=float(hv.mean()), mds=float(mv.mean()),
                     delta=float(mv.mean() - hv.mean()), p_mds_lower=float(p))
        print(f"    {f:<9} {hv.mean():>9.4f} {mv.mean():>9.4f} {mv.mean()-hv.mean():>+8.4f} {p:>8.4f}")
    out["h3_h4_per_fate"] = h3
    lym_d = h3["lymphoid"]["delta"]
    oth_d = np.mean([h3[f]["delta"] for f in FATES if f != "lymphoid"])
    specific = (lym_d < 0) and (lym_d < oth_d - 1e-9) and (h3["lymphoid"]["p_mds_lower"] < 0.05)
    out["h4_lymphoid_specific"] = bool(specific)
    print(f"\n    lymphoid delta = {lym_d:+.4f}   mean of the other three = {oth_d:+.4f}")
    print(f"    -> H4 {'PASSED — the drop IS lymphoid-specific' if specific else 'FAILED — the drop is NOT lymphoid-specific (global shift; see H5)'}")

    # ---------------- H5: are MDS HSCs simply off-manifold?
    oh = np.mean([offm[d] for d in donors if grp[d] == "healthy"])
    om = np.mean([offm[d] for d in donors if grp[d] == "mds"])
    u, p5 = mannwhitneyu([offm[d] for d in donors if grp[d] == "mds"],
                         [offm[d] for d in donors if grp[d] == "healthy"], alternative="greater")
    out["h5_offmanifold"] = dict(healthy=float(oh), mds=float(om), p=float(p5), per_donor=offm)
    print(f"\n  H5 — distance of a donor's HSCs to the training manifold (x median NN distance):")
    print(f"    healthy {oh:.2f}   MDS {om:.2f}   (MWU p={p5:.4f} for MDS farther)")
    if p5 < 0.05:
        print(f"    ** MDS HSCs ARE farther off-manifold -> a global accessibility drop would be an artifact;")
        print(f"       H4's specificity check is what decides whether the lymphoid result survives.")

    json.dump(out, open(f"{RESULTS}/mds_test_scgptbin.json", "w"), indent=1)
    print(f"\n  {'='*96}\n  VERDICT")
    print(f"    H1 (12-donor prediction)      : rho {r_a:+.3f}, p {pB:.4f}  -> {'PASS' if pB < 0.05 else 'FAIL'}")
    print(f"    H2 (adds beyond markers)      : partial {r_ar:+.3f}, p {pP:.4f}  -> {'PASS' if pP < 0.05 else 'FAIL'}")
    print(f"    H3 (MDS lymphoid lower)       : p {h3['lymphoid']['p_mds_lower']:.4f}  "
          f"-> {'PASS' if h3['lymphoid']['p_mds_lower'] < 0.05 else 'FAIL'}")
    print(f"    H4 (lymphoid-SPECIFIC)        : -> {'PASS' if specific else 'FAIL'}   <- the one that can kill it")
    print(f"\n[done] -> results/mds_test_scgptbin.json")


if __name__ == "__main__":
    main()
