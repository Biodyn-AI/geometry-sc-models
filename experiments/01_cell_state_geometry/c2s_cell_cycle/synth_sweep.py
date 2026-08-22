"""synth_sweep — CONSTRUCTIVE test of the cell-cycle manifold: build inputs at a target phase and see where
the representation lands. (Ihor's design, 2026-07-24.)

WHY. The manifold discovery in RESULTS_manifold.md is OBSERVATIONAL: real cells in, phase label computed from
those same cells, correlation with activations. The only input-side intervention was a DELETION (remove the 87
cell-cycle genes -> phase survives, 95%), i.e. a NECESSITY test. This is the SUFFICIENCY counterpart: construct
a cell that says "I am at phase theta" and ask whether the representation moves to phase theta.

DESIGN
  backbone   a real cell's ranked gene list with ALL cell-cycle genes stripped -> phase-neutral, on-distribution
  module     k genes whose EXTERNAL peak-phase annotation (Cyclebase / Whitfield 2002 synchronized time-course,
             degrees from G1/S) is nearest the target theta. Annotation is a TEMPORAL measurement, not this
             dataset's covariance, so the co-expression circularity of the observational design is broken.
  insertion  module genes are placed at FIXED rank positions, identical across every condition and theta, so
             token rank cannot drive any difference (the confound that bit Thread B).

CONDITIONS (per backbone, per theta)
  annotated  the real phase-theta module                      -> should track theta
  random     k random non-cycle genes                         -> should do nothing
  shuffled   the same 43 cycle genes, phase labels permuted   -> should do nothing (controls "cycle-ness")

READOUT. Project onto the phase plane fitted on REAL cells (ridge -> cos/sin). Because the frame comes from real
cells, a synthetic point landing at the right angle means it reached the SAME representation, not an artifact.

THE DISCRIMINATOR (from the ring-vs-disk result: angle = phase, radius = cycling strength)
  * a genuine phase manifold -> the sweep moves ANGULARLY at ~constant radius, and CLOSES THE LOOP
  * a mere "cell-cycle intensity" detector -> the sweep moves RADIALLY with little angular travel
Out: results/synth_sweep.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence
from cc_phase import phase_angle, S_GENES, G2M_GENES

# Cyclebase / Whitfield 2002 canonical peak phase in DEGREES from G1/S -- external time-course annotation.
PEAK = {
    "CCNE1": 0, "CCNE2": 0, "E2F1": 10, "PCNA": 20, "CDC6": 15, "CDT1": 15, "MCM2": 10, "MCM3": 10,
    "MCM4": 10, "MCM5": 10, "MCM6": 10, "MCM7": 10, "SLBP": 30, "RRM2": 45, "TYMS": 45, "DHFR": 40,
    "CDC45": 35, "GINS2": 40, "CLSPN": 35, "FEN1": 50, "RFC4": 45, "POLA1": 40,
    "CCNA2": 120, "TOP2A": 170, "NUSAP1": 175, "UBE2C": 200, "BIRC5": 205, "TPX2": 190,
    "CENPF": 185, "CENPE": 200, "CDK1": 180, "CCNB1": 210, "CCNB2": 210, "PLK1": 205,
    "BUB1": 195, "BUB1B": 195, "AURKA": 205, "AURKB": 215, "KIF11": 185, "KIF23": 210,
    "ANLN": 200, "ECT2": 195, "CDC20": 220, "PTTG1": 215, "CKS2": 190, "AURKAIP1": 200,
}


def circ_dist_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# ---- SECONDARY module source: co-expression-derived, uniform circle coverage -------------------------------
# EXPLICITLY WEAKER THAN `annotated`. The annotated source uses external time-course peak phases, so it is
# non-circular — but ~40% of the cycle (roughly 270-330 deg) has no phase-specific transcripts, so loop closure
# cannot be tested with it. This source instead defines each phase module from THIS dataset's own covariance
# (top genes enriched in cells at that phase), which covers all 12 phases uniformly and therefore makes loop
# closure testable — at the cost of reintroducing the co-expression circularity: the modules are built from the
# same covariance that defines the phase label. So the fact that the representation TRACKS is partly baked in;
# what is NOT baked in is the GEOMETRY — whether the model arranges the phases as a closed loop with correct
# adjacency. Read only the geometry from this arm.
import re as _re
_STOP_RE = _re.compile(r"^(MT-|MTRNR|RPL|RPS|MRPL|MRPS|LINC|AC[0-9]{6}|AL[0-9]{6}|RP[0-9]+-)")
_STOP_SET = {"MALAT1", "NEAT1", "RACK1", "EEF1A1", "EEF2", "ACTB", "ACTG1", "B2M", "TMSB4X", "TMSB10",
             "FTL", "FTH1", "TPT1", "GAPDH", "XIST"}


def coexpr_module_table(adata, theta, n_phases, k, df_max=0.5):
    """{target_deg: [k genes most enriched in cells at that phase]} — stopword-filtered."""
    import scipy.sparse as sp
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    df = (X > 0).mean(0)
    ok = np.array([not (_STOP_RE.match(s) or s in _STOP_SET) for s in var]) & (df <= df_max)
    edges = np.linspace(0, 2 * np.pi, n_phases + 1)
    b = np.clip(np.digitize(theta, edges) - 1, 0, n_phases - 1)
    gm = X.mean(0)
    table = {}
    for j in range(n_phases):
        m = b == j
        d = (X[m].mean(0) - gm) if m.sum() >= 10 else np.zeros_like(gm)
        d = np.where(ok, d, -np.inf)
        table[float(j * 360.0 / n_phases)] = [str(var[i]) for i in np.argsort(-d)[:k]]
    return table


def insert_at(backbone, module, positions):
    """Insert module genes at fixed rank positions -> rank is identical across conditions."""
    out = list(backbone)
    for g, p in sorted(zip(module, positions), key=lambda x: x[1]):
        out.insert(min(p, len(out)), g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--real-states", required=True, help="dir of real-cell cell-summary activations (for the frame)")
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 15, 21])
    ap.add_argument("--n-backbone", type=int, default=20)
    ap.add_argument("--n-phases", type=int, default=12)
    ap.add_argument("--module-k", type=int, default=8)
    ap.add_argument("--module-source", choices=["annotated", "coexpr"], default="annotated",
                    help="annotated = external time-course peak phase (PRIMARY, non-circular); "
                         "coexpr = this dataset's covariance (SECONDARY, circular; uniform circle "
                         "coverage so loop closure becomes testable — read GEOMETRY only)")
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/synth_sweep.json")
    a = ap.parse_args()

    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    var_up = set(np.char.upper(np.asarray(ad.var_names).astype(str)))
    peak = {g: d for g, d in PEAK.items() if g in var_up}
    cc_all = (set(S_GENES) | set(G2M_GENES) | set(PEAK)) & var_up
    print(f"annotated peak-phase genes usable: {len(peak)} | stripping {len(cc_all)} cc genes from backbones",
          flush=True)

    # ---- reference frame: fit (cos,sin) decoders on REAL cells, per layer ----
    theta_real, _, _, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.real_states, "row_cell_ids.npy"))
    frames = {}
    for L in a.layers:
        H = np.load(os.path.join(a.real_states, f"layer_{L:02d}_activations.npy")).astype(np.float64)
        t = theta_real[ids][:len(H)]; H = H[:len(t)]
        sc = StandardScaler().fit(H)
        m = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
        P = m.predict(sc.transform(H))
        frames[L] = dict(sc=sc, m=m, real_radius=float(np.median(np.linalg.norm(P - P.mean(0), axis=1))),
                         centre=P.mean(0))
        print(f"  L{L:02d} frame fitted on {len(H)} real cells (median radius {frames[L]['real_radius']:.3f})",
              flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    rng = np.random.default_rng(0)
    targets = np.arange(a.n_phases) * (360.0 / a.n_phases)
    positions = np.linspace(20, 120, a.module_k).astype(int)
    pgenes = list(peak); pdeg = np.array([peak[g] for g in pgenes], float)
    shuf_deg = rng.permutation(pdeg)                       # same genes, permuted phase labels
    ctab = coexpr_module_table(ad, theta_real, a.n_phases, a.module_k) if a.module_source == "coexpr" else None
    if ctab is not None:
        keys = sorted(ctab); shuf_map = dict(zip(keys, rng.permutation(keys)))
        print("co-expression modules (SECONDARY, circular by construction — read GEOMETRY only):", flush=True)
        for t in keys:
            print(f"    {t:5.0f}deg  {', '.join(ctab[t][:5])}", flush=True)

    # backbones: real cells with every cell-cycle gene removed
    order = rng.permutation(ad.n_obs)
    backbones = []
    for ci in order:
        g = [x for x in anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes) if x.upper() not in cc_all]
        if len(g) >= 200:
            backbones.append((int(ci), g[: a.max_genes - a.module_k]))
        if len(backbones) >= a.n_backbone:
            break
    noncc = [g for g in np.char.upper(np.asarray(ad.var_names).astype(str)) if g not in cc_all]
    print(f"{len(backbones)} phase-neutral backbones | {a.n_phases} target phases | k={a.module_k}", flush=True)

    def module_for(th, mode):
        if mode == "annotated":
            if ctab is not None:                                   # co-expression source (secondary)
                return list(ctab[min(ctab, key=lambda x: circ_dist_deg(x, th))]), 0.0
            idx = np.argsort(circ_dist_deg(pdeg, th))[: a.module_k]
            return [pgenes[i] for i in idx], float(np.mean(circ_dist_deg(pdeg[idx], th)))
        if mode == "shuffled":
            if ctab is not None:
                key = min(ctab, key=lambda x: circ_dist_deg(x, th))
                return list(ctab[shuf_map[key]]), None
            idx = np.argsort(circ_dist_deg(shuf_deg, th))[: a.module_k]
            return [pgenes[i] for i in idx], None
        return list(rng.choice(noncc, a.module_k, replace=False)), None

    recs = []
    for bi, (ci, bb) in enumerate(backbones):
        for mode in ["annotated", "random", "shuffled"]:
            for th in targets:
                mod, cov = module_for(th, mode)
                genes = insert_at(bb, mod, positions)
                cs = build_cell_sentence(genes)
                enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
                pos = int(enc["input_ids"].shape[1]) - 1
                with torch.no_grad():
                    out = model(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True)
                r = dict(backbone=ci, mode=mode, theta=float(th), coverage_deg=cov, proj={})
                for L in a.layers:
                    h = out.hidden_states[L + 1][0, pos].float().cpu().numpy()[None, :]
                    f = frames[L]
                    P = f["m"].predict(f["sc"].transform(h))[0] - f["centre"]
                    r["proj"][f"L{L:02d}"] = dict(ang=float(np.mod(np.degrees(np.arctan2(P[1], P[0])), 360)),
                                                  rad=float(np.linalg.norm(P)))
                recs.append(r)
        if (bi + 1) % 5 == 0:
            print(f"  backbone {bi+1}/{len(backbones)}", flush=True)

    # ---- analysis ----
    def circ_corr_deg(x, y):
        x, y = np.radians(x), np.radians(y)
        sx = np.sin(x - np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x))))
        sy = np.sin(y - np.arctan2(np.mean(np.sin(y)), np.mean(np.cos(y))))
        return float(np.sum(sx * sy) / (np.sqrt(np.sum(sx ** 2) * np.sum(sy ** 2)) + 1e-12))

    summary = {}
    for L in a.layers:
        key = f"L{L:02d}"; summary[key] = {}
        for mode in ["annotated", "random", "shuffled"]:
            sub = [r for r in recs if r["mode"] == mode]
            th = np.array([r["theta"] for r in sub]); an = np.array([r["proj"][key]["ang"] for r in sub])
            rd = np.array([r["proj"][key]["rad"] for r in sub])
            # per-backbone angular travel and loop closure
            travels, closures, ccs = [], [], []
            for ci, _ in backbones:
                s = [r for r in sub if r["backbone"] == ci]
                s.sort(key=lambda r: r["theta"])
                aa = np.array([r["proj"][key]["ang"] for r in s])
                if len(aa) < 3:
                    continue
                d = np.diff(np.unwrap(np.radians(aa)))
                travels.append(float(np.degrees(np.abs(d).sum())))
                closures.append(float(circ_dist_deg(aa[0], aa[-1])))
                ccs.append(circ_corr_deg(np.array([r["theta"] for r in s]), aa))
            summary[key][mode] = dict(
                circ_corr_theta_vs_angle=float(np.mean(ccs)) if ccs else None,
                mean_angular_travel_deg=float(np.mean(travels)) if travels else None,
                mean_loop_gap_deg=float(np.mean(closures)) if closures else None,
                radius_mean=float(rd.mean()), radius_cv=float(rd.std() / (rd.mean() + 1e-12)),
                radius_vs_real=float(rd.mean() / frames[L]["real_radius"]))
            s_ = summary[key][mode]
            print(f"  {key} {mode:<10} circ_corr(theta,angle)={s_['circ_corr_theta_vs_angle']:+.3f} "
                  f"travel={s_['mean_angular_travel_deg']:.0f}deg loop_gap={s_['mean_loop_gap_deg']:.0f}deg "
                  f"radius={s_['radius_mean']:.3f} ({s_['radius_vs_real']:.2f}x real)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, n_backbone=len(backbones), targets=targets.tolist(),
                   module_k=a.module_k, positions=positions.tolist(), records=recs),
              open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()


# ---- HELD-OUT, FAMILIARITY-MATCHED gene sets (fixes the two caveats of heldout_gene_placement.py) ----
# Caveat 1 there: "absent from the K562 panel" was confounded with "not expressed in K562", so a null could
#   mean the gene is merely unfamiliar. Fix: keep only held-out genes that are WELL EXPRESSED ELSEWHERE
#   (Tabula Sapiens document frequency > FAM_MIN), so the model has demonstrably seen them.
# Caveat 2 there: 1-8 gene prompts are far off-distribution (radius 1.15-1.76 vs real-cell 0.86). Fix: insert
#   them into a FULL 512-gene phase-neutral backbone, exactly as the constructive sweep does.
def heldout_familiar_sets(panel_syms, ts_h5ad, ens_path, g2g, fam_min=0.05):
    import pickle, numpy as np, anndata, scipy.sparse as sp
    sym2ens = {k.upper(): v for k, v in pickle.load(open(ens_path, "rb")).items()}
    panel_ens = {sym2ens[s] for s in panel_syms if s in sym2ens}
    g1s = {g.upper() for g, t in g2g.items() if "GO:0000082" in set(t)}
    g2m = {g.upper() for g, t in g2g.items() if "GO:0000086" in set(t)}
    both = g1s & g2m; g1s, g2m = g1s - both, g2m - both
    ts = anndata.read_h5ad(ts_h5ad)
    tv = np.char.upper(np.asarray(ts.var_names).astype(str))
    X = ts.X.toarray() if sp.issparse(ts.X) else np.asarray(ts.X)
    freq = {s: float(f) for s, f in zip(tv, (X > 0).mean(0))}
    def keep(S):
        return sorted({g for g in S if sym2ens.get(g) and sym2ens[g] not in panel_ens
                       and freq.get(g, 0.0) >= fam_min})
    return keep(g1s), keep(g2m), freq
