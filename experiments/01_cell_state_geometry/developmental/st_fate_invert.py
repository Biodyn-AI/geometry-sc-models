"""STATE-ST: does the PERTURBATION-TRAINED model know which knockdown moves a progenitor off a fate?

This is the one model that could overturn PERTURB_INVERT_CROSSMODEL.md. scGPT / MaxToki / STATE-SE are all
*embedding* models -- trained to represent cells, not to predict what a perturbation does. ST is trained on
Replogle K562 CRISPRi to predict exactly that. So it is the fair test of the CAPABILITY rather than of the
representation.

METHOD. ST consumes a "sentence" of S=64 basal X_state (2058-d) cell embeddings + a perturbation one-hot, and
predicts the perturbed states. Running it twice on the SAME basal cells -- once under perturbation p, once
under non-targeting -- gives the paired, pure perturbation effect (route_state_st/extract.py:120-128):

    Delta(p) = mean_cells [ out(p) - out(non-targeting) ]        (2058-d shift)

We feed it SETTY CD34+ progenitors (their X_state computed with STATE-SE, dim verified 2058), project Delta(p)
into the same 20-PC whitened space the fate directions live in, and score

    score(p, f) = -cos( Delta(p), d_f )       d_f = the on-manifold, FATE-SPECIFIC direction toward fate f

with a MINUS because CRISPRi is a KNOCKDOWN: knocking down a driver of fate f should push the cell AWAY from f.
(Note this is the opposite intervention to the activation screens on the other three models -- not a like-for-
like replication, and stated as such.)

*** WHY THIS IS NOT THE SAME BENCHMARK -- read before interpreting anything ***
The driver-recovery AUROC used for the other three models is NOT COMPUTABLE on ST. ST's perturbation vocabulary
is the Replogle K562 CRISPRi library (2024 genes), which targeted genes ESSENTIAL IN K562 -- and hematopoietic
lineage master regulators mostly are not. Coverage of our pre-registered driver sets:
      ery 1/7 (GATA1 only) | myeloid 0/8 (NONE) | lymphoid 1/8 (FOXO1) | mega 1/7 (GATA1)
You cannot compute an AUROC with zero positives. So instead we run the narrowest test ST CAN support, and
pre-register it here BEFORE looking at any output:

  GATE (validity, checked first). ST is being asked to extrapolate: it was trained on K562, and CD34+
    progenitors are out-of-distribution; its cell-type/batch tokens carry no Setty identity. If ST returns
    essentially the SAME shift no matter which perturbation is named, the ranking below is vacuous. So we first
    measure whether Delta(p) is PERTURBATION-SPECIFIC: mean pairwise cosine between the Delta(p) of different p,
    and the effective rank of the Delta matrix. If the shifts are all one direction, we stop and say so.

  H1 (primary). GATA1 -- THE erythroid master regulator, and one of the few real drivers ST knows -- should rank
     near the TOP of the 131 TFs by anti-erythroid score. Statistic: GATA1's rank / 131 = an exact empirical p
     against the other 130 TFs as the null.
  H2 (specificity). GATA1's anti-ERY score should exceed its anti-myeloid / anti-lymphoid score -- i.e. the
     effect is erythroid-specific, not a generic "moves the cell somewhere" effect.
  H3 (secondary, weaker prior). FOXO1 should rank high for lymphoid.

Run (.venv_state):  ../../.venv_state/bin/python st_fate_invert.py
Out: results/st_fate_invert_setty.json
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
import numpy as np
import anndata as ad
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "route_state"))
sys.path.insert(0, str(HERE.parent / "route_state_st"))
from tangent_diagnostic import unit, SEED, DIM  # noqa: E402
from local_steering import project_constant  # noqa: E402
from perturb_invert import FATE_CLUSTERS, DRIVERS  # noqa: E402

PROJ = f"{_DATA}"
SETTY = f"{PROJ}/data/hematopoiesis/setty19_cd34_bm.h5ad"
SE_NPZ = f"{PROJ}/data/branchpoint/state_setty.npz"          # defines WHICH Setty cells we use
XSTATE = f"{PROJ}/data/branchpoint/setty_xstate.h5ad"        # cached X_state (2058-d) for those cells
RESULTS = HERE / "results"
TF_DB = (f"{_DATA}/biodyn-work/network_inference/data/"
         "dorothea_trrust_union_immune.tsv")
S, N_SENT, B_SENT = 64, 8, 8                                  # sentence len, sentences per pert, batch


def build_xstate(cells):
    """X_state (2058-d) for the given Setty rows, via STATE-SE's official encoder. Cached."""
    if os.path.exists(XSTATE):
        a = ad.read_h5ad(XSTATE)
        if a.n_obs == len(cells):
            print(f"[cache] X_state {a.obsm['X_state'].shape}", flush=True)
            return np.asarray(a.obsm["X_state"], np.float64)
    from state_loader import load_state_se, load_protein_embeds
    from state.emb.inference import Inference
    dev = os.environ.get("ST_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg, _, _ = load_state_se(device=dev, dtype=torch.float32)
    pe, _, _ = load_protein_embeds()
    inf = Inference(cfg=cfg); inf.init_from_model(model, protein_embeds=pe)
    tmp = f"{PROJ}/data/branchpoint/_setty_sub.h5ad"
    ad.read_h5ad(SETTY)[cells].copy().write_h5ad(tmp)
    # SE-600M's default encode batch OOMs MPS (esp. when another job holds the GPU) -- force a small batch.
    bs = int(os.environ.get("SE_BATCH", "4"))
    print(f"[encode] X_state for {len(cells)} Setty cells (STATE-SE, dev={dev}, batch={bs})...", flush=True)
    inf.encode_adata(tmp, XSTATE, emb_key="X_state", batch_size=bs)
    a = ad.read_h5ad(XSTATE)
    print(f"[encode] -> {a.obsm['X_state'].shape}", flush=True)
    return np.asarray(a.obsm["X_state"], np.float64)


def main():
    RESULTS.mkdir(exist_ok=True)
    z = np.load(SE_NPZ, allow_pickle=True)
    y = z["pseudotime"].astype(np.float64); ok = np.isfinite(y)
    y = y[ok]; ci = z["cell_idx"].astype(int)[ok]; clusters = z["clusters"][ok]

    X = build_xstate(ci)                                       # (n, 2058) X_state for the SAME cells
    assert X.shape[0] == len(y), f"{X.shape} vs {len(y)}"

    # ---- whitened space + on-manifold FATE-SPECIFIC directions (identical construction to the other models)
    mean = X.mean(0)
    pca = PCA(min(DIM, X.shape[1], len(X) - 1), random_state=SEED).fit(X - mean)
    wsc = StandardScaler().fit(pca.transform(X - mean))
    to_z = lambda E: wsc.transform(pca.transform(np.atleast_2d(E) - mean))   # noqa: E731
    Xz = to_z(X)
    yz = (y - y.mean()) / (y.std() + 1e-9)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    clu_tr = clusters[tr]
    src_c = Xz[pool].mean(0)

    dirs, fl = {}, []
    for f, cls in FATE_CLUSTERS.items():
        m = np.isin(clu_tr, cls)
        if m.sum() < 20:
            print(f"  [skip fate {f}: {m.sum()} train cells]")
            continue
        w = unit(Xz[tr][m].mean(0) - src_c)
        D = project_constant(Xz[tr], w)(Xz[pool].mean(0)[None, :])[0]
        dirs[f] = unit(D); fl.append(f)
    Dm = np.mean([dirs[f] for f in fl], axis=0)
    spec = {f: unit(dirs[f] - Dm) for f in fl}                 # fate-specific = the bifurcation contrast

    # ---- ST + its perturbation vocabulary
    from st_loader import load_st
    import config as C
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, maps, _ = load_st(device=dev)
    pmap = maps["pert"]
    tfs_db = sorted({l.split("\t")[0] for l in open(TF_DB).read().splitlines()[1:] if l.strip()})
    tfs = [t for t in tfs_db if t in pmap]
    cov = {f: [d for d in DRIVERS[f] if d in pmap] for f in fl}
    print(f"[vocab] ST perturbations: {len(pmap)} | TF universe in it: {len(tfs)}/{len(tfs_db)}")
    print("[vocab] driver coverage: " + " | ".join(f"{f} {len(cov[f])}/{len(DRIVERS[f])} {cov[f]}" for f in fl))

    nt = pmap["non-targeting"].float().cpu().numpy()
    basal_pool = X[pool].astype(np.float32)

    def delta(gene):
        """Paired perturbation effect on Setty progenitors: mean_cells[ out(g) - out(non-targeting) ]."""
        oh = pmap[gene].float().cpu().numpy()
        acc = []
        g = np.random.default_rng(abs(hash(gene)) % (2 ** 31))
        for b in range(0, N_SENT, B_SENT):
            nb = min(B_SENT, N_SENT - b)
            rows = g.choice(len(basal_pool), nb * S, replace=True)
            bt = torch.from_numpy(basal_pool[rows]).to(dev)
            bi = torch.zeros(nb * S, dtype=torch.long, device=dev)     # Setty has no gem_group -> batch 0
            with torch.no_grad():
                op = model.forward(dict(pert_emb=torch.from_numpy(np.repeat(oh[None], nb * S, 0)).to(dev),
                                        ctrl_cell_emb=bt, batch=bi), padded=True)
                on = model.forward(dict(pert_emb=torch.from_numpy(np.repeat(nt[None], nb * S, 0)).to(dev),
                                        ctrl_cell_emb=bt, batch=bi), padded=True)
            acc.append((op - on).float().cpu().numpy())
        return np.concatenate(acc).mean(0)                             # (2058,)

    print(f"\n[run] {len(tfs)} perturbations x {N_SENT} sentences x {S} cells ...", flush=True)
    D = np.zeros((len(tfs), X.shape[1]))
    for k, t in enumerate(tfs):
        D[k] = delta(t)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(tfs)}", flush=True)

    # ---- GATE: is the predicted shift PERTURBATION-SPECIFIC on these OOD cells, or one generic direction?
    U = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    Cm = U @ U.T
    iu = np.triu_indices(len(tfs), 1)
    mean_cos = float(Cm[iu].mean())
    sv = np.linalg.svd(D, compute_uv=False)
    p_ = sv ** 2 / (sv ** 2).sum()
    erank = float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum()))
    norms = np.linalg.norm(D, axis=1)
    print("\n" + "=" * 96)
    print("GATE — is ST's predicted shift perturbation-SPECIFIC on out-of-distribution Setty progenitors?")
    print("=" * 96)
    print(f"  mean pairwise cosine between Delta(p) of different perturbations : {mean_cos:+.3f}"
          f"   (near 1.0 => one generic direction => the ranking below is VACUOUS)")
    print(f"  effective rank of the {len(tfs)}x{X.shape[1]} shift matrix            : {erank:.1f}")
    print(f"  ||Delta(p)||: median {np.median(norms):.3f}  min {norms.min():.3f}  max {norms.max():.3f}")
    gate = (mean_cos < 0.90) and (erank > 2.0)
    print(f"  -> GATE {'PASSED' if gate else 'FAILED'}")

    out = dict(model="state_st", n_perts=len(tfs), n_src=int(len(pool)), fates=fl,
               driver_coverage={f: cov[f] for f in fl},
               gate=dict(mean_pairwise_cos=mean_cos, eff_rank=erank, passed=bool(gate),
                         delta_norm_median=float(np.median(norms))))

    if not gate:
        print("\n  *** ST returns essentially ONE shift direction regardless of which perturbation is named.")
        print("      On out-of-distribution CD34+ progenitors it is not using the perturbation token, so no")
        print("      ranking derived from it is interpretable. Reporting the gate failure, not a ranking.")
        json.dump(out, open(RESULTS / "st_fate_invert_setty.json", "w"), indent=1)
        return

    # ---- scores: knockdown => a DRIVER of fate f should push AWAY from f (minus sign)
    Dz = np.array([to_z(X[pool].mean(0) + D[k])[0] - to_z(X[pool].mean(0))[0] for k in range(len(tfs))])
    Dz = Dz / (np.linalg.norm(Dz, axis=1, keepdims=True) + 1e-12)
    score = {f: -(Dz @ spec[f]) for f in fl}

    print("\n  ---- pre-registered tests ----")
    for f in fl:
        order = np.argsort(-score[f])
        rank = {tfs[i]: r + 1 for r, i in enumerate(order)}
        out.setdefault("ranks", {})[f] = {d: rank[d] for d in cov[f]}
        out.setdefault("top10", {})[f] = [tfs[i] for i in order[:10]]
        print(f"    ->{f:<9} top10: " + ", ".join(f"**{tfs[i]}**" if tfs[i] in cov[f] else tfs[i]
                                                  for i in order[:10]))
        if cov[f]:
            print(f"       known drivers: " + ", ".join(
                f"{d}=rank {rank[d]}/{len(tfs)} (empirical p={rank[d]/len(tfs):.3f})" for d in cov[f]))

    if "GATA1" in tfs:
        gi = tfs.index("GATA1")
        spec_row = {f: float(score[f][gi]) for f in fl}
        out["H2_gata1_specificity"] = spec_row
        print(f"\n    H2 (specificity) GATA1 anti-fate score: "
              + "  ".join(f"{f}={spec_row[f]:+.3f}" for f in fl))
        best = max(spec_row, key=spec_row.get)
        print(f"       strongest anti-fate effect: {best}"
              f"{'  ✓ (erythroid, as predicted)' if best == 'ery' else '  ✗ (NOT erythroid)'}")

    json.dump(out, open(RESULTS / "st_fate_invert_setty.json", "w"), indent=1)
    print(f"\n[done] -> results/st_fate_invert_setty.json")


if __name__ == "__main__":
    main()
