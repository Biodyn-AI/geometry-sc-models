"""Direction #2, next steps: (1) multi-seed robustness on the bilinear-vs-linear geometry margin, and
(2) the STRUCTURE SWEEP re-scored against the *geometry* ruler (not regulatory annotation).

Motivation: the original structure sweep (block-diagonal, nnz-sparsity, nuisance-sink, ...) was scored on
regulatory annotation and came back negative. But the real signal these models carry is non-regulatory
cell-state geometry (route_geometry/RESULTS.md), where bilinear modestly beats linear. This asks the right
question: which inductive bias best isolates that geometry — and, via the nuisance-SINK model (abundance
isolated into dedicated latents), whether the bilinear geometry lift SURVIVES abundance removal (the key
confound control) or was just total-count.

Trained AEs reused from route_b/runs (all scGPT L11, matched n_features=2048):
  linear_sae  = trained TopK atlas (baseline)
  atomic      = base bilinear (atomic_muon)
  composite   = base composite (composite_muon)
  block_diag  = block-diagonal composite (composite_muon_blk512)      <- earlier lone "win"
  nnz8        = sparsity control (composite_muon_nnz8)                 <- isolates sparsity from modularity
  sink        = nuisance-sink (composite_muon_sink5_32)               <- ABUNDANCE CONTROL

Run:  ../.venv/bin/python structure_geometry.py            (CPU; caches embeddings to results/emb_cache/)
"""
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROUTE_B = os.path.abspath(os.path.join(HERE, "..", "route_b"))
sys.path.insert(0, ROUTE_B); sys.path.insert(0, os.path.join(ROUTE_B, "baseline"))
import config as C
from bilinear_ae import BilinearAE
import sae_baseline as SB
from geometry_test import _z, BATCH, DEV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
CACHE = os.path.join(HERE, "results", "emb_cache"); os.makedirs(CACHE, exist_ok=True)

# scGPT L11 structure variants (name -> run dir). All BilinearAE, loaded via from_config.
SCGPT_BILIN = {
    "atomic": "scgpt_L11_atomic_muon",
    "composite": "scgpt_L11_composite_muon",
    "block_diag": "scgpt_L11_composite_muon_blk512",
    "nnz8": "scgpt_L11_composite_muon_nnz8",
    "sink": "scgpt_L11_composite_muon_sink5_32",
}


def _load_bilin(run):
    ck = torch.load(os.path.join(ROUTE_B, "runs", run, "latest.pt"), map_location="cpu", weights_only=False)
    m = BilinearAE.from_config(ck["config"]).to(DEV).eval()
    m.load_state_dict(ck["model"], strict=False)
    mean = ck["mean"].float() if torch.is_tensor(ck["mean"]) else torch.tensor(ck["mean"]).float()
    return m, mean


def pooled_embeddings(model_key, bilin_runs):
    """One streaming pass: per-cell mean-pooled embeddings for linear_sae + each bilinear variant."""
    cache = os.path.join(CACHE, f"{model_key}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    tgt = C.TARGETS[model_key]; d_model, n_feat = tgt["d_model"], tgt["n_features"]
    acts = np.load(tgt["acts"], mmap_mode="r"); cell_ids = np.load(tgt["cell_ids"]).astype(np.int64)
    n_cells = int(cell_ids.max()) + 1; n_pos = acts.shape[0]
    sae, scfg = SB.load_sae(str(tgt["linear_sae_dir"]), device=DEV)
    lin_mean = torch.tensor(scfg["activation_mean"], dtype=torch.float32) if scfg.get("activation_mean") is not None else torch.zeros(d_model)
    models = {name: _load_bilin(run) for name, run in bilin_runs.items()}
    counts = np.zeros(n_cells)
    sums = {"linear_sae": torch.zeros(n_cells, n_feat, dtype=torch.float64)}
    for name in models: sums[name] = torch.zeros(n_cells, n_feat, dtype=torch.float64)
    with torch.no_grad():
        for s in range(0, n_pos, BATCH):
            e = min(s + BATCH, n_pos)
            xb = torch.from_numpy(np.asarray(acts[s:e], dtype=np.float32)); cid = torch.from_numpy(cell_ids[s:e])
            h_lin, _ = sae.encode((xb - lin_mean).to(DEV)); sums["linear_sae"].index_add_(0, cid, h_lin.double().cpu())
            for name, (m, mean) in models.items():
                sums[name].index_add_(0, cid, m.encode((xb - mean).to(DEV)).double().cpu())
            np.add.at(counts, cell_ids[s:e], 1.0)
    cnt = torch.tensor(counts).clamp_min(1).unsqueeze(1)
    out = {k: (v / cnt).float().numpy() for k, v in sums.items()}
    np.savez(cache, **out); print(f"  cached {model_key} embeddings -> {cache}", flush=True)
    return out


def multiseed_class(E, labels, k=15, seeds=range(10)):
    y = np.asarray(labels); keep = np.array([np.sum(y == v) >= 5 for v in y]); Xz, yk = _z(E)[keep], y[keep]
    accs = []
    for sd in seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        accs.append(cross_val_score(KNeighborsClassifier(k), Xz, yk, cv=cv, scoring="balanced_accuracy").mean())
    return np.array(accs)


def multiseed_reg(E, target, k=15, seeds=range(10)):
    Xz = _z(E); t = np.asarray(target, float); r2 = []
    for sd in seeds:
        cv = KFold(5, shuffle=True, random_state=sd)
        r2.append(cross_val_score(KNeighborsRegressor(k), Xz, t, cv=cv, scoring="r2").mean())
    return np.array(r2)


def run_scgpt():
    print("\n===== scGPT L11: robustness + structure sweep on cell-type geometry =====", flush=True)
    E = pooled_embeddings("scgpt_L11", SCGPT_BILIN)
    r = np.load(os.path.join(HERE, "results", "scgpt_L11_ruler.npz"), allow_pickle=True)
    ct = r["cell_type"]
    reps = ["linear_sae", "atomic", "composite", "block_diag", "nnz8", "sink"]
    per = {name: multiseed_class(E[name], ct) for name in reps}
    lin = per["linear_sae"]
    out = {}
    print(f"  {'representation':12s}  cell-type bal-acc (10 seeds)   margin vs linear")
    for name in reps:
        a = per[name]; marg = a - lin
        out[name] = dict(mean=float(a.mean()), std=float(a.std()),
                         margin_vs_linear=float(marg.mean()), margin_std=float(marg.std()))
        tag = "" if name == "linear_sae" else f"  Δ={marg.mean():+.3f} ± {marg.std():.3f}"
        print(f"  {name:12s}  {a.mean():.3f} ± {a.std():.3f}{tag}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "scgpt_L11_structure_geometry.json"), "w"), indent=1)
    return out


def run_geneformer():
    print("\n===== Geneformer L11: robustness (bilinear vs linear on cell-cycle) =====", flush=True)
    E = pooled_embeddings("geneformer_L11", {"atomic": "geneformer_L11_atomic_muon"})
    r = np.load(os.path.join(HERE, "results", "geneformer_L11_ruler.npz"), allow_pickle=True)
    out = {}
    for tname, target in [("s_score", r["s_score"].astype(float)), ("g2m_score", r["g2m_score"].astype(float))]:
        lin = multiseed_reg(E["linear_sae"], target); bil = multiseed_reg(E["atomic"], target)
        marg = bil - lin
        out[tname] = dict(linear_mean=float(lin.mean()), linear_std=float(lin.std()),
                          bilinear_mean=float(bil.mean()), bilinear_std=float(bil.std()),
                          margin=float(marg.mean()), margin_std=float(marg.std()))
        print(f"  {tname:10s} linear {lin.mean():.3f}±{lin.std():.3f} | bilinear {bil.mean():.3f}±{bil.std():.3f}"
              f" | Δ={marg.mean():+.3f} ± {marg.std():.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "geneformer_L11_robustness.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("scgpt", "both"): run_scgpt()
    if which in ("geneformer", "both"): run_geneformer()
    print("\ndone.")
