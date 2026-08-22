"""Tier-A non-regulatory geometry test (direction #2).

Question: does an SCFM's L11 representation encode a *non-regulatory* biological ordering (cell-cycle for
Geneformer/K562, cell-type lineage for scGPT/Tabula Sapiens) that (a) is real under proper gates, and
(b) is recovered better by BILINEAR (interaction) latents than by a matched LINEAR SAE?

Method: build per-cell embeddings by mean-pooling each dictionary's token activations over the cell's tokens,
for four matched representations — raw-mean (d_model), SVD-50, linear TopK SAE (n_features), bilinear atomic
latents (n_features) — then score each embedding on ruler recovery with a shuffled-label null.

Metrics (5-fold CV, embeddings z-scored across cells):
  scGPT (cell-type):  k-NN balanced accuracy on cell_type (+ tissue);  Spearman(pairwise emb dist, ruler dist)
  Geneformer (cc):    k-NN regression R^2 on S-score & G2M-score;      Spearman(emb PC1, cc score)
  + trustworthiness (2D PCA vs full emb) as a neighbor-preservation sanity number
  + null: shuffle labels -> chance level; report margin (recovery - null)

Run:  ../.venv/bin/python geometry_test.py [scgpt|geneformer|both]   (CPU; co-tenant-friendly)
"""
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROUTE_B = os.path.abspath(os.path.join(HERE, "..", "route_b"))
sys.path.insert(0, ROUTE_B)
sys.path.insert(0, os.path.join(ROUTE_B, "baseline"))
import config as C                                   # route_b/config.py
from bilinear_ae import BilinearAE                   # route_b/bilinear_ae.py
import sae_baseline as SB                            # route_b/baseline/sae_baseline.py

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr

BATCH = 8192
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))   # leave headroom for the STATE agent


def per_cell_embeddings(model_key, run_name):
    """One streaming pass over L11 tokens -> per-cell mean-pooled embeddings for 4 representations."""
    tgt = C.TARGETS[model_key]
    d_model, n_feat = tgt["d_model"], tgt["n_features"]
    acts = np.load(tgt["acts"], mmap_mode="r")                       # (n_pos, d_model) float32
    cell_ids = np.load(tgt["cell_ids"]).astype(np.int64)            # (n_pos,)
    n_cells = int(cell_ids.max()) + 1
    n_pos = acts.shape[0]

    # linear SAE (atlas) + its centering mean
    sae, sae_cfg = SB.load_sae(str(tgt["linear_sae_dir"]), device=DEV)
    lin_mean = torch.tensor(sae_cfg["activation_mean"], dtype=torch.float32) \
        if sae_cfg.get("activation_mean") is not None else torch.zeros(d_model)
    # bilinear atomic AE + its centering mean
    ck = torch.load(os.path.join(ROUTE_B, "runs", run_name, "latest.pt"), map_location="cpu", weights_only=False)
    bil = BilinearAE.from_config(ck["config"]).to(DEV).eval()
    bil.load_state_dict(ck["model"])
    bil_mean = ck["mean"].float() if torch.is_tensor(ck["mean"]) else torch.tensor(ck["mean"]).float()

    counts = np.zeros(n_cells, dtype=np.float64)
    sum_raw = torch.zeros(n_cells, d_model, dtype=torch.float64)
    sum_lin = torch.zeros(n_cells, n_feat, dtype=torch.float64)
    sum_bil = torch.zeros(n_cells, n_feat, dtype=torch.float64)
    svd_sample = []                                                  # centered-token subsample for SVD basis

    with torch.no_grad():
        for s in range(0, n_pos, BATCH):
            e = min(s + BATCH, n_pos)
            xb = torch.from_numpy(np.asarray(acts[s:e], dtype=np.float32))     # (b, d_model)
            cid = torch.from_numpy(cell_ids[s:e])
            x_lin = xb - lin_mean
            x_bil = xb - bil_mean
            h_lin, _ = sae.encode(x_lin.to(DEV))                     # (b, n_feat) TopK sparse
            h_bil = bil.encode(x_bil.to(DEV))                        # (b, n_feat) atomic latents
            sum_raw.index_add_(0, cid, x_bil.double())
            sum_lin.index_add_(0, cid, h_lin.double().cpu())
            sum_bil.index_add_(0, cid, h_bil.double().cpu())
            np.add.at(counts, cell_ids[s:e], 1.0)
            if len(svd_sample) < 60 and (s // BATCH) % 7 == 0:
                svd_sample.append(x_bil.numpy().copy())

    cnt = torch.tensor(counts, dtype=torch.float64).clamp_min(1).unsqueeze(1)
    E_raw = (sum_raw / cnt).float().numpy()
    E_lin = (sum_lin / cnt).float().numpy()
    E_bil = (sum_bil / cnt).float().numpy()
    # SVD-50 basis from centered-token subsample, project the per-cell raw means
    S = np.concatenate(svd_sample, 0)
    S = S - S.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(S[: min(len(S), 60000)], full_matrices=False)
    V50 = Vt[:50].T
    E_svd = (E_raw - E_raw.mean(0, keepdims=True)) @ V50
    return dict(raw_mean=E_raw, svd50=E_svd, linear_sae=E_lin, bilinear=E_bil), int(n_cells)


def _z(X):
    X = np.nan_to_num(X.astype(np.float64))
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def score_classification(E, labels, k=15, seed=0):
    y = np.asarray(labels)
    keep = np.array([np.sum(y == v) >= 5 for v in y])               # drop ultra-rare types for stable CV
    Xz, yk = _z(E)[keep], y[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    knn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(knn, Xz, yk, cv=cv, scoring="balanced_accuracy").mean()
    rng = np.random.default_rng(seed)
    yperm = rng.permutation(yk)
    null = cross_val_score(knn, Xz, yperm, cv=cv, scoring="balanced_accuracy").mean()
    return dict(bal_acc=float(acc), null=float(null), margin=float(acc - null), n=int(keep.sum()))


def score_regression(E, target, k=15, seed=0):
    Xz = _z(E); t = np.asarray(target, dtype=np.float64)
    cv = KFold(5, shuffle=True, random_state=seed)
    knn = KNeighborsRegressor(n_neighbors=k)
    r2 = cross_val_score(knn, Xz, t, cv=cv, scoring="r2").mean()
    rng = np.random.default_rng(seed)
    null = cross_val_score(knn, Xz, rng.permutation(t), cv=cv, scoring="r2").mean()
    pc1 = _z(E) @ np.linalg.svd(_z(E), full_matrices=False)[2][0]
    rho = spearmanr(pc1, t).statistic
    return dict(r2=float(r2), null=float(null), margin=float(r2 - null), spearman_pc1=float(rho))


def ruler_dist_correlation(E, cell_type, tissue, n_pairs=40000, seed=0):
    """Spearman between pairwise embedding distance and a tissue+lineage ruler distance."""
    rng = np.random.default_rng(seed)
    Xz = _z(E); n = len(Xz)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    m = i != j; i, j = i[m], j[m]
    emb_d = np.linalg.norm(Xz[i] - Xz[j], axis=1)
    same_type = cell_type[i] == cell_type[j]
    same_tis = tissue[i] == tissue[j]
    ruler_d = np.where(same_type, 0.0, np.where(same_tis, 1.0, 2.0))
    return float(spearmanr(emb_d, ruler_d).statistic)


def run(model_key, run_name):
    print(f"\n===== {model_key} :: embeddings (streaming pass) =====", flush=True)
    embs, n_cells = per_cell_embeddings(model_key, run_name)
    out = {"model": model_key, "n_cells": n_cells, "representations": {}}
    reps = ["raw_mean", "svd50", "linear_sae", "bilinear"]

    if model_key.startswith("scgpt"):
        r = np.load(os.path.join(HERE, "results", "scgpt_L11_ruler.npz"), allow_pickle=True)
        cell_type, tissue = r["cell_type"], r["tissue"]
        for name in reps:
            E = embs[name]
            ct = score_classification(E, cell_type)
            ti = score_classification(E, tissue)
            tw = float(trustworthiness(_z(E), _z(E)[:, :2] if E.shape[1] > 2 else _z(E), n_neighbors=15))
            rc = ruler_dist_correlation(E, cell_type, tissue)
            out["representations"][name] = dict(dim=int(E.shape[1]), celltype=ct, tissue=ti,
                                                ruler_dist_spearman=rc, trustworthiness2d=tw)
            print(f"  {name:11s} dim={E.shape[1]:5d} | celltype bal_acc={ct['bal_acc']:.3f} "
                  f"(null {ct['null']:.3f}, +{ct['margin']:.3f}) | tissue={ti['bal_acc']:.3f} "
                  f"| ruler_rho={rc:+.3f}", flush=True)
    else:
        r = np.load(os.path.join(HERE, "results", "geneformer_L11_ruler.npz"), allow_pickle=True)
        s_score, g2m = r["s_score"].astype(float), r["g2m_score"].astype(float)
        for name in reps:
            E = embs[name]
            sc = score_regression(E, s_score); gc = score_regression(E, g2m)
            tw = float(trustworthiness(_z(E), _z(E)[:, :2] if E.shape[1] > 2 else _z(E), n_neighbors=15))
            out["representations"][name] = dict(dim=int(E.shape[1]), s_score=sc, g2m_score=gc,
                                                trustworthiness2d=tw)
            print(f"  {name:11s} dim={E.shape[1]:5d} | S-score R2={sc['r2']:+.3f} "
                  f"(null {sc['null']:+.3f}, +{sc['margin']:.3f}) | G2M R2={gc['r2']:+.3f} "
                  f"| PC1 rho(S)={sc['spearman_pc1']:+.3f}", flush=True)

    with open(os.path.join(HERE, "results", f"{model_key}_geometry.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("scgpt", "both"):
        run("scgpt_L11", "scgpt_L11_atomic_muon")
    if which in ("geneformer", "both"):
        run("geneformer_L11", "geneformer_L11_atomic_muon")
    print("\ndone.")
