"""Clean abundance control for the Geneformer cell-cycle bilinear lift.

Cell-cycle correlates with total RNA, so part of the recovery could be abundance, not interaction structure.
Test: partial out per-cell total-counts (and, as a 2nd proxy, the raw-mean abundance axis) from BOTH the
linear and bilinear per-cell embeddings, then re-score cell-cycle. If the bilinear-over-linear margin SURVIVES
residualization, the interaction lift is genuine (not total-count).

Uses cached embeddings (results/emb_cache/geneformer_L11.npz) + re-reads the 2000 cells' total counts from
replogle via the exact seed-42 selection (cheap). No big-tensor reload.
Run: ../.venv/bin/python abundance_control.py
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "route_b")))
from geometry_test import _z
from build_rulers import load_categorical_column, REPLOGLE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold


def total_counts_2000():
    import h5py
    with h5py.File(REPLOGLE, "r") as f:
        cell_genes = load_categorical_column(f["obs"], "gene")
        ctrl = np.where(np.isin(cell_genes, ["non-targeting", "Non-targeting", "non_targeting"]))[0]
        np.random.seed(42)
        sel = np.sort(np.random.choice(ctrl, 2000, replace=False)) if len(ctrl) > 2000 else np.sort(ctrl)
        tc = np.array([float(f["X"][int(r), :].sum()) for r in sel])
    return tc


def residualize(X, covars):
    """Return residuals of each column of X regressed on [1, covars...]."""
    C = np.column_stack([np.ones(len(X))] + [_z(c.reshape(-1, 1)).ravel() for c in covars])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def score(E, t, seeds=range(10)):
    Xz = _z(E); r2 = []
    for sd in seeds:
        cv = KFold(5, shuffle=True, random_state=sd)
        r2.append(cross_val_score(KNeighborsRegressor(15), Xz, t, cv=cv, scoring="r2").mean())
    return np.array(r2)


def main():
    emb = np.load(os.path.join(HERE, "results", "emb_cache", "geneformer_L11.npz"))
    ruler = np.load(os.path.join(HERE, "results", "geneformer_L11_ruler.npz"))
    s, g2m = ruler["s_score"].astype(float), ruler["g2m_score"].astype(float)
    print("reading per-cell total counts (2000 cells)...", flush=True)
    tc = total_counts_2000()
    log_tc = np.log1p(tc)
    raw_pc1 = _z(emb["linear_sae"]) @ np.linalg.svd(_z(emb["linear_sae"]), full_matrices=False)[2][0]  # abundance-ish axis
    covars = [log_tc, raw_pc1]
    print(f"corr(total_counts, S)={np.corrcoef(log_tc, s)[0,1]:+.3f}  "
          f"corr(total_counts, G2M)={np.corrcoef(log_tc, g2m)[0,1]:+.3f}\n", flush=True)

    out = {}
    for tname, t in [("s_score", s), ("g2m_score", g2m)]:
        row = {}
        for cond, cv in [("raw", None), ("resid_totalcount+pc1", covars)]:
            lin = emb["linear_sae"] if cv is None else residualize(emb["linear_sae"].astype(float), cv)
            bil = emb["atomic"] if cv is None else residualize(emb["atomic"].astype(float), cv)
            rl, rb = score(lin, t), score(bil, t)
            m = rb - rl
            row[cond] = dict(linear=float(rl.mean()), bilinear=float(rb.mean()),
                             margin=float(m.mean()), margin_std=float(m.std()))
            print(f"  {tname:10s} [{cond:22s}] linear {rl.mean():+.3f} | bilinear {rb.mean():+.3f} "
                  f"| Δ={m.mean():+.3f} ± {m.std():.3f}", flush=True)
        out[tname] = row
    json.dump(out, open(os.path.join(HERE, "results", "geneformer_abundance_control.json"), "w"), indent=1)
    print("\ndone.")


if __name__ == "__main__":
    main()
