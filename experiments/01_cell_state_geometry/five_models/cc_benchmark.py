"""Paper-style benchmark: do the cell-cycle MODEL manifolds beat an EXPRESSION baseline on downstream tasks?

Mirrors route_l2h5 / arxiv 2603.10261: reduce each representation to matched dimensionality, fit a small probe,
evaluate held-out. Three tasks (the cell-cycle analogues of the paper's pseudotime-ordering / classification /
subtype-AUROC):
  (A) PHASE ORDERING     circular R^2 predicting phi          (~ pseudotime-depth ordering)
  (B) PHASE CLASSIFY     balanced accuracy on G1/S/G2M        (~ branch/stage classification)
  (C) S-vs-G2M AUROC     the two cycling phases, harder       (~ subtype discrimination)

Each MODEL is compared to the expression baseline ON THE SAME CELLS (expression pulled from replogle at that
model's own cell_idx). Dimensionality swept k in {10,20,50} because the paper's model advantage appeared only
against a WEAK (low-dim) expression baseline.

CIRCULARITY, stated up front (as the paper did): phi and the G1/S/G2M call are DERIVED from marker expression,
so the expression baseline is circularly ADVANTAGED. A model that beats it anyway is meaningfully better; an
expression win is partly definitional. Read the table with that asymmetry in mind.

Run: ../../.venv/bin/python cc_benchmark.py
Out: results/cc_benchmark.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, balanced_accuracy_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import REPLOGLE, DATA, _load_categorical, to_lognorm, circ_mean, emb_path

MODELS = ["scgptcc", "geneformer", "maxtoki", "statecc", "ucecc"]
# Scored and printed, but held OUT of the headline win-count so UCE is not counted twice. UCE's is the only
# panel that is not controls-only (1000 non-targeting + 2000 CRISPRi perturbations); this row re-runs it on
# its 1000 controls for parity with the other four. See adapt_cached.py CACHES["ucecc_ctrl"].
SENSITIVITY = ["ucecc_ctrl"]
LABEL = {"scgptcc": "scGPT", "geneformer": "Geneformer", "maxtoki": "MaxToki", "statecc": "STATE-SE",
         "ucecc": "UCE", "ucecc_ctrl": "UCE ctrl-only"}
KS = [10, 20, 50]
SEED = 0


def pca_k(X, k):
    return StandardScaler().fit_transform(PCA(min(k, X.shape[1], X.shape[0] - 1),
                                              random_state=SEED).fit_transform(X - X.mean(0)))


def order_circ_r2(Xk, phi):
    Y = np.column_stack([np.cos(phi), np.sin(phi)]); P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xk):
        P[te] = Ridge(alpha=10.0).fit(Xk[tr], Y[tr]).predict(Xk[te])
    err = 1 - np.cos(np.arctan2(P[:, 1], P[:, 0]) - phi)
    base = np.mean(1 - np.cos(phi - circ_mean(phi)))
    return float(1 - err.mean() / (base + 1e-12))


def classify_bal(Xk, lab):
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(lab)); tr, te = perm[:len(lab)//2], perm[len(lab)//2:]
    m = LogisticRegression(max_iter=2000).fit(Xk[tr], lab[tr])
    return float(balanced_accuracy_score(lab[te], m.predict(Xk[te])))


def svg2m_auroc(Xk, lab):
    sel = np.isin(lab, ["S", "G2M"]); Xs, ys = Xk[sel], (lab[sel] == "G2M").astype(int)
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(ys)); tr, te = perm[:len(ys)//2], perm[len(ys)//2:]
    m = LogisticRegression(max_iter=2000).fit(Xs[tr], ys[tr])
    return float(roc_auc_score(ys[te], m.predict_proba(Xs[te])[:, 1]))


def run():
    out = {}
    for model in MODELS + SENSITIVITY:
        z = np.load(emb_path(model), allow_pickle=True)
        emb, phi, ci, cc = (z["emb"].astype(np.float64), z["phi"].astype(np.float64),
                            z["cell_idx"].astype(int), z["cc_phase"].astype(str))
        with h5py.File(REPLOGLE, "r") as f:
            X = np.stack([np.asarray(f["X"][int(i), :], dtype=np.float32) for i in ci])
        expr = to_lognorm(X)
        rec = {}
        for k in KS:
            Ee, Ex = pca_k(emb, k), pca_k(expr, k)
            rec[k] = dict(
                order_model=order_circ_r2(Ee, phi), order_expr=order_circ_r2(Ex, phi),
                classify_model=classify_bal(Ee, cc), classify_expr=classify_bal(Ex, cc),
                auroc_model=svg2m_auroc(Ee, cc), auroc_expr=svg2m_auroc(Ex, cc))
        out[model] = dict(n=int(len(phi)), d=int(emb.shape[1]), by_k=rec)

    # ---- tables ----
    for task, mk, xk, fmt in [("(A) PHASE ORDERING  circ-R^2", "order_model", "order_expr", "{:+.3f}"),
                              ("(B) PHASE CLASSIFY  bal-acc", "classify_model", "classify_expr", "{:.3f}"),
                              ("(C) S-vs-G2M        AUROC", "auroc_model", "auroc_expr", "{:.3f}")]:
        print(f"\n{task}   (model  vs  expression-baseline, same cells; expr is circularly ADVANTAGED)")
        print(f"  {'representation':<14}" + "".join(f"{'k='+str(k):>22}" for k in KS))
        # expression baseline row (per model's own cells -- shown as the value on scGPT's cells for reference,
        # but each model compares to expr on ITS cells; print delta per cell below)
        for model in MODELS + SENSITIVITY:
            deltas = []
            for k in KS:
                mv, xv = out[model]["by_k"][k][mk], out[model]["by_k"][k][xk]
                win = "model" if mv > xv else "EXPR"
                deltas.append(f"{fmt.format(mv)}/{fmt.format(xv)} {'+' if mv>xv else '-'}")
            tag = "  (sens.)" if model in SENSITIVITY else ""
            print(f"  {LABEL[model]:<14}" + "".join(f"{d:>22}" for d in deltas) + tag)
        print("   (each cell: model / expression ; +=model wins, -=expression wins)")

    # summary: how often does ANY model beat expression?
    wins = tot = 0
    for model in MODELS:
        for k in KS:
            for mk, xk in [("order_model", "order_expr"), ("classify_model", "classify_expr"),
                           ("auroc_model", "auroc_expr")]:
                tot += 1; wins += int(out[model]["by_k"][k][mk] > out[model]["by_k"][k][xk])
    print(f"\n  MODEL beats (circularly-advantaged) EXPRESSION in {wins}/{tot} model x k x task cells.")
    out["model_wins_over_expr"] = f"{wins}/{tot}"
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "cc_benchmark.json"), "w"), indent=1)
    print("[done] -> results/cc_benchmark.json")


if __name__ == "__main__":
    run()
