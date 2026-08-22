"""H-B2 — does the manifold geometry encode antagonistic regulators as opposed co-movement?

scGPT's MVC head M is linear in the cell embedding (pred_expr = M @ cell_emb, all weights from pretraining), so
MR = M @ R maps a whitened-20PC direction to a gene-change program with zero fitting (R = linear part of
whitened-PC -> raw-emb). At a branch-point cell the local manifold tangent is V (top-m local PCs); gene g's
achievable movement along the manifold is the m-vector  g_tan = V @ MR[g]  (its loading across the m tangent
directions). Two genes' CO-MOVEMENT along the manifold is  cos(g_a_tan, g_b_tan):

  positive => moving a real cell along the manifold tends to raise A and B together   (cooperative)
  negative => the manifold only lets you raise one at the expense of the other        (antagonistic toggle)

PREDICTION: known antagonistic pairs (GATA1/SPI1, KLF1/FLI1, ...) have MORE NEGATIVE tangent co-movement than
cooperative same-lineage pairs. That is the classic toggle switch learned as pure geometry. Then screen all
pairs by tangent co-movement and check whether known antagonists are the most anti-correlated.

WHY NOT absolute "is co-activation reachable": a 10-20 dim reachable subspace is tiny vs ~13k genes, so ANY
sparse gene direction is ~fully normal (normal_frac ~ 0.999 for every pair) — that measure is saturated and
cannot discriminate. It is reported as `co_normal_frac` for transparency; the co-movement cosine is the metric.

Control: tangent co-movement vs FULL-head co-movement cos(MR[a], MR[b]) (cell-independent, no manifold). If
antagonism shows up in the local tangent but not the full head, the geometry — not just the head — encodes it.

scGPT only (the model with a pretrained cell-emb->expression head).

Run:  ../../.venv/bin/python hb2_normal_space.py
Out:  results/hb2_normal_space.json
"""
import os, sys, json
from itertools import combinations
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from geom_common import RS, SEED  # noqa: E402
sys.path.insert(0, RS)
from tangent_diagnostic import path as npz_path, DIM  # noqa: E402
from native_decode import build_native_head, build_spaces  # noqa: E402
from gene_decode import load_expression  # noqa: E402
from perturb_invert import auroc  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
M_TAN = 10
K_NB = 100
PROGENITORS = ["HSC_1", "HSC_2", "Precursors"]

ANTAGONIST = [("GATA1", "SPI1"), ("KLF1", "FLI1"), ("GATA1", "CEBPA"), ("TAL1", "SPI1"),
              ("KLF1", "SPI1"), ("GATA1", "IRF8"), ("GATA2", "CEBPA")]
COOPERATIVE = [("GATA1", "KLF1"), ("GATA1", "TAL1"), ("GATA1", "ZFPM1"), ("SPI1", "CEBPA"),
               ("SPI1", "IRF8"), ("KLF1", "TAL1"), ("CEBPA", "IRF8")]
TF_SET = sorted({g for p in ANTAGONIST + COOPERATIVE for g in p})


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run():
    model, sfx = "scgptbin", "setty"
    z = np.load(npz_path(model, sfx), allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int)
    clu = z["clusters"]
    ok = np.isfinite(y); emb, y, ci, clu = emb[ok], y[ok], ci[ok], clu[ok]
    E, genes, clusters = load_expression(ci)
    M, keep = build_native_head(genes)
    genes_k = genes[keep]; gpos = {g: i for i, g in enumerate(genes_k)}
    tf_ok = [t for t in TF_SET if t in gpos]
    print(f"native head: {len(keep)}/{len(genes)} genes decodable | TFs available: {tf_ok}")

    Xz, to_raw = build_spaces(emb, dim=DIM)
    raw0 = to_raw(np.zeros((1, Xz.shape[1])))[0]
    R = np.column_stack([to_raw(e[None])[0] - raw0 for e in np.eye(Xz.shape[1])])   # (512, 20)
    MR = M @ R                                                                      # (n_genes, 20) head o PCmap

    branch = np.where(np.isin(clu, PROGENITORS))[0]
    rng = np.random.default_rng(SEED)
    if len(branch) > 300:
        branch = branch[rng.choice(len(branch), 300, replace=False)]
    _, idx = NearestNeighbors(n_neighbors=K_NB).fit(Xz).kneighbors(Xz[branch])
    Vs = []
    for j in range(len(branch)):
        Xc = Xz[idx[j]] - Xz[idx[j]].mean(0)
        Vs.append(np.linalg.svd(Xc, full_matrices=False)[2][:M_TAN])                # (m, 20) local tangent
    print(f"branch-point cells: {len(branch)} progenitors; tangent m={M_TAN} of {DIM} PCs")

    def metric(a, b):
        ia, ib = gpos[a], gpos[b]
        comov = np.mean([cos(V @ MR[ia], V @ MR[ib]) for V in Vs])                  # tangent co-movement
        full = cos(MR[ia], MR[ib])                                                  # full-head co-movement
        # saturated absolute-reachability control (documented, not the metric)
        ea = np.zeros(len(genes_k)); eb = np.zeros(len(genes_k)); ea[ia] = 1; eb[ib] = 1
        Q, _ = np.linalg.qr((MR @ Vs[0].T))
        nf = float(np.sqrt(max(0.0, 1 - np.linalg.norm(Q.T @ ((ea + eb) / np.sqrt(2))) ** 2)))
        return dict(comov_tan=float(comov), comov_full=float(full), co_normal_frac=nf)

    out = dict(model=model, n_branch=int(len(branch)), n_genes_decodable=int(len(keep)),
               tfs=tf_ok, antagonist={}, cooperative={})
    print(f"\n  {'pair':<16} {'kind':<5} | {'comov_tan':>10} {'comov_full':>11} {'co_normalfrac':>14}")
    for kind, pairs, store in (("antag", ANTAGONIST, out["antagonist"]), ("coop", COOPERATIVE, out["cooperative"])):
        for a, b in pairs:
            if a not in gpos or b not in gpos:
                continue
            r = metric(a, b); store[f"{a}/{b}"] = r
            print(f"  {a+'/'+b:<16} {kind:<5} | {r['comov_tan']:>+10.3f} {r['comov_full']:>+11.3f} "
                  f"{r['co_normal_frac']:>14.3f}")

    at = [v["comov_tan"] for v in out["antagonist"].values()]
    ct = [v["comov_tan"] for v in out["cooperative"].values()]
    af = [v["comov_full"] for v in out["antagonist"].values()]
    cf = [v["comov_full"] for v in out["cooperative"].values()]
    # AUROC: does low (negative) tangent co-movement flag antagonist pairs among the labelled set?
    labels = np.array([1] * len(at) + [0] * len(ct))
    scores_tan = -np.array(at + ct)                     # more negative comov -> higher antagonist score
    scores_full = -np.array(af + cf)
    out["summary"] = dict(antag_comov_tan_mean=float(np.mean(at)), coop_comov_tan_mean=float(np.mean(ct)),
                          antag_comov_full_mean=float(np.mean(af)), coop_comov_full_mean=float(np.mean(cf)),
                          auroc_labeled_tan=float(auroc(scores_tan, labels.astype(bool))),
                          auroc_labeled_full=float(auroc(scores_full, labels.astype(bool))))

    # SCREEN over all TF pairs
    known_ant = {frozenset(p) for p in ANTAGONIST}
    screen = [(f"{a}/{b}", metric(a, b)["comov_tan"], frozenset((a, b)) in known_ant)
              for a, b in combinations(tf_ok, 2)]
    screen.sort(key=lambda t: t[1])                    # most anti-correlated first
    scr = -np.array([s[1] for s in screen]); pos = np.array([s[2] for s in screen])
    out["screen"] = dict(n_pairs=len(screen), n_antagonist=int(pos.sum()),
                         auroc_antagonist=float(auroc(scr, pos.astype(bool))),
                         most_antagonistic_top8=[(s[0], round(s[1], 3)) for s in screen[:8]],
                         antagonist_ranks=[i + 1 for i, s in enumerate(screen) if s[2]])

    s = out["summary"]
    print(f"\n  tangent co-movement: antagonist mean {np.mean(at):+.3f} vs cooperative {np.mean(ct):+.3f}"
          f"  ({'ANTAG MORE OPPOSED' if np.mean(at) < np.mean(ct) else 'no separation'})")
    print(f"  full-head co-movement (control): antagonist {np.mean(af):+.3f} vs cooperative {np.mean(cf):+.3f}")
    print(f"  labelled-set AUROC (antagonist by -comov): tangent {s['auroc_labeled_tan']:.3f} | "
          f"full-head {s['auroc_labeled_full']:.3f}")
    print(f"  screen {out['screen']['n_pairs']} pairs: AUROC={out['screen']['auroc_antagonist']:.3f}"
          f" | antagonist ranks {out['screen']['antagonist_ranks']}")
    print(f"  most-opposed pairs: {out['screen']['most_antagonistic_top8']}")
    json.dump(out, open(os.path.join(RESULTS, "hb2_normal_space.json"), "w"), indent=1)
    print("[done] -> results/hb2_normal_space.json")
    return out


if __name__ == "__main__":
    run()

