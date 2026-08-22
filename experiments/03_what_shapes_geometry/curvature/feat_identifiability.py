"""Route Q #1 — are the NAMED products identified? (the make-or-break control)

The raw-residual gene lists were NOT identified: top-15 gene lists overlapped 1/15 across (r,lambda) fits
that predicted equally well. The whole premise of #1 is that fitting Q on SPARSE, NAMED atlas features
(instead of dense raw directions) makes the interaction identifiable. This tests that premise directly:

fit Q on h_bar at several (r, lambda) with statistically indistinguishable held-out Delta_Q, extract the
top-K interacting feature PAIRS (largest |Q_ij|) and the top eigen-direction feature SET from each, and
measure overlap (Jaccard) across configs + across random seeds. If the named products are stable, #1
delivered an identified, reportable answer; if not, the interaction is diffuse in feature space too and
no specific product is reportable (still a clean result: "redundant AND non-identified").

Out: results/feat_identifiability_scgpt_setty.json
"""
import os, json
import numpy as np, torch
import qfit, feat_naming as FN
from run_features import load_feats

torch.set_num_threads(6)
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
CFGS = [(8, 1e-2), (16, 1e-2), (32, 1e-2), (8, 1e-3), (16, 1e-3), (16, 1e-1)]
SEEDS = (0, 1)
TOPK = 30


def top_pairs(Q, alive_idx, k=TOPK):
    A = np.abs(Q).copy(); np.fill_diagonal(A, 0.0)
    n = Q.shape[0]; iu = np.triu_indices(n, k=1)
    order = np.argsort(-A[iu])[:k]
    return set(frozenset((int(alive_idx[iu[0][o]]), int(alive_idx[iu[1][o]]))) for o in order)


def top_eigfeats(Q, alive_idx, k=TOPK):
    w, V = np.linalg.eigh(Q)
    v = V[:, np.argmax(np.abs(w))]
    return set(int(alive_idx[o]) for o in np.argsort(-np.abs(v))[:k])


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def main(model="scgpt", dataset="setty"):
    H, alive_idx, y = load_feats(model, dataset)
    xh, c, _, _ = qfit.preprocess(H, H, True)
    lin = qfit.LinearPart(xh, c, y)

    pairs, eigs = {}, {}
    for r, lam in CFGS:
        # average the Q over seeds is not meaningful (sign), so take seed 0 for the ranking but
        # also measure seed stability separately below
        Q = qfit.fit_q(lin, xh, y, r, lam, seed=qfit.SEED).Q()
        pairs[(r, lam)] = top_pairs(Q, alive_idx)
        eigs[(r, lam)] = top_eigfeats(Q, alive_idx)

    # seed stability at the modal config
    seed_pairs = [top_pairs(qfit.fit_q(lin, xh, y, 16, 1e-2, seed=qfit.SEED + 11 * s).Q(), alive_idx)
                  for s in SEEDS]

    ref = (16, 1e-2)
    print(f"[{model}/{dataset}] identifiability of top-{TOPK} named products (feature space)")
    print(f"  reference config {ref}")
    print("  config          Jaccard(pairs vs ref)   Jaccard(eig-feats vs ref)")
    pair_j, eig_j = [], []
    for k in CFGS:
        pj, ej = jac(pairs[k], pairs[ref]), jac(eigs[k], eigs[ref])
        pair_j.append(pj); eig_j.append(ej)
        print(f"  {str(k):15s} {pj:19.2f} {ej:26.2f}")
    seed_j = jac(seed_pairs[0], seed_pairs[1])
    print(f"  seed-to-seed Jaccard(pairs) at {ref}: {seed_j:.2f}")

    # how many pairs are shared across ALL configs (the robust core)?
    core = set.intersection(*pairs.values())
    print(f"\n  pairs in the top-{TOPK} of ALL {len(CFGS)} configs (robust core): {len(core)}")
    for fp in list(core)[:10]:
        i, j = tuple(fp)
        print(f"    {FN.short(i)}  ×  {FN.short(j)}")

    out = dict(model=model, dataset=dataset, topk=TOPK, ref=list(ref),
               pair_jaccard_vs_ref={str(k): jac(pairs[k], pairs[ref]) for k in CFGS},
               eig_jaccard_vs_ref={str(k): jac(eigs[k], eigs[ref]) for k in CFGS},
               seed_jaccard=seed_j,
               mean_pair_jaccard=float(np.mean(pair_j)), mean_eig_jaccard=float(np.mean(eig_j)),
               robust_core_size=len(core),
               robust_core=[[tuple(fp)[0], tuple(fp)[1], FN.short(tuple(fp)[0]), FN.short(tuple(fp)[1])]
                            for fp in core])
    json.dump(out, open(os.path.join(RES, f"feat_identifiability_{model}_{dataset}.json"), "w"), indent=1)
    print(f"\n  -> results/feat_identifiability_{model}_{dataset}.json")


if __name__ == "__main__":
    main()
