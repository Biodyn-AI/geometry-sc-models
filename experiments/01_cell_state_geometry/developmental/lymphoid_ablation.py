"""Was "lymphoid is the weak branch" a MODEL failure or a thin MARKER PANEL?

NATIVE_DECODE_RESULTS.md states: "The lymphoid branch is genuinely the weak one. It is 4/4 only AFTER
column-centering; raw it is 3/4, and the Ridge decoder also misses lymphoid on scgptbin. CLP is a small, rare
compartment (4 markers). Do not present lymphoid as a clean hit."

Note what it says in its own parenthesis: **4 markers**. Every other program in that panel has 8-9. A program
score is a mean of z-scored marker genes, so a 4-gene program is roughly twice as noisy as an 8-gene one — and
lymphoid is the one that fails. That is a measurement-precision explanation, not a model-failure explanation,
and it was never tested.

This isolates it: hold the model, the geometry, the steer, the fates and the readout EXACTLY fixed, and change
ONE thing — the lymphoid gene list.

  OLD (route_steering, 4 genes) : IL7R, JCHAIN, EBF1, VPREB1
  NEW (canonical B/T-lymphoid)  : + DNTT, CD79A, CD79B, RAG1, RAG2, LTB

The added genes are textbook lymphoid-progenitor markers (DNTT/TdT and RAG1/RAG2 are the definitive V(D)J
machinery of committed lymphoid progenitors; CD79A/CD79B the B-cell receptor complex). They are canonical and
external to this dataset — not genes chosen because they helped.

If lymphoid flips from miss to hit on the panel change alone, the caveat is retired: the branch was never weak,
the instrument reading it was.

Run:  ../../.venv/bin/python lymphoid_ablation.py
Out:  results/lymphoid_ablation.json
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
from itertools import permutations
import numpy as np
from sklearn.neighbors import NearestNeighbors

import common                                                                                # noqa: E402
from common import RESULTS, FATE_CLUSTERS, ROOT_CLUSTERS, load_expression, panel_index       # noqa: E402
from tangent_diagnostic import path as npz_path, zscore, unit, steer, SEED, BUDGETS, T_STAR  # noqa: E402
from local_steering import project_constant                                                  # noqa: E402
from native_decode import build_native_head, build_spaces                                    # noqa: E402
from fate_matrix import dominance                                                            # noqa: E402

OLD_LYMPHOID = ["IL7R", "JCHAIN", "EBF1", "VPREB1"]                                   # route_steering's panel
NEW_LYMPHOID = ["IL7R", "JCHAIN", "EBF1", "VPREB1", "DNTT", "CD79A", "CD79B", "RAG1", "RAG2", "LTB"]

# hold the fate set at route_steering's ORIGINAL four, so the ONLY thing that changes is the lymphoid gene list
FOUR = ["ery", "myeloid", "lymphoid", "mega"]


def matrix(lymph_panel, fates):
    common.PANEL["setty"]["lymphoid"] = list(lymph_panel)          # the one thing that varies
    z = np.load(npz_path("scgptbin", "setty"), allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int); clu = z["clusters"]
    ok = np.isfinite(y); emb, y, ci, clu = emb[ok], y[ok], ci[ok], clu[ok]

    E_full, genes, _ = load_expression("setty", ci)
    M, keep = build_native_head(genes)
    pi = panel_index(genes[keep], "setty")

    Xz, to_raw = build_spaces(emb)
    yz = zscore(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    clu_tr = clu[tr]
    native = lambda ce: ce @ M.T                       # noqa: E731
    P_real = native(emb)
    mu, sd = P_real[tr].mean(0), P_real[tr].std(0) + 1e-9
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))

    src = Xz[te[yz[te] <= np.quantile(yz[te], 1 / 3)]]
    if len(src) > 200:
        src = src[rng.choice(len(src), 200, replace=False)]
    roots = np.isin(clu_tr, ROOT_CLUSTERS["setty"])
    src_c = Xz[tr][roots].mean(0)

    rows = []
    for f in fates:
        w_f = unit(Xz[tr][np.isin(clu_tr, FATE_CLUSTERS["setty"][f])].mean(0) - src_c)
        pth = steer(src, project_constant(Xz[tr], w_f), 0.25 * d0, BUDGETS)
        p0, p8 = native(to_raw(pth[0])), native(to_raw(pth[T_STAR]))
        rows.append([float((((p8[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)
                            - ((p0[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)).mean()) for k in fates])
    A = np.array(rows)
    Ac = A - A.mean(0, keepdims=True)
    D = dominance(Ac)
    perms = [dominance(Ac[list(p)]) for p in permutations(range(len(fates)))]
    return dict(A=A, Ac=Ac, D=D, p=float(np.mean([d >= D - 1e-12 for d in perms])),
                n_perms=len(perms), n_markers=len(pi["lymphoid"]),
                hits_raw=sum(int(np.argmax(A[i]) == i) for i in range(len(fates))),
                hits_ctr=sum(int(np.argmax(Ac[i]) == i) for i in range(len(fates))),
                lymph_raw_ok=bool(np.argmax(A[fates.index("lymphoid")]) == fates.index("lymphoid")),
                lymph_ctr_ok=bool(np.argmax(Ac[fates.index("lymphoid")]) == fates.index("lymphoid")))


def main():
    out = {}
    print(f"\n{'='*100}\n=== Is 'lymphoid is the weak branch' a model failure, or a 4-gene panel?")
    print("=== Everything held fixed (scGPT's own MVC head, same geometry, same steer, same 4 fates).")
    print("=== The ONLY variable: the lymphoid marker list.\n")
    for tag, panel in [("OLD (4 genes — route_steering)", OLD_LYMPHOID),
                       ("NEW (10 genes — canonical)", NEW_LYMPHOID)]:
        r = matrix(panel, FOUR)
        out[tag] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        print(f"  {tag}")
        print(f"    lymphoid markers in scGPT vocab : {r['n_markers']}")
        print(f"    lymphoid is its own top riser   : raw {'YES' if r['lymph_raw_ok'] else 'NO '}"
              f"   centered {'YES' if r['lymph_ctr_ok'] else 'NO '}")
        print(f"    whole diagonal                  : raw {r['hits_raw']}/4, centered {r['hits_ctr']}/4")
        print(f"    diagonal dominance D            : {r['D']:+.3f}   exact p = {r['p']:.4f}\n")

    a, b = out["OLD (4 genes — route_steering)"], out["NEW (10 genes — canonical)"]
    flipped = (not a["lymph_raw_ok"]) and b["lymph_raw_ok"]
    print(f"  ---- verdict ----")
    if flipped:
        print("  Lymphoid flips MISS -> HIT on the RAW matrix from the panel change alone. The branch was never")
        print("  weak; the 4-gene panel reading it was. route_steering's caveat can be retired.")
    else:
        print(f"  Lymphoid raw: {a['lymph_raw_ok']} -> {b['lymph_raw_ok']}. The panel is not the whole story;")
        print("  report the caveat as it stands, with the D/p change as the honest effect of the wider panel.")
    print(f"  D: {a['D']:+.3f} -> {b['D']:+.3f}   |   raw hits: {a['hits_raw']}/4 -> {b['hits_raw']}/4"
          f"   |   p: {a['p']:.4f} -> {b['p']:.4f}")
    json.dump(out, open(os.path.join(RESULTS, "lymphoid_ablation.json"), "w"), indent=1)
    print(f"\n[done] -> results/lymphoid_ablation.json")


if __name__ == "__main__":
    main()
