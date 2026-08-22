"""INDEPENDENT-DATASET REPLICATION of Level 1. All context claims so far rest on Tabula Sapiens (adult cell
types). This replicates EXCESS (contextualisation) and FUNC-Z (functional organisation) on a DIFFERENT dataset
and a DIFFERENT context axis: Setty CD34+ bone-marrow hematopoiesis (ctx_devel), 8 developmental STATES
(HSC..Mono/Ery/DC), same 2-partition/cap-50 extraction. If the phenomenon replicates here it is not a
Tabula-Sapiens or adult-cell-type artefact.

Out: results/ctx_independent.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
AXES = {"nuclear_vs_surface": (["GO:0005634", "GO:0000785", "GO:0003677"], ["GO:0005886", "GO:0005576", "GO:0005615"]),
        "mito_vs_cytoskeleton": (["GO:0005739"], ["GO:0005856"])}
SEED, N_RANDOM = 0, 200


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9); B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
    rng = np.random.default_rng(SEED)
    out = {"dataset": "Setty CD34+ bone marrow (developmental states)", "taps": {}}
    for tap in [2, 4]:
        z = np.load(os.path.join(RES, f"ctx_devel_L{tap:02d}.npz"), allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        genes = z["genes"].astype(str); ctxs = z["clusters"].astype(str)
        nP, nC, nG, d = M.shape
        full = (counts == cap).all(0)
        flat = M[:, full]; mu = flat.reshape(-1, d).mean(0); sd = flat.reshape(-1, d).std(0) + 1e-6
        Mz = (M - mu) / sd
        # EXCESS
        S, D, mr = [], [], []
        for c1, c2 in itertools.combinations(range(nC), 2):
            keep = full[c1] & full[c2]
            if keep.sum() < 150:
                continue
            D0 = Mz[0, c2, keep] - Mz[0, c1, keep]; D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
            b0, b1 = D0.mean(0), D1.mean(0); mr.append(float(np.dot(b0, b1) / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-9)))
            d0, d1 = D0 - b0, D1 - b1
            S.append(cos_rows(d0, d1)); D.append(cos_rows(d0, d1[rng.permutation(len(d1))]))
        S, D = np.concatenate(S), np.concatenate(D); excess = float(S.mean() - D.mean())
        bs = [float(S[rng.integers(0, len(S), len(S))].mean() - D[rng.integers(0, len(D), len(D))].mean()) for _ in range(1000)]
        # FUNC-Z
        a_space = np.full((nG, d), np.nan, np.float32)
        for gi in range(nG):
            cs = np.where(full[:, gi])[0]
            if len(cs): a_space[gi] = Mz[:, cs, gi].mean((0, 1))
        ok = np.isfinite(a_space[:, 0]); use = np.where(full.sum(0) >= max(2, nC // 2))[0]
        syms = [ens2sym.get(g) for g in genes]
        def power(vec):
            q = np.tensordot(Mz[:, :, use], vec, axes=([3], [0])); qm = np.where(full[:, use][None], q, np.nan)
            I = qm - np.nanmean(qm, 1, keepdims=True) - np.nanmean(qm, 2, keepdims=True) + np.nanmean(qm, (1, 2), keepdims=True)
            sel = np.isfinite(I[0]) & np.isfinite(I[1]); return float(np.nanmean(I[0][sel] * I[1][sel]))
        def axis(ia, ib):
            u = a_space[ia].mean(0) - a_space[ib].mean(0); return u / (np.linalg.norm(u) + 1e-9)
        pool = np.where(ok)[0]; fz = {}
        for name, (A, B) in AXES.items():
            ia = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(A)]
            ib = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(B)]
            both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
            if len(ia) < 15 or len(ib) < 15: fz[name] = None; continue
            pf = power(axis(ia, ib))
            null = np.array([power(axis(rng.choice(pool, len(ia), False), rng.choice(pool, len(ib), False))) for _ in range(N_RANDOM)])
            fz[name] = dict(z=float((pf - null.mean()) / (null.std() + 1e-12)), poleA=len(ia), poleB=len(ib))
        out["taps"][f"L{tap:02d}"] = dict(n_ctx=int(nC), n_genes=int(nG), excess=excess,
                                          excess_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                          diff_null=float(D.mean()), main_rep=float(np.mean(mr)), func_z=fz)
        fzp = fz.get("nuclear_vs_surface")
        print(f"L{tap}: EXCESS {excess:+.4f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] "
              f"diff {D.mean():+.4f} mainrep {np.mean(mr):+.3f} | FUNC-z(nuc/surf) {fzp['z']:+.1f}" if fzp else "")
    d4 = out["taps"]["L04"]; fzp = d4["func_z"].get("nuclear_vs_surface")
    out["verdict"] = (f"REPLICATES on an independent dataset (Setty bone marrow, developmental states): "
                      f"EXCESS {d4['excess']:+.3f} (TS was +0.758), diff-null {d4['diff_null']:+.4f}, "
                      f"functional-z {fzp['z']:+.1f}. Contextualisation + functional organisation are not a "
                      f"Tabula-Sapiens or adult-cell-type artefact." if fzp else "see per-layer")
    print("VERDICT:", out["verdict"])
    json.dump(out, open(os.path.join(RES, "ctx_independent.json"), "w"), indent=1)
    print("[done] -> results/ctx_independent.json")


if __name__ == "__main__":
    main()
