"""CROSS-MODEL / SCALING COMPARISON of the contextualisation phenomenon (a model-fact, ceiling-immune).

Level 1 established (on MaxToki-217M): a gene's representation undergoes gene-specific, reproducible,
abundance-independent context modulation, organised along functional axes -- though that organisation IS
co-expression (fails the co-expression null). Level 2 (novel biology) is null three ways. So the remaining
useful, co-expression-ceiling-IMMUNE direction is to turn the phenomenon into a MODEL METRIC and compare
architectures and scale.

Two metrics, both model-facts, computed identically on every model's own gene vocabulary:

  EXCESS  (contextualisation strength) -- how gene-SPECIFIC is the context response, beyond the averaging null.
     For each context PAIR, delta_p(g) = [v_p(g,c2)-v_p(g,c1)] - crowd_mean, in cell partition p.
     same = mean_g cos(delta_0(g), delta_1(g));  diff = mean_g cos(delta_0(g), delta_1(perm g));  EXCESS=same-diff.
     0 = attention only averages (no gene-specific context response); higher = more contextualisation.
     Reported with the CONTEXT-MAIN-EFFECT replication as a positive control (does the crowd shift itself
     reproduce -- i.e. is the measurement working at all).

  FUNC-Z  (functional organisation) -- is the context modulation aligned with a functional axis (nuclear vs
     surface, built from GO) more than matched RANDOM-partition axes. z over the random-axis null.

FAIRNESS. Caps differ (MaxToki cap 50; scGPT/STATE cap 20) and vocabularies differ, so cross-ARCHITECTURE
numbers are compared with that caveat. The clean SCALING comparison is ctx217m600 vs ctx1b: same cells, same
600-cell/cap-50 settings, differing only in model size. Per-dimension z-scoring (Timkey) before any projection.

Reads results/<prefix>_L{tap}.npz. Out: results/ctx_cross_model.json
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
SEED, N_RANDOM, MIN_GENES = 0, 200, 150

MODELS = [   # (label, prefix, tap, note)
    ("scGPT",            "ctx_scgpt",   4, "value-binned, 512-d"),
    ("STATE-SE",         "ctx_state",   4, "ESM2-init gene tokens, 2048-d"),
    ("MaxToki-217M",     "ctx217m600",  4, "600-cell matched"),
    ("MaxToki-1B",       "ctx1b",       4, "600-cell matched"),
    ("MaxToki-217M-1k",  "ctx_maxtoki", 4, "1000-cell headline"),
    ("MaxToki-217M-random", "ctxrand",  4, "RANDOM-INIT control (untrained, same arch)"),
    ("MaxToki-217M-cap20", "ctx217m_cap20", 4, "cap-matched to scGPT/STATE (cap 20)"),
]
AXES = {"nuclear_vs_surface": (["GO:0005634", "GO:0000785", "GO:0003677"],
                               ["GO:0005886", "GO:0005576", "GO:0005615"])}


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def analyse(prefix, tap, ens2sym, g2g, rng):
    z = np.load(os.path.join(RES, f"{prefix}_L{tap:02d}.npz"), allow_pickle=True)
    M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
    genes = z["genes"].astype(str); ctxs = z["contexts"].astype(str)
    nP, nC, nG, d = M.shape
    full = (counts == cap).all(0)
    flat = M[:, full]
    mu = flat.reshape(-1, d).mean(0); sd = flat.reshape(-1, d).std(0) + 1e-6
    Mz = (M - mu) / sd

    # ---- EXCESS (contextualisation) over context pairs ----
    same_all, diff_all, main_rep = [], [], []
    for c1, c2 in itertools.combinations(range(nC), 2):
        keep = full[c1] & full[c2]
        if keep.sum() < MIN_GENES:
            continue
        D0 = Mz[0, c2, keep] - Mz[0, c1, keep]; D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
        b0, b1 = D0.mean(0), D1.mean(0)
        main_rep.append(float(np.dot(b0, b1) / (np.linalg.norm(b0) * np.linalg.norm(b1) + 1e-9)))
        d0, d1 = D0 - b0, D1 - b1
        same_all.append(cos_rows(d0, d1))
        diff_all.append(cos_rows(d0, d1[rng.permutation(len(d1))]))
    S, Dg = np.concatenate(same_all), np.concatenate(diff_all)
    excess = float(S.mean() - Dg.mean())
    bs = [float(S[rng.integers(0, len(S), len(S))].mean() - Dg[rng.integers(0, len(Dg), len(Dg))].mean())
          for _ in range(1000)]
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    # ---- FUNC-Z (functional organisation vs random-axis null) ----
    a_space = np.full((nG, d), np.nan, np.float32)
    for gi in range(nG):
        cs = np.where(full[:, gi])[0]
        if len(cs):
            a_space[gi] = Mz[:, cs, gi].mean((0, 1))
    ok = np.isfinite(a_space[:, 0])
    use = np.where(full.sum(0) >= max(2, nC // 2))[0]
    syms = [ens2sym.get(g) for g in genes]

    def power(vec):
        q = np.tensordot(Mz[:, :, use], vec, axes=([3], [0]))
        qm = np.where(full[:, use][None], q, np.nan)
        I = qm - np.nanmean(qm, 1, keepdims=True) - np.nanmean(qm, 2, keepdims=True) + np.nanmean(qm, (1, 2), keepdims=True)
        sel = np.isfinite(I[0]) & np.isfinite(I[1])
        return float(np.nanmean(I[0][sel] * I[1][sel]))

    def axis(ia, ib):
        u = a_space[ia].mean(0) - a_space[ib].mean(0); return u / (np.linalg.norm(u) + 1e-9)

    fz = {}
    pool = np.where(ok)[0]
    for name, (A, B) in AXES.items():
        ia = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(A)]
        ib = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(B)]
        both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
        if len(ia) < 15 or len(ib) < 15:
            fz[name] = None; continue
        pf = power(axis(ia, ib))
        null = np.array([power(axis(rng.choice(pool, len(ia), replace=False),
                                    rng.choice(pool, len(ib), replace=False))) for _ in range(N_RANDOM)])
        fz[name] = dict(z=float((pf - null.mean()) / (null.std() + 1e-12)), poleA=len(ia), poleB=len(ib))

    return dict(n_ctx=int(nC), n_genes=int(nG), dim=int(d), cap=cap,
                pairs_scored=len(same_all), excess=excess, excess_ci=ci,
                main_effect_replication=float(np.mean(main_rep)),
                anisotropy=float(np.abs(np.corrcoef(Mz[:, full].reshape(-1, d)[rng.integers(0, full.sum() * 2, 3000)].T)).mean()) if False else None,
                func_z=fz)


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
    rng = np.random.default_rng(SEED)
    out = {}
    print(f"{'model':<18} {'ctx':>4} {'genes':>6} {'dim':>5} {'cap':>4} {'EXCESS':>9} {'95% CI':>18} "
          f"{'main-rep':>9} {'FUNC-z':>8}")
    print("-" * 92)
    for label, prefix, tap, note in MODELS:
        try:
            r = analyse(prefix, tap, ens2sym, g2g, rng); r["note"] = note
        except Exception as e:
            print(f"{label:<18} ERR {repr(e)[:60]}"); out[label] = {"error": repr(e)[:150]}; continue
        out[label] = r
        fz = r["func_z"].get("nuclear_vs_surface")
        fzs = f"{fz['z']:+.1f}" if fz else "n/a"
        print(f"{label:<18} {r['n_ctx']:>4} {r['n_genes']:>6} {r['dim']:>5} {r['cap']:>4} "
              f"{r['excess']:>+9.4f} [{r['excess_ci'][0]:+.3f},{r['excess_ci'][1]:+.3f}] "
              f"{r['main_effect_replication']:>+9.3f} {fzs:>8}")

    # scaling verdict from the matched pair
    a, b = out.get("MaxToki-217M"), out.get("MaxToki-1B")
    if a and b and "error" not in a and "error" not in b:
        out["scaling"] = dict(excess_217m=a["excess"], excess_1b=b["excess"],
                              delta=b["excess"] - a["excess"])
        print(f"\nSCALING (matched 600-cell): EXCESS 217M {a['excess']:+.4f} -> 1B {b['excess']:+.4f} "
              f"(Δ {b['excess']-a['excess']:+.4f})")
    json.dump(out, open(os.path.join(RES, "ctx_cross_model.json"), "w"), indent=1)
    print("\n[done] -> results/ctx_cross_model.json")


if __name__ == "__main__":
    main()
