"""IS THE FUNCTIONAL-AXIS MODULATION JUST REPRESENTATION-SPACE COMPACTNESS? (the reviewer's other null)

`ctx_coexpr_null.py` showed the functional-axis context modulation exceeds a CO-EXPRESSION-coherence-matched
null. A reviewer will ask for the other obvious match variable: maybe functional gene sets simply CLUSTER TIGHTLY
in the model's representation space, and ANY tightly-clustered set gives a strong axis regardless of function.

Same dose-response design as the co-expression control, but the match variable is REPRESENTATION-SPACE
TIGHTNESS: mean pairwise cosine of the axis-defining genes in the gene-main-effect space a(g). We generate null
axes spanning the tightness range (loose random sets -> tight a(g)-neighbour modules), fit power~tightness, and
place each functional axis on that curve. If functional axes sit ABOVE the tightness curve, their modulation is
not merely a consequence of being a compact cluster in representation space.

This is the strict companion to the co-expression control: co-expression matches the DATA structure of the gene
set, tightness matches its REPRESENTATION structure. Surviving both is the strong claim.

Power is the same rank-controlled reproducible interaction power used throughout.
Out: results/ctx_tightness_null.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ctx_position_confound as CP
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
TAPS = [4, 8]
MIN_CTX = 9
N_NULL = 400
SEED = 0
from scipy.stats import spearmanr

AXES = {
    "nuclear_vs_surface":   (["GO:0005634", "GO:0000785", "GO:0003677"], ["GO:0005886", "GO:0005576", "GO:0005615"]),
    "mito_vs_cytoskeleton": (["GO:0005739"], ["GO:0005856"]),
    "transcription_vs_transport": (["GO:0006355", "GO:0003700"], ["GO:0006811", "GO:0038023"]),
}


def tightness(S, idx):
    if len(idx) < 2:
        return 0.0
    sub = S[np.ix_(idx, idx)]; iu = np.triu_indices(len(idx), 1)
    return float(sub[iu].mean())


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
    rng = np.random.default_rng(SEED)

    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    ctxs = z0["contexts"].astype(str); genes = z0["genes"].astype(str)
    syms = [ens2sym.get(g) for g in genes]
    tokmap = json.load(open(f"{CP.MSETUP}/token_dictionary.json")); ens2tid = {k: int(v) for k, v in tokmap.items()}
    tids = np.array([ens2tid.get(g, -1) for g in genes])

    print("[1/2] per-context mean rank (abundance control)", flush=True)
    MR = CP.mean_ranks(set(ctxs))
    rank = np.full((len(ctxs), len(genes)), np.nan)
    for ci, c in enumerate(ctxs):
        d = MR.get(c, {})
        for gi, t in enumerate(tids):
            if t in d:
                rank[ci, gi] = d[t]

    out = {"taps": {}}
    for tap in TAPS:
        z = np.load(os.path.join(RES, f"ctx_maxtoki_L{tap:02d}.npz"), allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        full = (counts == cap).all(0)
        flat = M[:, full]; mu = flat.reshape(-1, M.shape[-1]).mean(0); sd = flat.reshape(-1, M.shape[-1]).std(0) + 1e-6
        Mz = (M - mu) / sd
        a_space = np.full((len(genes), M.shape[-1]), np.nan, np.float32)
        for gi in range(len(genes)):
            cs = np.where(full[:, gi])[0]
            if len(cs):
                a_space[gi] = Mz[:, cs, gi].mean((0, 1))
        gene_ok = np.isfinite(a_space[:, 0])
        use = np.where(full.sum(0) >= MIN_CTX)[0]
        Ruse = rank[:, use]; muse = full[:, use]; Msub = Mz[:, :, use]

        # tightness similarity matrix S = cosine in a(g) space (representation compactness)
        A = np.nan_to_num(a_space)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        pool = np.where(gene_ok)[0]

        def power(vec):
            q = np.tensordot(Msub, vec, axes=([3], [0]))
            qm = np.where(muse[None], q, np.nan)
            fin = np.isfinite(qm[0]) & np.isfinite(Ruse)
            r = Ruse[fin]; B = np.column_stack([np.ones_like(r), r, r ** 2])
            for p in range(2):
                yv = qm[p][fin]; qm[p][fin] = yv - B @ np.linalg.lstsq(B, yv, rcond=None)[0]
            I = qm - np.nanmean(qm, 1, keepdims=True) - np.nanmean(qm, 2, keepdims=True) + np.nanmean(qm, (1, 2), keepdims=True)
            sel = np.isfinite(I[0]) & np.isfinite(I[1])
            return float(np.nanmean(I[0][sel] * I[1][sel]))

        def axis(ia, ib):
            u = a_space[ia].mean(0) - a_space[ib].mean(0); return u / (np.linalg.norm(u) + 1e-9)

        def tset_tightness(idx):
            if len(idx) < 2:
                return 0.0
            sub = An[idx]; g = (sub @ sub.T); iu = np.triu_indices(len(idx), 1)
            return float(g[iu].mean())

        def module(size, loose):
            """gene set of `size`, tightness tuned by growing from a seed's a(g)-nearest neighbours."""
            seed = rng.choice(pool)
            sims = An[pool] @ An[seed]
            nb = pool[np.argsort(-sims)]
            top = nb[: min(len(nb), int(size * loose))]
            return rng.choice(top, min(size, len(top)), replace=False)

        print(f"\n=== layer {tap}: representation-tightness baseline curve ({N_NULL} null axes) ===", flush=True)
        nx, ny = [], []
        for _ in range(N_NULL):
            loose = rng.choice([1, 1, 2, 3, 5, 8, 15, 40, 120])
            ga, gb = module(400, loose), module(400, loose)
            nx.append(0.5 * (tset_tightness(ga) + tset_tightness(gb))); ny.append(power(axis(ga, gb)))
        nx, ny = np.array(nx), np.array(ny)
        B = np.column_stack([np.ones_like(nx), nx, nx ** 2]); coef, *_ = np.linalg.lstsq(B, ny, rcond=None)
        resid_sd = float((ny - B @ coef).std()); rho = float(spearmanr(nx, ny).statistic)
        print(f"   null tightness range [{nx.min():+.3f}, {nx.max():+.3f}]; power~tightness Spearman {rho:+.2f}"
              f"; residual scatter {resid_sd:.3f}")

        out["taps"][f"L{tap:02d}"] = {"power_tightness_rho": rho, "axes": {}}
        for name, (Ag, Bg) in AXES.items():
            ia = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Ag)]
            ib = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Bg)]
            both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
            tg = 0.5 * (tset_tightness(ia) + tset_tightness(ib))
            pw = power(axis(ia, ib))
            pred = float(np.array([1, tg, tg ** 2]) @ coef); zz = (pw - pred) / (resid_sd + 1e-9)
            pctl = float((nx < tg).mean())
            print(f"   {name:<28} tightness {tg:+.3f} (pctl {pctl:.2f})  power {pw:+.3f}  "
                  f"curve-predicts {pred:+.3f}  -> ABOVE z = {zz:+.1f}")
            out["taps"][f"L{tap:02d}"]["axes"][name] = dict(tightness=tg, tightness_pctl=pctl, power=pw,
                                                            curve_pred=pred, z_above_tightness=float(zz))

    allz = [(t, n, d["z_above_tightness"]) for t, tv in out["taps"].items() for n, d in tv["axes"].items()]
    best = max(allz, key=lambda x: x[2])
    out["verdict"] = (
        f"strongest: {best[0]}/{best[1]}, {best[2]:+.1f} sigma above the representation-tightness curve. " +
        ("SURVIVES the tightness null too — the functional-axis modulation is not merely a consequence of the "
         "gene set being compact in representation space. Combined with the co-expression control, the "
         "functional structure survives both the data-structure and representation-structure matched nulls."
         if best[2] > 3 else
         "DOES NOT clear the tightness null — the modulation may be explained by the functional gene sets simply "
         "being compact clusters in representation space, independent of their function."))
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_tightness_null.json"), "w"), indent=1)
    print("[done] -> results/ctx_tightness_null.json")


if __name__ == "__main__":
    main()
