"""LEVEL-2 FEASIBILITY: does a gene move toward its OWN functional pole in contexts that share that function?

The results so far (ctx_polysemy .. ctx_coexpr_null) reach Level 1: genes undergo gene-specific, reproducible,
abundance-independent context modulation organised along functional axes beyond co-expression. Level 2 -- "the
model tells you a gene's role in a new cell" -- needs the movement to be DIRECTIONALLY appropriate: a
functionally-surface gene should move toward the surface pole specifically in surface/secretory cell types, not
everywhere. This probe asks, cheaply, on the existing extraction, whether any such directional signal exists.

THE STATISTIC -- a cross-validated Tukey non-additivity test along a functional axis u.
For a gene g and context c, project the representation onto u:  q(g,c) = <v(g,c), u>. Decompose into
  f(g)  gene loading    (mean over contexts)   -- the gene's own functional identity on u
  h(c)  context loading (mean over genes)      -- the context's functional character on u
  e(g,c) interaction    (what is left)
The DIRECTIONAL / context-appropriate prediction is that the interaction follows the PRODUCT of the loadings:
e(g,c) ~ f(g) * h(c), with positive slope -- i.e. a gene moves further toward its own pole exactly in contexts
whose character shares that pole. beta = corr(e, f*h) measures this.

NO CIRCULARITY: f, h (hence the predictor f*h) are built from ONE cell partition; the interaction e is measured
on the OTHER partition; beta averages the two cross-directions. A predictor fit on one half cannot manufacture
correlation with an independent half's residual.

CONTROLS: projection rank-residualised (abundance); and the decisive one -- functional u is compared against
CO-EXPRESSION-MODULE axes spanning the coherence range (as in ctx_coexpr_null). If functional beta sits ABOVE
the coherence-beta curve, the directional congruence is beyond co-expression -> Level 2 has a spark. If it sits
on the curve, the direction is co-expression and Level 2 is not reachable this way.

Reads the headline ctx_maxtoki_L*.npz. Out: results/ctx_directional_probe.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ctx_position_confound as CP
from ctx_coexpr_null import coexpr_matrix, coherence
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
TAPS = [4, 8]   # headline ctx_maxtoki set has L0/L4/L8/L11; L4 is where the null comparisons were run
MIN_CTX = 9
N_NULL = 300
SEED = 0
from scipy.stats import spearmanr

AXES = {
    "nuclear_vs_surface":   (["GO:0005634", "GO:0000785", "GO:0003677"], ["GO:0005886", "GO:0005576", "GO:0005615"]),
    "mito_vs_cytoskeleton": (["GO:0005739"], ["GO:0005856"]),
    "transcription_vs_transport": (["GO:0006355", "GO:0003700"], ["GO:0006811", "GO:0038023"]),
}


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
    rng = np.random.default_rng(SEED)

    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    genes = z0["genes"].astype(str); ctxs = z0["contexts"].astype(str)
    syms = [ens2sym.get(g) for g in genes]
    tokmap = json.load(open(f"{CP.MSETUP}/token_dictionary.json")); ens2tid = {k: int(v) for k, v in tokmap.items()}
    tids = np.array([ens2tid.get(g, -1) for g in genes])

    print("[1/3] per-context mean rank (abundance control)", flush=True)
    MR = CP.mean_ranks(set(ctxs))
    rank = np.full((len(ctxs), len(genes)), np.nan)
    for ci, c in enumerate(ctxs):
        for gi, t in enumerate(tids):
            if t in MR.get(c, {}):
                rank[ci, gi] = MR[c][t]
    print("[2/3] co-expression matrix", flush=True)
    C = coexpr_matrix(list(genes))

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
        muse = full[:, use]; Ruse = rank[:, use]

        def proj_resid(vec):
            """per-partition projection onto vec, rank-residualised; returns (2, nctx, |use|) with nan mask."""
            q = np.tensordot(Mz[:, :, use], vec, axes=([3], [0]))
            qm = np.where(muse[None], q, np.nan)
            fin = np.isfinite(qm[0]) & np.isfinite(Ruse)
            r = Ruse[fin]; A = np.column_stack([np.ones_like(r), r, r ** 2])
            for p in range(2):
                yv = qm[p][fin]; qm[p][fin] = yv - A @ np.linalg.lstsq(A, yv, rcond=None)[0]
            return qm

        def beta(vec):
            """cross-validated non-additivity: predictor f*h from one partition, interaction from the other."""
            qm = proj_resid(vec)
            F, H, E, P = {}, {}, {}, {}
            for p in range(2):
                Q = qm[p]
                grand = np.nanmean(Q)
                F[p] = np.nanmean(Q, 0)                      # gene loading  (|use|,)
                H[p] = np.nanmean(Q, 1)                      # context loading (nctx,)
                E[p] = Q - F[p][None, :] - H[p][:, None] + grand
                P[p] = np.outer(H[p], F[p])                  # predictor f*h  (nctx, |use|)
            out = []
            for a, b in [(0, 1), (1, 0)]:
                sel = np.isfinite(E[a]) & np.isfinite(P[b])
                x, y = P[b][sel], E[a][sel]
                if x.std() < 1e-9:
                    continue
                out.append(float(np.corrcoef(x, y)[0, 1]))
            return float(np.mean(out)) if out else 0.0

        def axis(ia, ib):
            u = a_space[ia].mean(0) - a_space[ib].mean(0); return u / (np.linalg.norm(u) + 1e-9)

        pool = np.where(gene_ok)[0]

        def module(size, loose):
            seed = rng.choice(pool); nb = pool[np.argsort(-C[seed, pool])]
            top = nb[: min(len(nb), int(size * loose))]
            return rng.choice(top, min(size, len(top)), replace=False)

        # co-expression-module beta curve
        nx, nb_ = [], []
        for _ in range(N_NULL):
            loose = rng.choice([1, 1, 2, 3, 5, 8, 15, 40, 120])
            ga, gb = module(400, loose), module(400, loose)
            nx.append(0.5 * (coherence(C, ga) + coherence(C, gb))); nb_.append(beta(axis(ga, gb)))
        nx, nb_ = np.array(nx), np.array(nb_)
        A = np.column_stack([np.ones_like(nx), nx, nx ** 2]); coef, *_ = np.linalg.lstsq(A, nb_, rcond=None)
        rsd = float((nb_ - A @ coef).std())
        print(f"\n=== layer {tap}: null beta {nb_.mean():+.3f}±{nb_.std():.3f}; coherence~beta rho "
              f"{spearmanr(nx, nb_).statistic:+.2f} ===")

        out["taps"][f"L{tap:02d}"] = {"null_beta_mean": float(nb_.mean()), "axes": {}}
        for name, (Ag, Bg) in AXES.items():
            ia = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Ag)]
            ib = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Bg)]
            both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
            u = axis(ia, ib); bt = beta(u)
            coh = 0.5 * (coherence(C, ia) + coherence(C, ib))
            pred = float(np.array([1, coh, coh ** 2]) @ coef); zc = (bt - pred) / (rsd + 1e-9)
            znull = (bt - nb_.mean()) / (nb_.std() + 1e-9)
            print(f"   {name:<28} beta={bt:+.3f}  null={nb_.mean():+.3f}  z_vs_null={znull:+.1f}  "
                  f"z_above_coexpr_curve={zc:+.1f}")
            out["taps"][f"L{tap:02d}"]["axes"][name] = dict(beta=bt, z_vs_null=float(znull),
                                                            z_above_coexpr=float(zc), coherence=coh)

    allc = [(t, n, d) for t, tv in out["taps"].items() for n, d in tv["axes"].items()]
    best = max(allc, key=lambda x: x[2]["z_above_coexpr"])
    t, n, d = best
    out["verdict"] = (
        f"strongest: {t}/{n} beta={d['beta']:+.3f}, {d['z_above_coexpr']:+.1f} sigma above the "
        f"co-expression-beta curve (z_vs_null {d['z_vs_null']:+.1f}). " +
        ("SPARK — genes move toward their OWN functional pole more in contexts that share that function, beyond "
         "co-expression. Level 2 (context-appropriate directional readout) has a measurable signal; a curated "
         "directional test + re-extraction is now justified."
         if d["z_above_coexpr"] > 3 and d["beta"] > 0 else
         "NO SPARK — the directional congruence does not exceed co-expression (or is absent). The movement is "
         "functionally organised (Level 1) but not context-APPROPRIATE beyond co-expression. Level 2 likely "
         "not reachable for this expression-only model; do NOT spend the re-extraction.")
    )
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_directional_probe.json"), "w"), indent=1)
    print("[done] -> results/ctx_directional_probe.json")


if __name__ == "__main__":
    main()
