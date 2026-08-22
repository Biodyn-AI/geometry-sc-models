"""c2s_gm_lib — route_genemanifold Thread-A machinery, ported to C2S-Scale-Gemma-2.

FIDELITY NOTE. The STATISTICS (order/circle/grid/antipodal/margin_boot and their nulls) are copied
VERBATIM from biotensor/codebase/route_genemanifold/gm_lib.py — they encode the mandated discipline:
NEVER feature-shuffle; the null PERMUTES THE ANNOTATION over the same genes (holds gene set,
co-expression, abundance fixed). Do not "improve" them.

WHAT CHANGES FOR C2S. C2S-Scale is a Gemma-2 TEXT LLM: a cell = a rank-ordered "cell sentence" of gene
names, standard tokenizer, NO learned gene-embedding table. So the route's context-free WEIGHT bases
(scgpt_we / maxtoki_we / geneformer_we) have no analog. The model basis here is the CONTEXT-AWARE
ACTIVATION basis `c2s_ctx_L{k}` — per-gene mean of the residual at that gene's token position, layer k,
over a diverse Tabula Sapiens panel (the direct analog of gm_lib's `ctx_L{k}`, and of build_ctx). The two
reference bases are unchanged and model-independent:
  coexpr : each gene's log1p-CP10k profile across the SAME TS panel (kidney+lung+immune). DATA baseline.
  esm2   : UCE/STATE frozen ESM2 protein embeddings. SEQUENCE control.
A hypothesis counts as a C2S result only if a c2s_ctx basis beats BOTH.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("C2S_GM_CACHE", os.path.join(HERE, "cache"))
BASES_DIR = os.environ.get("C2S_GM_BASES", os.path.join(HERE, "bases"))   # c2s_ctx_L{k}.npz live here
os.makedirs(CACHE, exist_ok=True)

# ---- reference data (local paths; override with env for the pod) ----
TS_RAW = os.environ.get("C2S_TS_RAW",
    f"{_DATA}/raw")
TS_FILES = ["tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad",
            "tabula_sapiens_immune_subset_20000.h5ad"]
_UCE = os.environ.get("C2S_UCE_SNAP", os.path.expanduser(
    "~/.cache/huggingface/hub/models--minwoosun--uce-misc/snapshots/"
    "bffb91084e4476698984e7e01f6170ce291f4074"))
ESM2_PT = os.path.join(_UCE, "protein_embeddings",
                       "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
CHROM_CSV = os.path.join(_UCE, "species_chrom.csv")

N_CELLS_PER_TISSUE = int(os.environ.get("C2S_TS_NCELLS", "2500"))
MIN_GENE_CELLS = 10
AUTOSOMES = [str(i) for i in range(1, 23)]
_cache = {}


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


# ---------------------------------------------------------------- reference bases
def build_coexpr(force=False):
    """DATA BASELINE: each gene's log1p-CP10k expression profile across the diverse TS panel
    (kidney+lung+immune, N_CELLS_PER_TISSUE each, one shared gene space). Rows = genes.
    Ported verbatim from gm_lib.build_coexpr."""
    import h5py
    out = os.path.join(CACHE, "coexpr_ts.npz")
    if os.path.exists(out) and not force:
        return out
    rng = np.random.default_rng(0)
    mats, syms0 = [], None
    for fn in TS_FILES:
        with h5py.File(os.path.join(TS_RAW, fn), "r") as f:
            X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
            fnm = f["var"]["feature_name"]
            syms = _dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]]
            if syms0 is None:
                syms0 = syms
            assert np.array_equal(syms, syms0), f"{fn}: gene space differs"
            sel = np.sort(rng.choice(n, min(N_CELLS_PER_TISSUE, n), replace=False))
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            D = np.zeros((len(sel), g), dtype=np.float32)
            for i, r in enumerate(sel):
                s, e = int(indptr[r]), int(indptr[r + 1])
                D[i, idx[s:e]] = data[s:e]
            tot = D.sum(1, keepdims=True); tot[tot == 0] = 1.0
            mats.append(np.log1p(D / tot * 1e4))
            print(f"[coexpr] {fn}: {len(sel)} cells", flush=True)
    C = np.concatenate(mats, 0)
    keep = (C > 0).sum(0) >= MIN_GENE_CELLS
    C, syms0 = C[:, keep], np.char.upper(syms0[keep].astype(str))
    np.savez(out, profiles=C.T.astype(np.float32), symbols=syms0)
    print(f"[coexpr] -> {out}  genes={int(keep.sum())} cells={C.shape[0]}", flush=True)
    return out


def _coexpr():
    z = np.load(build_coexpr(), allow_pickle=True)
    return z["profiles"].astype(np.float64), z["symbols"].astype(str)


def _esm2():
    """SEQUENCE control: UCE/STATE frozen ESM2 protein embeddings, keyed by gene symbol.
    Ported verbatim from gm_lib._esm2."""
    import torch
    d = torch.load(ESM2_PT, map_location="cpu", weights_only=False)
    syms = np.array(sorted(str(k).upper() for k in d.keys()))
    key = {str(k).upper(): k for k in d.keys()}
    M = np.stack([np.asarray(d[key[s]], dtype=np.float32) for s in syms]).astype(np.float64)
    return M, syms


def _c2s_ctx(layer):
    """C2S MODEL BASIS: per-gene mean residual activation at the gene's token position, layer `layer`,
    over the TS panel. Built on the GPU pod by build_ctx_basis.py -> bases/c2s_ctx_L{layer:02d}.npz."""
    p = os.path.join(BASES_DIR, f"c2s_ctx_L{layer:02d}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not built yet (run build_ctx_basis.py on the pod)")
    z = np.load(p, allow_pickle=True)
    return z["M"].astype(np.float64), np.char.upper(z["symbols"].astype(str))


def coords():
    """gene_symbol -> chromosome + start (UCE species_chrom, human, autosomes handled by caller).
    Ported verbatim from genome_wide.coords."""
    import pandas as pd
    d = pd.read_csv(CHROM_CSV)
    d = d[d.species == "human"].copy()
    d["gene_symbol"] = d.gene_symbol.astype(str).str.upper()
    d = d[~d.gene_symbol.duplicated(keep="first")]
    d["chromosome"] = d.chromosome.astype(str)
    return d.set_index("gene_symbol")


def basis(name):
    if name in _cache:
        return _cache[name]
    if name == "coexpr":
        v = _coexpr()
    elif name == "esm2":
        v = _esm2()
    elif name.startswith("c2s_ctx_L"):
        v = _c2s_ctx(int(name[9:11]))
    else:
        raise ValueError(f"unknown basis {name}")
    _cache[name] = v
    return v


def subset(name, genes):
    """Rows of basis(name) for `genes` (upper symbols), in order. Returns (M, mask found)."""
    M, syms = basis(name)
    pos = {s: i for i, s in enumerate(syms)}
    ok = np.array([g.upper() in pos for g in genes], dtype=bool)
    rows = np.array([pos[g.upper()] for g in np.asarray(genes)[ok]], dtype=int)
    return M[rows], ok


# ================================================================ STATISTICS (verbatim from gm_lib)
def _pcs(M, k=3):
    X = np.asarray(M, dtype=np.float64)
    X = X - X.mean(0)
    if min(X.shape) <= 64 or X.shape[0] < 200:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        return (U * s)[:, :k]
    from sklearn.utils.extmath import randomized_svd
    U, s, _ = randomized_svd(X, n_components=k, n_iter=4, random_state=0)
    return U * s


def _spear(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def _circ_mean(a):
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _circ_corr(a, b):
    sa, sb = np.sin(a - _circ_mean(a)), np.sin(b - _circ_mean(b))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def order_score(M, idx, n_perm=2000, seed=0, n_pc=3):
    P = _pcs(M, n_pc)
    stat = max(abs(_spear(P[:, k], idx)) for k in range(P.shape[1]))
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(idx)
        null[i] = max(abs(_spear(P[:, k], p)) for k in range(P.shape[1]))
    return dict(stat=float(stat), p=float((null >= stat).mean()),
                z=float((stat - null.mean()) / (null.std() + 1e-12)), null_mean=float(null.mean()), n=len(idx))


def circle_score(M, phase, n_perm=2000, seed=0):
    P = _pcs(M, 2)
    ang = np.arctan2(P[:, 1], P[:, 0])
    stat = abs(_circ_corr(ang, phase))
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = abs(_circ_corr(ang, rng.permutation(phase)))
    return dict(stat=float(stat), p=float((null >= stat).mean()),
                z=float((stat - null.mean()) / (null.std() + 1e-12)), null_mean=float(null.mean()), n=len(phase))


def grid_score(M, c1, c2, n_perm=2000, seed=0):
    P = _pcs(M, 3)
    best = 0.0
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            best = max(best, min(abs(_spear(P[:, i], c1)), abs(_spear(P[:, j], c2))))
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for t in range(n_perm):
        p1, p2 = rng.permutation(c1), rng.permutation(c2)
        b = 0.0
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                b = max(b, min(abs(_spear(P[:, i], p1)), abs(_spear(P[:, j], p2))))
        null[t] = b
    return dict(stat=float(best), p=float((null >= best).mean()),
                z=float((best - null.mean()) / (null.std() + 1e-12)), null_mean=float(null.mean()), n=len(c1))


def antipodal_score(name, a, b, axis_genes=None, axis_sign=None, n_null=2000, seed=0):
    M, syms = basis(name)
    pos = {s: i for i, s in enumerate(syms)}
    if a.upper() not in pos or b.upper() not in pos:
        return None
    va, vb = M[pos[a.upper()]], M[pos[b.upper()]]
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    nrm = np.linalg.norm(M, axis=1)
    da = np.digitize(np.linalg.norm(va), np.percentile(nrm, np.arange(10, 100, 10)))
    db = np.digitize(np.linalg.norm(vb), np.percentile(nrm, np.arange(10, 100, 10)))
    dec = np.digitize(nrm, np.percentile(nrm, np.arange(10, 100, 10)))
    ca, cb = np.where(dec == da)[0], np.where(dec == db)[0]
    rng = np.random.default_rng(seed)
    ia, ib = rng.choice(ca, n_null), rng.choice(cb, n_null)
    Na, Nb = M[ia], M[ib]
    null = np.sum(Na * Nb, 1) / (np.linalg.norm(Na, axis=1) * np.linalg.norm(Nb, axis=1) + 1e-12)
    out = dict(cos=cos, null_mean=float(null.mean()),
               p_more_negative=float((null <= cos).mean()),
               z=float((cos - null.mean()) / (null.std() + 1e-12)))
    if axis_genes is not None:
        axis = va - vb
        g, ok = subset(name, axis_genes)
        if ok.sum() >= 6:
            proj = g @ axis / (np.linalg.norm(axis) + 1e-12)
            out["axis_rho"] = _spear(proj, np.asarray(axis_sign)[ok])
            out["axis_n"] = int(ok.sum())
    return out


def _raw_stat(M, coord, kind, n_pc=3):
    if kind == "circle":
        P = _pcs(M, 2)
        return abs(_circ_corr(np.arctan2(P[:, 1], P[:, 0]), coord))
    if kind == "grid":
        P = _pcs(M, 3); c1, c2 = coord
        return max(min(abs(_spear(P[:, i], c1)), abs(_spear(P[:, j], c2)))
                   for i in range(3) for j in range(3) if i != j)
    P = _pcs(M, n_pc)
    return max(abs(_spear(P[:, k], coord)) for k in range(P.shape[1]))


def margin_boot(M_model, M_base, coord, kind, n_boot=1000, seed=0):
    """Paired bootstrap over GENES: is the model's margin over a reference REAL? Verbatim from gm_lib."""
    n = len(M_model)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        c = (np.asarray(coord[0])[i], np.asarray(coord[1])[i]) if kind == "grid" else np.asarray(coord)[i]
        try:
            d[b] = _raw_stat(M_model[i], c, kind) - _raw_stat(M_base[i], c, kind)
        except Exception:
            d[b] = np.nan
    d = d[np.isfinite(d)]
    return dict(margin=float(_raw_stat(M_model, coord, kind) - _raw_stat(M_base, coord, kind)),
                ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                frac_le0=float((d <= 0).mean()))
