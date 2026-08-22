"""gm_lib — shared machinery for the GENE-TOKEN manifold hunt.

The question (Ihor, 2026-07-17), after the pooled cell manifolds were shown redundant with expression
(`route_utility/RESULTS.md`, 0/20): forget cells. Do the GENE TOKENS carry geometric structure — a circle, an
ordered curve, a grid — the way days-of-the-week form a circle in an LLM (Engels et al. / Goodfire)?

We already know biology lives in the gene-embedding table: gene2vec excess +0.245 (MaxToki) / +0.074 (scGPT),
cross-model, an ORIGINAL-model property (`weightpapers/RESULTS.md` 06, 07); and W_E beats every layer on
TF-vs-target (08). What has never been tested is whether that structure is GEOMETRIC (ordered, low-dim, aligned
to an external coordinate) rather than merely CATEGORICAL (similar genes cluster). This library provides the
bases and the statistics; hypotheses live in `gene_sets.py`.

THE DESIGN PRINCIPLE (carried from route_utility, where it decided everything). Any structure in a gene basis
can be explained by the CO-EXPRESSION CONFOUND -- "genes co-expressed across cells get similar embeddings". So
every statistic is computed on the model bases AND on two reference bases:
  * `coexpr` -- each gene's expression profile across a diverse cell panel. THE DATA BASELINE. If it recovers
    the annotation as well as the model does, the model added nothing (exactly the route_utility verdict).
  * `esm2`   -- UCE/STATE's frozen ESM2 PROTEIN embeddings, never trained on expression. THE SEQUENCE CONTROL.
    If a structure shows here too, it is protein-sequence similarity (paralogy), not learned biology.
A hypothesis only counts if a model basis beats BOTH.

BASES (`basis(name)` -> (M, symbols), rows aligned to symbols):
  context-free WEIGHTS      scgpt_we, maxtoki_we, geneformer_we
  OUTPUT (unembedding)      maxtoki_lmhead   -- untied from embed_tokens, a genuinely separate readout basis
  SEQUENCE control          esm2
  DATA baseline             coexpr
  context-aware ACTIVATIONS ctx_L{k}         -- scGPT layer k, per-gene mean over the cached TS atlas
  cell-type conditioned     ctx_L{k}@{cell type}

Nulls: NEVER feature-shuffle (a known trap in this program -- it destroys the covariance that makes any
structure, so it "passes" everything). The null here PERMUTES THE ANNOTATION over the same genes, which holds
the gene set, its co-expression structure and its abundance prior fixed and asks only whether the embedding's
arrangement is aligned to the annotated order. Statistics that select over PCs recompute that same selection
inside the null, so the selection is paid for.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, struct, pickle
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = f"{_DATA}"
sys.path.insert(0, os.path.join(ROOT, "codebase", "route_a", "scgpt", "weightpapers"))

CACHE = os.path.join(ROOT, "data", "genemanifold")
os.makedirs(CACHE, exist_ok=True)

ATLAS = (f"{BASE}/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map/"
         "experiments/scgpt_atlas/activations")
TS_RAW = f"{BASE}/biomechinterp/biodyn-work/single_cell_mechinterp/data/raw"
TS_FILES = ["tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad", "tabula_sapiens_immune_subset_20000.h5ad"]
MT_ST = f"{ROOT}/../maxtoki/setup/MaxToki-217M-HF/model.safetensors"
MT_TOK = f"{ROOT}/../maxtoki/setup/token_dictionary.json"
# Geneformer -- the THIRD learned gene-embedding table in the portfolio, and the decisive generalisation test
# for the HOX-cluster finding (scGPT and MaxToki are the other two; UCE and STATE-SE have no learned gene table
# at all -- their gene tokens ARE the frozen ESM2 vectors of the `esm2` basis, so the finding cannot generalise
# to them by construction). Geneformer shares MaxTokif's tokenizer (20,274/20,275 identical ensembl->row
# mappings; same lab), so HOX genes land on the same token IDs and the comparison needs no re-alignment.
GF_DIR = "{_MODELS}/Geneformer"
GF = {"geneformer_we":    (f"{GF_DIR}/Geneformer-V2-104M/model.safetensors",
                           f"{GF_DIR}/geneformer/token_dictionary_gc104M.pkl"),                 # 20275 x 768
      "geneformer_v1_we": (f"{GF_DIR}/Geneformer-V1-10M/model.safetensors",
                           f"{GF_DIR}/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")}  # 25426x256
ENSMAP = "{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
ESM2_PT = (f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--minwoosun--uce-misc/snapshots/"
           "bffb91084e4476698984e7e01f6170ce291f4074/protein_embeddings/"
           "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")

N_CELLS_PER_TISSUE = 2500
MIN_GENE_CELLS = 10
_DT = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("bf16", 2), "F64": ("<f8", 8)}
_cache = {}


# ---------------------------------------------------------------- safetensors (from weightpapers/06)
class ST_Reader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.base = 8 + n

    def get(self, key):
        h = self.header[key]; dt, sh = h["dtype"], h["shape"]
        b0, b1 = h["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + b0); raw = f.read(b1 - b0)
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        else:
            npdt, _ = _DT[dt]; arr = np.frombuffer(raw, dtype=npdt).astype(np.float32)
        return arr.reshape(sh).astype(np.float64)


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


# ---------------------------------------------------------------- bases
def _scgpt_we():
    import wp_lib as L
    W = L.gene_embedding()
    s2r = L.symbol_to_row()
    syms = np.array(sorted(s2r.keys()))
    return W[np.array([s2r[s] for s in syms])], syms


def _maxtoki(which):
    """which='embed' -> model.embed_tokens.weight (input); 'lmhead' -> lm_head.weight (untied OUTPUT)."""
    R = ST_Reader(MT_ST)
    M = R.get("model.embed_tokens.weight" if which == "embed" else "lm_head.weight")
    tok = json.load(open(MT_TOK))                      # ensembl id -> token row
    ens2sym = {}
    for sym, ens in pickle.load(open(ENSMAP, "rb")).items():
        ens2sym[ens] = sym.upper()
    rows, syms = [], []
    for ens, row in tok.items():
        s = ens2sym.get(ens)
        if s is not None and row < M.shape[0]:
            rows.append(row); syms.append(s)
    o = np.argsort(syms)
    rows, syms = np.array(rows)[o], np.array(syms)[o]
    _, keep = np.unique(syms, return_index=True)       # first occurrence per symbol
    return M[rows[keep]], syms[keep]


def _geneformer(name):
    """Geneformer's learned input gene-embedding table (`bert.embeddings.word_embeddings.weight`).

    Geneformer is a BERT-style MLM over rank-value-encoded expression -- no protein sequence, no genomic
    coordinates, a different objective and a different corpus from MaxToki. It is therefore the clean
    independent replication for the HOX-cluster claim. Same loader shape as _maxtoki (shared tokenizer), except
    the token dict is a pickle keyed by ensembl id rather than JSON.
    """
    st, tk = GF[name]
    M = ST_Reader(st).get("bert.embeddings.word_embeddings.weight")
    tok = pickle.load(open(tk, "rb"))                  # ensembl id -> token row
    ens2sym = {}
    for sym, ens in pickle.load(open(ENSMAP, "rb")).items():
        ens2sym[ens] = sym.upper()
    rows, syms = [], []
    for ens, row in tok.items():
        s = ens2sym.get(ens)
        if s is not None and isinstance(row, (int, np.integer)) and row < M.shape[0]:
            rows.append(int(row)); syms.append(s)
    o = np.argsort(syms)
    rows, syms = np.array(rows)[o], np.array(syms)[o]
    _, keep = np.unique(syms, return_index=True)       # first occurrence per symbol
    return M[rows[keep]], syms[keep]


def _esm2():
    """UCE/STATE gene tokens = frozen ESM2 protein embeddings, keyed by gene symbol. The SEQUENCE control.

    NOTE (2026-07-17): this table IS UCE's gene vocabulary, byte for byte -- 19,790 human symbols x 5,120 dims,
    matching route_uce/MODEL_NOTES.md exactly. UCE and STATE-SE build no learned gene table; they feed these
    frozen ESM2 vectors straight into a TransformerEncoder. So for those two models this basis is not a
    "control" at all -- it IS their gene representation, and their HOX-cluster score is this row.
    """
    import torch
    d = torch.load(ESM2_PT, map_location="cpu", weights_only=False)
    syms = np.array(sorted(str(k).upper() for k in d.keys()))
    key = {str(k).upper(): k for k in d.keys()}
    M = np.stack([np.asarray(d[key[s]], dtype=np.float32) for s in syms]).astype(np.float64)
    return M, syms


def build_coexpr(force=False):
    """THE DATA BASELINE: each gene's log1p-CP10k expression profile across a diverse TS panel
    (kidney + lung + immune, N_CELLS_PER_TISSUE each, one shared 60,606-gene space). Rows = genes."""
    out = os.path.join(CACHE, "coexpr_ts.npz")
    if os.path.exists(out) and not force:
        return out
    rng = np.random.default_rng(0)
    mats, syms0 = [], None
    for fn in TS_FILES:
        with h5py.File(os.path.join(TS_RAW, fn), "r") as f:
            X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
            # var/_index is the ENSEMBL id; the symbols live in the categorical var/feature_name.
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
            print(f"[coexpr] {fn}: {len(sel)} cells")
    C = np.concatenate(mats, 0)                                    # (cells, genes)
    keep = (C > 0).sum(0) >= MIN_GENE_CELLS
    C, syms0 = C[:, keep], np.char.upper(syms0[keep].astype(str))
    np.savez(out, profiles=C.T.astype(np.float32), symbols=syms0)  # (genes, cells)
    print(f"[coexpr] -> {out}  genes={keep.sum()} cells={C.shape[0]}")
    return out


def _coexpr():
    z = np.load(build_coexpr(), allow_pickle=True)
    return z["profiles"].astype(np.float64), z["symbols"].astype(str)


def _replogle_path():
    sys.path.insert(0, os.path.join(ROOT, "codebase", "route_cellcycle"))
    from cc_common import REPLOGLE as R          # single source of truth for this path
    return R


def build_coexpr_k562(n_cells=8000, force=False):
    """A SECOND data baseline, on CYCLING cells (Replogle K562).

    WHY THIS EXISTS. `coexpr` is TS kidney+lung+immune -- tissues that barely proliferate. For a cell-cycle
    question that panel has almost no cycling variance, so its co-expression baseline is under-powered and the
    model beats it for free. That is EXACTLY the trap route_utility caught (a linear probe was the wrong
    baseline and flattered the model by +0.06 until a fair one beat every model). A cell-cycle claim must be
    tested against co-expression measured where the cell cycle actually varies. K562 is a proliferating line.
    """
    out = os.path.join(CACHE, "coexpr_k562.npz")
    if os.path.exists(out) and not force:
        return out
    rng = np.random.default_rng(0)
    with h5py.File(_replogle_path(), "r") as f:
        X = f["X"]; n = X.shape[0]
        v = f["var"]["gene_name_index"]
        syms = _dec(v[:]).astype(str) if v.dtype.kind in "OS" else np.asarray(v).astype(str)
        sel = np.sort(rng.choice(n, min(n_cells, n), replace=False))
        D = np.stack([np.asarray(X[int(i), :], dtype=np.float32) for i in sel])
    tot = D.sum(1, keepdims=True); tot[tot == 0] = 1.0
    C = D if (D.max() < 20.0 and not np.allclose(D, np.round(D))) else np.log1p(D / tot * 1e4)
    keep = (C > 0).sum(0) >= MIN_GENE_CELLS
    np.savez(out, profiles=C[:, keep].T.astype(np.float32), symbols=np.char.upper(syms[keep].astype(str)))
    print(f"[coexpr_k562] -> {out}  genes={int(keep.sum())} cells={C.shape[0]}")
    return out


def _coexpr_k562():
    z = np.load(build_coexpr_k562(), allow_pickle=True)
    return z["profiles"].astype(np.float64), z["symbols"].astype(str)


FETAL_GUT = os.path.join(ROOT, "data", "pancreas", "fetal_gut.h5ad")


def build_coexpr_devel(n_cells=8000, force=False):
    """A THIRD data baseline, on DEVELOPMENTAL cells (human fetal gut, 62,849 cells).

    WHY THIS EXISTS -- same reason as build_coexpr_k562, one hypothesis later. The TS kidney+lung+immune panel
    expresses HOX in ~3% of cells (housekeeping: 82%; 8/29 HOX genes in <1%). So `coexpr`'s 0.08 on the HOX grid
    measures MY PANEL'S BLINDNESS, not the data -- and it handed MaxToki an apparently-surviving win. Fetal gut
    is embryonic endoderm, where HOX is strongly expressed AND spatially patterned along the anterior-posterior
    axis, i.e. exactly the coordinate the hypothesis is about. This is the baseline the HOX claim requires.

    Third instance of the same trap in this program (route_utility: linear-vs-nonlinear; route_genemanifold:
    cycling cells for the cell cycle; here: developmental cells for HOX). The rule is now unconditional:
    THE BASELINE PANEL MUST EXPRESS THE BIOLOGY THE HYPOTHESIS IS ABOUT.
    """
    out = os.path.join(CACHE, "coexpr_devel.npz")
    if os.path.exists(out) and not force:
        return out
    rng = np.random.default_rng(0)
    with h5py.File(FETAL_GUT, "r") as f:
        X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
        fnm = f["var"]["feature_name"]
        syms = _dec(fnm["categories"][:]).astype(str)[fnm["codes"][:]]
        sel = np.sort(rng.choice(n, min(n_cells, n), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        D = np.zeros((len(sel), g), dtype=np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            D[i, idx[s:e]] = data[s:e]
    tot = D.sum(1, keepdims=True); tot[tot == 0] = 1.0
    C = D if (D.max() < 20.0 and not np.allclose(D, np.round(D))) else np.log1p(D / tot * 1e4)
    keep = (C > 0).sum(0) >= MIN_GENE_CELLS
    np.savez(out, profiles=C[:, keep].T.astype(np.float32), symbols=np.char.upper(syms[keep].astype(str)))
    print(f"[coexpr_devel] -> {out}  genes={int(keep.sum())} cells={C.shape[0]}")
    return out


def _coexpr_devel():
    z = np.load(build_coexpr_devel(), allow_pickle=True)
    return z["profiles"].astype(np.float64), z["symbols"].astype(str)


def _atlas_meta():
    m = json.load(open(os.path.join(ATLAS, "extraction_metadata.json")))
    return m["cell_data"]


def build_ctx(layer, cell_type=None, n_chunks=60, chunk=8000, min_cnt=12, force=False):
    """CONTEXT-AWARE basis: per-gene mean of the scGPT residual at that gene's TOKEN POSITION, layer `layer`,
    over the cached TS atlas (3.56M tokens / 3000 cells / immune+kidney+lung). Optionally restricted to cells
    of one `cell_type` -- the 'different programs on/off' probe.

    CONVENTION CAVEAT (inherited, flagged in weightpapers/08): the atlas was extracted with the raw-count value
    convention (the project's 51-bin bug), so inputs are off-distribution. Averaging over cells washes out the
    mis-binned VALUE channel and leaves gene identity, which is what a per-gene basis reads -- 08 judged the
    across-layer PROFILE usable on exactly this ground. Read the depth/cell-type CONTRASTS, not absolutes.
    """
    tag = f"ctx_L{layer:02d}" + (f"@{cell_type.replace(' ', '_').replace(',', '')}" if cell_type else "")
    out = os.path.join(CACHE, f"{tag}.npz")
    if os.path.exists(out) and not force:
        return out
    import wp_lib as L
    names = L.gene_names()
    cd = _atlas_meta()
    keep_cells = None
    if cell_type is not None:
        keep_cells = np.array([i for i, c in enumerate(cd) if c["cell_type"] == cell_type])
        assert len(keep_cells) >= 20, f"{cell_type}: only {len(keep_cells)} cells"
    A = np.load(os.path.join(ATLAS, f"layer_{layer:02d}_activations.npy"), mmap_mode="r")
    gid = np.load(os.path.join(ATLAS, f"layer_{layer:02d}_gene_ids.npy"))
    cid = np.load(os.path.join(ATLAS, f"layer_{layer:02d}_cell_ids.npy"))
    N, d = A.shape
    starts = np.linspace(0, max(0, N - chunk), n_chunks).astype(int)
    sums, cnts = {}, {}
    for s in starts:
        sl = slice(int(s), int(s) + chunk)
        g, c = gid[sl], cid[sl]
        m = np.isin(c, keep_cells) if keep_cells is not None else np.ones(len(g), bool)
        if not m.any():
            continue
        a = np.asarray(A[sl][m], dtype=np.float64); g = g[m]
        for u in np.unique(g):
            w = g == u
            sums[u] = sums.get(u, 0) + a[w].sum(0)
            cnts[u] = cnts.get(u, 0) + int(w.sum())
    ids = np.array([u for u in sorted(sums) if cnts[u] >= min_cnt])
    M = np.stack([sums[u] / cnts[u] for u in ids])
    syms = np.char.upper(np.array([names[i] for i in ids]).astype(str))
    o = np.argsort(syms); M, syms = M[o], syms[o]
    _, keep = np.unique(syms, return_index=True)
    np.savez(out, M=M[keep].astype(np.float32), symbols=syms[keep],
             counts=np.array([cnts[i] for i in ids])[o][keep])
    print(f"[{tag}] genes={len(keep)} (>= {min_cnt} tokens)"
          + (f" cells={len(keep_cells)}" if keep_cells is not None else "") + f" -> {out}")
    return out


def basis(name):
    """(M, symbols) for any basis name. Cached in-process."""
    if name in _cache:
        return _cache[name]
    if name == "scgpt_we":
        v = _scgpt_we()
    elif name == "maxtoki_we":
        v = _maxtoki("embed")
    elif name == "maxtoki_lmhead":
        v = _maxtoki("lmhead")
    elif name in GF:
        v = _geneformer(name)
    elif name == "esm2":
        v = _esm2()
    elif name == "coexpr":
        v = _coexpr()
    elif name == "coexpr_k562":
        v = _coexpr_k562()
    elif name == "coexpr_devel":
        v = _coexpr_devel()
    elif name.startswith("ctx_L"):
        ct = None
        if "@" in name:
            name, ct = name.split("@", 1)
            ct = {c["cell_type"].replace(" ", "_").replace(",", ""): c["cell_type"]
                  for c in _atlas_meta()}.get(ct, ct)
        z = np.load(build_ctx(int(name[5:7]), ct), allow_pickle=True)
        v = (z["M"].astype(np.float64), z["symbols"].astype(str))
    else:
        raise ValueError(f"unknown basis {name}")
    _cache[name] = v
    return v


def subset(name, genes):
    """Rows of `basis(name)` for `genes` (upper symbols), in the given order. Returns (M, mask of found)."""
    M, syms = basis(name)
    pos = {s: i for i, s in enumerate(syms)}
    ok = np.array([g.upper() in pos for g in genes], dtype=bool)
    rows = np.array([pos[g.upper()] for g in np.asarray(genes)[ok]], dtype=int)  # dtype: empty list -> float
    return M[rows], ok


# ---------------------------------------------------------------- statistics
def _pcs(M, k=3):
    """Top-k principal coordinates. We only ever need k<=3, so a full SVD is pure waste at the n where this
    question is actually answerable: on the TRRUST set (2780 x 1232) a full SVD costs ~3 s, which put one
    bootstrap sweep at ~4 h. randomized_svd on the same matrix is ~50x faster and identical for the leading
    components (they are what it solves for). Small matrices keep the exact path -- randomized SVD is only an
    approximation worth taking when the matrix is big enough to pay for it."""
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


def order_score(M, idx, n_perm=2000, seed=0, n_pc=3):
    """Is the annotated 1-D ORDER recovered? stat = max_k |Spearman(PC_k, idx)| over the first n_pc PCs.
    Null PERMUTES the annotation over the same genes (holds gene set / co-expression / abundance fixed) and
    recomputes the same max, so the over-PCs selection is paid for."""
    P = _pcs(M, n_pc)
    stat = max(abs(_spear(P[:, k], idx)) for k in range(P.shape[1]))
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(idx)
        null[i] = max(abs(_spear(P[:, k], p)) for k in range(P.shape[1]))
    return dict(stat=float(stat), p=float((null >= stat).mean()),
                z=float((stat - null.mean()) / (null.std() + 1e-12)), null_mean=float(null.mean()), n=len(idx))


def _circ_corr(a, b):
    """Jammalamadaka-SenGupta circular correlation."""
    sa, sb = np.sin(a - _circ_mean(a)), np.sin(b - _circ_mean(b))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def _circ_mean(a):
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def circle_score(M, phase, n_perm=2000, seed=0):
    """Is the annotated CIRCLE recovered? stat = |circular corr| between the PC1-PC2 angle and the annotated
    phase. Handedness is free (we take |.|), origin is free (circ_corr centres both)."""
    P = _pcs(M, 2)
    ang = np.arctan2(P[:, 1], P[:, 0])
    stat = abs(_circ_corr(ang, phase))
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = abs(_circ_corr(ang, rng.permutation(phase)))
    return dict(stat=float(stat), p=float((null >= stat).mean()),
                z=float((stat - null.mean()) / (null.std() + 1e-12)), null_mean=float(null.mean()), n=len(phase))


def grid_score(M, c1, c2, n_perm=2000, seed=0):
    """2-D grid: recover BOTH coordinates from the first 3 PCs. stat = min(best|rho| for c1, best|rho| for c2)
    using disjoint PCs -- min, so a grid must encode both, not one twice."""
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
    """Is the antagonistic pair (a,b) ANTIPODAL -- opposite along a shared axis -- rather than merely far apart?
    Two statistics:
      cos       : cosine(W[a], W[b]); null = cosines of random gene pairs matched on vector norm decile.
      axis_rho  : if `axis_genes`/`axis_sign` given, do OTHER lineage genes order along the a-b difference axis?
                  (that is what makes it an AXIS and not just two distant points)
    """
    M, syms = basis(name)
    pos = {s: i for i, s in enumerate(syms)}
    if a.upper() not in pos or b.upper() not in pos:
        return None
    va, vb = M[pos[a.upper()]], M[pos[b.upper()]]
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    nrm = np.linalg.norm(M, axis=1)
    da, db = np.digitize(np.linalg.norm(va), np.percentile(nrm, np.arange(10, 100, 10))), \
        np.digitize(np.linalg.norm(vb), np.percentile(nrm, np.arange(10, 100, 10)))
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


def compare(stat_fn, gene_sets_M, bases, **kw):
    """Run one statistic across bases. Returns {basis: result}. `gene_sets_M` maps basis -> (M, coord)."""
    return {b: stat_fn(*gene_sets_M[b], **kw) for b in bases if b in gene_sets_M}


def _raw_stat(M, coord, kind, n_pc=3):
    """The point statistic only (no null) -- for bootstrapping margins."""
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
    """Is the model's margin over a reference basis REAL? Paired bootstrap over GENES.

    A bigger number is not a win. Both bases are re-scored on the SAME resampled gene set (the whole pipeline
    -- PCA and statistic -- is recomputed inside each resample, so the margin's uncertainty includes the
    instability of fitting a geometry to a handful of genes, which is the dominant term at n~10-40).
    """
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
