"""Shared config for route_steering2 — the three follow-ups to route_steering.

route_steering established: (a) the pseudotime tangent rotates 48-113 deg (theta_far, headroom-free);
(b) a constant linear direction + LOCAL TANGENT PROJECTION steers on-manifold 16/16; (c) aiming that steer at a
terminal branch raises that branch's program, 4/4, read out through scGPT's OWN pretrained MVC head
(circularity-free), exact permutation p = 0.042.

Three things that result needs, and this folder does:

  1. baseline.py       THE MISSING CONTROL. Every number in route_steering lives inside SOME model's embedding
                       space. `random_proj` shows the DIRECTION matters within scGPT's space; nothing shows that
                       scGPT's SPACE matters. Baseline = 20-D whitened PCA of log1p-CP10k expression, identical
                       pipeline. If it steers as well, the result is about single-cell data geometry, not about
                       foundation models. Sharpened by a real tension in the project's own findings: theta_far
                       reproduces 16/16 across four architecturally unrelated models (which argues the geometry
                       is in the DATA), while route_benchmark finds geometry BUILDS WITH DEPTH (which argues the
                       model adds it). This adjudicates.

  2. fate_matrix.py    THE POWER FIX. With 4 fates the exact permutation over 4! = 24 target->fate assignments
                       has a FLOOR of p = 1/24 = 0.042 — the reported p is the floor, not evidence of a large
                       effect. Two free fixes: (i) Setty has a 5th terminal branch (DCs) that was never used ->
                       5! = 120, floor 0.0083; (ii) scGPT's MVC head is a fixed matrix over the gene vocabulary,
                       so it is gene-universal and transfers unchanged to lung/gut/pancreas — whose scgptbin
                       embeddings ALREADY EXIST. Four independent diagonals, Fisher-combined.

  3. maxtoki_native.py THE "scGPT-ONLY" FIX. NATIVE_DECODE_RESULTS.md says the circularity-free readout is
                       scGPT-only. It is not. MaxToki-217M is a LlamaForCausalLM with tie_word_embeddings=false
                       and a real lm_head.weight of shape (20275, 1232) — a pretrained hidden-state -> gene-token
                       decoder in which no parameter was fit on Setty. Same gate, same fate matrix, second
                       architecture.

All geometry stays in the project-standard 20-D whitened PCA space (tangent_diagnostic.prep).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
import numpy as np
import h5py

ROOT = f"{_DATA}"
RS = os.path.join(ROOT, "codebase", "route_steering")          # reuse route_steering's code, unchanged
sys.path.insert(0, RS)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
DATA = os.path.join(ROOT, "data", "branchpoint")

TISSUES = ["setty", "pancreas", "lung", "gut"]

H5AD = {
    "setty":    os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad"),
    "pancreas": os.path.join(ROOT, "data/pancreas/pancreas_setty_schema.h5ad"),
    "lung":     os.path.join(ROOT, "data/pancreas/lung_airway_setty_schema.h5ad"),
    "gut":      os.path.join(ROOT, "data/pancreas/gut_setty_schema.h5ad"),
}

# ---------------------------------------------------------------------------------------------------------
# Terminal fates per tissue, and the marker programs that read them out.
#
# The panels are hand-curated CANONICAL lineage markers taken from standard cell-type biology, NOT derived from
# these datasets — that is what keeps the fate diagonal circularity-free (same principle as route_steering's
# blood PANEL, which these extend). `stem` is the root program that should FALL as a cell commits.
#
# BLOOD: route_steering used 4 fates (ery/myeloid/lymphoid/mega) and hit the 1/24 permutation floor. Setty's
# annotation carries a 5th terminal branch — DCs, n=303 — that was simply never used. Adding it takes the
# attainable floor from 0.042 to 0.0083 at zero cost.
# ---------------------------------------------------------------------------------------------------------
FATE_CLUSTERS = {
    "setty": {
        "ery":      ["Ery_1", "Ery_2"],
        "myeloid":  ["Mono_1", "Mono_2"],
        "lymphoid": ["CLP"],
        "mega":     ["Mega"],
        "dc":       ["DCs"],                       # the 5th branch route_steering left on the table
    },
    "pancreas": {                                  # Bastidas-Ponce endocrinogenesis; root = Ductal/Ngn3-low
        "alpha":   ["Alpha"],
        "beta":    ["Beta"],
        "delta":   ["Delta"],
        "epsilon": ["Epsilon"],
    },
    "lung": {                                      # Tabula Sapiens airway; root = basal cell
        "club":    ["club cell"],
        "goblet":  ["respiratory tract goblet cell"],
        "ciliated": ["lung multiciliated epithelial cell"],
    },
    "gut": {                                       # fetal gut; root = stem cell / progenitor cell
        "enterocyte": ["enterocyte"],
        "goblet":     ["intestine goblet cell"],
        "eec":        ["enteroendocrine cell"],
    },
}

# Root clusters — the source compartment a steer starts from (used for the fate direction's origin).
ROOT_CLUSTERS = {
    "setty":    ["HSC_1", "HSC_2"],
    "pancreas": ["Ductal", "Ngn3 low EP"],
    "lung":     ["basal cell"],
    "gut":      ["stem cell", "progenitor cell"],
}

PANEL = {
    "setty": {          # unchanged from route_steering/gene_decode.py PANEL, + a dc program for the 5th branch
        "stem":     ["CD34", "AVP", "HLF", "CRHBP", "MEIS1", "HOXA9", "MLLT3", "SPINK2", "PRSS57"],
        "ery":      ["GATA1", "KLF1", "GYPA", "HBB", "HBD", "TFRC", "CD36", "GATA2", "APOC1"],
        "myeloid":  ["SPI1", "CEBPA", "MPO", "ELANE", "LYZ", "CD14", "CSF1R", "AZU1"],
        "lymphoid": ["IL7R", "JCHAIN", "EBF1", "VPREB1", "DNTT", "CD79A", "CD79B", "RAG1", "RAG2", "LTB"],
        "mega":     ["PF4", "PPBP", "ITGA2B", "VWF", "GP9", "THBS1", "TUBB1"],
        "dc":       ["IRF8", "SPIB", "TCF4", "CLEC4C", "IL3RA", "LILRA4", "BCL11A"],
    },
    "pancreas": {
        "stem":    ["SOX9", "HES1", "SPP1", "NEUROG3", "HNF1B", "ONECUT1"],
        "alpha":   ["GCG", "ARX", "IRX2", "TTR", "PEMT"],
        "beta":    ["INS1", "INS2", "PDX1", "IAPP", "NKX6-1", "UCN3"],
        "delta":   ["SST", "HHEX", "LEPR", "PCSK1"],
        "epsilon": ["GHRL", "ACSL1", "SERPINA6"],
    },
    "lung": {
        "stem":     ["KRT5", "TP63", "KRT14", "DLK2", "KRT15"],
        "club":     ["SCGB1A1", "SCGB3A2", "CYP2F1", "BPIFB1"],
        "goblet":   ["MUC5AC", "MUC5B", "TFF3", "SPDEF", "TFF1"],
        "ciliated": ["FOXJ1", "TPPP3", "DNAI1", "TEKT1", "PIFO", "CAPS"],
    },
    "gut": {
        "stem":       ["LGR5", "OLFM4", "ASCL2", "SOX9", "MKI67"],
        "enterocyte": ["FABP1", "APOA1", "ALPI", "APOA4", "RBP2"],
        "goblet":     ["MUC2", "TFF3", "SPDEF", "CLCA1", "FCGBP"],
        "eec":        ["CHGA", "CHGB", "PYY", "NEUROD1", "SCG3"],
    },
}

N_HVG = 2000


def load_expression(tissue, cell_idx):
    """log1p CP10k expression for the given h5ad rows. Returns (E[n,g] float32, genes, clusters).

    Generalises route_steering/gene_decode.load_expression (which is hardcoded to Setty) to all four tissues.
    Handles both CSR-in-group and dense X layouts.
    """
    p = H5AD[tissue]
    with h5py.File(p, "r") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"]["index"][:]])
        cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["__categories"]["clusters"][:]]
        clu_codes = f["obs"]["clusters"][:]
        X = f["X"]
        if isinstance(X, h5py.Group):                       # CSR
            ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
            E = np.zeros((len(cell_idx), len(genes)), dtype=np.float32)
            for r, i in enumerate(cell_idx):
                s, e = ip[i], ip[i + 1]
                E[r, ind[s:e]] = dat[s:e]
        else:                                               # dense
            order = np.argsort(cell_idx)
            E = np.zeros((len(cell_idx), len(genes)), dtype=np.float32)
            E[order] = X[np.sort(cell_idx), :]
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    E = np.log1p(E / tot * 1e4)
    clusters = np.array([cats[c] for c in clu_codes])[cell_idx]
    return E, genes, clusters


def panel_index(genes, tissue):
    """Map the tissue's marker panel onto an array of gene symbols; drop programs with no genes present."""
    gsym = {s: i for i, s in enumerate(genes)}
    pi = {k: np.array([gsym[s] for s in v if s in gsym]) for k, v in PANEL[tissue].items()}
    return {k: v for k, v in pi.items() if len(v)}


def fisher_combine(pvals):
    """Fisher's method over independent p-values -> (chi2, df, combined p)."""
    from scipy.stats import chi2
    pv = np.asarray([max(p, 1e-12) for p in pvals], float)
    stat = float(-2.0 * np.log(pv).sum())
    df = 2 * len(pv)
    return stat, df, float(chi2.sf(stat, df))
