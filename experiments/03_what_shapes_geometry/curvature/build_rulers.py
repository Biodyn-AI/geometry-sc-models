"""Tier-A ruler builder for the non-regulatory-geometry test (direction #2).

Builds the *biological ruler* labels for both substrates, matched to the cell-index order used by the
cached L11 activations (see route_b/config.py, bio-sae extraction). No activation load here — cheap linchpin.

Substrates:
  scGPT  L11  -> cell-TYPE / lineage ruler   (Tabula Sapiens immune+kidney+lung, 3000 cells, 63 types)
  Geneformer L11 -> cell-CYCLE ruler          (Replogle K562 non-targeting controls, 2000 cells)

Cell-index alignment (verified against bio-sae extraction code):
  - scGPT: activation cell index i == i-th entry of `cell_data` in
      <BASE>/phase3_multitissue/ts_activations/extraction_metadata.json  (fields: tissue, cell_type, cell_idx)
  - Geneformer: activation cell index i == i-th element of the *sorted* array
      np.random.seed(42); ctrl = where(obs['gene'] == 'non-targeting'); np.random.choice(ctrl, 2000, F).sort()
    over <PAPER>/.../replogle_concat.h5ad  (dense /X, obs['gene'], var['gene_name_index'])

Outputs (route_geometry/results/):
  scgpt_L11_ruler.npz       : cell_type[3000], tissue[3000]
  geneformer_L11_ruler.npz  : cell_idx[2000] (source rows), s_score[2000], g2m_score[2000], cc_phase[2000]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json
import os
import numpy as np

BASE = f"{_DATA}/biodyn-work/subproject_42_sparse_autoencoder_biological_map/experiments"
REPLOGLE = f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
OUT = os.path.join(os.path.dirname(__file__), "results")

# Tirosh et al. 2016 cell-cycle gene sets (standard S / G2M markers).
S_GENES = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL","PRIM1",
    "UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3",
    "MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN",
    "POLA1","CHAF1B","BRIP1","E2F8"]
G2M_GENES = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B","MKI67",
    "TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B",
    "GTSE1","KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2",
    "CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3",
    "CBX5","CENPA"]


def load_categorical_column(h5group, col_name):
    import h5py
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        cats = col['categories'][:]
        codes = col['codes'][:]
        if cats.dtype.kind in ('O', 'S'):
            cats = np.array([x.decode() if isinstance(x, bytes) else x for x in cats])
        return cats[codes]
    data = col[:]
    if data.dtype.kind in ('O', 'S'):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
    return data


def build_scgpt_ruler():
    meta = os.path.join(BASE, "phase3_multitissue/ts_activations/extraction_metadata.json")
    cd = json.load(open(meta))["cell_data"]          # order == activation cell index
    cell_type = np.array([c["cell_type"] for c in cd])
    tissue = np.array([c["tissue"] for c in cd])
    np.savez(os.path.join(OUT, "scgpt_L11_ruler.npz"), cell_type=cell_type, tissue=tissue)
    print(f"[scGPT] {len(cd)} cells | tissues={dict(zip(*np.unique(tissue, return_counts=True)))} | "
          f"{len(np.unique(cell_type))} cell types")
    return cell_type, tissue


def _score(Xlog, var_upper, gene_set):
    """Mean z-scored expression over the marker genes present (Tirosh-style, simplified)."""
    idx = [np.where(var_upper == g)[0][0] for g in gene_set if g in set(var_upper)]
    if not idx:
        return np.zeros(Xlog.shape[0]), 0
    sub = Xlog[:, idx]
    z = (sub - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-8)
    return z.mean(1), len(idx)


def build_geneformer_ruler():
    import h5py
    with h5py.File(REPLOGLE, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_total = f['X'].shape[0]
        ctrl = np.where(np.isin(cell_genes, ['non-targeting', 'Non-targeting', 'non_targeting']))[0]
        np.random.seed(42)
        sel = np.sort(np.random.choice(ctrl, 2000, replace=False)) if len(ctrl) > 2000 else np.sort(ctrl)
        # load the 2000 selected rows (dense X), mirroring the extraction's per-row read
        X = np.empty((len(sel), f['X'].shape[1]), dtype=np.float32)
        for i, r in enumerate(sel):
            X[i] = f['X'][int(r), :]
    # library-size normalize + log1p (same as extraction), then z-score marker genes
    rs = X.sum(1, keepdims=True); rs[rs == 0] = 1
    Xlog = np.log1p(X / rs * 1e4)
    var_upper = np.char.upper(var_genes.astype(str))
    s_score, ns = _score(Xlog, var_upper, S_GENES)
    g2m_score, ng2 = _score(Xlog, var_upper, G2M_GENES)
    phase = np.where(np.maximum(s_score, g2m_score) < 0, "G1",
                     np.where(s_score >= g2m_score, "S", "G2M"))
    np.savez(os.path.join(OUT, "geneformer_L11_ruler.npz"),
             cell_idx=sel, s_score=s_score, g2m_score=g2m_score, cc_phase=phase)
    from collections import Counter
    print(f"[Geneformer] {len(sel)} ctrl cells | S markers found {ns}/{len(S_GENES)}, "
          f"G2M {ng2}/{len(G2M_GENES)} | phases={dict(Counter(phase))} | "
          f"S range [{s_score.min():.2f},{s_score.max():.2f}]")
    return sel, s_score, g2m_score, phase


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_scgpt_ruler()
    build_geneformer_ruler()
    print("rulers written to", OUT)
