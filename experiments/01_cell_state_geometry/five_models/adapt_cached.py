"""Pair EXISTING cached K562 embeddings with a corrected cell-cycle phase, in the cc schema.

Some models already have K562 cell-cycle embeddings on disk from earlier routes -- there is no reason to re-run
the model. This adapter reproduces the exact cell selection each cache was built on, recomputes phi with the
corrected preprocessing (cc_common.to_lognorm -- the double-normalisation bug that corrupts the ruler is fixed
here), and writes data/cellcycle/<model>_k562.npz so cc_geometry.py / cc_steering.py run unchanged.

Cross-model note: the cells need NOT be identical across models. The claim under test -- "a fixed direction
cannot steer a loop, a rotating one can" -- is per-model geometry; each model is judged on its own cells. Using
each model's own cached cells is therefore valid and avoids a needless re-extraction.

GENEFORMER: data/route_manifold/geneformer_percell_L11.npy is (2000, 1152), built by route_geometry on the
Replogle K562 non-targeting controls selected as (build_rulers.py:84-85):
    np.random.seed(42); ctrl = where(gene == 'non-targeting'); sel = np.sort(np.random.choice(ctrl, 2000, False))
Row i of the cache == sel[i]. Geneformer's tokenizer is rank-based, so the embeddings are invariant to the
expression-normalisation bug; only phi needs recomputing.

Run:  ../../.venv_state/bin/python adapt_cached.py geneformer
Out:  data/cellcycle/<model>_k562.npz  (emb, phi, s_score, g2m_score, cc_phase, cell_idx)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import (REPLOGLE, DATA, _load_categorical, to_lognorm, _score, phase_angle, circ_mean,
                       build_substrate, S_GENES, G2M_GENES)  # noqa: E402

ROOT = f"{_DATA}"

# For each cache: either an .npy of embeddings whose rows follow a reproducible RandomState draw ('reproduce'),
# or an .npz that already carries its own cell_idx ('npz' + field names). Either way we recompute phi with the
# corrected preprocessing on the cache's OWN cells.
CACHES = {
    # route_geometry's Geneformer L11 on the seed-42 2000-cell draw; .npy has no cell_idx, so reproduce it.
    "geneformer": dict(kind="reproduce", emb="data/route_manifold/geneformer_percell_L11.npy", n=2000, seed=42),
    # route_maxtoki's MaxToki-217M L8; the npz carries cell_idx, so use it directly.
    "maxtoki": dict(kind="npz", npz="data/maxtoki_acts/maxtoki_k562.npz", emb_field="raw_mean"),
    # STATE-SE L11, freshly extracted on the substrate cells. NB: extract_state.py saves LOCAL h5ad row
    # numbers (0..2999) as cell_idx, not replogle indices -- but the embeddings are in substrate ORDER (the
    # h5ad was built from the substrate in order, extraction preserves row order), so pair by POSITION with the
    # already-correct substrate phi rather than by the (bogus) saved cell_idx.
    "statecc": dict(kind="ordered", npz="data/cellcycle/state_raw_k562.npz", emb_field="emb"),
    # UCE-100M L2, from route_uce/extract_percell.py on replogle_k562_subset.h5ad. Like STATE it stores LOCAL
    # subset rows (0..2999) in cell_idx rather than replogle indices -- but UNLIKE STATE its cells are NOT the
    # substrate: only 1000 are non-targeting, the other 2000 are CRISPRi perturbations. So the 'ordered' trick
    # (pair by position with the substrate) would attach the wrong phase to every cell while looking fine. The
    # subset is indexed by cell_barcode, unique in replogle, so recover the true replogle rows by joining on it
    # and score phi on UCE's OWN cells.
    "ucecc": dict(kind="barcode", npz="data/cellcycle/uce_k562.npz", emb_field="emb",
                  subset="data/state_activations/replogle_k562_subset.h5ad"),
    # Parity check for the row above: the other four models are scored on non-targeting CONTROLS only, so
    # restrict UCE to its 1000 controls and re-score phi within that subset. Guards the worry that UCE's
    # perturbed cells (not its representation) drive whatever the full-panel row shows.
    "ucecc_ctrl": dict(kind="barcode", npz="data/cellcycle/uce_k562.npz", emb_field="emb",
                       subset="data/state_activations/replogle_k562_subset.h5ad", ctrl_only=True),
}


def subset_rows_to_replogle(local_rows, subset_rel):
    """LOCAL h5ad row numbers in a replogle subset -> true replogle row indices, joined on cell_barcode."""
    with h5py.File(os.path.join(ROOT, subset_rel), "r") as f:
        bc = np.asarray(f["obs"]["cell_barcode"]).astype(str)[local_rows]
    with h5py.File(REPLOGLE, "r") as f:
        rep = np.asarray(f["obs"]["cell_barcode"]).astype(str)
    pos = {b: i for i, b in enumerate(rep)}
    missing = [b for b in bc if b not in pos]
    assert not missing, f"{len(missing)}/{len(bc)} subset barcodes absent from replogle"
    return np.array([pos[b] for b in bc], dtype=int)


def reproduce_ctrl_selection(n, seed):
    with h5py.File(REPLOGLE, "r") as f:
        cell_gene = _load_categorical(f["obs"], "gene")
    ctrl = np.where(np.isin(cell_gene, ["non-targeting", "Non-targeting", "non_targeting"]))[0]
    np.random.seed(seed)                                      # exactly build_rulers.py's np.random.seed(42)
    return np.sort(np.random.choice(ctrl, n, replace=False))


def build(model):
    spec = CACHES[model]
    if spec["kind"] == "ordered":
        # embeddings are in substrate row order; take phi/scores/cell_idx straight from the substrate cache
        emb = np.load(os.path.join(ROOT, spec["npz"]), allow_pickle=True)[spec["emb_field"]].astype(np.float32)
        sub = build_substrate()
        assert emb.shape[0] == len(sub["cell_idx"]), f"{emb.shape[0]} embs vs {len(sub['cell_idx'])} substrate"
        out = os.path.join(DATA, f"{model}_k562.npz")
        np.savez(out, emb=emb, phi=sub["phi"], s_score=sub["s_score"], g2m_score=sub["g2m_score"],
                 cc_phase=sub["cc_phase"].astype(str), cell_idx=sub["cell_idx"])
        cc_phase = sub["cc_phase"]; ns, ng = int(sub["n_s_markers"]), int(sub["n_g2m_markers"])
        phi = sub["phi"]
        from collections import Counter
        print(f"[{model}] {emb.shape[0]} cells x {emb.shape[1]}-d (paired by ORDER with substrate) | "
              f"S {ns}/{len(S_GENES)}, G2M {ng}/{len(G2M_GENES)} | phase {dict(Counter(cc_phase))}")
        for p in ("G1", "S", "G2M"):
            m = cc_phase == p
            if m.sum():
                print(f"    {p:>4}: n={m.sum():4d}  circ-mean phi = {np.degrees(circ_mean(phi[m])):+7.1f} deg")
        print(f"    -> {out}")
        return

    if spec["kind"] == "reproduce":
        emb = np.load(os.path.join(ROOT, spec["emb"])).astype(np.float32)
        assert emb.shape[0] == spec["n"], f"cache has {emb.shape[0]} rows, expected {spec['n']}"
        sel = reproduce_ctrl_selection(spec["n"], spec["seed"])
    elif spec["kind"] == "barcode":
        z = np.load(os.path.join(ROOT, spec["npz"]), allow_pickle=True)
        emb = z[spec["emb_field"]].astype(np.float32)
        sel = subset_rows_to_replogle(z["cell_idx"].astype(int), spec["subset"])
        if spec.get("ctrl_only"):
            with h5py.File(REPLOGLE, "r") as f:
                g = _load_categorical(f["obs"], "gene").astype(str)[sel]
            keep = np.isin(g, ["non-targeting", "Non-targeting", "non_targeting"])
            emb, sel = emb[keep], sel[keep]
            print(f"[{model}] ctrl_only: kept {int(keep.sum())}/{len(keep)} non-targeting cells")
    else:
        z = np.load(os.path.join(ROOT, spec["npz"]), allow_pickle=True)
        emb = z[spec["emb_field"]].astype(np.float32)
        sel = z["cell_idx"].astype(int)
    assert len(sel) == emb.shape[0]

    with h5py.File(REPLOGLE, "r") as f:
        genes = _load_categorical(f["var"], "gene_name_index").astype(str)
        X = np.stack([np.asarray(f["X"][int(i), :], dtype=np.float32) for i in sel])
    Xlog = to_lognorm(X)
    gu = np.char.upper(genes)
    s_score, ns = _score(Xlog, gu, S_GENES)
    g2m_score, ng = _score(Xlog, gu, G2M_GENES)
    cc_phase = np.where(np.maximum(s_score, g2m_score) < 0, "G1",
                        np.where(s_score >= g2m_score, "S", "G2M"))
    phi = phase_angle(Xlog, gu, cc_phase)

    out = os.path.join(DATA, f"{model}_k562.npz")
    np.savez(out, emb=emb, phi=phi, s_score=s_score, g2m_score=g2m_score,
             cc_phase=cc_phase.astype(str), cell_idx=sel)
    from collections import Counter
    print(f"[{model}] {emb.shape[0]} cells x {emb.shape[1]}-d | S markers {ns}/{len(S_GENES)}, "
          f"G2M {ng}/{len(G2M_GENES)} | phase {dict(Counter(cc_phase))}")
    for p in ("G1", "S", "G2M"):
        m = cc_phase == p
        if m.sum():
            print(f"    {p:>4}: n={m.sum():4d}  circ-mean phi = {np.degrees(circ_mean(phi[m])):+7.1f} deg")
    print(f"    -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "geneformer")
