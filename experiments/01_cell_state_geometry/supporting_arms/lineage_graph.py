"""Curated hematopoietic lineage graph for the Tabula-Sapiens immune cells cached at scGPT L11.

The immune tissue slice (1000 cells) of route_geometry/results/scgpt_L11_ruler.npz spans the
hematopoietic differentiation TREE. We map each ontology `cell_type` string to:
  - a terminal BRANCH label (the concept for supervised probing / erasure), and
  - a coarse lineage MAJOR axis (myeloid vs lymphoid vs erythroid vs stem),
  - a nominal tree DEPTH (stem=0, progenitor=1, mature=2) used only for description.

Branching structure (source hematopoietic-manifold paper's nonlinearity was exactly this tree):

    HSC (stem) ─┬─ CMP ─┬─ granulocyte/monocyte ─> MYELOID  (neutrophil, macrophage, monocyte, mast, ...)
                │       └─ MEP ─────────────────> ERYTHROID (erythrocyte, erythroid progenitor, platelet)
                └─ CLP ─┬───────────────────────> T   (CD4/CD8/Treg/gd/NKT/thymocyte)
                        ├───────────────────────> B   (B cell, plasma cell)
                        └───────────────────────> NK  (natural killer, innate lymphoid cell)

Read-only consumer of route_geometry; defines no numbers that aren't ontology facts.
"""
import numpy as np

# terminal-branch assignment by substring match (lowercased). Order matters: first hit wins.
# Each rule: (substring, branch, major, depth)
_RULES = [
    # --- stem / multipotent progenitors (branch point; under-sampled, kept separate) ---
    ("hematopoietic stem cell", "stem", "stem", 0),
    ("hematopoietic precursor", "stem", "stem", 1),
    ("common myeloid progenitor", "stem", "stem", 1),
    ("hematopoietic cell", "stem", "stem", 0),
    # --- erythroid (MEP sub-branch of myeloid) ---
    ("erythroid progenitor", "erythroid", "erythroid", 1),
    ("erythrocyte", "erythroid", "erythroid", 2),
    ("platelet", "erythroid", "erythroid", 2),
    ("megakaryocyte", "erythroid", "erythroid", 2),
    # --- NK / innate lymphoid (before T, so NKT stays T; ILC->NK) ---
    ("natural killer cell", "NK", "lymphoid", 2),
    ("innate lymphoid cell", "NK", "lymphoid", 2),
    # --- B lineage ---
    ("plasma cell", "B", "lymphoid", 2),
    ("plasmablast", "B", "lymphoid", 2),
    ("b cell", "B", "lymphoid", 2),
    # --- T lineage (NKT and thymocytes are T-lineage) ---
    ("nk t", "T", "lymphoid", 2),
    ("thymocyte", "T", "lymphoid", 1),
    ("t cell", "T", "lymphoid", 2),
    ("regulatory t", "T", "lymphoid", 2),
    ("gamma-delta t", "T", "lymphoid", 2),
    # --- myeloid: monocyte / macrophage / granulocyte / DC ---
    ("neutrophil", "myeloid", "myeloid", 2),
    ("granulocyte", "myeloid", "myeloid", 2),
    ("basophil", "myeloid", "myeloid", 2),
    ("mast cell", "myeloid", "myeloid", 2),
    ("monocyte", "myeloid", "myeloid", 2),
    ("macrophage", "myeloid", "myeloid", 2),
    ("mononuclear phagocyte", "myeloid", "myeloid", 2),
    ("dendritic cell", "myeloid", "myeloid", 2),
    ("langerhans cell", "myeloid", "myeloid", 2),
    ("microglial cell", "myeloid", "myeloid", 2),
    ("myeloid", "myeloid", "myeloid", 2),   # catch-all myeloid leukocyte / myeloid cell
]

# broad terms with no lineage commitment -> unassigned (dropped from lineage tests)
_UNASSIGNED = {"leukocyte"}


def assign(cell_types):
    """Return (branch, major, depth) arrays; branch=='' for unassigned / non-hematopoietic."""
    ct = np.asarray(cell_types)
    branch = np.array([""] * len(ct), dtype=object)
    major = np.array([""] * len(ct), dtype=object)
    depth = np.full(len(ct), -1, dtype=int)
    for i, name in enumerate(ct):
        s = str(name).lower()
        if s in _UNASSIGNED:
            continue
        for sub, br, mj, dp in _RULES:
            if sub in s:
                branch[i] = br; major[i] = mj; depth[i] = dp
                break
    return branch, major, depth


if __name__ == "__main__":
    import os
    RG = os.path.join(os.path.dirname(__file__), "..", "route_geometry", "results", "scgpt_L11_ruler.npz")
    d = np.load(RG, allow_pickle=True)
    ct, tis = d["cell_type"], d["tissue"]
    m = tis == "immune"
    br, mj, dp = assign(ct[m])
    from collections import Counter
    print("BRANCH:", dict(Counter(br[br != ""])))
    print("MAJOR :", dict(Counter(mj[mj != ""])))
    unassigned = sorted(set(str(x) for x in ct[m][br == ""]))
    print("unassigned:", unassigned)
    print("total immune", int(m.sum()), "assigned", int((br != "").sum()))
