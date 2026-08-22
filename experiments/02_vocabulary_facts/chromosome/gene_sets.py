"""gene_sets — the hand-crafted hypotheses for the gene-token manifold hunt.

Each entry is one falsifiable geometric claim about a specific, biologically-motivated gene set, with an
EXTERNAL coordinate that does not come from expression data. See gm_lib.py for the design principle; the short
version is the selection rule:

    prefer structures where CO-EXPRESSION predicts a BLOB but the ANNOTATION predicts an ORDER.

Enzymes in a linear pathway are co-regulated as a block -> co-expression gives a featureless cluster with no
internal ordering, while KEGG assigns each an exact step index. That gap is where a model can actually win.
By the same rule the obvious hypothesis (cell-cycle genes form a circle ordered by peak phase) is the WEAKEST,
because peak phase IS the co-expression structure -- so it is carried as the POSITIVE CONTROL, not a headline.

Schema per hypothesis:
  kind   : "order" (1-D ordered curve) | "circle" | "grid" (2-D) | "antipodal" | "blob" (negative control)
  genes  : upper HGNC symbols
  coord  : the external coordinate, aligned to `genes`
           order -> step/stage index ; circle -> phase in RADIANS ; grid -> (coord1, coord2)
  source : where the coordinate comes from (must not be expression)
  note   : the honest confound/risk statement
"""
import numpy as np

H = {}


def _add(name, kind, genes, coord, source, note, role="test"):
    H[name] = dict(name=name, kind=kind, genes=[g.upper() for g in genes], coord=coord,
                   source=source, note=note, role=role)


# ---------------------------------------------------------------- TIER 1: annotation ⊥ co-expression
# Linear metabolic pathways. Co-expression -> block/blob. KEGG -> exact step order.

_add("glycolysis", "order",
     ["HK1", "HK2", "HK3", "GCK", "GPI", "PFKL", "PFKM", "PFKP", "ALDOA", "ALDOB", "ALDOC", "TPI1",
      "GAPDH", "PGK1", "PGK2", "PGAM1", "PGAM2", "ENO1", "ENO2", "ENO3", "PKM", "PKLR"],
     [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10],
     "KEGG hsa00010 step order (glucose -> pyruvate)",
     "Isozymes share a step index (ties). Co-expression cannot order steps: glycolytic enzymes rise and fall "
     "together as one program. Sequence control matters for isozyme families (HK1/2/3 are paralogs).")

_add("heme_biosynth", "order",
     ["ALAS1", "ALAS2", "ALAD", "HMBS", "UROS", "UROD", "CPOX", "PPOX", "FECH"],
     [1, 1, 2, 3, 4, 5, 6, 7, 8],
     "KEGG hsa00860 / textbook heme pathway step order (succinyl-CoA+glycine -> heme)",
     "The cleanest textbook 8-step chain; erythroid cells express the whole chain co-ordinately, so "
     "co-expression predicts a tight blob with no internal order. Small n (9) is the main power risk.")

_add("tca", "order",
     ["CS", "ACO2", "IDH2", "IDH3A", "IDH3B", "OGDH", "DLST", "SUCLA2", "SUCLG1", "SDHA", "SDHB", "SDHC",
      "SDHD", "FH", "MDH2"],
     [1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 8],
     "KEGG hsa00020 TCA cycle step order",
     "NB the TCA is a CYCLE -- we test it as an ORDER (one turn) because the model has no reason to close it; "
     "a circle here would be a striking bonus. SDH subunits are one complex (ties).")

_add("urea_cycle", "order",
     ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS"],
     [1, 2, 3, 4, 5, 0],
     "KEGG hsa00220 urea cycle step order (NAGS = activator, step 0)",
     "Very small (6). Liver-restricted, so atlas coverage (kidney/lung/immune) is thin -> embeddings may be "
     "undertrained. Included because the order is textbook-exact.")

_add("secretory_pathway", "order",
     # ER -> ERGIC -> cis-Golgi -> trans-Golgi -> secretory vesicle -> plasma membrane
     ["SEC61A1", "SEC61B", "SSR1", "CALR", "CANX", "PDIA3", "HSPA5",          # 1 ER
      "SEC23A", "SEC24A", "SEC13", "SEC31A", "LMAN1", "SURF4",                # 2 ER exit / ERGIC
      "GOLGA2", "GOLPH3", "GOSR2", "STX5",                                     # 3 cis-Golgi
      "GOLGB1", "B4GALT1", "ST6GAL1", "TGOLN2",                                # 4 trans-Golgi
      "VAMP3", "VAMP8", "RAB27A", "SCAMP1",                                    # 5 secretory vesicle
      "STX4", "SNAP23", "VAMP2"],                                              # 6 plasma membrane fusion
     [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
     "GO-CC compartment order along the secretory route (ER->ERGIC->cis/trans-Golgi->vesicle->PM)",
     "MY TOP CANDIDATE. The order is a SPATIAL/transit fact, orthogonal to co-expression (secretory machinery "
     "co-expresses as a block). And there is prior evidence of signal: weightpapers/01 found a GO-CC secretory "
     "co-pole in W_E, and 08 found GO-CC is the ONE category early layers ENHANCE (L1 z10.9 > W_E 6.7).")

_add("complement_cascade", "order",
     ["C1QA", "C1QB", "C1QC", "C1R", "C1S", "C4A", "C4B", "C2", "C3", "C5", "C6", "C7", "C8A", "C8B",
      "C8G", "C9"],
     [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6],
     "Classical complement activation cascade order (C1 -> C4/C2 -> C3 -> C5 -> C6-C9 MAC)",
     "Cascade order is a proteolytic-activation fact, not an expression gradient. C1Q subunits and C8 subunits "
     "are complexes (ties). Liver-biased expression is the coverage risk.")

_add("coagulation_cascade", "order",
     ["F12", "F11", "F9", "F8", "F10", "F5", "F2", "FGA", "FGB", "FGG", "F13A1"],
     [1, 2, 3, 3, 4, 4, 5, 6, 6, 6, 7],
     "Intrinsic coagulation cascade order (F12 -> F11 -> F9/F8 -> F10/F5 -> thrombin -> fibrinogen -> F13)",
     "Same logic as complement. Liver-restricted -> coverage risk in a kidney/lung/immune atlas.")

# ---------------------------------------------------------------- TIER 1: 2-D grid with hard ground truth
_HOX = [(f"HOX{c}{n}", c, n) for c, ns in
        [("A", [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]), ("B", [1, 2, 3, 4, 5, 6, 7, 8, 9, 13]),
         ("C", [4, 5, 6, 8, 9, 10, 11, 12, 13]), ("D", [1, 3, 4, 8, 9, 10, 11, 12, 13])]
        for n in ns]
_add("hox_grid", "grid",
     [g for g, _, _ in _HOX],
     ([n for _, _, n in _HOX], [ord(c) - 65 for _, c, _ in _HOX]),
     "HGNC HOX nomenclature: paralog number 1..13 (= anterior-posterior axis by spatial colinearity) x "
     "cluster A/B/C/D (= chromosome). Both coordinates are sequence/position facts, not expression.",
     "Paralog number maps to the A-P body axis by colinearity -- a developmental/spatial fact, not an "
     "expression gradient in adult tissue. SEQUENCE CONFOUND IS SEVERE: paralogs share homeodomains, so ESM2 "
     "will order them; the esm2 basis is the essential control here. HOX is lowly expressed in "
     "kidney/lung/immune -> undertrained-embedding risk; check token counts before believing a null.")

# ---------------------------------------------------------------- TIER 1: antipodal axes
# (gene_a, gene_b, axis_genes, axis_sign)  -- sign +1 = pole of gene_a, -1 = pole of gene_b
ANTIPODAL = {
    "gata1_spi1": dict(
        a="GATA1", b="SPI1",
        axis_genes=["KLF1", "GATA2", "EPOR", "HBB", "SLC4A1", "AHSP", "ALAS2",     # erythroid (GATA1 pole)
                    "CEBPA", "CSF1R", "ITGAM", "LYZ", "MPO", "ELANE", "FCGR3A"],   # myeloid   (SPI1 pole)
        axis_sign=[1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
        note="The textbook bistable switch (mutual repression, erythroid vs myeloid)."),
    "tbx21_gata3": dict(
        a="TBX21", b="GATA3",
        axis_genes=["IFNG", "STAT4", "IL12RB2", "CXCR3",                            # Th1 (TBX21)
                    "IL4", "IL5", "IL13", "STAT6", "CCR4"],                         # Th2 (GATA3)
        axis_sign=[1, 1, 1, 1, -1, -1, -1, -1, -1],
        note="Th1/Th2 master regulators, mutually antagonistic."),
    "rorc_foxp3": dict(
        a="RORC", b="FOXP3",
        axis_genes=["IL17A", "IL17F", "IL23R", "CCR6", "STAT3",                     # Th17 (RORC)
                    "IL2RA", "CTLA4", "IKZF2", "TNFRSF18"],                         # Treg (FOXP3)
        axis_sign=[1, 1, 1, 1, 1, -1, -1, -1, -1],
        note="Th17/Treg reciprocal fate switch."),
    "pax5_prdm1": dict(
        a="PAX5", b="PRDM1",
        axis_genes=["CD19", "MS4A1", "CD79A", "CD79B", "BACH2",                     # B cell (PAX5)
                    "XBP1", "IRF4", "SDC1", "MZB1", "DERL3"],                       # plasma (PRDM1)
        axis_sign=[1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        note="PAX5 and PRDM1/BLIMP1 mutually repress: B-cell identity vs plasma-cell differentiation."),
    "sox2_cdx2": dict(
        a="SOX2", b="CDX2",
        axis_genes=["POU5F1", "NANOG", "LIN28A", "CDH1", "GATA3", "KRT18", "TFAP2C"],
        axis_sign=[1, 1, 1, 1, -1, -1, -1],
        note="ICM/pluripotency vs trophectoderm. Coverage risk: not expressed in adult atlases."),
}
for k, v in ANTIPODAL.items():
    _add(f"antipodal_{k}", "antipodal", [v["a"], v["b"]] + v["axis_genes"], None,
         "textbook mutual-repression pairs; axis membership from lineage marker biology", v["note"])
    H[f"antipodal_{k}"].update({q: v[q] for q in ("a", "b", "axis_genes", "axis_sign")})

# ---------------------------------------------------------------- TIER 2: strong truth, heavier confound
_CC = {  # Cyclebase / Whitfield 2002 canonical peak phase; degrees around the cycle from G1/S = 0
    "CCNE1": 0, "CCNE2": 0, "E2F1": 10, "PCNA": 20, "CDC6": 15, "CDT1": 15, "MCM2": 10, "MCM3": 10,
    "MCM4": 10, "MCM5": 10, "MCM6": 10, "MCM7": 10, "SLBP": 30, "RRM2": 45, "TYMS": 45, "DHFR": 40,
    "CDC45": 35, "GINS2": 40, "CLSPN": 35, "FEN1": 50, "RFC4": 45, "POLA1": 40,          # G1/S -> S
    "CCNA2": 120, "TOP2A": 170, "NUSAP1": 175, "UBE2C": 200, "BIRC5": 205, "TPX2": 190,
    "CENPF": 185, "CENPE": 200, "CDK1": 180, "CCNB1": 210, "CCNB2": 210, "PLK1": 205,
    "BUB1": 195, "BUB1B": 195, "AURKA": 205, "AURKB": 215, "KIF11": 185, "KIF23": 210,
    "ANLN": 200, "ECT2": 195, "CDC20": 220, "PTTG1": 215, "CKS2": 190, "AURKAIP1": 200,  # G2 -> M
}
_add("cellcycle_circle", "circle", list(_CC), np.deg2rad(np.array(list(_CC.values()), float)),
     "Cyclebase / Whitfield 2002 canonical peak phase (degrees), G1/S origin",
     "POSITIVE CONTROL, NOT a headline. The direct days-of-the-week analogue and the strongest prior (the "
     "model certainly sees cell-cycle variation) -- but MAXIMALLY confounded: peak phase IS the co-expression "
     "structure, exactly the degeneracy that made the cell-cycle CELL manifold a dead testbed "
     "(route_cellcycle/RESULTS.md 11). Its job is to tell us the instrument has a pulse: if no basis shows a "
     "circle here, the whole gene-manifold programme is probably dead.",
     role="positive_control")

_add("globin_switch", "order",
     ["HBZ", "HBE1", "HBG1", "HBG2", "HBA1", "HBA2", "HBD", "HBB"],
     [1, 1, 2, 2, 3, 3, 3, 3],
     "developmental globin switching order (embryonic HBZ/HBE1 -> fetal HBG -> adult HBA/HBD/HBB)",
     "Fetal/embryonic globins are essentially ABSENT from adult atlases, so co-expression in the pretraining "
     "corpus cannot easily supply the order -- if a basis still orders them, it is sequence/paralogy (ESM2 "
     "control decides). Small n (8).")

# ---------------------------------------------------------------- TIER 3: negative / confound controls
_add("ribosomal_blob", "blob",
     ["RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL8", "RPL9", "RPL10", "RPL11", "RPL13", "RPL15",
      "RPS2", "RPS3", "RPS4X", "RPS5", "RPS6", "RPS7", "RPS8", "RPS9", "RPS11", "RPS13", "RPS15"],
     [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15],
     "the numeral in the HGNC symbol -- an arbitrary naming index with NO biological order",
     "NEGATIVE CONTROL. Ribosomal proteins are the most perfectly co-expressed set in the genome, so "
     "co-expression is maximal while the annotation (the name numeral) is meaningless. Any basis that "
     "'recovers' this order is reporting an artifact, and calibrates our false-positive rate.",
     role="negative_control")

_add("histone_blob", "blob",
     ["H1-2", "H1-3", "H1-4", "H2AC11", "H2AC12", "H2AC13", "H2BC11", "H2BC12", "H3C1", "H3C2",
      "H4C1", "H4C2", "H4C3"],
     [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5],
     "histone family index (H1/H2A/H2B/H3/H4) -- a stoichiometric family label with no transit order",
     "NEGATIVE CONTROL / naming check. Replication-dependent histones are tightly co-expressed and share a "
     "cluster; family is real but is NOT an ordered coordinate. Symbols follow the 2019 HGNC renaming and may "
     "miss older vocabularies -- a coverage probe as much as a control.",
     role="negative_control")


def by_kind(kind):
    return {k: v for k, v in H.items() if v["kind"] == kind}


def summary():
    for k, v in H.items():
        print(f"  {k:<24} {v['kind']:<9} n={len(v['genes']):<3} role={v['role']}")


if __name__ == "__main__":
    summary()
