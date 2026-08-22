"""HYPOTHESIS REGISTRY for the steerable-directions research program.

Each hypothesis maps GENE SYMBOL -> LABEL, plus a kind:
    "categorical"  multi-class (chromosome, cell-cycle phase, ...)      -> balanced accuracy
    "binary"       one-vs-rest membership (a GO term, ribosomal, ...)   -> ROC-AUC
    "continuous"   a real-valued gene property (abundance, position)    -> Spearman rho

Every hypothesis carries `expect`, our prior about the outcome, so the screen can be CALIBRATED: a screen that
passes everything is broken. Deliberate expected-negatives are included (family-driven properties that the
protein-sequence control should win; expression-derived properties the co-expression control should win).

All labels come from data already on this machine (GO, DoRothEA/TRRUST, UCE coordinates, expression panels).
Add a hypothesis by writing a loader returning (labels, kind, meta) and registering it in REGISTRY.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES

GENE2GO = f"{_DATA}/perturb/gene2go_all.pkl"
DOROTHEA = f"{_DATA}/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv"
MIN_PER_CLASS = 40
_cache = {}


def _go():
    if "go" not in _cache:
        d = pickle.load(open(GENE2GO, "rb"))
        _cache["go"] = {str(k).upper(): set(v) for k, v in d.items() if isinstance(v, (set, list, tuple))}
    return _cache["go"]


def _panel():
    """co-expression panel -> per-gene breadth (fraction of cells expressing) and mean abundance."""
    if "panel" not in _cache:
        M, syms = G.basis("coexpr")            # rows = genes, cols = cells
        _cache["panel"] = (np.asarray(M), np.asarray(syms))
    return _cache["panel"]


# ---------------------------------------------------------------- hypothesis loaders
def h_chromosome():
    C = coords()
    lab = {s: C.chromosome[s] for s in C.index if C.chromosome[s] in AUTOSOMES}
    return lab, "categorical", dict(
        source="UCE species_chrom", expect="POSITIVE (validated: model-specific vs NORMAL panels)",
        unfair_refs=["coexpr_k562"],
        unfair_reason="K562 is an aneuploid cancer line: subclonal copy-number variation makes same-chromosome "
                      "genes co-vary across cells (verified: same-chr correlation excess +0.0074 vs +0.0003 in "
                      "normal panels), so it decodes chromosome (0.585) through DNA DOSAGE, a different physical "
                      "mechanism from the co-regulation this hypothesis tests. Reported, not gated on. NB this "
                      "is itself a live alternative mechanism for the finding -- see memory cnv-alternative-mechanism.")


def h_position():
    C = coords()
    lab = {s: float(C.start[s]) for s in C.index if C.chromosome[s] in AUTOSOMES}
    return lab, "continuous", dict(source="UCE species_chrom", group_by="chromosome",
                                   unfair_refs=["coexpr_k562"],
                                   unfair_reason="same copy-number confound as h_chromosome",
                                   expect="POSITIVE (validated: model-specific, scored within chromosome)")


def h_expr_breadth():
    M, syms = _panel()
    breadth = (M > 0).mean(1)
    return {s: float(b) for s, b in zip(syms, breadth)}, "continuous", dict(
        source="co-expression panel", expect="NEGATIVE for model-specificity — label is derived FROM the "
        "co-expression panel, so the co-expression baseline should win by construction (calibration)")


def h_expr_abundance():
    M, syms = _panel()
    mu = M.mean(1)
    return {s: float(v) for s, v in zip(syms, mu)}, "continuous", dict(
        source="co-expression panel", expect="NEGATIVE for model-specificity (same circularity as breadth)")


def h_tf_role():
    """transcription factor vs regulated target vs neither (DoRothEA)."""
    tfs, tgts = set(), set()
    with open(DOROTHEA) as f:
        head = f.readline().rstrip("\n").split("\t")
        it, ig = (head.index("tf") if "tf" in head else 0), (head.index("target") if "target" in head else 2)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > max(it, ig):
                tfs.add(p[it].upper()); tgts.add(p[ig].upper())
    lab = {}
    for g in tfs: lab[g] = "TF"
    for g in tgts - tfs: lab[g] = "target"
    return lab, "categorical", dict(source="DoRothEA", expect="uncertain — regulatory role may be sequence "
                                    "(DNA-binding domains) so ESM2 may win")


def h_ribosomal():
    M, syms = _panel()
    lab = {s: ("ribosomal" if (s.startswith("RPL") or s.startswith("RPS")) else "other") for s in syms}
    return lab, "binary", dict(positive="ribosomal", source="symbol prefix",
                               expect="POSITIVE decodability (a tight co-regulated module); model-specificity "
                               "uncertain — co-expression should also find it")


def h_mito():
    M, syms = _panel()
    lab = {s: ("mito" if s.startswith("MT-") else "other") for s in syms}
    return lab, "binary", dict(positive="mito", source="symbol prefix",
                               expect="POSITIVE decodability (mitochondrial genes are extreme co-expressors)")


def h_paralog_family():
    """crude gene-family size via symbol prefix (e.g. COL1A1 -> COL). Sequence families -> ESM2 should win."""
    M, syms = _panel()
    import re
    pref = {s: re.match(r"^([A-Z]+)", s).group(1) if re.match(r"^([A-Z]+)", s) else s for s in syms}
    cnt = Counter(pref.values())
    lab = {s: ("large_family" if cnt[pref[s]] >= 8 else "small_family") for s in syms}
    return lab, "binary", dict(positive="large_family", source="symbol prefix",
                               expect="NEGATIVE for model-specificity — this is gene-family identity, the "
                               "protein-sequence control (ESM2) should win (calibration)")


def _go_term_hypothesis(term, name):
    def loader():
        go = _go()
        lab = {g: (name if term in t else "other") for g, t in go.items()}
        return lab, "binary", dict(positive=name, source=f"GO {term}",
                                   expect="uncertain — functional program; co-expression is the baseline to beat")
    return loader


def _top_go_terms(n=10, min_genes=150, max_genes=3000):
    """well-populated but not universal GO terms (skip GO:0005515 'protein binding' etc.)."""
    go = _go()
    c = Counter(t for terms in go.values() for t in terms)
    picked = [(t, k) for t, k in c.most_common() if min_genes <= k <= max_genes]
    return picked[:n]


REGISTRY = {
    "chromosome": h_chromosome,
    "position": h_position,
    "expr_breadth": h_expr_breadth,
    "expr_abundance": h_expr_abundance,
    "tf_role": h_tf_role,
    "ribosomal": h_ribosomal,
    "mito": h_mito,
    "paralog_family": h_paralog_family,
}


def register_go_terms(n=10):
    """add the n most populated informative GO terms as one-vs-rest hypotheses."""
    for t, k in _top_go_terms(n):
        key = f"go_{t.replace(':', '')}"
        REGISTRY[key] = _go_term_hypothesis(t, key)
    return [f"go_{t.replace(':', '')}" for t, _ in _top_go_terms(n)]


def load(name):
    labels, kind, meta = REGISTRY[name]()
    # drop classes that are too small to test
    if kind in ("categorical", "binary"):
        c = Counter(labels.values())
        keep = {k for k, v in c.items() if v >= MIN_PER_CLASS}
        labels = {g: l for g, l in labels.items() if l in keep}
    return labels, kind, meta


if __name__ == "__main__":
    added = register_go_terms(10)
    print(f"{len(REGISTRY)} hypotheses registered ({len(added)} GO terms)\n")
    for nm in REGISTRY:
        try:
            lab, kind, meta = load(nm)
            if kind == "continuous":
                desc = f"{len(lab)} genes, continuous"
            else:
                c = Counter(lab.values()); desc = f"{len(lab)} genes, {len(c)} classes, min={min(c.values())}"
            print(f"  {nm:<18} {kind:<12} {desc}")
        except Exception as e:
            print(f"  {nm:<18} LOAD FAILED: {repr(e)[:80]}")
# ================================================================ BATCH 2 =====================================
# Genome-organisation, neighbourhood, and dynamics hypotheses. Screen 1 found that everything determined by
# protein sequence loses to ESM2 and everything derived from expression loses to co-expression; the only
# survivors were WHERE A GENE SITS (chromosome, position). This batch pushes on that gap and includes two
# deliberate calibration negatives so a screen that passes everything is detectable.
import re
from collections import defaultdict

_gwc = {}                      # batch-local cache; keeps out of _cache's key space
DEVEL_CELLS = 2500             # cell subsample for the fetal-gut panel (195 MB float32 vs gm_lib's 1.25 GB f64)
NEIGH_K, NEIGH_SPAN, NEIGH_MIN = 10, 2_000_000, 8
DETREND_W = 101
TRRUST = f"{_DATA}/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"
OMNIPATH = f"{_DATA}/biodyn-work/network_inference/data/omnipath_interactions.tsv"
DEVEL_NPZ = os.path.join(G.CACHE, "coexpr_devel.npz")


# ---------------------------------------------------------------- shared helpers
def _gorder():
    """chromosome -> (symbols, starts) SORTED ascending by coordinate, autosomes only.

    The sort is load-bearing: species_chrom.csv is grouped by chromosome but 21 of 22 autosomes are
    out of coordinate order, so np.diff / searchsorted / st[0] on file order silently return garbage.
    """
    if "ord" not in _gwc:
        C = coords()
        C = C[C.chromosome.isin(AUTOSOMES)]
        out = {}
        for ch, g in C.groupby("chromosome"):
            st = g.start.to_numpy().astype(float)
            sy = g.index.to_numpy().astype(str)
            ok = np.isfinite(st)                                  # a NaN would sort last and corrupt every window
            st, sy = st[ok], sy[ok]
            o = np.argsort(st, kind="mergesort")                  # stable -> reproducible on tied coordinates
            out[str(ch)] = (sy[o], st[o])
        _gwc["ord"] = out
    return _gwc["ord"]


def _detrend_on_position(per_chrom, deg):
    """Regress a per-gene value on a degree-`deg` polynomial in within-chromosome position RANK.

    deg=5 for quantities that merely correlate with position (density); deg=1 for quantities that are a
    deterministic function of it (the chromosome-end fold), where a flexible fit would eat the signal itself.
    """
    from scipy.stats import rankdata
    out = {}
    for ch, (sy, st, v) in per_chrom.items():
        if len(v) < deg + 2:
            continue
        u = (rankdata(st) - 0.5) / len(st)
        V = np.vander(u, deg + 1)
        r = v - V @ np.linalg.lstsq(V, v, rcond=None)[0]
        for s, x in zip(sy, r):
            out[s] = float(x)
    return out


def _devel_panel():
    """(z-scored rows, breadth, symbol->row, chromosome-ordered live genes) for the fetal-gut panel.

    Read straight from the npz as float32: G.basis('coexpr_devel') upcasts to float64 (1.25 GB) and parks
    it in G._cache for the life of the process. Zero-variance genes are EXCLUDED rather than zero-filled --
    a zero row correlates 0 with everything and would drag its neighbours' scores down, turning a
    neighbourhood label into a proxy for local expression sparsity, which is what coexpr reads best.
    """
    if "devel" not in _gwc:
        z = np.load(DEVEL_NPZ, allow_pickle=True)
        P, syms = z["profiles"], z["symbols"].astype(str)
        if P.shape[1] > DEVEL_CELLS:
            sel = np.sort(np.random.default_rng(0).choice(P.shape[1], DEVEL_CELLS, replace=False))
            M = np.ascontiguousarray(P[:, sel], dtype=np.float32)
        else:
            M = np.ascontiguousarray(P, dtype=np.float32)
        del P, z
        breadth = (M > 0).mean(1)
        sd = M.std(1)
        live = sd > 0
        Z = np.zeros_like(M)
        Z[live] = (M[live] - M[live].mean(1, keepdims=True)) / sd[live, None]
        del M
        row = {s: i for i, s in enumerate(syms)}
        order = {}
        for ch, (sy, st) in _gorder().items():
            order[ch] = [(s, p) for s, p in zip(sy, st) if s in row and live[row[s]]]
        _gwc["devel"] = (Z, breadth, row, order, syms)
    return _gwc["devel"]


def _neighbours(seq, i, k=NEIGH_K, span=NEIGH_SPAN):
    """indices of the +-k genomic neighbours of seq[i] within `span` bp. The gene ITSELF is excluded."""
    p0 = seq[i][1]
    return [j for j in range(max(0, i - k), min(len(seq), i + k + 1))
            if j != i and abs(seq[j][1] - p0) <= span]


# ---------------------------------------------------------------- TIER 1: genome organisation
def _density(win=1.0e6):
    if "dens" not in _gwc:
        per = {}
        for ch, (sy, st) in _gorder().items():
            lo = np.searchsorted(st, st - win, "left")
            hi = np.searchsorted(st, st + win, "right")
            per[ch] = (sy, st, np.log1p(hi - lo - 1).astype(float))   # -1 excludes the gene itself
        _gwc["dens"] = per
    return _gwc["dens"]


def h_nb_gene_density():
    """log1p count of annotated genes within +-1 Mb of this gene's start.

    Local gene density is the strongest coordinate-only proxy for large-scale chromatin state: dense
    neighbourhoods are GC-rich, early-replicating, A-compartment; sparse ones are B-compartment gene
    deserts. Physical mechanism an expression-only model could learn: co-located genes share a
    compartment, so a gene in a dense cluster has systematically more co-occurring partners in a cell's
    token set. Neither baseline has a channel -- no expression value and no residue enters the label.
    """
    lab = {s: float(v) for ch, (sy, st, d) in _density().items() for s, v in zip(sy, d)}
    return lab, "continuous", dict(
        source="count of annotated genes within +-1 Mb of gene start, log1p; UCE species_chrom "
               "(n=18916, range 0.00-5.05, median 3.14)",
        group_by="chromosome",
        expect="PASS expected — best candidate in the batch. Same family as both existing winners and "
               "invisible to both baselines by construction. CAVEAT: within-chromosome |rho(position, "
               "density)| = 0.254 (max 0.544) and screen.py averages ABSOLUTE per-chromosome rho, so part "
               "of any win re-detects the already-won `position`. Interpret jointly with the _resid twin.")


def h_nb_gene_density_resid():
    """nb_gene_density with the within-chromosome positional trend removed. The decisive twin."""
    return _detrend_on_position(_density(), 5), "continuous", dict(
        source="nb_gene_density minus a deg-5 polynomial in within-chromosome position rank; residual "
               "mean |rho| vs position falls 0.254 -> 0.030 (max 0.106), label sd 0.594",
        group_by="chromosome",
        expect="DECISIVE for the pair. If this ALSO beats both baselines, the model carries local gene "
               "density independently of where the gene sits — a genuine SECOND genome-architecture "
               "channel. If it collapses while raw density passes, density was position in disguise and "
               "the family has one spatial direction, not two. Informative either way.")


def _tandem_raw(win=1.0e6, min_fam=5, min_near=2, min_prefix=2):
    """gene -> (tandem|dispersed, family prefix, chromosome) for members of large symbol-prefix families."""
    if "tand" not in _gwc:
        C = coords()
        C = C[C.chromosome.isin(AUTOSOMES)]
        sym = C.index.to_numpy().astype(str)
        chrom = C.chromosome.to_numpy().astype(str)
        start = C.start.to_numpy(dtype=float)
        fam = defaultdict(list)
        for i, s in enumerate(sym):
            m = re.match(r"^([A-Z]+)", s)
            if m and len(m.group(1)) >= min_prefix:          # >=2 chars kills "C"/"A" junk pseudo-families
                fam[m.group(1)].append(i)
        out = {}
        for p, idx in fam.items():
            if len(idx) < min_fam:
                continue
            by = defaultdict(list)
            for i in idx:
                by[chrom[i]].append(start[i])
            by = {k: np.sort(np.asarray(v)) for k, v in by.items()}
            for i in idx:
                a = by[chrom[i]]
                near = int(np.searchsorted(a, start[i] + win, "right")
                           - np.searchsorted(a, start[i] - win, "left")) - 1   # -1 self; dup-coord safe
                out[sym[i]] = ("tandem" if near >= min_near else "dispersed", p, chrom[i])
        _gwc["tand"] = out
    return _gwc["tand"]


def _balance(genes, raw, keyfn, rng, min_pairs):
    """within each stratum keep equal tandem/dispersed counts, so the stratum key carries no label info."""
    strata = defaultdict(list)
    for g in genes:
        strata[keyfn(g)].append(g)
    kept = []
    for _, mem in sorted(strata.items()):
        t = sorted(g for g in mem if raw[g][0] == "tandem")
        d = sorted(g for g in mem if raw[g][0] == "dispersed")
        n = min(len(t), len(d))
        if n < min_pairs:
            continue
        kept += list(rng.choice(t, n, replace=False)) + list(rng.choice(d, n, replace=False))
    return sorted(kept)


def h_tandem_vs_dispersed():
    """Clustered in a tandem array vs dispersed — BALANCED WITHIN FAMILY.

    Converts screen 1's `paralog_family` FAILURE (esm2 0.733) into a controlled contrast: same families,
    same sequence signal, only LOCATION varying. Balancing within family is load-bearing — on raw labels
    family-prefix identity alone predicts tandem-ness at AUROC 0.962 (the tandem class is OR/ZNF/KRTAP/
    IGHV-dominated), so esm2 would win by reading family, not location. After balancing: 0.500.
    """
    raw = _tandem_raw()
    rng = np.random.default_rng(0)
    kept = _balance(sorted(raw), raw, lambda g: raw[g][1], rng, 3)
    return {g: raw[g][0] for g in kept}, "binary", dict(
        positive="tandem",
        source="species_chrom.csv; symbol-prefix families (>=2-char prefix, >=5 autosomal members); tandem "
               "= >=2 other family members within 1 Mb; equalised WITHIN family (n=1670, 96 families, "
               "family-only oracle AUROC 0.500 from 0.962)",
        expect="PASS plausible, but read against a DECLARED confound: chromosome identity alone still "
               "scores AUROC 0.642 here, and chromosome is already a model-specific win. Only a score "
               "materially above 0.642 is evidence of clustering beyond chromosome identity. The _strict "
               "twin removes that confound outright.")


def h_tandem_vs_dispersed_strict():
    """Balanced within family AND within chromosome (3 alternating passes). The honest test."""
    raw = _tandem_raw()
    rng = np.random.default_rng(0)
    kept = sorted(raw)
    for _ in range(3):
        kept = _balance(kept, raw, lambda g: raw[g][1], rng, 3)
        kept = _balance(kept, raw, lambda g: raw[g][2], rng, 10)
    return {g: raw[g][0] for g in kept}, "binary", dict(
        positive="tandem",
        source="as tandem_vs_dispersed, additionally balanced within chromosome (n=910 -> ~719 after the "
               "4-basis intersection; family oracle 0.516, chromosome oracle 0.500)",
        expect="GENUINELY UNCERTAIN — the cleanest test in the batch. Both nuisance readouts are "
               "neutralised, so the only remaining signal is local physical clustering itself. A win means "
               "genomic neighbourhood is encoded independently of family AND chromosome identity. A drop "
               "to chance means the chromosome/position wins are the whole story and there is no separate "
               "sub-chromosomal clustering direction.")


def h_subtelomeric():
    """q-terminal genes vs CHROMOSOME-MATCHED interior controls.

    The naive version (outer 2 Mb of either end) has a per-chromosome positive rate spanning 1.97% (chr6)
    to 17.8% (chr19), and an oracle with access to NOTHING but chromosome identity scores AUROC 0.675 on
    it — so the model would pass by re-expressing its known chromosome code. Here each autosome contributes
    exactly K positives and K negatives, driving that oracle to exactly 0.500 (verified).

    q-arm only: chr13/14/15/21/22 are acrocentric, so their low-coordinate extreme is pericentromeric, not
    telomeric. The high-coordinate end is a real telomere on all 22 autosomes.
    """
    K = 50
    rng = np.random.default_rng(0)
    lab = {}
    for ch, (sy, st) in _gorder().items():
        if len(sy) < 6 * K:
            continue
        n = len(sy)
        for s in sy[-K:]:
            lab[str(s)] = "subtelomeric"
        for s in rng.choice(sy[int(0.20 * n):int(0.80 * n)], K, replace=False):
            lab[str(s)] = "interior"
    return lab, "binary", dict(
        positive="subtelomeric",
        source="species_chrom.csv; K=50 q-terminal genes per autosome vs K=50 chromosome-matched interior "
               "controls drawn from the 20-80% band (n=2000, chromosome-only oracle AUROC 0.500 exactly)",
        expect="UNCERTAIN, leaning PASS. Chromosome identity is neutralised by construction, so this is a "
               "clean test of within-chromosome TERMINAL position, independent of the chromosome result. "
               "Expect esm2 above chance: subtelomeric regions are ZNF/OR/IGHV tandem-array enriched. If "
               "the model wins, re-run with those prefixes dropped before believing it. NOTE: group_by is "
               "deliberately NOT set — screen.py honours it only for continuous kinds, so it would be "
               "silently inert meta; the matching does that job instead.")


def _nn():
    if "nn" not in _gwc:
        d, nb = {}, {}
        for ch, (sy, st) in _gorder().items():
            prev = np.r_[np.inf, np.diff(st)]
            nxt = np.r_[np.diff(st), np.inf]
            i = np.arange(len(st))
            j = np.clip(np.where(prev <= nxt, i - 1, i + 1), 0, len(st) - 1)
            for s, v, k in zip(sy, np.minimum(prev, nxt), sy[j]):
                d[s] = float(np.log10(1.0 + v)); nb[s] = k
        _gwc["nn"] = (d, nb)
    return _gwc["nn"]


def _stem(s):
    m = re.match(r"^([A-Z]+)", s)
    return m.group(1) if m and len(m.group(1)) >= 3 else None


def h_nn_spacing():
    """log10(1 + bp) to the nearest neighbouring gene start on the same chromosome.

    The finest-scale coordinate feature available: separates genes packed head-to-tail in tandem arrays
    from isolated genes in intergenic space. RESOLUTION-FLOOR PROBE — chromosome (1e8 bp) and position
    (1e6 bp) both passed, so the open question is whether 1e4 bp survives.
    """
    d, _ = _nn()
    return dict(d), "continuous", dict(
        source="nearest-neighbour gene-start gap, coordinate-sorted (n=18916, median 4.47, max 6.57)",
        group_by="chromosome",
        expect="UNCERTAIN, leaning FAIL — deliberately placed to find where the positional code runs out. "
               "esm2 is the competitive baseline here, not coexpr, because the tightest spacings are "
               "tandem paralog arrays. Read jointly with nn_spacing_nonparalog.")


def h_nn_spacing_nonparalog():
    """nn_spacing with genes whose nearest neighbour shares a family stem dropped (12.0% of genes)."""
    d, nb = _nn()
    lab = {s: v for s, v in d.items() if not (_stem(s) and _stem(s) == _stem(nb[s]))}
    return lab, "continuous", dict(
        source="nn_spacing, tandem-paralog neighbours excluded (n=16651, median 4.50)",
        group_by="chromosome",
        expect="CONTROL, not an independent hypothesis. If nn_spacing passes, this must also pass for the "
               "result to mean 'fine-scale positional code' rather than 'paralog-array detector'.")


def h_dist_to_chrom_end_resid():
    """log1p(Mb to the nearer chromosome end), with the LINEAR position component removed.

    A FOLD of position, not a monotone map of it: an even coordinate where `position` is odd. Chromosome
    extent is approximated by the first/last annotated gene, so no telomere table is needed.

    Detrended at degree 1, not 5, on purpose. The fold is a deterministic function of position, so a
    flexible polynomial approximates the tent shape and eats the signal itself (label sd collapses
    0.95 -> 0.17 going deg1 -> deg5). Degree 1 removes exactly what the ESTABLISHED linear position code
    could supply — genome_position_direction.log has linear ridge +0.412 vs MLP +0.071 — while preserving
    the fold. Raw |rho| vs position 0.203 (max 0.743 on chr21) -> detrended 0.080 (max 0.248).
    """
    per = {}
    for ch, (sy, st) in _gorder().items():
        per[ch] = (sy, st, np.log1p(np.minimum(st - st[0], st[-1] - st) / 1.0e6))
    return _detrend_on_position(per, 1), "continuous", dict(
        source="log1p Mb to nearer chromosome end (extent = first/last annotated gene), linear "
               "within-chromosome position trend regressed out (n=18916)",
        group_by="chromosome",
        expect="UNCERTAIN, leaning FAIL. Right family, but the position code is essentially LINEAR "
               "(cubic R^2 0.187 vs linear 0.181; MLP and GBR both worse than ridge), so an even "
               "coordinate is unlikely to be separately readable. A PASS = a genuinely new polar "
               "coordinate (how far from an end) on top of the validated linear ordinate. A FAIL while "
               "`position` passes is the informative outcome: linear ordinate, no end-proximity notion.")


# ---------------------------------------------------------------- TIER 2: position x expression hybrid
def _domain():
    """per-chromosome mean pairwise co-expression among each gene's genomic neighbours (focal excluded).

    Banded: consecutive genes share nearly all neighbours and any pair inside a +-K window is at most 2K
    apart, so corr(i, i+d) for d=1..2K computed once per chromosome makes every neighbourhood mean a
    lookup. Unit-tested identical to the brute-force triu mean (max abs diff 1.4e-17).
    """
    if "dom" not in _gwc:
        Z, _, row, order, _ = _devel_panel()
        ncell = Z.shape[1]
        per = {}
        for ch, seq in order.items():
            n = len(seq)
            if n < NEIGH_MIN + 2:
                continue
            sy = np.array([s for s, _ in seq]); pos = np.array([p for _, p in seq])
            Zc = Z[[row[s] for s in sy]]
            band = np.full((2 * NEIGH_K + 1, n), np.nan)
            for d in range(1, 2 * NEIGH_K + 1):
                if d >= n:
                    break
                band[d, :n - d] = (Zc[:n - d] * Zc[d:]).sum(1) / ncell
            vals = np.full(n, np.nan)
            for i in range(n):
                nb = _neighbours(seq, i)
                if len(nb) < NEIGH_MIN:
                    continue
                acc = cnt = 0.0
                for a in range(len(nb)):
                    for b in range(a + 1, len(nb)):
                        v = band[nb[b] - nb[a], nb[a]]
                        if np.isfinite(v):
                            acc += v; cnt += 1
                if cnt:
                    vals[i] = acc / cnt
            m = np.isfinite(vals)
            per[ch] = (sy[m], pos[m], vals[m])
        _gwc["dom"] = per
    return _gwc["dom"]


def h_nb_coexpr_domain():
    """Does the gene sit in a coherently co-regulated chromosomal domain, or a patchwork?

    Every correlation entering the score is between two OTHER genes, so the focal gene's own profile never
    enters its own label. Computed on the fetal-gut panel — NOT the Tabula Sapiens panel screen.py uses as
    its co-expression baseline.
    """
    lab = {s: float(v) for ch, (sy, _, vv) in _domain().items() for s, v in zip(sy, vv)}
    return lab, "continuous", dict(
        source=f"mean pairwise co-expression among the +-{NEIGH_K} genomic neighbours (<=2 Mb, >={NEIGH_MIN} "
               f"neighbours), fetal-gut panel, {DEVEL_CELLS} cells, focal gene excluded from every pair "
               "(n=14553)",
        group_by="chromosome",
        expect="PASS plausible. esm2 has no route — genomic adjacency is not in protein sequence. coexpr "
               "should sit above chance via domain-bleed but should not match a table with a real "
               "positional coordinate. CAVEAT: |rho vs position| = 0.158 (max 0.595), so a model that "
               "merely re-reads `position` can pass. Read with the _local twin; pass HERE and fail THERE "
               "means position, not domain structure.")


def h_nb_coexpr_domain_local():
    """Same label with the smooth along-chromosome trend removed. THE LOAD-BEARING VARIANT.

    `position` is already an established model-specific direction that the baselines lack, so ANY label
    varying smoothly with genomic coordinate is passable by re-reading it — which would look like a new
    finding and be the old one. A running mean over DETREND_W genes in genomic order kills the smooth
    component (|rho vs position| 0.158 -> 0.051); what remains is whether THIS gene's immediate
    neighbourhood is more or less coherent than the surrounding megabases.
    """
    half = DETREND_W // 2
    lab = {}
    for ch, (sy, _, v) in _domain().items():
        n = len(v)
        if n < DETREND_W:
            continue
        cs = np.concatenate([[0.0], np.cumsum(v)])
        idx = np.arange(n)
        lo = np.maximum(0, idx - half); hi = np.minimum(n, idx + half + 1)
        for s, r in zip(sy, v - (cs[hi] - cs[lo]) / (hi - lo)):
            lab[s] = float(r)
    return lab, "continuous", dict(
        source=f"nb_coexpr_domain minus a {DETREND_W}-gene running mean along the chromosome (n=14553)",
        group_by="chromosome",
        expect="GENUINELY UNCERTAIN, which is the point. esm2 still blind; coexpr still holds only the "
               "gene's own profile. The model can no longer win by re-reading `position`, so a PASS is a "
               "NEW genome-organisation direction (local domain structure), while a FAIL with the raw "
               "version passing localises that pass to the known position coordinate.")


def h_nb_breadth():
    """Mean expression breadth of the gene's genomic NEIGHBOURS (gene itself excluded), fetal-gut panel.

    Requires knowing BOTH who your genomic neighbours are and what they express — a hybrid neither
    baseline holds. Measured, not assumed: rho(label, the gene's OWN fetal-gut breadth) = +0.042, so the
    self-similarity shortcut a co-expression baseline could exploit is very weak. That is far cleaner than
    expected and makes this a real test rather than the partial-leak foil it was first designed as.
    """
    _, breadth, row, order, _ = _devel_panel()
    lab = {}
    for ch, seq in order.items():
        for i in range(len(seq)):
            nb = _neighbours(seq, i)
            if len(nb) < 6:
                continue
            lab[seq[i][0]] = float(np.mean([breadth[row[seq[j][0]]] for j in nb]))
    return lab, "continuous", dict(
        source=f"mean fraction-of-cells-expressing over the +-{NEIGH_K} genomic neighbours within 2 Mb, "
               "fetal-gut panel, gene itself EXCLUDED (n=14871, self-leak rho +0.042)",
        group_by="chromosome",
        expect="UNCERTAIN. Scored WITHIN chromosome so it cannot re-win on the chromosome direction, and "
               "the self-leak is negligible. Residual confound that no local data can remove: the label "
               "panel (fetal gut) differs from the baseline panel (Tabula Sapiens), so a model win could "
               "be panel mismatch rather than biology — which is exactly what devel_breadth_own is "
               "registered to detect. Do not interpret this without that control.")


# ---------------------------------------------------------------- TIER 3: long shots
def _dorothea_indegree(tiers=None):
    """(in-degree over pure non-TF targets, TF set). tiers=None -> all confidence levels."""
    key = f"ind_{tiers}"
    if key not in _gwc:
        srcs, ind = set(), defaultdict(set)
        with open(DOROTHEA) as f:
            head = f.readline().rstrip("\n").split("\t")
            i_s = head.index("source") if "source" in head else 0
            i_t = head.index("target") if "target" in head else 1
            i_c = head.index("confidence") if "confidence" in head else None
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) <= max(i_s, i_t):
                    continue
                if tiers is not None and i_c is not None and not (set(p[i_c].split(";")) & set(tiers)):
                    continue
                srcs.add(p[i_s].upper()); ind[p[i_t].upper()].add(p[i_s].upper())
        _gwc[key] = ({g: v for g, v in ind.items() if g not in srcs}, srcs)
    return _gwc[key]


def _panel_stats():
    """gene -> (breadth, log mean abundance) on the co-expression panel."""
    if "pstat" not in _gwc:
        M, syms = _panel()
        b = (M > 0).mean(1); a = np.log1p(M.mean(1))
        _gwc["pstat"] = {str(s): (float(x), float(y)) for s, x, y in zip(syms, b, a)}
    return _gwc["pstat"]


def h_target_in_degree_resid():
    """How many distinct TFs regulate this non-TF gene, with expression level regressed out.

    A proxy for regulatory-region architecture (promoter/enhancer complexity), which is genomic
    organisation rather than protein function — the family that produced both winners.

    Two controls are load-bearing. (a) A/B/C tiers only: 88.3% of DoRothEA edges are level D, INFERRED
    from bulk co-expression, so an all-tier label is a co-expression statistic wearing a network costume.
    (b) in-degree >= 2: 46% of A/B/C genes have in-degree exactly 1, and inside a tie block the residual
    ordering is exactly MINUS the fitted expression value — residualising a tie-heavy discrete label
    RE-INJECTS expression, inverted, as the tiebreaker.
    """
    ind, _ = _dorothea_indegree(("A", "B", "C"))
    raw = {g: float(np.log1p(len(v))) for g, v in ind.items() if len(v) >= 2}
    ps = _panel_stats()
    g = sorted(k for k in raw if k in ps)
    if len(g) < 200:
        raise ValueError(f"target_in_degree_resid: only {len(g)} genes overlap the panel")
    y = np.array([raw[k] for k in g])
    F = np.array([ps[k] for k in g])
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    D = np.column_stack([np.ones(len(g)), F, F ** 2])
    r = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    return {k: float(v) for k, v in zip(g, r)}, "continuous", dict(
        source="DoRothEA A/B/C in-degree over non-TF targets, in-degree>=2 (n=4791), residualised on "
               "panel breadth+abundance (linear+quadratic)",
        expect="LEAN FAIL — promoter complexity co-varies with expression breadth. But the residual is "
               "exactly the part co-expression should not be able to hand over, and the "
               "expression-inferred D-tier edges are excluded, so a model win here would be strong. "
               "Asymmetric: only a MODEL win is interpretable — coexpr retains the full profile and can "
               "still recover in-degree through it, which is a fair fight, not a leak.")


def h_repression_bias():
    """NON-TF targets predominantly REPRESSED vs predominantly ACTIVATED by their TRRUST regulators.

    A property of how a gene is WRITTEN TO, not of the protein it encodes — the only signed-regulation
    label available locally. Both TRRUST and DoRothEA TFs are excluded, so the DNA-binding-domain channel
    that let esm2 win `tf_role` (0.794) is unavailable here.
    """
    sign, tfs = defaultdict(list), set()
    with open(TRRUST) as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")               # headerless: TF, target, mode, PMID
            if len(p) < 3:
                continue
            tfs.add(p[0].strip().upper())
            if p[2].strip() == "Activation":
                sign[p[1].strip().upper()].append(1)
            elif p[2].strip() == "Repression":
                sign[p[1].strip().upper()].append(0)
    _, dor_tfs = _dorothea_indegree(None)
    tfs |= dor_tfs
    lab = {}
    for g, v in sign.items():
        if g in tfs or len(v) < 2:          # drop TFs; >=2 signed edges so one paper cannot set the label
            continue
        f = sum(v) / len(v)
        if f > 0.5:
            lab[g] = "activated"
        elif f < 0.5:
            lab[g] = "repressed"            # exact ties dropped: no majority sign
    return lab, "binary", dict(
        positive="repressed",
        source="TRRUST signed edges (3149 activation / 1922 repression / 4325 unknown discarded); non-TF "
               "targets with >=2 signed regulators, majority sign (n=585: 449 activated / 136 repressed)",
        expect="LEAN FAIL. The likelier adversary is esm2, not coexpr: TRRUST repression edges are "
               "dominated by the p53/RB/MYC literature, so repressed targets are enriched for cell-cycle "
               "and apoptosis genes — and esm2 won all 8 GO terms in screen 1. If this passes decodability "
               "but loses to esm2, read it as the GO channel, not regulatory logic. If it loses to coexpr, "
               "check the breadth/abundance split across the classes before concluding anything.")


def h_cc_phase():
    """G1/S vs G2/M peak phase — a TIMING label, not a function/localisation label.

    READ `expect` BEFORE BELIEVING A VERDICT. screen.py's baselines are hardcoded and its co-expression
    panel is Tabula Sapiens kidney+lung+immune, which barely proliferates. Peak phase IS the co-expression
    structure of a CYCLING population, so a win over TS-coexpr measures our panel's blindness, not model
    specificity. Settled only by cc_phase_matched_ref(), which screen.py does not perform.
    """
    go = _go()
    G1S = {"GO:0000082", "GO:0006260", "GO:0006270"}     # G1/S transition, replication, replication initiation
    G2M = {"GO:0000086", "GO:0007059", "GO:0000070"}     # G2/M transition, chr / sister-chromatid segregation
    a = {g for g, t in go.items() if t & G1S}
    b = {g for g, t in go.items() if t & G2M}
    lab = {g: "G1S" for g in a - b}
    lab.update({g: "G2M" for g in b - a})                # dual-annotated genes dropped, not forced
    import gene_sets
    for g, deg in gene_sets._CC.items():                 # curated core wins where it disagrees with GO
        lab[g.upper()] = "G1S" if deg < 90 else "G2M"
    return lab, "categorical", dict(
        source="GO phase-transition terms (G1/S vs G2/M) + gene_sets._CC Cyclebase/Whitfield core "
               "(n=1245: 647 G1S / 598 G2M; 329 survive the 4-basis intersection)",
        expect="TRAP DETECTOR — a MODEL-SPECIFIC verdict from screen.py is NOT a hit for this hypothesis. "
               "Predicted: model beats coexpr(TS), LOSES to esm2 on protein-family grounds (replication "
               "machinery MCM/ORC/POLA vs segregation machinery kinesin/CENP/AURK — every GO term in "
               "screen 1 lost to esm2), and would also lose to a cycling panel. The esm2 loss is the most "
               "likely outcome and kills the gate before the panel question is reached. If it beats esm2, "
               "run cc_phase_matched_ref() before claiming anything.")


def h_context_lability():
    """SD of a gene's standardised expression rank across three biologically distinct panels.

    Fetal gut (developmental), K562 (cycling cancer line), Setty19 CD34 (haematopoietic progenitors).
    The screen's coexpr baseline is Tabula Sapiens — none of these three — and the label is a CONTRAST no
    single panel can supply. Checked rather than asserted: rho(label, TS mean abundance) = -0.168 and
    rho(label, |z(TS abundance)|) = -0.105, so this is not a re-skin of the already-won expr_abundance.

    Ranks are computed AFTER intersecting the three panels; ranking within each panel's own gene universe
    (19492 / 6546 / 11258) makes the label half universe-mismatch artefact.
    """
    def mean_by_symbol(tag):
        z = np.load(os.path.join(G.CACHE, f"{tag}.npz"), allow_pickle=True)
        P, sy = z["profiles"], z["symbols"].astype(str)
        mu = np.empty(P.shape[0])
        for i in range(0, P.shape[0], 2000):
            mu[i:i + 2000] = P[i:i + 2000].mean(1)
        del P, z                                          # free before the next panel loads
        out = {}
        for s, v in zip(sy, mu):
            out[s] = max(out.get(s, -np.inf), float(v))    # duplicate symbols: keep the higher-expressed row
        return out

    def zrank(v):
        r = (np.argsort(np.argsort(v)).astype(float) + 0.5) / len(v)
        return (r - r.mean()) / (r.std() + 1e-12)

    mus = {t: mean_by_symbol(t) for t in ("coexpr_devel", "coexpr_k562", "coexpr_hsc")}
    common = sorted(set.intersection(*(set(v) for v in mus.values())))
    Z = np.vstack([zrank(np.array([mus[t][g] for g in common])) for t in mus])
    return {g: float(s) for g, s in zip(common, Z.std(0))}, "continuous", dict(
        source="sd of standardised mean-expression rank across fetal gut / K562 / Setty19 CD34, ranked "
               "within the 3-panel intersection (n=4790, median 0.313)",
        expect="UNCERTAIN — a fair test, not a favourite. No single-panel basis holds a cross-panel "
               "contrast and the abundance confound is weak, so coexpr has no construction-level route, "
               "and a model pretrained across many tissues must predict the same gene at very different "
               "ranks per context. Against it: sd over 3 points has 2 df, and the panels differ in assay, "
               "depth and cell-line-vs-tissue status, so the label conflates biological lability with "
               "panel technical difference. If it passes, re-run dropping MT-/RP genes (only 16 present, "
               "so it should barely move).")


# ---------------------------------------------------------------- TIER 4: calibration negatives
def h_devel_breadth_own():
    """CALIBRATION NEGATIVE #1 (co-expression side) — GUARDS THE ENTIRE FETAL-GUT FAMILY.

    The gene's OWN expression breadth on the fetal-gut panel: the direct analogue of screen 1's
    `expr_breadth` (which coexpr won at ~1.00) but computed on a DIFFERENT panel from the baseline.
    rho(fetal-gut breadth, TS breadth) = +0.765, so coexpr(TS) should still win comfortably.

    This is the control that makes nb_breadth / nb_coexpr_domain interpretable. If the MODEL wins here,
    then the panel swap ALONE confers a spurious advantage and every fetal-gut-derived result in this
    batch is an artefact of baseline mismatch rather than a finding.
    """
    _, breadth, _, _, syms = _devel_panel()
    return {s: float(b) for s, b in zip(syms, breadth)}, "continuous", dict(
        source=f"fraction of cells expressing, fetal-gut panel, {DEVEL_CELLS} cells (n=19490)",
        expect="NEGATIVE for model-specificity, by construction. coexpr(TS) should win: breadth is an "
               "expression statistic and the two panels' breadths correlate at rho 0.765. A model win "
               "INVALIDATES nb_breadth and nb_coexpr_domain — it would show the fetal-gut/Tabula-Sapiens "
               "panel mismatch is by itself enough to beat the baseline.")


def h_signalling_direction():
    """CALIBRATION NEGATIVE #2 (sequence side) — upstream source vs downstream sink in OmniPath.

    Cascade position (kinase/ligand/receptor vs substrate/effector) is protein architecture, so esm2 is
    expected to win. If the model tables beat esm2 here, suspect the screen, not the biology.

    ~62% of OmniPath rows use underscore-joined complex symbols (PPP2CA_PPP2R5A_PTPA). Each subunit is
    credited with the edge additively; treating the literal string as a symbol would silently drop those
    rows and misclassify every obligate-complex enzyme (PP2A, CK2, AMPK, SCF) as a sink.
    """
    import csv
    src, dst = Counter(), Counter()
    with open(OMNIPATH) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            a = {g for g in row["source_genesymbol"].strip().upper().split("_") if g}
            b = {g for g in row["target_genesymbol"].strip().upper().split("_") if g}
            if not a or not b:
                continue
            for x in a - b:                                # skip self / shared-subunit autoloops
                src[x] += 1
            for y in b - a:
                dst[y] += 1
    lab = {}
    for g in set(src) | set(dst):
        if src[g] + dst[g] < 3:                            # need a few edges before 'direction' means anything
            continue
        if src[g] > dst[g]:
            lab[g] = "source"
        elif dst[g] > src[g]:
            lab[g] = "sink"                                # exact ties dropped
    return lab, "binary", dict(
        positive="source",
        source="OmniPath directed interactions (85217 edges, complexes expanded to member genes; "
               "n=5199: 2072 source / 3127 sink)",
        expect="NEGATIVE for model-specificity BY DESIGN. esm2 should win, ~0.75-0.90 AUROC by analogy "
               "with tf_role (0.794). Verified NOT redundant with tf_role: DoRothEA TFs are only 3.2% of "
               "'source' and 8.8% of 'sink'. Caveat: if coexpr wins instead of esm2, that is a "
               "degree-hubness artefact, not a statement about sequence.")


REGISTRY["nb_gene_density"]           = h_nb_gene_density
REGISTRY["nb_gene_density_resid"]     = h_nb_gene_density_resid
REGISTRY["tandem_vs_dispersed"]       = h_tandem_vs_dispersed
REGISTRY["tandem_vs_dispersed_strict"] = h_tandem_vs_dispersed_strict
REGISTRY["subtelomeric"]              = h_subtelomeric
REGISTRY["nn_spacing"]                = h_nn_spacing
REGISTRY["nn_spacing_nonparalog"]     = h_nn_spacing_nonparalog
REGISTRY["dist_to_chrom_end_resid"]   = h_dist_to_chrom_end_resid
REGISTRY["nb_coexpr_domain"]          = h_nb_coexpr_domain
REGISTRY["nb_coexpr_domain_local"]    = h_nb_coexpr_domain_local
REGISTRY["nb_breadth"]                = h_nb_breadth
REGISTRY["target_in_degree_resid"]    = h_target_in_degree_resid
REGISTRY["repression_bias"]           = h_repression_bias
REGISTRY["cc_phase"]                  = h_cc_phase
REGISTRY["context_lability"]          = h_context_lability
REGISTRY["devel_breadth_own"]         = h_devel_breadth_own
REGISTRY["signalling_direction"]      = h_signalling_direction


def cc_phase_matched_ref():
    """STANDALONE adjudicator for cc_phase — deliberately NOT a registry entry.

    screen.py's BASES are hardcoded and `matched_ref`/`extra_refs` keys are inert, so the cycling-panel
    comparison cannot be expressed as meta. Adding coexpr_k562 to BASES globally would shrink the common
    gene set for EVERY hypothesis (cc_phase alone: 329 -> 244), so run this separately, alone, when the
    box is idle:  ../../.venv/bin/python -c "import hypotheses as h; h.cc_phase_matched_ref()"
    """
    import screen as S
    labels, kind, _ = load("cc_phase")
    bases = ["maxtoki_lmhead", "maxtoki_we", "coexpr", "coexpr_k562", "esm2"]
    tabs, common = {}, None
    for b in bases:
        M, syms = G.basis(b)
        pi = {s: i for i, s in enumerate(syms)}
        tabs[b] = (M, pi)
        s = set(pi) & set(labels)
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"cc_phase matched-ref: {len(common)} genes shared across {len(bases)} bases")
    if len(common) < 200:
        print("  ABORT: below the 200-gene floor; result would not be trustworthy")
        return None
    y = np.array([labels[g] for g in common])
    print("  ", dict(Counter(y)))
    out = {b: S.score(tabs[b][0][[tabs[b][1][g] for g in common]], y, kind) for b in bases}
    for b in bases:
        print(f"  {b:<16} {out[b]:.3f}")
    best = max(("maxtoki_lmhead", "maxtoki_we"), key=lambda b: out[b])
    ok = out[best] > out["coexpr_k562"] and out[best] > out["esm2"]
    print(f"  -> {'SURVIVES the cycling-panel control' if ok else 'FAILS — a matched baseline wins'} "
          f"(model {out[best]:.3f} vs coexpr_k562 {out['coexpr_k562']:.3f}, esm2 {out['esm2']:.3f})")
    return out
