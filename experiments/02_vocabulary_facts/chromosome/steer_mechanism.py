"""DOES THE STEERING DESTINATION MATCH REAL BIOLOGY? — closing the mechanism loop (Ihor, 2026-07-18).

THE QUESTION. steer_where.py showed that pushing the chromosome variable CONFIDENTLY RELABELS cells, and that
the destination depends on WHICH chromosome you push (destination agreement 0.35 for chromosomes vs 0.49 for
meaningless shams). The obvious mechanistic hypothesis, and Ihor's question:

    chr-C steering sends cells toward cell type T *because* T genuinely over-expresses chr-C genes.

If that holds, the model's "chromosome" variable is not an abstract genomic coordinate at all -- it is a handle
on the CO-REGULATED EXPRESSION PROGRAM that chromosome carries, and the whole finding (chromosome decodable
from an expression-only model) gets a concrete mechanism.

THE TEST. Two matrices over (chromosome x cell type):
  DEST[c, t]  -- from the model: fraction of cells that land in type t when steered toward chromosome c
                 (steer_where.py, saved as dest_chr).
  ENRICH[c, t]-- from the DATA ONLY, no model involved: how much cell type t over-expresses chromosome c's
                 genes. Built as: log1p-CP10k -> mean per (gene, cell type) -> z-score EACH GENE across cell
                 types (removes gene-level abundance, leaving "which type expresses this gene relatively most")
                 -> average that z over the genes of each chromosome.
Then ask whether DEST and ENRICH agree ROW-WISE (matched chromosome) more than they do for MISMATCHED pairs.

WHY A MATCHED-VS-MISMATCHED PERMUTATION IS THE RIGHT NULL. "mesodermal cell" is a common attractor and a
common enrichment winner, so a naive argmax match-rate is inflated by shared marginals. Permuting which
chromosome's DEST row is compared to which chromosome's ENRICH row holds BOTH marginals fixed and tests only
whether the PAIRING carries information. Reported two ways:
  * mean matched correlation minus mean mismatched correlation (+ permutation p);
  * argmax agreement (does the modal destination equal the top-enriched type?) vs its permutation null.

Run: ../../.venv_state/bin/python -u steer_mechanism.py     (needs results/steer_where.json)
Out: results/steer_mechanism.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
from genome_wide import coords, AUTOSOMES

# NB: this analysis is DATA-ONLY -- no forward passes -- so it deliberately does not import the model stack
# (steer_lib/transformers). These two readers are inlined from steer_classifier for that reason.


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _cat(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group) and "categories" in g:
        return _dec(g["categories"][:]).astype(str)[g["codes"][:]]
    return _dec(g[:]).astype(str)

MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
MDIR = f"{MAXTOKI_SETUP}/MaxToki-217M-HF"
TOKMAP = f"{MAXTOKI_SETUP}/token_dictionary.json"
N_CELLS_ENRICH = 6000        # data-side only: no model forward passes, so this can be large
MIN_GENES_PER_CHR = 30
SEED = 0


def direction_genes(top_k=150):
    """The genes that actually DEFINE each chromosome steering direction.

    WHY. Averaging expression z-scores over ALL ~1000 genes of a chromosome is dominated by global cell-type
    effects, not chromosome-specific programs: measured, that construction makes 18/22 chromosomes share the
    same top-enriched cell type, i.e. the data side carries almost no chromosome-specific signal to match
    against. But the steering push is a centroid difference, and only some genes sit far along it. Ranking
    genes by their projection onto d_C isolates the ones the push is actually made of -- a far sharper thing to
    ask the expression data about. Uses gm_lib's numpy safetensors reader: no torch, no forward passes.
    """
    C = coords()
    R = G.ST_Reader(f"{MDIR}/model.safetensors")
    EMB = R.get("model.embed_tokens.weight")
    tokmap = json.load(open(TOKMAP))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tok2sym, tok2chr = {}, {}
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and t < EMB.shape[0]:
            tok2sym[t] = s; tok2chr[t] = str(C.loc[s, "chromosome"])
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    rng = np.random.default_rng(SEED)
    is_tr = rng.random(len(tids)) < 0.5          # SAME split as the steering run
    gcen = EMB[tids[is_tr]].mean(0)
    out = {}
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        d = EMB[tids[m]].mean(0) - gcen
        d = d / (np.linalg.norm(d) + 1e-12)
        proj = EMB[tids] @ d                      # every gene's projection onto the push direction
        top = tids[np.argsort(-proj)[:top_k]]
        out[c] = [tok2sym[t] for t in top]
    return out


def build_enrichment(classes, gene_subsets=None):
    """ENRICH[c, t] from real expression only. No model.
    gene_subsets: optional {chrom: [symbols]} -- if given, average over THOSE genes instead of all genes on
    the chromosome (the sharper, direction-defined version)."""
    C = coords()
    with h5py.File(G.FETAL_GUT, "r") as f:
        fn = f["var"]["feature_name"]
        syms = _dec(fn["categories"][:]).astype(str)[fn["codes"][:]] if isinstance(fn, h5py.Group) \
            else _dec(fn[:]).astype(str)
        ct = _cat(f, "cell_type")
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], min(N_CELLS_ENRICH, shape[0]), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            E[i, idx[s:e]] = data[s:e]
    ct = ct[sel]
    up = np.char.upper(syms.astype(str))

    # normalise per cell, then mean per (gene, cell type)
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    L = np.log1p(E / tot * 1e4)
    keep_t = [t for t in classes if (ct == t).sum() >= 10]
    M = np.stack([L[ct == t].mean(0) for t in keep_t])           # (n_types, n_genes)

    # z-score EACH GENE across cell types -> "which type expresses this gene relatively most"
    mu, sd = M.mean(0), M.std(0)
    ok = sd > 1e-8
    Zg = np.zeros_like(M)
    Zg[:, ok] = (M[:, ok] - mu[ok]) / sd[ok]

    # average per chromosome -- over all its genes, or over a supplied direction-defined subset
    gene_chr = np.array([C.loc[s, "chromosome"] if s in C.index else "" for s in up], dtype=object)
    sym_pos = {}
    for i, s in enumerate(up):
        sym_pos.setdefault(s, i)
    rows, chrs_used = [], []
    for c in AUTOSOMES:
        if gene_subsets is not None:
            idx = np.array([sym_pos[s] for s in gene_subsets.get(c, []) if s in sym_pos and ok[sym_pos[s]]],
                           dtype=int)
            if len(idx) < 20:
                continue
            rows.append(Zg[:, idx].mean(1))
        else:
            m = (gene_chr == c) & ok
            if m.sum() < MIN_GENES_PER_CHR:
                continue
            rows.append(Zg[:, m].mean(1))
        chrs_used.append(c)
    return np.stack(rows), chrs_used, keep_t          # (n_chr, n_types)


def compare(DEST, chroms_d, classes, ENR, chrs_e, types_e, tag):
    """Match DEST rows to ENRICH rows; matched-vs-mismatched permutation test."""
    # align to the chromosomes and cell types present in BOTH
    ci = [i for i, c in enumerate(chroms_d) if c in chrs_e]
    cj = [chrs_e.index(chroms_d[i]) for i in ci]
    ti = [i for i, t in enumerate(classes) if t in types_e]
    tj = [types_e.index(classes[i]) for i in ti]
    D = DEST[np.ix_(ci, ti)]
    Eh = ENR[np.ix_(cj, tj)]
    used_chr = [chroms_d[i] for i in ci]; used_t = [classes[i] for i in ti]
    print(f"[aligned]  {D.shape[0]} chromosomes x {D.shape[1]} cell types\n")

    # --- correlation of each DEST row with each ENRICH row
    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else v * 0
    Dz = np.stack([z(r) for r in D]); Ez = np.stack([z(r) for r in Eh])
    Ccorr = Dz @ Ez.T / D.shape[1]                              # (n_chr, n_chr)
    matched = np.diag(Ccorr)
    off = Ccorr[~np.eye(len(matched), dtype=bool)]
    gap = float(matched.mean() - off.mean())

    rng = np.random.default_rng(SEED)
    null = np.array([np.diag(Ccorr[rng.permutation(len(matched))]).mean() - off.mean()
                     for _ in range(20000)])
    p_corr = float(((null >= gap).sum() + 1) / (len(null) + 1))

    # --- argmax agreement, against the same marginal-preserving null
    dest_top = D.argmax(1); enr_top = Eh.argmax(1)
    agree = float((dest_top == enr_top).mean())
    null_a = np.array([float((dest_top[rng.permutation(len(dest_top))] == enr_top).mean())
                       for _ in range(20000)])
    p_arg = float(((null_a >= agree).sum() + 1) / (len(null_a) + 1))

    # DEGENERACY DIAGNOSTIC: if the enrichment matrix points nearly every chromosome at the same cell type,
    # it carries no chromosome-specific signal and the whole comparison is underpowered -- a significant but
    # tiny gap must NOT then be read as support.
    n_distinct = len(set(enr_top.tolist()))
    modal_share = float(np.bincount(enr_top, minlength=D.shape[1]).max() / len(enr_top))

    print(f"\n=== [{tag}] does the steering destination match real expression enrichment? ===")
    print(f"  enrichment distinct top-types    : {n_distinct}/{len(enr_top)} chromosomes "
          f"(one type takes {modal_share:.0%})" + ("   <-- DEGENERATE" if modal_share > 0.5 else ""))
    print(f"  matched-chromosome correlation   : {matched.mean():+.4f}")
    print(f"  mismatched (off-diagonal)        : {off.mean():+.4f}")
    print(f"  GAP (matched - mismatched)       : {gap:+.4f}   permutation p = {p_corr:.4f}")
    print(f"  argmax agreement                 : {agree:.3f}  (null {null_a.mean():.3f})  p = {p_arg:.4f}")

    print(f"  per chromosome: modal steering destination  vs  most enriched cell type")
    for k, c in enumerate(used_chr):
        mark = "MATCH" if dest_top[k] == enr_top[k] else ""
        print(f"    chr{c:<3} steer-> {str(used_t[dest_top[k]]):<26} data-> {str(used_t[enr_top[k]]):<26} "
              f"r={Ccorr[k, k]:+.2f} {mark}")

    strong = (p_corr < 0.05) and (gap > 0.05) and (modal_share <= 0.5)
    weak = (p_corr < 0.05) and not strong
    verdict = ("SUPPORTED: destination tracks chromosome-specific expression" if strong else
               "NOT SUPPORTED: no matched-pair signal" if p_corr >= 0.05 else
               "INCONCLUSIVE: gap is significant but tiny and/or the enrichment side is degenerate "
               "-- not evidence for the mechanism")
    print(f"  VERDICT: {verdict}")
    return dict(tag=tag, n_chrom=int(D.shape[0]), n_types=int(D.shape[1]),
                matched_corr=float(matched.mean()), mismatched_corr=float(off.mean()), gap=gap,
                p_corr=p_corr, argmax_agreement=agree, argmax_null=float(null_a.mean()), p_argmax=p_arg,
                enrich_distinct_top=n_distinct, enrich_modal_share=modal_share, verdict=verdict,
                per_chrom={c: dict(steer=str(used_t[dest_top[k]]), data=str(used_t[enr_top[k]]),
                                   r=float(Ccorr[k, k])) for k, c in enumerate(used_chr)})


def main():
    W = json.load(open(os.path.join(HERE, "results", "steer_where.json")))
    classes = np.array(W["classes"])
    chroms_d = list(map(str, W["chroms"]))
    DEST = np.array(W["dest_chr"])
    print(f"[steering] destinations for {len(chroms_d)} chromosomes over {len(classes)} classes "
          f"(n={W['n_cells']} cells, alpha={W['alpha']})")

    runs = []
    # (a) coarse: all genes on the chromosome
    ENR, ce, te = build_enrichment(list(classes))
    runs.append(compare(DEST, chroms_d, classes, ENR, ce, te, "all chromosome genes"))
    # (b) sharp: only the genes that define the steering direction
    subs = direction_genes(top_k=150)
    ENR2, ce2, te2 = build_enrichment(list(classes), gene_subsets=subs)
    runs.append(compare(DEST, chroms_d, classes, ENR2, ce2, te2, "top-150 direction genes"))

    print("\n" + "=" * 92)
    for r in runs:
        print(f"  [{r['tag']:<24}] gap {r['gap']:+.4f} p={r['p_corr']:.4f}  "
              f"argmax {r['argmax_agreement']:.2f}  -> {r['verdict'].split(':')[0]}")
    json.dump(dict(alpha=W["alpha"], n_cells=W["n_cells"], runs=runs),
              open(os.path.join(HERE, "results", "steer_mechanism.json"), "w"), indent=1)
    print("\n[done] -> results/steer_mechanism.json")


if __name__ == "__main__":
    main()
