"""FEASIBILITY: how many genes, and how balanced, would a MaxToki cell-type-stratified extraction actually give?

WHY THIS RUNS BEFORE THE EXTRACTION. The scGPT pilot (`ctx_interaction.py`) died of two design faults that
were invisible until measured, and both are pure TOKENISATION properties — no forward pass required:

  1. THE STOPWORD TRAP. Requiring a gene to clear a token-count floor in EVERY context left 65 genes, and they
     were the ubiquitous housekeeping core (ACTB, B2M, FTL, HLA-*, ribosomal, hnRNP). Ethayarajh 2019 found
     stopwords are the MOST context-specific words in NLP, so that panel maximises the confound while
     containing the genes least likely to switch biological role. We therefore measure PAIRWISE context
     overlap, not the all-context intersection, and report both.
  2. HETEROSCEDASTIC INTERACTION. A per-gene representation is a MEAN over that gene's token occurrences, so
     its SEM scales 1/sqrt(n). In the scGPT caches the same gene's count varied a median 6.1x across contexts,
     which dumps context-dependent sampling noise straight into the interaction term. We therefore measure the
     count-imbalance distribution, and how many genes survive SUBSAMPLING every context to a common count --
     the fix that makes the interaction term honest.

So this script answers, before spending any GPU time: how many usable genes, at what count floor, with what
residual imbalance, and what the forward pass will cost.

Tokenisation replicates maxtoki_layers.py exactly (log1p CP10K, divide by Geneformer gene medians, rank-order
descending, truncate to MAX_LEN) so the numbers transfer directly to the real extraction.

Out: results/ctx_feasibility.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)

TS = f"{_DATA}/raw"
PANELS = ["tabula_sapiens_immune_subset_20000.h5ad", "tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad"]
# Sequence length is the entire cost driver: measured MaxToki-217M throughput on MPS is 4.9 cells/s at 512,
# 2.4 at 1024, 1.0 at 2048. But shortening biases the visible gene panel toward abundant genes -- the very
# stopword trap that sank the pilot -- so run this at several lengths and read the tradeoff, do not assume.
MAX_LEN = int(os.environ.get("MAXLEN", 2048))
MIN_CELLS = 250         # a context needs this many cells to be worth extracting
COUNT_FLOORS = [12, 25, 50, 100]
SEED = 0


def load_panel(path, max_cells=None):
    """stream a CSR h5ad, returning (ensembl_ids, list_of_(idx,val) per cell, cell_type array)."""
    with h5py.File(path, "r") as f:
        ens = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]).astype(str)
        ens = np.array([e.split(".")[0] for e in ens])
        ct = f["obs"]["cell_type"]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in ct["categories"][:]]).astype(str)
        ctypes = cats[ct["codes"][:]]
        X = f["X"]; n = int(X.attrs["shape"][0])
        sel = np.arange(n)
        if max_cells and n > max_cells:
            sel = np.sort(np.random.default_rng(SEED).choice(n, max_cells, replace=False))
        indptr = X["indptr"][:]
        cells = []
        for r in sel:
            s, e = int(indptr[r]), int(indptr[r + 1])
            cells.append((X["indices"][s:e], X["data"][s:e].astype(np.float32)))
    return ens, cells, ctypes[sel]


def main():
    from maxtoki_adapter import MaxTokiTokenizer
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)

    per_ctx = collections.defaultdict(collections.Counter)   # cell_type -> Counter(token -> occurrences)
    n_cells_ctx = collections.Counter()
    tot_tokens = 0

    for p in PANELS:
        path = os.path.join(TS, p)
        if not os.path.exists(path):
            print(f"  [skip] {p} not found"); continue
        ens, cells, ctypes = load_panel(path, max_cells=20000)
        var_idx, token_ids, medians = tok.make_var_mapping(list(ens))
        print(f"[{p}] {len(cells)} cells, {len(ens)} vars -> {len(var_idx)} map to MaxToki vocab", flush=True)

        pos = np.full(len(ens), -1, np.int64)               # var row -> position within var_idx
        pos[var_idx] = np.arange(len(var_idx))
        for (idx, val), c in zip(cells, ctypes):
            rs = float(val.sum()) or 1.0
            keep = pos[idx] >= 0
            if not keep.any():
                continue
            j = pos[idx[keep]]
            en = np.log1p(val[keep] / rs * 1e4)
            nz = en > 0
            if not nz.any():
                continue
            norm = en[nz] / np.maximum(medians[j[nz]], 1e-9)
            order = np.argsort(-norm)[: MAX_LEN - 2]
            # token_ids/medians are indexed by POSITION WITHIN var_idx (as in maxtoki_layers.py), not by var row
            toks = token_ids[j[nz][order]]
            per_ctx[c].update(int(t) for t in toks)
            n_cells_ctx[c] += 1
            tot_tokens += len(toks)

    ctx = {c: v for c, v in per_ctx.items() if n_cells_ctx[c] >= MIN_CELLS}
    print(f"\n=== {len(ctx)} contexts with >= {MIN_CELLS} cells "
          f"(of {len(per_ctx)} cell types seen); {tot_tokens:,} gene tokens total ===")
    for c in sorted(ctx, key=lambda c: -n_cells_ctx[c])[:14]:
        print(f"  {n_cells_ctx[c]:>6} cells   {len(ctx[c]):>6} distinct genes   {c[:52]}")

    res = dict(n_contexts=len(ctx), total_tokens=int(tot_tokens),
               cells_per_context={c: int(n_cells_ctx[c]) for c in ctx}, floors={})

    names = sorted(ctx, key=lambda c: -n_cells_ctx[c])
    print(f"\n{'floor':<7} {'genes/ctx (median)':<20} {'ALL-context ∩':<16} {'PAIRWISE ∩ (median)':<22} "
          f"{'count imbalance max/min (median)'}")
    for K in COUNT_FLOORS:
        sets = {c: {g for g, n in ctx[c].items() if n >= K} for c in names}
        per = [len(sets[c]) for c in names]
        allint = len(set.intersection(*sets.values())) if sets else 0
        pw, imb = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sh = sets[names[i]] & sets[names[j]]
                pw.append(len(sh))
                if sh:
                    r = [max(ctx[names[i]][g], ctx[names[j]][g]) / max(1, min(ctx[names[i]][g], ctx[names[j]][g]))
                         for g in list(sh)[:4000]]
                    imb.append(float(np.median(r)))
        res["floors"][K] = dict(genes_per_ctx_median=float(np.median(per)), all_intersection=allint,
                                pairwise_median=float(np.median(pw)) if pw else 0,
                                pairwise_max=int(np.max(pw)) if pw else 0,
                                imbalance_median=float(np.median(imb)) if imb else float("nan"))
        print(f"{K:<7} {np.median(per):<20.0f} {allint:<16} {np.median(pw) if pw else 0:<22.0f} "
              f"{np.median(imb) if imb else float('nan'):.2f}x")

    # What does count-EQUALISED subsampling cost? For each context pair, cap both to the shared min count.
    K = 25
    sets = {c: {g for g, n in ctx[c].items() if n >= K} for c in names}
    surv, kept_frac = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sh = sets[names[i]] & sets[names[j]]
            if not sh:
                continue
            m = [min(ctx[names[i]][g], ctx[names[j]][g]) for g in sh]
            surv.append(sum(1 for v in m if v >= K))
            kept_frac.append(np.mean([mi / max(ctx[names[i]][g], ctx[names[j]][g])
                                      for mi, g in zip(m, sh)]))
    print(f"\ncount-equalised subsampling at floor {K}: median {np.median(surv):.0f} genes survive per context "
          f"pair, retaining {100*np.mean(kept_frac):.0f}% of tokens on the richer side")
    res["equalised"] = dict(floor=K, median_genes=float(np.median(surv)),
                            token_retention=float(np.mean(kept_frac)))

    # forward-pass cost
    n_cells_total = sum(n_cells_ctx[c] for c in ctx)
    res["cost"] = dict(cells=int(n_cells_total), tokens=int(tot_tokens))
    print(f"\nextraction cost: {n_cells_total:,} cells / {tot_tokens:,} tokens through MaxToki-217M "
          f"with output_hidden_states (12 taps x 1232 dims)")

    best = max(COUNT_FLOORS, key=lambda K: res["floors"][K]["pairwise_median"] if
               res["floors"][K]["imbalance_median"] < 3.0 else -1)
    res["verdict"] = (
        f"Pairwise design gives a median {res['floors'][best]['pairwise_median']:.0f} shared genes per context "
        f"pair at floor {best}, versus {res['floors'][best]['all_intersection']} for the all-context "
        f"intersection that sank the scGPT pilot. Residual count imbalance "
        f"{res['floors'][best]['imbalance_median']:.1f}x is handled by equalised subsampling.")
    print(f"\nVERDICT: {res['verdict']}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "ctx_feasibility.json"), "w"), indent=1)
    print("[done] -> results/ctx_feasibility.json")


if __name__ == "__main__":
    main()
