"""EXTRACT MaxToki per-gene CONTEXTUAL representations, stratified by cell type, with honest noise control.

This is the extraction the scGPT pilot could not support. Four design choices, each fixing a specific measured
failure of `ctx_interaction.py` (see that file's header and the gene-polysemy memory):

 1. PAIRWISE, NOT ALL-CONTEXT. Requiring a count floor in every context left 65 genes, all housekeeping -- the
    stopword trap (Ethayarajh 2019: stopwords are the MOST context-specific words, so that panel maximises the
    confound while holding the genes least likely to switch role). Feasibility measured at MAX_LEN=1024:
    5,677 shared genes per context PAIR at floor 25, vs 231 for the all-context intersection. We keep any gene
    reaching the floor in >= 2 contexts and do all statistics pairwise.

 2. CELL-LEVEL SPLIT HALVES. Each cell is assigned to partition 0 or 1 up front, and every (gene, context)
    mean is accumulated separately per partition. This gives a genuinely independent noise estimate. The pilot
    split by DIMENSION instead, which is worthless here because dimensions within a layer are correlated
    (anisotropy 0.79 at L0, 0.93 at L8) -- both halves saw the same nuisance and "reliability" hit 0.98 under a
    null that itself sat at 0.82.

 3. OCCURRENCE CAP -- the heteroscedasticity fix. A per-gene representation is a MEAN over its tokens, so SEM
    scales 1/sqrt(n). In the scGPT caches the same gene's count varied a median 6.1x across contexts, dumping
    context-dependent sampling noise straight into the interaction term (this, not biology, is what the pilot's
    z=+10.4 was measuring). We therefore accumulate AT MOST `CAP` occurrences per (gene, context, partition),
    over a shuffled cell order, so every well-covered gene is estimated from exactly the same number of tokens
    in every context. Genes that never reach CAP are recorded with their true count so analysis can drop them.

 4. TWO-PASS. Pass 1 tokenises only (CPU, minutes) to choose the gene panel and the contexts; pass 2 runs the
    forward pass accumulating ONLY panel genes. Accumulating all 19,982 vocab genes would need terabytes.

Throughput measured on this box (MaxToki-217M, MPS, float32): 4.9 cells/s at len 512, 2.4 at 1024, 1.0 at 2048.

Out: results/ctx_maxtoki_L{tap}.npz  — M[partition, context, gene, dim], counts[partition, context, gene]
Run: ../../.venv_state/bin/python -u ctx_extract_maxtoki.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, collections, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)

TS = f"{_DATA}/raw"
PANELS = ["tabula_sapiens_immune_subset_20000.h5ad", "tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad"]
MDIR = f"{MSETUP}/{os.environ.get('MODEL', 'MaxToki-217M-HF')}"   # MODEL=MaxToki-1B-HF for the scaling pass

MAX_LEN   = int(os.environ.get("MAXLEN", 1024))
N_CTX     = int(os.environ.get("NCTX", 12))     # contexts, by descending cell count
CELLS_CTX = int(os.environ.get("CELLSCTX", 1000))
CAP       = int(os.environ.get("CAP", 50))      # max occurrences accumulated per (gene, ctx, partition)
FLOOR     = int(os.environ.get("FLOOR", 25))    # panel: gene must reach this in >= 2 contexts
MAX_GENES = int(os.environ.get("MAXGENES", 6000))
TAPS      = [int(x) for x in os.environ.get("TAPS", "0,4,8,11").split(",")]
OUTPREFIX = os.environ.get("OUTPREFIX", "ctx_maxtoki")   # override so a layer-scan run won't clobber headline data
BATCH, SEED, NPART = 4, 0, 2


def stream_panel(path, tok):
    """yield (token_ids_for_cell, cell_type) using MaxToki's exact tokenisation."""
    with h5py.File(path, "r") as f:
        ens = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]).astype(str)
        ens = np.array([e.split(".")[0] for e in ens])
        ctg = f["obs"]["cell_type"]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in ctg["categories"][:]]).astype(str)
        ctypes = cats[ctg["codes"][:]]
        X = f["X"]; n = int(X.attrs["shape"][0]); indptr = X["indptr"][:]
        var_idx, token_ids, medians = tok.make_var_mapping(list(ens))
        pos = np.full(len(ens), -1, np.int64); pos[var_idx] = np.arange(len(var_idx))
        for r in range(n):
            s, e = int(indptr[r]), int(indptr[r + 1])
            idx, val = X["indices"][s:e], X["data"][s:e].astype(np.float32)
            keep = pos[idx] >= 0
            if not keep.any():
                continue
            j = pos[idx[keep]]
            en = np.log1p(val[keep] / (float(val.sum()) or 1.0) * 1e4)
            nz = en > 0
            if not nz.any():
                continue
            norm = en[nz] / np.maximum(medians[j[nz]], 1e-9)
            order = np.argsort(-norm)[: MAX_LEN - 2]
            yield token_ids[j[nz][order]].astype(np.int64), ctypes[r]


def main():
    from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    rng = np.random.default_rng(SEED)

    # ---- PASS 1: tokenise only, choose contexts and gene panel -------------------------------------------
    print("[pass 1] tokenising to choose contexts and gene panel", flush=True)
    cells = collections.defaultdict(list)
    for p in PANELS:
        path = os.path.join(TS, p)
        if not os.path.exists(path):
            print(f"  [skip] {p}"); continue
        k = 0
        for toks, ct in stream_panel(path, tok):
            cells[ct].append(toks); k += 1
        print(f"  {p}: {k} cells", flush=True)

    ctx_names = sorted([c for c in cells if len(cells[c]) >= 250], key=lambda c: -len(cells[c]))[:N_CTX]
    for c in ctx_names:                       # shuffle then cap, so the occurrence cap is a random subsample
        rng.shuffle(cells[c])
        cells[c] = cells[c][:CELLS_CTX]
    cnt = {c: collections.Counter(int(t) for s in cells[c] for t in s) for c in ctx_names}
    reach = collections.Counter()
    for c in ctx_names:
        for g, n in cnt[c].items():
            if n >= FLOOR:
                reach[g] += 1
    panel = [g for g, k in reach.items() if k >= 2]
    panel.sort(key=lambda g: -sum(cnt[c].get(g, 0) for c in ctx_names))
    panel = sorted(panel[:MAX_GENES])
    gpos = {g: i for i, g in enumerate(panel)}
    print(f"[pass 1] {len(ctx_names)} contexts x {sum(len(cells[c]) for c in ctx_names)} cells; "
          f"panel = {len(panel)} genes reaching {FLOOR} in >=2 contexts", flush=True)
    for c in ctx_names:
        print(f"    {len(cells[c]):>5} cells  {c[:58]}")

    # ---- PASS 2: forward pass, accumulate with an occurrence cap ------------------------------------------
    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32); dev = xt.device
    d = xt.model.config.hidden_size
    acc = {L: np.zeros((NPART, len(ctx_names), len(panel), d), np.float32) for L in TAPS}
    cnts = np.zeros((NPART, len(ctx_names), len(panel)), np.int32)
    print(f"[pass 2] forward pass on {dev}; taps {TAPS}; accumulator "
          f"{sum(a.nbytes for a in acc.values())/2**30:.2f} GB", flush=True)

    work = [(ci, si, s) for ci, c in enumerate(ctx_names) for si, s in enumerate(cells[c])]
    rng.shuffle(work)
    done = 0
    for a in range(0, len(work), BATCH):
        chunk = work[a:a + BATCH]
        L = max(len(s) for _, _, s in chunk) + 2
        ids = np.full((len(chunk), L), tok.EOS, np.int64); am = np.zeros((len(chunk), L), np.int64)
        for j, (_, _, s) in enumerate(chunk):
            sq = np.concatenate([[tok.BOS], s, [tok.EOS]]); ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
        with torch.no_grad():
            out = xt.model(input_ids=torch.from_numpy(ids).to(dev),
                           attention_mask=torch.from_numpy(am).to(dev), output_hidden_states=True)
            hs = {L_: out.hidden_states[L_].to("cpu", torch.float32).numpy() for L_ in TAPS}
        for j, (ci, si, s) in enumerate(chunk):
            part = si % NPART                                   # cell-level split half
            for p_, t in enumerate(s):
                gi = gpos.get(int(t))
                if gi is None or cnts[part, ci, gi] >= CAP:      # occurrence cap == heteroscedasticity fix
                    continue
                cnts[part, ci, gi] += 1
                for L_ in TAPS:
                    acc[L_][part, ci, gi] += hs[L_][j, 1 + p_]
        done += len(chunk)
        if done % 400 < BATCH:
            frac = float((cnts >= CAP).mean())
            print(f"    {done}/{len(work)} cells | {frac:.1%} of (gene,ctx,part) cells at cap", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    tid2ens = {int(v): k for k, v in tokmap.items()}
    for L_ in TAPS:
        M = acc[L_] / np.maximum(cnts[..., None], 1)
        out = os.path.join(HERE, "results", f"{OUTPREFIX}_L{L_:02d}.npz")
        np.savez_compressed(out, M=M.astype(np.float16), counts=cnts,
                            genes=np.array([tid2ens.get(g, str(g)) for g in panel]),
                            contexts=np.array(ctx_names), cap=CAP, max_len=MAX_LEN)
        print(f"  wrote {out}  M{M.shape}")
    print(f"\n[done] {len(panel)} genes x {len(ctx_names)} contexts x {NPART} partitions x {len(TAPS)} taps; "
          f"{float((cnts >= CAP).mean()):.1%} of cells reached the {CAP}-occurrence cap")


if __name__ == "__main__":
    main()
