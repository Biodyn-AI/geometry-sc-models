"""EXTRACT MaxToki contextual gene representations across HEMATOPOIETIC DIFFERENTIATION STAGES.

Same machinery as ctx_extract_maxtoki.py, but the "contexts" are developmental branch clusters from the Setty
CD34 hematopoiesis data (HSC_1/2, Precursors, CLP, Ery_1, Mono_1/2, DCs), each carrying a palantir pseudotime.
This lets us ask how a gene's contextual representation moves along a KNOWN differentiation trajectory with a
KNOWN branch structure -- the developmental analogue of the mature-cell-type analysis.

Design choices identical to the mature extraction: two independent cell partitions; an occurrence cap per
(gene, cluster, partition) so counts are balanced (kills heteroscedasticity); pairwise-comparable panel; taps at
the layers where the effect lives (L2 peak, L4 where the functional axes validated).

Genes are Setty symbols -> Ensembl (Geneformer name_id) -> MaxToki token. X is integer counts (verified), so the
Geneformer log1p-CP10k / gene-median / rank tokenisation applies directly.

Out: results/ctx_devel_L{tap}.npz  (M[part, cluster, gene, dim], counts, genes, clusters, pseudotime per cluster)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, collections, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
DATA = f"{_DATA}/branchpoint/setty_xstate.h5ad"
MDIR = f"{MSETUP}/MaxToki-217M-HF"

MAX_LEN = 1024
MIN_CELLS = 130
CAP = 25
FLOOR = 15
MAX_GENES = 5000
TAPS = [int(x) for x in os.environ.get("TAPS", "2,4").split(",")]
BATCH, SEED, NPART = 4, 0, 2


def load():
    with h5py.File(DATA, "r") as f:
        syms = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]]).astype(str)
        cl = f["obs"]["clusters"]
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in cl["categories"][:]]).astype(str)
        clusters = cats[cl["codes"][:]]
        pt = f["obs"]["palantir_pseudotime"][:]
        X = f["X"]; n = int(X.attrs["shape"][0]); indptr = X["indptr"][:]
        cells = []
        for r in range(n):
            s, e = int(indptr[r]), int(indptr[r + 1])
            cells.append((X["indices"][s:e], X["data"][s:e].astype(np.float32)))
    return syms, cells, clusters, pt


def main():
    from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    name_id = {k.upper(): v for k, v in pickle.load(open(NAME_ID, "rb")).items()}
    rng = np.random.default_rng(SEED)

    syms, cells, clusters, pt = load()
    ens = [name_id.get(s.upper()) for s in syms]                       # symbol -> ensembl (None if unmapped)
    var_idx, token_ids, medians = tok.make_var_mapping(ens)            # positions within var that map to vocab
    pos = np.full(len(syms), -1, np.int64); pos[var_idx] = np.arange(len(var_idx))

    keep_clusters = [c for c, k in collections.Counter(clusters).items() if k >= MIN_CELLS]
    keep_clusters = sorted(keep_clusters, key=lambda c: np.median(pt[clusters == c]))   # order by pseudotime
    ptime = {c: float(np.mean(pt[clusters == c])) for c in keep_clusters}
    print(f"[setup] {len(keep_clusters)} branch clusters (pseudotime-ordered): "
          + ", ".join(f"{c}({ptime[c]:.2f})" for c in keep_clusters), flush=True)

    # tokenise every cell in kept clusters
    seqs, cell_ctx = [], []
    for (idx, val), c in zip(cells, clusters):
        if c not in keep_clusters:
            continue
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
        seqs.append(token_ids[j[nz][order]].astype(np.int64)); cell_ctx.append(c)
    cidx = {c: i for i, c in enumerate(keep_clusters)}
    print(f"[setup] {len(seqs)} cells tokenised", flush=True)

    # panel: genes reaching FLOOR in >= 2 clusters
    cnt = {c: collections.Counter() for c in keep_clusters}
    for sq, c in zip(seqs, cell_ctx):
        cnt[c].update(int(t) for t in sq)
    reach = collections.Counter()
    for c in keep_clusters:
        for g, k in cnt[c].items():
            if k >= FLOOR:
                reach[g] += 1
    panel = sorted([g for g, k in reach.items() if k >= 2],
                   key=lambda g: -sum(cnt[c].get(g, 0) for c in keep_clusters))[:MAX_GENES]
    panel = sorted(panel); gpos = {g: i for i, g in enumerate(panel)}
    print(f"[setup] panel = {len(panel)} genes reaching {FLOOR} in >=2 clusters", flush=True)

    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32); dev = xt.device
    d = xt.model.config.hidden_size
    acc = {L: np.zeros((NPART, len(keep_clusters), len(panel), d), np.float32) for L in TAPS}
    cnts = np.zeros((NPART, len(keep_clusters), len(panel)), np.int32)

    part_of = {}
    seen = collections.Counter()
    order_cells = list(range(len(seqs))); rng.shuffle(order_cells)
    for a in range(0, len(order_cells), BATCH):
        chunk = order_cells[a:a + BATCH]
        L = max(len(seqs[i]) for i in chunk) + 2
        ids = np.full((len(chunk), L), tok.EOS, np.int64); am = np.zeros((len(chunk), L), np.int64)
        for jb, i in enumerate(chunk):
            sq = np.concatenate([[tok.BOS], seqs[i], [tok.EOS]]); ids[jb, :len(sq)] = sq; am[jb, :len(sq)] = 1
        with torch.no_grad():
            out = xt.model(input_ids=torch.from_numpy(ids).to(dev),
                           attention_mask=torch.from_numpy(am).to(dev), output_hidden_states=True)
            hs = {L_: out.hidden_states[L_].to("cpu", torch.float32).numpy() for L_ in TAPS}
        for jb, i in enumerate(chunk):
            c = cell_ctx[i]; ci = cidx[c]
            seen[c] += 1; part = seen[c] % NPART                      # cell-level split within each cluster
            for p_, t in enumerate(seqs[i]):
                gi = gpos.get(int(t))
                if gi is None or cnts[part, ci, gi] >= CAP:
                    continue
                cnts[part, ci, gi] += 1
                for L_ in TAPS:
                    acc[L_][part, ci, gi] += hs[L_][jb, 1 + p_]
        if (a // BATCH) % 100 == 0:
            print(f"    {a}/{len(seqs)} cells | {float((cnts >= CAP).mean()):.1%} at cap", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json")); tid2ens = {int(v): k for k, v in tokmap.items()}
    for L_ in TAPS:
        M = acc[L_] / np.maximum(cnts[..., None], 1)
        out = os.path.join(HERE, "results", f"ctx_devel_L{L_:02d}.npz")
        np.savez_compressed(out, M=M.astype(np.float16), counts=cnts,
                            genes=np.array([tid2ens.get(g, str(g)) for g in panel]),
                            clusters=np.array(keep_clusters),
                            pseudotime=np.array([ptime[c] for c in keep_clusters]), cap=CAP)
        print(f"  wrote {out}  M{M.shape}")
    print(f"[done] {len(panel)} genes x {len(keep_clusters)} clusters x {NPART} partitions x {len(TAPS)} taps; "
          f"{float((cnts >= CAP).mean()):.1%} at cap")


if __name__ == "__main__":
    main()
