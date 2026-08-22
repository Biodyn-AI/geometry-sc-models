"""BUILD ctx_scgpt_L*.npz from the existing scGPT per-gene contextual atlas — for the cross-model comparison.

The scGPT whole-human atlas (subproject_42/scgpt_atlas/activations) already stores per-gene residual-stream
activations for all 12 layers, over the SAME 3000 Tabula Sapiens cells (immune+kidney+lung) used across the
project, with a cell_type per cell. We reformat it into the same schema as ctx_maxtoki_L*.npz, applying the
Level-1 discipline: contexts = cell types with >= MIN_CELLS; two independent cell PARTITIONS; an occurrence CAP
per (gene, context, partition) so counts are balanced (kills heteroscedasticity).

CAVEAT (inherited, documented in gm_lib.build_ctx): the atlas was extracted under scGPT's raw-count value
convention (the project's 51-bin issue), so absolute activations are off-distribution. The Level-1 analyses read
CONTRASTS (gene x context interaction, functional-axis projections), which average out the mis-binned value
channel and leave gene identity -- usable for a cross-model comparison, but read this scGPT result as
directional relative to MaxToki, not as a calibrated absolute.

d_model = 512, 12 layers. Genes are scGPT tokens -> symbol -> Ensembl (stored, so the shared analysis works).
Out: results/ctx_scgpt_L{tap}.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
ATLAS = (f"{_DATA}/biodyn-work/subproject_42_sparse_autoencoder_biological_map/"
         "experiments/scgpt_atlas/activations")
META = (f"{_DATA}/biodyn-work/subproject_42_sparse_autoencoder_biological_map/"
        "experiments/phase3_multitissue/ts_activations/extraction_metadata.json")
TAPS = [int(x) for x in os.environ.get("TAPS", "0,4,8").split(",")]
MIN_CELLS = 100
CAP = 20
FLOOR = 12
MAX_GENES = 5000
NPART, SEED = 2, 0


def capped_group(pig, cap):
    """given gene-index per position (unsorted), return (positions_to_keep, gene_of_those) capping each gene at
    `cap` occurrences. Vectorised via sort + within-group rank."""
    order = np.argsort(pig, kind="stable")
    pig_s = pig[order]
    start = np.searchsorted(pig_s, pig_s, side="left")
    within = np.arange(len(pig_s)) - start
    keep = within < cap
    return order[keep], pig_s[keep]


def main():
    sym2ens = {}
    for s, e in pickle.load(open(NAME_ID, "rb")).items():
        sym2ens.setdefault(s.upper(), e)
    tid2sym = {int(k): v for k, v in json.load(open(os.path.join(ATLAS, "token_id_to_gene_name.json"))).items()}
    cell_data = json.load(open(META))["cell_data"]
    ctypes = np.array([c["cell_type"] for c in cell_data])

    contexts = [c for c, n in collections.Counter(ctypes).items() if n >= MIN_CELLS]
    contexts = sorted(contexts, key=lambda c: -(ctypes == c).sum())
    print(f"[setup] {len(contexts)} contexts (>= {MIN_CELLS} cells): "
          + ", ".join(f"{c[:18]}({int((ctypes==c).sum())})" for c in contexts), flush=True)
    cidx = {c: i for i, c in enumerate(contexts)}

    # cell -> context index and partition (split each context's cells into 2)
    rng = np.random.default_rng(SEED)
    cell_ctx = np.full(len(cell_data), -1, np.int64); cell_part = np.full(len(cell_data), -1, np.int64)
    for c in contexts:
        cells = np.where(ctypes == c)[0]; rng.shuffle(cells)
        cell_ctx[cells] = cidx[c]; cell_part[cells[: len(cells) // 2]] = 0; cell_part[cells[len(cells) // 2:]] = 1

    # panel: count gene token occurrences per context (use layer 4 gid/cid), keep genes reaching FLOOR in >=2
    gid = np.load(os.path.join(ATLAS, "layer_04_gene_ids.npy"), mmap_mode="r")
    cid = np.load(os.path.join(ATLAS, "layer_04_cell_ids.npy"), mmap_mode="r")
    gid = np.asarray(gid); cid = np.asarray(cid)
    ctx_of_pos = cell_ctx[cid]
    keeppos = ctx_of_pos >= 0
    per_ctx_cnt = collections.defaultdict(collections.Counter)
    G, Cp = gid[keeppos], ctx_of_pos[keeppos]
    for g, c in zip(G, Cp):
        per_ctx_cnt[int(c)][int(g)] += 1
    reach = collections.Counter()
    for c in range(len(contexts)):
        for g, k in per_ctx_cnt[c].items():
            if k >= FLOOR:
                reach[g] += 1
    panel_tok = [g for g, k in reach.items() if k >= 2 and int(g) in tid2sym and tid2sym[int(g)].upper() in sym2ens]
    panel_tok.sort(key=lambda g: -sum(per_ctx_cnt[c].get(g, 0) for c in range(len(contexts))))
    panel_tok = panel_tok[:MAX_GENES]
    gpos = {int(t): i for i, t in enumerate(panel_tok)}
    panel_ens = np.array([sym2ens[tid2sym[int(t)].upper()] for t in panel_tok])
    print(f"[setup] panel = {len(panel_tok)} genes (reach {FLOOR} in >=2 contexts, mappable to Ensembl)", flush=True)

    # map full gid -> panel index once
    maxtok = int(gid.max()) + 1
    tok2panel = np.full(maxtok, -1, np.int64)
    for t, i in gpos.items():
        tok2panel[t] = i

    for tap in TAPS:
        act = np.load(os.path.join(ATLAS, f"layer_{tap:02d}_activations.npy"), mmap_mode="r")
        gidL = np.asarray(np.load(os.path.join(ATLAS, f"layer_{tap:02d}_gene_ids.npy"), mmap_mode="r"))
        cidL = np.asarray(np.load(os.path.join(ATLAS, f"layer_{tap:02d}_cell_ids.npy"), mmap_mode="r"))
        d = act.shape[1]
        acc = np.zeros((NPART, len(contexts), len(panel_tok), d), np.float32)
        cnts = np.zeros((NPART, len(contexts), len(panel_tok)), np.int32)
        cc = cell_ctx[cidL]; pp = cell_part[cidL]; pig_all = tok2panel[gidL]
        for c in range(len(contexts)):
            for p in range(NPART):
                sel = np.where((cc == c) & (pp == p) & (pig_all >= 0))[0]
                if len(sel) == 0:
                    continue
                keep_local, genes_k = capped_group(pig_all[sel], CAP)
                pos = np.sort(sel[keep_local])                     # sorted for faster memmap gather
                a = np.asarray(act[pos], dtype=np.float32)
                gk = pig_all[pos]
                np.add.at(acc[p, c], gk, a)
                np.add.at(cnts[p, c], gk, 1)
            print(f"  L{tap:02d} context {contexts[c][:22]:<24} done", flush=True)
        M = acc / np.maximum(cnts[..., None], 1)
        out = os.path.join(HERE, "results", f"ctx_scgpt_L{tap:02d}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, M=M.astype(np.float16), counts=cnts, genes=panel_ens,
                            contexts=np.array(contexts), cap=CAP)
        print(f"  wrote {out}  M{M.shape}  ({float((cnts>=CAP).mean()):.0%} at cap)", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
