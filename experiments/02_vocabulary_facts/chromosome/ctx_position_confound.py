"""IS THE GENE-SPECIFIC CONTEXT RESPONSE JUST RANK-POSITION ENCODING?

`ctx_polysemy.py` found a large, reproducible gene-specific context response (EXCESS +0.758 at L4, +0.665 at
L8, +0.627 at L11; exactly 0.000 at L0, the context-free embedding layer, which validates the pipeline).

BEFORE THAT COUNTS AS BIOLOGY, THE OBVIOUS CONFOUND MUST DIE. MaxToki reads a cell as a rank-ordered gene list
and uses RoPE, and this project has already established at length that the model encodes genomic and sequence
POSITION. A gene ranks 5th in a macrophage and 500th in a T cell purely because it is expressed more there.
That rank change is gene-specific, context-specific, and perfectly reproducible across cell partitions --
i.e. it would produce exactly the signal we measured, with no functional content whatsoever.

THE TEST. Recompute each gene's mean rank per context from tokenisation (CPU only, no forward pass), then:

  1. CORRELATION. Does the per-gene shift magnitude ||delta(g)|| track |mean_rank(g,c2) - mean_rank(g,c1)|?
     A strong positive correlation means the effect is substantially positional.
  2. STRATIFICATION. Split genes by how much their rank moved between the two contexts and recompute EXCESS in
     each stratum. If EXCESS is large only where rank moved a lot, and collapses where rank is stable, the
     finding is position. If EXCESS survives in the rank-stable stratum, something else is going on.
  3. RESIDUALISATION. Regress the per-gene shift on rank change (linear, on ranks) and recompute directional
     agreement on the residual.

Stratum (3) is the honest headline number: gene-specific context response AT MATCHED RANK.

Out: results/ctx_position_confound.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, collections, itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
RES = os.path.join(HERE, "results")
TS = f"{_DATA}/raw"
PANELS = ["tabula_sapiens_immune_subset_20000.h5ad", "tabula_sapiens_kidney.h5ad", "tabula_sapiens_lung.h5ad"]
MAX_LEN, CELLS_CTX, SEED = 1024, 1000, 0
TAPS = [4, 8, 11]
from scipy.stats import spearmanr


def mean_ranks(ctx_names):
    """per (context, token) mean rank position, replicating the extractor's tokenisation exactly."""
    from maxtoki_adapter import MaxTokiTokenizer
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    rng = np.random.default_rng(SEED)
    cells = collections.defaultdict(list)
    for p in PANELS:
        path = os.path.join(TS, p)
        if not os.path.exists(path):
            continue
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
                if ctypes[r] not in ctx_names:
                    continue
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
                cells[ctypes[r]].append(token_ids[j[nz][order]].astype(np.int64))
    acc = {}
    for c in ctx_names:
        rng.shuffle(cells[c])
        s = collections.Counter(); n = collections.Counter()
        for toks in cells[c][:CELLS_CTX]:
            for rank, t in enumerate(toks):
                s[int(t)] += rank; n[int(t)] += 1
        acc[c] = {g: s[g] / n[g] for g in s}
    return acc


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def main():
    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    ctxs = z0["contexts"].astype(str); genes = z0["genes"].astype(str)
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    ens2tid = {k: int(v) for k, v in tokmap.items()}
    tids = np.array([ens2tid.get(g, -1) for g in genes])

    print("[1/2] recomputing per-gene mean rank per context (tokenisation only)", flush=True)
    MR = mean_ranks(set(ctxs))
    rank = np.full((len(ctxs), len(genes)), np.nan)
    for ci, c in enumerate(ctxs):
        d = MR.get(c, {})
        for gi, t in enumerate(tids):
            if t in d:
                rank[ci, gi] = d[t]
    print(f"      rank table filled for {np.isfinite(rank).mean():.1%} of (context, gene) cells")

    out = {"taps": {}}
    rng = np.random.default_rng(SEED)
    for tap in TAPS:
        z = np.load(os.path.join(RES, f"ctx_maxtoki_L{tap:02d}.npz"), allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        full = (counts == cap).all(0)
        flat = M[:, full]
        mu = flat.reshape(-1, M.shape[-1]).mean(0); sd = flat.reshape(-1, M.shape[-1]).std(0) + 1e-6
        Mz = (M - mu) / sd

        mags, drank, S_lo, D_lo, S_hi, D_hi, S_res, D_res = [], [], [], [], [], [], [], []
        for c1, c2 in itertools.combinations(range(len(ctxs)), 2):
            keep = full[c1] & full[c2] & np.isfinite(rank[c1]) & np.isfinite(rank[c2])
            if keep.sum() < 200:
                continue
            D0 = Mz[0, c2, keep] - Mz[0, c1, keep]; D1 = Mz[1, c2, keep] - Mz[1, c1, keep]
            d0, d1 = D0 - D0.mean(0), D1 - D1.mean(0)
            dr = np.abs(rank[c2, keep] - rank[c1, keep])
            mags.append(np.linalg.norm((d0 + d1) / 2, axis=1)); drank.append(dr)
            med = np.median(dr)
            lo, hi = dr <= med, dr > med                       # rank-stable vs rank-moving genes
            perm = rng.permutation(keep.sum())
            S_lo.append(cos_rows(d0[lo], d1[lo])); D_lo.append(cos_rows(d0[lo], d1[perm][lo]))
            S_hi.append(cos_rows(d0[hi], d1[hi])); D_hi.append(cos_rows(d0[hi], d1[perm][hi]))
            # residualise both halves on rank change (linear in rank), then re-test direction
            A = np.column_stack([np.ones(keep.sum()), rank[c1, keep], rank[c2, keep], dr])
            proj = lambda V: V - A @ np.linalg.lstsq(A, V, rcond=None)[0]
            r0, r1 = proj(d0), proj(d1)
            S_res.append(cos_rows(r0, r1)); D_res.append(cos_rows(r0, r1[perm]))

        mags = np.concatenate(mags); drank = np.concatenate(drank)
        rho = float(spearmanr(mags, drank).statistic)
        ex = lambda S, D: float(np.concatenate(S).mean() - np.concatenate(D).mean())
        e_lo, e_hi, e_res = ex(S_lo, D_lo), ex(S_hi, D_hi), ex(S_res, D_res)
        print(f"\n=== layer {tap} ===")
        print(f"  shift magnitude vs |rank change|        rho = {rho:+.3f}")
        print(f"  EXCESS, rank-STABLE genes (below median) : {e_lo:+.4f}")
        print(f"  EXCESS, rank-MOVING genes (above median) : {e_hi:+.4f}")
        print(f"  EXCESS after residualising on rank       : {e_res:+.4f}   <-- the honest number")
        out["taps"][f"L{tap:02d}"] = dict(rho_mag_vs_rank=rho, excess_rank_stable=e_lo,
                                          excess_rank_moving=e_hi, excess_residualised=e_res)

    best = max(out["taps"].values(), key=lambda v: v["excess_residualised"])
    out["verdict"] = (
        f"residualised EXCESS {best['excess_residualised']:+.4f}. " +
        ("SURVIVES rank control — the gene-specific context response is not merely token-position encoding."
         if best["excess_residualised"] > 0.05 else
         "COLLAPSES under rank control — the apparent gene-specific context response is substantially "
         "token-rank position, which this model is already known to encode. Not biology."))
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_position_confound.json"), "w"), indent=1)
    print("[done] -> results/ctx_position_confound.json")


if __name__ == "__main__":
    main()
