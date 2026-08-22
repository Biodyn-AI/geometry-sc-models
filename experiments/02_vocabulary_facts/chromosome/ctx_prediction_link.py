"""DOES CONTEXTUALISATION SERVE THE MODEL'S OBJECTIVE? (the 'so what', and it closes the loop)

We have shown genes are contextualised, and that the contextualisation IS co-expression. The missing link is
WHY the model does it. MaxToki is autoregressive over rank-ordered genes: at each position it predicts the next
gene. If contextualisation is functional for that objective, then the genes the model contextualises MORE should
be the genes whose prediction benefits MORE from having the real (co-expressed) context present.

INTERVENTION (a clean causal measure of 'does context help predict gene g'):
  For a target gene g at rank-position p in a real cell, compare the model's log-probability of g given
    - REAL prefix:     [BOS] + the cell's own genes ranked 1..p-1     (the true co-expression context)
    - CHIMERIC prefix: [BOS] + a random OTHER cell's genes ranked 1..p-1  (a context lacking g's co-expression)
  context_benefit(g) = logP(g | real) - logP(g | chimeric), averaged over occurrences.
Reading logits only at the final prefix position = the model's own next-gene distribution; no probe, no
circularity. The gene set differs between real and chimeric, so this isolates the contribution of WHICH genes
co-occur, not rank order alone.

CONTEXTUALISATION(g): from the extraction, how much g's representation moves across cell types --
  mean over the contexts where g is count-balanced of || z(g,c) - mean_c z(g,c) ||  (z-scored L4 space).

THE TEST: partial Spearman of context_benefit(g) vs contextualisation(g), controlling for log gene frequency
(frequent genes are both better predicted and more sampled). A POSITIVE partial correlation means the model
contextualises exactly the genes whose prediction its context improves -- i.e. contextualisation is the model
encoding the co-expression it predicts from, unifying the representation phenomenon with the co-expression
ceiling and with the training objective.

Out: results/ctx_prediction_link.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import steer_lib as SL
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
TS = f"{_DATA}/raw"
PANEL = "tabula_sapiens_immune_subset_20000.h5ad"
N_CELLS, POS_PER_CELL, MAX_LEN, SEED = 350, 20, 512, 0
from scipy.stats import spearmanr, rankdata


def tokenise(tok, n, rng):
    with h5py.File(os.path.join(TS, PANEL), "r") as f:
        ens = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]).astype(str)
        ens = np.array([e.split(".")[0] for e in ens])
        var_idx, token_ids, medians = tok.make_var_mapping(list(ens))
        pos = np.full(len(ens), -1, np.int64); pos[var_idx] = np.arange(len(var_idx))
        X = f["X"]; N = int(X.attrs["shape"][0]); indptr = X["indptr"][:]
        sel = np.sort(rng.choice(N, min(n, N), replace=False)); seqs = []
        for r in sel:
            s, e = int(indptr[r]), int(indptr[r + 1]); idx, val = X["indices"][s:e], X["data"][s:e].astype(np.float32)
            keep = pos[idx] >= 0
            if not keep.any():
                continue
            j = pos[idx[keep]]; en = np.log1p(val[keep] / (float(val.sum()) or 1.0) * 1e4); nz = en > 0
            if nz.sum() < 60:
                continue
            norm = en[nz] / np.maximum(medians[j[nz]], 1e-9); order = np.argsort(-norm)[: MAX_LEN - 2]
            seqs.append(token_ids[j[nz][order]].astype(np.int64))
    return seqs


def contextualisation_per_gene():
    """per-gene across-context movement in z-scored L4 space; returns {tid: value}."""
    z = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    M, counts, cap, genes = z["M"].astype(np.float32), z["counts"], int(z["cap"]), z["genes"].astype(str)
    full = (counts == cap).all(0); d = M.shape[-1]
    flat = M[:, full]; mu = flat.reshape(-1, d).mean(0); sd = flat.reshape(-1, d).std(0) + 1e-6
    Mz = (M - mu) / sd; Mavg = Mz.mean(0)                      # (nctx, ngene, d)
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json")); ens2tid = {k: int(v) for k, v in tokmap.items()}
    out = {}
    for gi in range(len(genes)):
        cs = np.where(full[:, gi])[0]
        if len(cs) < 4:
            continue
        V = Mavg[cs, gi]; ctr = V - V.mean(0)
        out[ens2tid.get(genes[gi], -1)] = float(np.linalg.norm(ctr, axis=1).mean())
    out.pop(-1, None)
    return out


def main():
    st = SL.Steerer()
    rng = np.random.default_rng(SEED)
    seqs = tokenise(st.tok, N_CELLS, rng)
    print(f"[setup] {len(seqs)} cells", flush=True)
    ctxal = contextualisation_per_gene()
    print(f"[setup] contextualisation for {len(ctxal)} genes", flush=True)

    BOS, EOS = st.tok.BOS, st.tok.EOS
    benefit = {}                    # tid -> list of logP(real) - logP(chimeric)
    freq = {}
    def logprob(prefix_tokens, target):
        ids = np.concatenate([[BOS], prefix_tokens]).astype(np.int64)
        lg = st.logits(ids)                                   # torch tensor (seq, vocab), float32 on CPU
        lp = torch.log_softmax(lg[-1], -1)
        return float(lp[int(target)])

    for si, s in enumerate(seqs):
        if len(s) < 20:
            continue
        positions = rng.choice(np.arange(5, len(s)), min(POS_PER_CELL, len(s) - 5), replace=False)
        other = seqs[rng.integers(0, len(seqs))]
        for p in positions:
            g = int(s[p])
            if g not in ctxal:
                continue
            real_lp = logprob(s[:p], g)
            oth = other[:p] if len(other) >= p else np.concatenate([other, s[:p - len(other)]])
            chim_lp = logprob(oth[:p], g)
            benefit.setdefault(g, []).append(real_lp - chim_lp)
            freq[g] = freq.get(g, 0) + 1
        if si % 25 == 0:
            print(f"    cell {si}/{len(seqs)}", flush=True)

    genes = [g for g, v in benefit.items() if len(v) >= 3 and g in ctxal]
    ben = np.array([np.mean(benefit[g]) for g in genes])
    ctx = np.array([ctxal[g] for g in genes])
    frq = np.array([np.log(freq[g]) for g in genes])
    # global sanity: does context help at all?
    all_ben = np.concatenate([benefit[g] for g in genes])
    print(f"\n[result] {len(genes)} target genes; mean context benefit (logP real - chimeric) = "
          f"{all_ben.mean():+.3f} (positive => real context helps prediction)")

    def partial(a, b, c):
        ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
        A = np.column_stack([np.ones_like(rc), rc]); res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        return float(spearmanr(res(ra), res(rb)).statistic)
    raw = float(spearmanr(ben, ctx).statistic)
    par = partial(ben, ctx, frq)
    rng2 = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        i = rng2.integers(0, len(genes), len(genes))
        try: bs.append(partial(ben[i], ctx[i], frq[i]))
        except Exception: pass
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print(f"[result] benefit ~ contextualisation: raw rho {raw:+.3f}; partial (control log-freq) {par:+.3f} "
          f"95% CI [{lo:+.3f},{hi:+.3f}]")
    out = dict(n_genes=len(genes), mean_context_benefit=float(all_ben.mean()),
               raw_rho=raw, partial_rho=par, partial_ci=[lo, hi], n_cells=len(seqs))
    out["verdict"] = (
        (f"CONTEXT HELPS PREDICTION (mean benefit {all_ben.mean():+.2f} nats) AND the model contextualises the "
         f"genes it helps most (partial rho {par:+.3f}, CI [{lo:+.3f},{hi:+.3f}] controlling frequency). "
         "Contextualisation serves the objective: the model encodes the co-expression it predicts from — "
         "unifying the representation phenomenon, the co-expression ceiling, and the training objective."
         if lo > 0 else
         f"Context helps prediction (mean benefit {all_ben.mean():+.2f}) but its per-gene strength does NOT track "
         f"contextualisation once frequency is controlled (partial rho {par:+.3f}, CI [{lo:+.3f},{hi:+.3f}]). "
         "Contextualisation and context-benefit are separate; do not claim the objective link."))
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_prediction_link.json"), "w"), indent=1)
    print("[done] -> results/ctx_prediction_link.json")


if __name__ == "__main__":
    main()
