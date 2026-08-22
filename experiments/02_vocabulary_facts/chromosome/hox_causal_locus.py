"""DOES MAXTOKI'S COMPUTATION *USE* THE HOX CLUSTER STRUCTURE, OR ONLY CARRY IT AS A READABLE CORRELATE?
(Ihor, 2026-07-17 — "ablate the axis, don't just read it", the recommendation the paper (materials/paper.pdf)
and this route jointly arrive at.)

The HEADLINE (RESULTS.md) is read entirely off the WEIGHTS: lm_head's HOX rows separate the four genomic
clusters at held-out 4-class acc 0.884 (217M). That is a statement about what is DECODABLE, not about what the
forward pass DOES. Section 6 made exactly this distinction for the secretory axis and found the model does not
use it. This file points that test at the headline.

------------------------------------------------------------------------------------------------------------
WHY THE OBVIOUS ABLATION IS CIRCULAR (documented; see hox_causal_locus_circular note in RESULTS.md).
------------------------------------------------------------------------------------------------------------
Build the rank-3 cluster subspace U_clu from lm_head's HOX rows, ablate it from residual h, watch CE rise. It
DOES rise, but so does ablating the PARALOG subspace (both are built from HOX rows), while a random rank-3 does
nothing. To predict any specific HOX gene g the model must load h with mass along g's lm_head row; HOX rows are
cluster-clustered, so ablating U_clu drops g's logit BY ARITHMETIC even for a model with no cluster concept.
Two near-orthogonal HOX-derived subspaces both hurt => gene-identity tautology, not evidence. (§6's "removing
any direction hurts", sharpened for a categorical weight-derived subspace.) So CE-ablation cannot be the test.

------------------------------------------------------------------------------------------------------------
THE VALID TEST — subspace-free, no lm_head projection, so nothing to be circular about.
------------------------------------------------------------------------------------------------------------
At a non-HOX position p whose context already contains some HOX tokens, read the logit the model assigns to
every HOX gene k that is ABSENT from the whole cell (absent -> no repeat-suppression, no leakage). Regress:

    logit(k @ p)  ~  beta * [cluster(k) in context]  +  gamma * [paralog(k) in context]
                     +  gene_FE(k)  +  (cell x position)_FE

  * gene_FE absorbs each HOX gene's baseline favourability (beta/gamma are NOT "which genes exist").
  * (cell x position)_FE absorbs the local context's general HOX-friendliness (the co-expression *level*).
  * beta and gamma are estimated JOINTLY, so beta is the cluster effect CONTROLLING FOR paralog (§3 done
    causally): does seeing HOXA genes raise OTHER absent HOXA genes, beyond the paralog co-expression axis
    colinearity (§2) guarantees?

  gamma > 0 is EXPECTED (paralog co-expresses across clusters; a model that uses expression must show it) — the
  positive control. beta is the question: a "chr7 context -> predict chr7 genes" mechanism gives beta > 0; a
  merely-readable table gives beta ~ 0.

CONTROLS.
  PLACEBO NULL — permute the gene->cluster labelling over the 39 genes, recompute `match` from each row's stored
    context-cluster bitmask, refit beta. A real cluster effect COLLAPSES; an FE/paralog artifact survives.
  PER-CLUSTER beta — is the effect broad or driven by one well-populated cluster?

Datasets (arg 2). The forward pass conditions on cluster only where the TISSUE activates that cluster, so the
mechanism test is to rotate the tissue and see if the USED cluster follows the dominant one:
  * fetal_gut  — HOXB-dominant (co-occur B 54% / A 45% / C 13% / D 6%). Headline run. beta_B=+0.26, beta_A~0.
  * agingmds   — bone marrow, HOXA-dominant (A 23% / B 6%, C&D absent). THE FLIP TEST: predict beta_A lights up.
  * aging      — bone marrow, HOXA-dominant (A 17% / B 3%). Replication of the flip.
(Setty19 CD34 BM, causal_ablate.py's dataset, is HOX-DEAD — A co-occur 4%, C&D absent; too thin, not used.)
The tokenizer ranks genes by normalised expression, and rank is invariant to a monotone re-scaling, so log1p on
either raw counts or an already-normalised matrix gives the same gene ordering — both dataset conventions work.

Run: ../../.venv_state/bin/python -u hox_causal_locus.py [n_cells] [dataset]  (default 2500 fetal_gut; ~1 s/cell)
Out: results/hox_causal_locus_<dataset>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, f"{_DATA}/maxtoki/setup")
import gm_lib as G, gene_sets as S
from maxtoki_adapter import MaxTokiAttentionExtractor, MaxTokiTokenizer

MDIR = f"{_DATA}/maxtoki/setup/MaxToki-217M-HF"
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
TOKMAP = f"{_DATA}/maxtoki/setup/token_dictionary.json"
N_CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
DATASET = sys.argv[2] if len(sys.argv) > 2 else "fetal_gut"
MAX_LEN = 2048
POS_STRIDE = 7
N_PLACEBO = 200
SEED = 0
CN = "ABCD"

_D = f"{_DATA}"
DATASETS = {                                    # dominant HOX cluster in parentheses -- the flip axis
    "fetal_gut": G.FETAL_GUT,                   # HOXB
    "agingmds": f"{_D}/aging/agingmds_setty_schema.h5ad",   # HOXA
    "aging": f"{_D}/aging/aging_setty_schema.h5ad",         # HOXA
}


def load_syms(f):
    """Robust gene-symbol read: categorical feature_name (fetal gut) or a plain array (setty_schema)."""
    for key in ("feature_name", "gene_name", "gene_symbols"):
        if key in f["var"]:
            v = f["var"][key]
            if isinstance(v, h5py.Group) and "categories" in v:
                return np.char.upper(_dec(v["categories"][:]).astype(str)[v["codes"][:]])
            return np.char.upper(_dec(v[:]).astype(str))
    idxk = "_index" if "_index" in f["var"] else "index"
    return np.char.upper(_dec(f["var"][idxk][:]).astype(str))

h = S.H["hox_grid"]
HOXG = np.array(h["genes"]); CLU = np.asarray(h["coord"][1], int); PAR = np.asarray(h["coord"][0], int)
sym2clu = {g: c for g, c in zip(HOXG, CLU)}; sym2par = {g: p for g, p in zip(HOXG, PAR)}


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def make_demean(g1, g2):
    """Precompute group indices ONCE; return a fast two-way (g1 FE + g2 FE) residualiser via alternating
    projection. g1/g2 are integer label arrays. bincount on precomputed inverse indices avoids re-running
    np.unique per call — the placebo refits the demeaner hundreds of times, so this matters."""
    u1, i1 = np.unique(g1, return_inverse=True)
    u2, i2 = np.unique(g2, return_inverse=True)
    c1 = np.bincount(i1); c2 = np.bincount(i2)

    def demean(v, n_iter=40):
        v = v.copy()
        for _ in range(n_iter):
            v -= (np.bincount(i1, weights=v) / c1)[i1]
            v -= (np.bincount(i2, weights=v) / c2)[i2]
        return v
    return demean


def fit_beta_gamma(Y, M1, M2):
    Xd = np.stack([M1, M2], 1)
    b, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return Xd, b


def main():
    torch.manual_seed(SEED)
    name_id = pickle.load(open(NAME_ID, "rb"))
    tokmap = json.load(open(TOKMAP))
    hox_tid = {g: int(tokmap[name_id[g]]) for g in HOXG if g in name_id and name_id[g] in tokmap}
    HG = sorted(hox_tid)
    HT = np.array([hox_tid[g] for g in HG])
    HC = np.array([sym2clu[g] for g in HG]); HP = np.array([sym2par[g] for g in HG])
    HTset = {int(t): k for k, t in enumerate(HT)}
    print(f"[setup] {len(HG)} HOX genes tokenised; per cluster {np.bincount(HC).tolist()}", flush=True)

    # ---- ROW CACHE. The forward pass is the ONLY expensive part (~1 s/cell = ~40-70 min); the regression is
    # seconds. Cache the extracted rows keyed by (dataset, n_cells) so a regression-code bug never costs a
    # re-extraction again (learned the hard way: a per-cluster SE crash discarded a 67-min run). A cache hit
    # skips the model, the tokenizer AND the h5ad load entirely — so refits are instant and never contend with a
    # concurrent forward pass for GPU/RAM (this box crashes on concurrent heavy jobs).
    cache = os.path.join(HERE, "results", f"cache_rows_{DATASET}_{N_CELLS}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        R, dt = z["R"], float(z["dt"])
        print(f"[cache] loaded {len(R):,} rows from {os.path.basename(cache)} — no model/data load", flush=True)
        rows = None
    else:
        xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32); dev = xt.device
        tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
        dpath = DATASETS[DATASET]
        print(f"[data] {DATASET}: {dpath}", flush=True)
        with h5py.File(dpath, "r") as f:
            syms = load_syms(f)
            X = f["X"]; n, g = (int(v) for v in X.attrs["shape"])
            rng = np.random.default_rng(SEED); sel = np.sort(rng.choice(n, min(N_CELLS, n), replace=False))
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            E = np.zeros((len(sel), g), np.float32)
            for i, r in enumerate(sel):
                s, e = int(indptr[r]), int(indptr[r + 1]); E[i, idx[s:e]] = data[s:e]
        var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in syms])
        seqs = []
        for i in range(len(E)):
            rs = E[i].sum() or 1.0
            en = np.log1p(E[i] / rs * 1e4)[var_idx]; nz = en > 0
            nm = np.zeros_like(en); nm[nz] = en[nz] / medians[nz]
            seqs.append(np.nonzero(nz)[0][np.argsort(-nm[nz])][:MAX_LEN - 2])
        rows = []
    t0 = time.time()
    for a in ([] if rows is None else range(0, len(seqs), 8)):
        ch = seqs[a:a + 8]; L = max(len(s) for s in ch) + 2
        ids = np.full((len(ch), L), tok.EOS, np.int64); am = np.zeros((len(ch), L), np.int64)
        for j, s in enumerate(ch):
            sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]]); ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
        with torch.no_grad():
            lg = xt.model(input_ids=torch.from_numpy(ids).to(dev),
                          attention_mask=torch.from_numpy(am).to(dev), use_cache=False).logits.float().cpu().numpy()
        for j in range(len(ch)):
            hx = [(p, HTset[int(ids[j, p])]) for p in range(L) if am[j, p] and int(ids[j, p]) in HTset]
            if not hx:
                continue
            seen = {k for _, k in hx}; hpos = np.array([p for p, _ in hx])
            lo, hi = hpos.min() + 1, min(hpos.max() + 50, int(am[j].sum()) - 1)
            cand = [p for p in range(lo, hi + 1) if am[j, p] and int(ids[j, p]) not in HTset]
            for p in cand[::POS_STRIDE]:
                ctx = [k for q, k in hx if q < p]
                if not ctx:
                    continue
                cclu = set(int(HC[k]) for k in ctx); cpar = set(int(HP[k]) for k in ctx)
                gmask = 0
                for k in ctx:
                    gmask |= (1 << int(k))            # 39-bit context-GENE bitmask (for the exact placebo null)
                for k in range(len(HG)):
                    if k in seen:                     # only genes ABSENT from the whole cell
                        continue
                    rows.append((a + j, p, k, float(lg[j, p, HT[k]]),
                                 int(int(HC[k]) in cclu), int(int(HP[k]) in cpar), gmask))
        if a % 160 == 0:
            print(f"  cells {a + len(ch)}/{len(seqs)}  rows {len(rows):,}  {time.time() - t0:.0f}s", flush=True)

    if rows is not None:
        dt = time.time() - t0
        R = np.array(rows, dtype=np.float64)
        np.savez(cache, R=R, dt=dt)                 # persist BEFORE any regression, so a fit bug is free to fix
        print(f"[cache] saved {len(R):,} rows -> {os.path.basename(cache)}", flush=True)

    cell = R[:, 0].astype(int); posn = R[:, 1].astype(int); gene = R[:, 2].astype(int)
    y = R[:, 3]; match = R[:, 4]; parm = R[:, 5]; gmask = R[:, 6].astype(np.int64)
    cp = cell * 1_000_000 + posn
    nHG = len(HG)
    # context-membership matrix Ctx[i,k] = 1 iff HOX gene k is in row i's context (unpack the 39-bit gmask)
    Ctx = ((gmask[:, None] >> np.arange(nHG)[None, :]) & 1).astype(np.float32)
    print(f"\n[data] {len(y):,} rows over {len(np.unique(cell))} cells in {dt:.0f}s; "
          f"match {match.mean():.3f}, parmatch {parm.mean():.3f}, "
          f"corr(match,parm)={np.corrcoef(match, parm)[0, 1]:+.3f}", flush=True)

    demean = make_demean(gene, cp)
    Y = demean(y); M2 = demean(parm); M1 = demean(match)
    Xd, b = fit_beta_gamma(Y, M1, M2)
    res = Y - Xd @ b
    u, inv = np.unique(cell, return_inverse=True); XtX = np.linalg.inv(Xd.T @ Xd)
    meat = np.zeros((2, 2))
    for c in range(len(u)):
        m = inv == c; s = (Xd[m] * res[m, None]).sum(0); meat += np.outer(s, s)
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    beta, gamma = float(b[0]), float(b[1]); se_b, se_g = float(se[0]), float(se[1])

    # ---- PLACEBO NULL (exact): permute the gene->cluster labelling over the 39 HOX genes (preserving the
    # 11/10/9/9 sizes), then recompute `match` CONSISTENTLY on both sides — the candidate's cluster AND the
    # context's clusters both move under the permuted labels. This is the gm_lib annotation-permuting null.
    # A real cluster-conditioning effect must COLLAPSE toward 0; an FE/demeaning artifact survives.
    HC_by_gene = HC.astype(int)
    rngp = np.random.default_rng(SEED)
    placebo = np.empty(N_PLACEBO)
    ridx = np.arange(len(y))
    for t in range(N_PLACEBO):
        sig = rngp.permutation(HC_by_gene)                     # length-39 relabelling, sizes preserved
        Sigma = np.zeros((nHG, 4), np.float32)
        Sigma[np.arange(nHG), sig] = 1.0
        present = (Ctx @ Sigma) > 0                            # (n_rows, 4) context-cluster presence under sig
        mk = present[ridx, sig[gene]].astype(float)            # candidate's permuted cluster present in context?
        Mp = demean(mk, 20)
        _, bp = fit_beta_gamma(Y, Mp, M2)
        placebo[t] = bp[0]
    placebo_mean, placebo_sd = float(placebo.mean()), float(placebo.std())
    placebo_p = float(((placebo >= beta).sum() + 1) / (N_PLACEBO + 1))

    # ---- PER-CLUSTER beta WITH cell-clustered SE: the flip test (does the USED cluster track the tissue?)
    # needs real inference on beta_A vs beta_B, not point estimates. Refit match split by the candidate gene's
    # cluster (one-vs pooled paralog) and cluster the SE by cell, exactly as the aggregate above.
    percluster = {}
    for c in range(4):
        mc = (match * (HC_by_gene[gene] == c)).astype(float)   # same-cluster match, only cluster-c candidates
        idmask = mc != 0
        n_ident = int(idmask.sum())                            # rows identifying this cluster's beta
        n_cells_id = int(len(np.unique(cell[idmask])))         # DISTINCT cells behind them -> the real power
        # a cluster the tissue does not co-activate (C/D in bone marrow) has its identifying variation in a
        # handful of cells; the cell-clustered SE is then meaningless (bone-marrow HOXC: 2,664 rows but only 16
        # cells -> a spurious +0.32). Guard on distinct CELLS, not rows, and also the singular case.
        if n_ident < 500 or n_cells_id < 40:
            percluster[CN[c]] = dict(beta=None, se=None, t=None, n_ident=n_ident, n_cells=n_cells_id)
            continue
        Mc = demean(mc)
        Xc, bc = fit_beta_gamma(Y, Mc, M2)
        resc = Y - Xc @ bc
        XtXc = np.linalg.inv(Xc.T @ Xc); meatc = np.zeros((2, 2))
        for cc in range(len(u)):
            m = inv == cc; s = (Xc[m] * resc[m, None]).sum(0); meatc += np.outer(s, s)
        se_c = float(np.sqrt((XtXc @ meatc @ XtXc)[0, 0]))
        percluster[CN[c]] = dict(beta=float(bc[0]), se=se_c, t=float(bc[0] / (se_c + 1e-12)),
                                 n_ident=n_ident, n_cells=n_cells_id)

    delta = 0.25 * abs(gamma)
    ci_b = (beta - 1.96 * se_b, beta + 1.96 * se_b)
    used = ci_b[0] > 0                                          # cluster causally used at all
    strong = ci_b[0] > delta                                   # used at >= a quarter of paralog
    gamma_nonzero = (gamma - 1.96 * se_g) > 0
    placebo_clean = placebo_p < 0.05 and abs(placebo_mean) < 0.02
    if used and placebo_clean:
        verdict = ("USED — cluster is causally conditioned on"
                   + (" (>=1/4 of paralog)" if strong else " (weak: <1/4 of paralog, but > 0)"))
    elif (not used) and gamma_nonzero:
        verdict = "READABLE CORRELATE — cluster not causally used, paralog is"
    else:
        verdict = "UNRESOLVED"

    print("=" * 96)
    print(f"  BETA  cluster-in-context -> absent same-cluster logit : {beta:+.4f}  SE {se_b:.4f}  "
          f"t={beta / se_b:+.2f}  95%CI [{ci_b[0]:+.4f}, {ci_b[1]:+.4f}]")
    print(f"  GAMMA paralog-in-context -> absent same-paralog logit : {gamma:+.4f}  SE {se_g:.4f}  "
          f"t={gamma / se_g:+.2f}")
    print(f"  PLACEBO (permuted cluster labels) beta : {placebo_mean:+.4f} +- {placebo_sd:.4f}  "
          f"p(|null|>=|beta|)={placebo_p:.4f}   {'COLLAPSES (good)' if placebo_clean else 'DID NOT COLLAPSE'}")
    print("  PER-CLUSTER beta (cell-clustered SE; the flip test):")
    for k, v in percluster.items():
        if v["beta"] is None:
            print(f"    HOX{k}: unidentified (n_ident={v['n_ident']:,}, {v.get('n_cells', 0)} cells — "
                  f"cluster not co-activated in this tissue)")
        else:
            print(f"    HOX{k}: beta={v['beta']:+.3f}  SE {v['se']:.3f}  t={v['t']:+.2f}  "
                  f"(n_ident={v['n_ident']:,}, {v['n_cells']:,} cells)")
    print(f"  cluster effect = {abs(beta) / (abs(gamma) + 1e-9) * 100:.0f}% of paralog;  delta(1/4 gamma)={delta:.4f}")
    print(f"\n  VERDICT: {verdict}")

    out = dict(n_cells=int(len(np.unique(cell))), n_rows=int(len(y)), seconds=dt,
               beta=beta, se_beta=se_b, t_beta=beta / se_b, ci_beta=list(ci_b),
               gamma=gamma, se_gamma=se_g, t_gamma=gamma / se_g,
               placebo_mean=placebo_mean, placebo_sd=placebo_sd, placebo_p=placebo_p,
               per_cluster_beta=percluster, delta=delta,
               used=bool(used), strong=bool(strong), placebo_clean=bool(placebo_clean),
               verdict=verdict, corr_match_parm=float(np.corrcoef(match, parm)[0, 1]))
    out["dataset"] = DATASET
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    # fetal_gut keeps the canonical name (the headline, cited in RESULTS.md §6B); rotations are suffixed.
    fn = "hox_causal_locus.json" if DATASET == "fetal_gut" else f"hox_causal_locus_{DATASET}.json"
    json.dump(out, open(os.path.join(HERE, "results", fn), "w"), indent=1)
    print(f"\n[done] -> results/{fn}")


if __name__ == "__main__":
    main()
