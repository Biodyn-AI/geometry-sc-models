"""DO DOCUMENTED CONTEXT-SWITCHING TFs MOVE MORE ACROSS CONTEXTS THAN MATCHED CONTROLS?

The stronger, per-gene form of the gene-polysemy claim. Positive class: 14 high-confidence context-switching
transcription factors -- lineage-determining collaborators with documented cell-type-specific partners/cistromes
(SPI1, IRF4, IRF8, GATA3, MAFB, BCL6, PRDM1, TBX21, RUNX1, RUNX3, BATF, MAF, BACH2, ZEB2), curated and
adversarially checked.

THE CENTRAL THREAT, flagged independently by all three sourcing agents: this class is CO-EXPRESSION-CONFOUNDED
by construction. MaxToki sees only rank-ordered gene identity, so a TF's context-switch reaches it ONLY through
its co-expressed neighbourhood differing by cell type -- which IS co-expression. So "switchers move more than
random genes" would deflate to co-expression, the project's recurring result. The informative quantity is
switcher movement OVER a control matched on how much its co-expression neighbourhood changes across contexts.

DESIGN -- a CONTROL LADDER, reporting how the effect shrinks as controls tighten:
  vs ALL genes                          naive; expected positive, uninformative
  vs abundance-matched genes            removes the expression-level confound (our metric is abundance-sensitive)
  vs abundance + NEIGHBOURHOOD-DIVERGENCE matched genes   the decisive control: matched on how much each gene's
      co-expression partners change across contexts. Surviving THIS is movement beyond co-expression.
  vs other TFs (abundance-matched)      isolates "being a context-switcher" from "being a TF"

METRIC -- per-gene context lability L(g) = mean over contexts of <E_0(g,c), E_1(g,c)>, where E_p is the
gene x context interaction tensor (representation minus gene main effect minus context main effect) in cell
partition p. The dot product across the two INDEPENDENT cell partitions keeps only reproducible movement (noise
averages out). Higher = the gene's representation moves more, gene-specifically and reproducibly, across cells.

NEIGHBOURHOOD DIVERGENCE nd(g): per context, gene g's expression correlation to 500 landmark genes (computed on
that context's own cells); nd(g) = mean pairwise (1 - cosine) of g's 12 per-context neighbourhood vectors. High
= g's co-expression partners change a lot across cell types. This is the co-expression signal to match on.

Non-independence: TFs fall in paralog families (GATA/IRF/RUNX/MAF/BACH); the permutation test blocks by family.
Asymmetry (pre-registered): a positive surviving neighbourhood-matching is strong evidence; a null is weak
(the switch may be post-translational, invisible to an expression model).

Out: results/ctx_switcher_test.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, csv, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ctx_position_confound as CP
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
NET = f"{_DATA}/biodyn-work/single_cell_mechinterp/external/networks"
TS = CP.TS; PANELS = CP.PANELS
TAPS = [4, 8]
MIN_CTX = 9
N_LANDMARK = 500
N_PERM = 5000
SEED = 0

SWITCHERS = ["SPI1", "IRF4", "IRF8", "GATA3", "MAFB", "BCL6", "PRDM1", "TBX21", "RUNX1", "RUNX3",
             "BATF", "MAF", "BACH2", "ZEB2"]
# paralog/family blocks for the block-permutation (single-member families are their own block)
FAMILY = {"IRF4": "IRF", "IRF8": "IRF", "RUNX1": "RUNX", "RUNX3": "RUNX", "MAF": "MAF", "MAFB": "MAF",
          "GATA3": "GATA", "BACH2": "BACH", "BATF": "BATF", "SPI1": "ETS", "BCL6": "BCL6",
          "PRDM1": "PRDM1", "TBX21": "TBX", "ZEB2": "ZEB"}


def tf_census():
    tf = set()
    for r in csv.reader(open(f"{NET}/trrust_human.tsv"), delimiter="\t"):
        if r:
            tf.add(r[0].upper())
    for i, r in enumerate(csv.reader(open(f"{NET}/dorothea_human.tsv"), delimiter="\t")):
        if i and r:
            tf.add(r[0].upper())
    return tf


def neighbourhood_divergence(genes_ens, syms, use):
    """per-gene divergence of its co-expression neighbourhood across the 12 contexts (co-expression control)."""
    from maxtoki_adapter import MaxTokiTokenizer  # noqa (not needed but keeps env parity)
    # per-context standardized expression for panel genes, then neighbourhood vectors to landmark genes
    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    ctxs = z0["contexts"].astype(str)
    # collect per-context cell x gene matrices
    percx = {c: [] for c in ctxs}
    ctset = set(ctxs)
    for p in PANELS:
        path = os.path.join(TS, p)
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            ens = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]).astype(str)
            ens = np.array([e.split(".")[0] for e in ens])
            vcol = {e: j for j, e in enumerate(ens)}
            take = np.array([vcol.get(g.split(".")[0], -1) for g in genes_ens])
            v2p = np.full(len(ens), -1, np.int64)
            for pj, vr in enumerate(take):
                if vr >= 0:
                    v2p[vr] = pj
            ctg = f["obs"]["cell_type"]
            cats = np.array([x.decode() if isinstance(x, bytes) else x for x in ctg["categories"][:]]).astype(str)
            ctypes = cats[ctg["codes"][:]]
            X = f["X"]; n = int(X.attrs["shape"][0]); indptr = X["indptr"][:]
            for r in range(n):
                c = ctypes[r]
                if c not in ctset or len(percx[c]) >= 800:
                    continue
                s, e = int(indptr[r]), int(indptr[r + 1]); ii, vv = X["indices"][s:e], X["data"][s:e].astype(np.float32)
                pj = v2p[ii]; keep = pj >= 0
                row = np.zeros(len(genes_ens), np.float32)
                row[pj[keep]] = np.log1p(vv[keep] / (float(vv.sum()) or 1.0) * 1e4)
                percx[c].append(row)
    # landmark genes = most variable among `use` genes (pooled)
    pooled = np.vstack([np.array(v) for c in ctxs for v in percx[c][:200]])
    var = pooled.var(0); land = use[np.argsort(-var[use])[:N_LANDMARK]]
    # per-gene neighbourhood vector per context
    nbr = {}
    for c in ctxs:
        Z = np.array(percx[c]); Z = Z - Z.mean(0); Z = Z / (Z.std(0) + 1e-8)
        nbr[c] = (Z.T @ Z[:, land]) / len(Z)          # (n_gene, n_landmark) correlation to landmarks
    nd = np.full(len(genes_ens), np.nan)
    for gi in use:
        V = np.array([nbr[c][gi] for c in ctxs])       # (12, n_landmark)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        G = Vn @ Vn.T; iu = np.triu_indices(len(ctxs), 1)
        nd[gi] = 1.0 - float(G[iu].mean())
    return nd


def lability(Mz, full, use, MIN):
    """per-gene reproducible context lability L(g) = mean_c <E0(g,c), E1(g,c)>."""
    a = np.full((Mz.shape[0], Mz.shape[2], Mz.shape[3]), np.nan, np.float32)  # (part, gene, d) main effect
    for gi in use:
        cs = np.where(full[:, gi])[0]
        a[:, gi] = Mz[:, cs, gi].mean(1)
    L = np.full(Mz.shape[2], np.nan)
    b = np.full((Mz.shape[0], Mz.shape[1], Mz.shape[3]), np.nan, np.float32)  # context main effect over `use`
    for p in range(2):
        for c in range(Mz.shape[1]):
            m = full[c, use]
            if m.sum() > 5:
                b[p, c] = Mz[p, c, use][m].mean(0)
    grand = np.nanmean(a[:, use], 1)                                          # (part, d)
    for gi in use:
        cs = np.where(full[:, gi])[0]
        acc = 0.0
        for c in cs:
            E0 = Mz[0, c, gi] - a[0, gi] - b[0, c] + grand[0]
            E1 = Mz[1, c, gi] - a[1, gi] - b[1, c] + grand[1]
            acc += float(E0 @ E1)
        L[gi] = acc / max(1, len(cs))
    return L


def matched_control(target_idx, pool_idx, covars, rng, k=1):
    """for each target gene pick k nearest pool genes in standardized covariate space (without replacement)."""
    Z = (covars - np.nanmean(covars, 0)) / (np.nanstd(covars, 0) + 1e-9)
    used = set(); ctrl = []
    for t in target_idx:
        d = np.linalg.norm(Z[pool_idx] - Z[t], axis=1)
        order = np.argsort(d)
        picked = 0
        for j in order:
            g = pool_idx[j]
            if g in used or g in target_idx:
                continue
            used.add(g); ctrl.append(g); picked += 1
            if picked >= k:
                break
    return np.array(ctrl)


def block_perm_p(L, pos_idx, ctrl_pool, families, rng, nperm=N_PERM):
    """difference in mean lability, positives vs a control set, with block(family) permutation of labels."""
    obs = np.nanmean(L[pos_idx]) - np.nanmean(L[ctrl_pool])
    # permute which genes are 'positive' within the combined pool, keeping family blocks intact
    combined = list(pos_idx) + list(ctrl_pool)
    fam = np.array([families.get(g, f"solo{g}") for g in combined])
    Lc = L[combined]; npos = len(pos_idx)
    blocks = {}
    for i, fv in enumerate(fam):
        blocks.setdefault(fv, []).append(i)
    cnt = 0
    for _ in range(nperm):
        # assign whole blocks to positive until we reach ~npos genes
        order = list(blocks.values()); rng.shuffle(order)
        sel, take = [], 0
        for blk in order:
            sel += blk; take += len(blk)
            if take >= npos:
                break
        mask = np.zeros(len(combined), bool); mask[sel] = True
        stat = np.nanmean(Lc[mask]) - np.nanmean(Lc[~mask])
        if stat >= obs:
            cnt += 1
    return float(obs), float((cnt + 1) / (nperm + 1))


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    sym2ens = {}
    for e, s in ens2sym.items():
        sym2ens.setdefault(s, e)
    rng = np.random.default_rng(SEED)
    tfset = tf_census()

    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    genes = z0["genes"].astype(str); ctxs = z0["contexts"].astype(str)
    syms = np.array([ens2sym.get(g, "") for g in genes])
    sidx = {s: i for i, s in enumerate(syms)}

    tokmap = json.load(open(f"{CP.MSETUP}/token_dictionary.json")); ens2tid = {k: int(v) for k, v in tokmap.items()}
    tids = np.array([ens2tid.get(g, -1) for g in genes])
    print("[1/4] per-context mean rank (abundance covariate)", flush=True)
    MR = CP.mean_ranks(set(ctxs))
    rank = np.full((len(ctxs), len(genes)), np.nan)
    for ci, c in enumerate(ctxs):
        for gi, t in enumerate(tids):
            if t in MR.get(c, {}):
                rank[ci, gi] = MR[c][t]
    abund = np.nanmean(rank, 0)

    out = {"switchers_on_panel": [], "taps": {}}
    for tap in TAPS:
        z = np.load(os.path.join(RES, f"ctx_maxtoki_L{tap:02d}.npz"), allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        full = (counts == cap).all(0)
        flat = M[:, full]; mu = flat.reshape(-1, M.shape[-1]).mean(0); sd = flat.reshape(-1, M.shape[-1]).std(0) + 1e-6
        Mz = (M - mu) / sd
        use = np.where(full.sum(0) >= MIN_CTX)[0]

        if tap == TAPS[0]:
            print("[2/4] neighbourhood divergence (co-expression control covariate)", flush=True)
            nd = neighbourhood_divergence(list(genes), syms, use)
        print(f"[3/4] per-gene lability, layer {tap}", flush=True)
        L = lability(Mz, full, use, MIN_CTX)

        pos = np.array([sidx[s] for s in SWITCHERS if s in sidx and sidx[s] in use and np.isfinite(L[sidx[s]])])
        on_panel = [syms[i] for i in pos]
        out["switchers_on_panel"] = on_panel
        fam_by_idx = {sidx[s]: FAMILY.get(s, s) for s in SWITCHERS if s in sidx}

        all_use = np.array([g for g in use if np.isfinite(L[g])])
        tf_use = np.array([g for g in all_use if syms[g] in tfset and g not in set(pos)])
        covar_abund = abund[:, None]
        covar_both = np.column_stack([abund, nd])

        res = {"n_switchers": len(pos), "median_L_switchers": float(np.nanmedian(L[pos]))}
        # ladder
        ctrl_all = np.array([g for g in all_use if g not in set(pos)])
        obs, p = block_perm_p(L, pos, ctrl_all, fam_by_idx, np.random.default_rng(1))
        res["vs_all_genes"] = dict(delta=obs, p=p, n_ctrl=len(ctrl_all))
        cA = matched_control(pos, ctrl_all, covar_abund, rng, k=5)
        obs, p = block_perm_p(L, pos, cA, fam_by_idx, np.random.default_rng(2))
        res["vs_abundance_matched"] = dict(delta=obs, p=p, n_ctrl=len(cA))
        cB = matched_control(pos, ctrl_all, covar_both, rng, k=5)
        obs, p = block_perm_p(L, pos, cB, fam_by_idx, np.random.default_rng(3))
        res["vs_abundance_plus_neighdiv_matched"] = dict(delta=obs, p=p, n_ctrl=len(cB))
        cT = matched_control(pos, tf_use, covar_abund, rng, k=5)
        obs, p = block_perm_p(L, pos, cT, fam_by_idx, np.random.default_rng(4))
        res["vs_other_TFs_abundance_matched"] = dict(delta=obs, p=p, n_ctrl=len(cT))

        # is the neighbourhood-divergence itself higher for switchers? (sanity on the confound)
        res["neighdiv_switchers"] = float(np.nanmedian(nd[pos]))
        res["neighdiv_all"] = float(np.nanmedian(nd[all_use]))
        out["taps"][f"L{tap:02d}"] = res
        print(f"\n=== layer {tap}: {len(pos)} switchers on panel ({', '.join(on_panel)}) ===")
        for kk in ["vs_all_genes", "vs_abundance_matched", "vs_abundance_plus_neighdiv_matched",
                   "vs_other_TFs_abundance_matched"]:
            d = res[kk]
            print(f"   {kk:<42} Δlability {d['delta']:+.3f}  p={d['p']:.4f}  (n_ctrl {d['n_ctrl']})")
        print(f"   neighbourhood divergence  switchers {res['neighdiv_switchers']:.3f} vs all {res['neighdiv_all']:.3f}")

    l4 = out["taps"].get("L04", {})
    n_sw = l4.get("n_switchers", 0)
    dec = l4.get("vs_abundance_plus_neighdiv_matched", {})
    if n_sw < 5:
        out["verdict"] = (
            f"UNDERPOWERED / MEASUREMENT WALL, not a result. Only {n_sw} of {len(SWITCHERS)} documented "
            "context-switching TFs are count-balanced in >=9 contexts on this panel (they are low-abundance "
            "regulators; only 3 reach the 6000-gene panel at all). A group comparison with family-blocking on "
            "n<5 is meaningless. This is NOT evidence switchers do or do not move more -- the class is simply "
            "not adequately sampled in a 12-cell-type, abundance-selected panel. A real test needs a targeted "
            "re-extraction that forces the switcher genes + a matched control pool into the panel with more "
            "cells per context so low-expressed TFs reach the count floor in enough contexts.")
    else:
        out["verdict"] = (
            f"L4 switchers vs abundance+neighbourhood-matched controls: Δ {dec.get('delta', float('nan')):+.3f}, "
            f"p={dec.get('p', float('nan')):.4f}. " +
            ("POSITIVE beyond co-expression — documented context-switching TFs move more across contexts than "
             "controls matched on abundance AND on how much their co-expression neighbourhood changes."
             if dec.get("p", 1) < 0.05 and dec.get("delta", 0) > 0 else
             "NOT beyond co-expression — any switcher signal is explained once abundance and co-expression-"
             "neighbourhood change are matched. Per pre-registered asymmetry this null is WEAK evidence."))
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_switcher_test.json"), "w"), indent=1)
    print("[done] -> results/ctx_switcher_test.json")


if __name__ == "__main__":
    main()
