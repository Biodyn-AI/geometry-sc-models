"""LEVEL-2, CURATED GROUND TRUTH: do TFs sit near their ChIP/literature targets, beyond co-expression, and more
where the target programme is active?

The axis-based feasibility probe (ctx_directional_probe.py) was flat, but it used each gene's AVERAGE functional
position as the target — which blurs exactly the per-context distinction a context-switch is about. This test
replaces that with EXTERNAL, per-TF ground truth: each TF's curated target set from TRRUST (literature/ChIP) and
DoRothEA (A/B confidence). Crucially these targets are defined INDEPENDENTLY of co-expression in our data, so
"TF sits near its targets" is not automatically a co-expression restatement.

Two questions, pooled across all measurable TFs for power:
  (A) BEYOND CO-EXPRESSION, STATIC. In each cell type, is a TF's context representation closer (cosine, after
      removing the context main effect) to the centroid of its CURATED targets than to the centroid of
      CO-EXPRESSION-MATCHED non-targets (random genes matched on how co-expressed they are with the TF)? If yes,
      the model places a TF near its regulatory targets beyond what co-expression explains.
  (B) CONTEXT-APPROPRIATE, DIRECTIONAL (the Level-2 claim). Does that excess closeness INCREASE in cell types
      where the target programme is more active (targets more highly expressed)? A TF drawn toward its targets
      specifically when those targets are on is context-appropriate regulatory representation.

Controls: per-dimension z-scoring (done in the extraction load) + per-context centring (removes the averaging
null / anisotropy); the co-expression-matched control set is the decisive one for (A); target-activity is the
per-context modulator for (B). Abundance enters only through rank, reported.

Reads headline ctx_maxtoki_L{04,08}.npz. Out: results/ctx_curated_targets.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, csv, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ctx_position_confound as CP
from ctx_coexpr_null import coexpr_matrix
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
NET = f"{_DATA}/biodyn-work/single_cell_mechinterp/external/networks"
TAPS = [4, 8]
MIN_TF_CTX = 6         # TF must be count-balanced in >= this many contexts
MIN_TARGETS = 15       # ... with at least this many measurable curated targets
N_CTRL = 40            # co-expression-matched control redraws
SEED = 0
from scipy.stats import spearmanr


def load_targets():
    """TF -> set(targets) from TRRUST (all) + DoRothEA (A/B confidence). Literature/ChIP, NOT co-expression."""
    tgt = collections.defaultdict(set)
    for r in csv.reader(open(f"{NET}/trrust_human.tsv"), delimiter="\t"):
        if len(r) >= 2:
            tgt[r[0].upper()].add(r[1].upper())
    for i, r in enumerate(csv.reader(open(f"{NET}/dorothea_human.tsv"), delimiter="\t")):
        if i and len(r) >= 3 and r[2].strip().upper() in ("A", "B"):
            tgt[r[0].upper()].add(r[1].upper())
    return tgt


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    tgtmap = load_targets()
    rng = np.random.default_rng(SEED)

    z0 = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
    genes = z0["genes"].astype(str); ctxs = z0["contexts"].astype(str)
    syms = np.array([ens2sym.get(g, "") for g in genes]); sidx = {s: i for i, s in enumerate(syms) if s}
    tokmap = json.load(open(f"{CP.MSETUP}/token_dictionary.json")); ens2tid = {k: int(v) for k, v in tokmap.items()}
    tids = np.array([ens2tid.get(g, -1) for g in genes])

    print("[1/3] per-context mean rank (target-activity + abundance)", flush=True)
    MR = CP.mean_ranks(set(ctxs))
    rank = np.full((len(ctxs), len(genes)), np.nan)
    for ci, c in enumerate(ctxs):
        for gi, t in enumerate(tids):
            if t in MR.get(c, {}):
                rank[ci, gi] = MR[c][t]
    # activity = -rank (lower rank number = more highly expressed = more active); use max_rank - rank
    print("[2/3] co-expression matrix (for the matched control)", flush=True)
    C = coexpr_matrix(list(genes))

    out = {"taps": {}}
    for tap in TAPS:
        z = np.load(os.path.join(RES, f"ctx_maxtoki_L{tap:02d}.npz"), allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        full = (counts == cap).all(0)
        flat = M[:, full]; mu = flat.reshape(-1, M.shape[-1]).mean(0); sd = flat.reshape(-1, M.shape[-1]).std(0) + 1e-6
        Mz = ((M - mu) / sd).mean(0)                      # (nctx, ngene, d) partition-averaged
        # per-context centring: remove the averaging null so cosine reflects gene-specific proximity
        for c in range(len(ctxs)):
            m = full[c]
            if m.sum() > 5:
                Mz[c, m] -= Mz[c, m].mean(0)

        # eligible TFs
        elig = []
        for s, gi in sidx.items():
            if s not in tgtmap or full[:, gi].sum() < MIN_TF_CTX:
                continue
            tg = [sidx[t] for t in tgtmap[s] if t in sidx and t != s]
            if len(tg) >= MIN_TARGETS:
                elig.append((s, gi, np.array(tg)))
        print(f"\n=== layer {tap}: {len(elig)} measurable TFs with >= {MIN_TARGETS} measurable targets ===", flush=True)
        if not elig:
            continue

        exc_static, exc_null, modul, modul_null = [], [], [], []
        for s, gi, tg in elig:
            cs = np.where(full[:, gi])[0]                 # contexts where the TF is measured
            # co-expression profile of the TF to all genes (for matched control)
            cprof = C[gi]
            pool = np.array([j for j in range(len(genes)) if full[:, j].any() and j != gi and j not in set(tg)])
            e_by_c, a_by_c = [], []
            en_draws = [[] for _ in range(N_CTRL)]
            for c in cs:
                mt = tg[full[c, tg]]                       # targets measured in this context
                if len(mt) < 8:
                    continue
                v = Mz[c, gi]; vt = Mz[c, mt].mean(0)
                ct = float(v @ vt / (np.linalg.norm(v) * np.linalg.norm(vt) + 1e-9))
                # co-expression-matched control: genes with similar co-expression-to-TF as the targets
                mpool = pool[full[c, pool]]
                # match on cprof: for each target pick nearest-in-cprof pool gene (this draw uses random ties)
                order = np.argsort(cprof[mpool])
                cp_sorted = cprof[mpool][order]
                cc_draw = []
                for k in range(N_CTRL):
                    idx = []
                    for tv in cprof[mt]:
                        j = int(np.searchsorted(cp_sorted, tv))
                        j = min(max(j + rng.integers(-3, 4), 0), len(mpool) - 1)
                        idx.append(mpool[order[j]])
                    vc = Mz[c, np.array(idx)].mean(0)
                    cc = float(v @ vc / (np.linalg.norm(v) * np.linalg.norm(vc) + 1e-9))
                    en_draws[k].append(ct - cc)
                    cc_draw.append(cc)
                e_by_c.append(ct - float(np.mean(cc_draw)))
                a_by_c.append(float(np.nanmean(rank[c, mt])))    # lower = more active
            if len(e_by_c) < 3:
                continue
            exc_static.append(float(np.mean(e_by_c)))
            exc_null.append(float(np.mean([np.mean(d) for d in en_draws if d])))
            # (B) does excess closeness track target activity across this TF's contexts? (activity = -rank)
            act = -np.array(a_by_c)
            if np.std(act) > 0 and np.std(e_by_c) > 0:
                modul.append(float(spearmanr(e_by_c, act).statistic))
                modul_null.append(float(spearmanr(e_by_c, rng.permutation(act)).statistic))

        exc_static = np.array(exc_static)
        # (A) static: TF closer to curated targets than co-expression-matched controls?
        mean_exc = float(np.nanmean(exc_static))
        # sign test across TFs (each TF one observation, so non-independence is handled)
        frac_pos = float(np.mean(exc_static > 0))
        from math import comb
        n = len(exc_static); k = int((exc_static > 0).sum())
        p_sign = float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n <= 900 else float("nan")
        modul = np.array(modul); modul_null = np.array(modul_null)
        mean_mod = float(np.nanmean(modul)) if len(modul) else float("nan")
        p_mod = float((np.nanmean(modul_null) >= np.nanmean(modul))) if len(modul) else float("nan")
        # proper: compare modul distribution to modul_null via paired sign
        if len(modul):
            kk = int((modul > 0).sum()); nn = len(modul)
            p_mod_sign = float(sum(comb(nn, i) for i in range(kk, nn + 1)) / 2 ** nn) if nn <= 900 else float("nan")
        else:
            p_mod_sign = float("nan")

        print(f"  (A) STATIC excess closeness to curated targets vs co-expression-matched controls:")
        print(f"        mean {mean_exc:+.4f} over {n} TFs; {100*frac_pos:.0f}% of TFs positive (sign p={p_sign:.1e})")
        print(f"  (B) CONTEXT modulation: excess closeness vs target activity, mean rho {mean_mod:+.3f} "
              f"over {len(modul)} TFs; {100*np.mean(modul>0):.0f}% positive (sign p={p_mod_sign:.2f})")
        out["taps"][f"L{tap:02d}"] = dict(n_tfs=n, static_excess_mean=mean_exc, static_frac_pos=frac_pos,
                                          static_sign_p=p_sign, modulation_mean_rho=mean_mod,
                                          modulation_frac_pos=float(np.mean(modul > 0)) if len(modul) else float("nan"),
                                          modulation_sign_p=p_mod_sign, n_modul=len(modul))

    l4 = out["taps"].get("L04", {})
    A_pos = l4.get("static_excess_mean", 0) > 0 and l4.get("static_sign_p", 1) < 0.05
    B_pos = l4.get("modulation_mean_rho", 0) > 0 and l4.get("modulation_sign_p", 1) < 0.05
    out["verdict"] = (
        f"L4: (A) static excess {l4.get('static_excess_mean', float('nan')):+.4f} "
        f"(sign p {l4.get('static_sign_p', float('nan')):.1e}); (B) activity modulation rho "
        f"{l4.get('modulation_mean_rho', float('nan')):+.3f} (sign p {l4.get('modulation_sign_p', float('nan')):.2f}). " +
        ("LEVEL-2 SPARK — TFs sit closer to their curated (ChIP/literature) targets than to co-expression-matched "
         "controls AND more so where the target programme is active. Context-appropriate regulatory "
         "representation beyond co-expression; the curated directional programme is worth building out."
         if A_pos and B_pos else
         "PARTIAL — TFs sit closer to curated targets than co-expression-matched controls (A positive) but the "
         "effect is NOT modulated by where the programme is active (B null): near-targets, but not "
         "context-appropriately so." if A_pos else
         "NO LEVEL-2 SIGNAL — even with ChIP/literature targets as ground truth, TFs are not placed near their "
         "targets beyond co-expression, nor context-appropriately. The curated route does not rescue Level 2; "
         "the ceiling argument holds. Level 1 (functional organisation beyond co-expression) remains the result.")
    )
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "ctx_curated_targets.json"), "w"), indent=1)
    print("[done] -> results/ctx_curated_targets.json")


if __name__ == "__main__":
    main()
