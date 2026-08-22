"""GEOMETRIC screen on C2S-Scale — port of route_genemanifold/run_contextfree.py.

For each gene_sets hypothesis, score every basis with the matching gm_lib statistic and decide the
verdict: a C2S activation basis (c2s_ctx_L*) counts as a MODEL result only if it beats BOTH references
(coexpr = data, esm2 = sequence) with significance. margin_boot confirms the best model basis's margin
over each reference is real (paired bootstrap over genes), the route's proper model-specificity test.

MODEL_BASES default = every c2s_ctx_L*.npz found in the bases dir. Run after build_ctx_basis.py.
Out: results/geometric.json
"""
import os, sys, glob, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2s_gm_lib as G
import gene_sets as S

BASES_DIR = os.environ.get("C2S_GM_BASES", os.path.join(os.path.dirname(__file__), "bases"))
MODEL_BASES = sorted(f"c2s_ctx_L{int(os.path.basename(p)[9:11]):02d}"
                     for p in glob.glob(os.path.join(BASES_DIR, "c2s_ctx_L*.npz")))
REF_BASES = ["coexpr", "esm2"]
BASES = MODEL_BASES + REF_BASES
N_PERM = 2000
if not MODEL_BASES:
    print("WARNING: no c2s_ctx_L*.npz in bases dir — running reference-only.", flush=True)


def score(b, h):
    M, ok = G.subset(b, h["genes"])
    if ok.sum() < 6:
        return dict(error=f"only {int(ok.sum())} genes"), ok
    k = h["kind"]
    if k in ("order", "blob"):
        r = G.order_score(M, np.asarray(h["coord"], float)[ok], n_perm=N_PERM)
    elif k == "circle":
        r = G.circle_score(M, np.asarray(h["coord"], float)[ok], n_perm=N_PERM)
    elif k == "grid":
        r = G.grid_score(M, np.asarray(h["coord"][0], float)[ok],
                         np.asarray(h["coord"][1], float)[ok], n_perm=N_PERM)
    elif k == "antipodal":
        r = G.antipodal_score(b, h["a"], h["b"], h["axis_genes"], h["axis_sign"])
        r = r or dict(error="poles absent")
    r["n_found"] = int(ok.sum())
    return r, ok


def coord_for(h, ok):
    k = h["kind"]
    if k == "grid":
        return (np.asarray(h["coord"][0], float)[ok], np.asarray(h["coord"][1], float)[ok])
    if k == "antipodal":
        return None
    return np.asarray(h["coord"], float)[ok]


def main():
    res = {}
    for name, h in S.H.items():
        by = {}
        okref = None
        for b in BASES:
            r, ok = score(b, h)
            by[b] = r
            if b in MODEL_BASES and "error" not in r:
                okref = ok
        res[name] = dict(kind=h["kind"], role=h["role"], n=len(h["genes"]), by_basis=by)
        # margin bootstrap: best model basis vs each reference, PAIRED over the genes present in BOTH
        # (the route's mandated discipline — a bigger number is not a win; the margin's CI must clear 0).
        if h["kind"] != "antipodal" and MODEL_BASES:
            valid = [b for b in MODEL_BASES if "stat" in by.get(b, {})]
            if valid:
                bm = max(valid, key=lambda b: by[b]["stat"])
                _, symsM = G.basis(bm); setM = set(symsM)
                kind = "grid" if h["kind"] == "grid" else ("circle" if h["kind"] == "circle" else "order")
                mb = {}
                for ref in REF_BASES:
                    _, symsR = G.basis(ref); setR = set(symsR)
                    mask = np.array([g.upper() in setM and g.upper() in setR for g in h["genes"]])
                    gsh = [g for g, m in zip(h["genes"], mask) if m]
                    if len(gsh) < 6:
                        continue
                    Mm, _ = G.subset(bm, gsh); Mr, _ = G.subset(ref, gsh)
                    if h["kind"] == "grid":
                        coord = (np.asarray(h["coord"][0], float)[mask], np.asarray(h["coord"][1], float)[mask])
                    else:
                        coord = np.asarray(h["coord"], float)[mask]
                    mb[ref] = dict(n_shared=len(gsh), **G.margin_boot(Mm, Mr, coord, kind))
                res[name]["best_model_basis"] = bm
                res[name]["margin_boot"] = mb
        # console line
        cells = "  ".join(f"{b.replace('c2s_ctx_','')}:{by[b]['stat']:.2f}" for b in BASES
                          if "stat" in by.get(b, {}))
        if h["kind"] == "antipodal":
            cells = "  ".join(f"{b.replace('c2s_ctx_','')}:cos{by[b]['cos']:+.2f}" for b in BASES
                              if "cos" in by.get(b, {}))
        print(f"[{name:<24}] {cells}", flush=True)

    # ---- verdicts ----
    print("\n" + "=" * 90)
    print("VERDICT: a c2s_ctx basis must beat BOTH coexpr (data) and esm2 (sequence), significantly")
    print("=" * 90)
    for name, v in res.items():
        by = v["by_basis"]
        if v["kind"] == "antipodal":
            key, better = "cos", (lambda a, b: a < b)
        else:
            key, better = "stat", (lambda a, b: a > b)
        mvals = {b: by[b][key] for b in MODEL_BASES if by.get(b) and key in by[b]}
        rvals = {b: by[b][key] for b in REF_BASES if by.get(b) and key in by[b]}
        if not mvals or len(rvals) < len(REF_BASES):
            v["verdict"] = "INCOMPLETE"; print(f"  {name:<24} INCOMPLETE"); continue
        mb = (min if v["kind"] == "antipodal" else max)(mvals, key=mvals.get)
        m = mvals[mb]; ce, es = rvals["coexpr"], rvals["esm2"]
        sig = (by[mb].get("p", 1.0) < 0.05) if v["kind"] != "antipodal" else (by[mb].get("p_more_negative", 1) < 0.05)
        wins = better(m, ce) and better(m, es) and sig
        # MODEL STRUCTURE requires the PAIRED margin bootstrap to clear 0 vs BOTH refs (antipodal has no boot).
        mbok = False
        if v["kind"] == "antipodal":
            mbok = True
        elif v.get("margin_boot") and len(v["margin_boot"]) == len(REF_BASES):
            mbok = all(v["margin_boot"][r]["ci_lo"] > 0 for r in v["margin_boot"])
        verdict = ("** MODEL STRUCTURE **" if wins and mbok else
                   "model>both (pt) but margin CI touches 0" if wins else
                   "sequence (esm2)" if better(es, m) and better(es, ce) else
                   "co-expression" if better(ce, m) else "null / n.s.")
        v["verdict"] = verdict; v["best_model_basis"] = mb
        print(f"  {name:<24}{v['role']:<18} {mb.replace('c2s_ctx_','model_')} {m:+.3f}  "
              f"coexpr {ce:+.3f}  esm2 {es:+.3f}   {verdict}", flush=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "results", "geometric.json"), "w"), indent=1)
    print("\n[done] -> results/geometric.json", flush=True)


if __name__ == "__main__":
    main()
