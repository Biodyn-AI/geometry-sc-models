"""$0 de-risk: build coexpr + esm2 bases locally and validate the ported machinery BEFORE any pod time.
Checks: (1) both reference bases build/load; (2) coords loads; (3) each hypothesis has >=6 genes in both
bases; (4) the four statistics execute; (5) baseline SANITY — esm2 (sequence) should tend to win the
sequence-confounded HOX/globin/isozyme sets, coexpr (data) the cell-cycle circle."""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2s_gm_lib as G
import gene_sets as S

REF = ["coexpr", "esm2"]
print("=== building/loading reference bases ===", flush=True)
for b in REF:
    M, syms = G.basis(b)
    print(f"  {b:<8} shape={M.shape} n_syms={len(syms)}", flush=True)
C = G.coords()
print(f"  coords: {len(C)} human genes, autosomes present={sorted(set(C.chromosome)&set(G.AUTOSOMES))[:5]}...", flush=True)


def score(h, b):
    M, ok = G.subset(b, h["genes"])
    if ok.sum() < 6:
        return dict(err=f"{int(ok.sum())} genes"), int(ok.sum())
    k = h["kind"]
    if k in ("order", "blob"):
        r = G.order_score(M, np.asarray(h["coord"], float)[ok], n_perm=500)
    elif k == "circle":
        r = G.circle_score(M, np.asarray(h["coord"], float)[ok], n_perm=500)
    elif k == "grid":
        r = G.grid_score(M, np.asarray(h["coord"][0], float)[ok], np.asarray(h["coord"][1], float)[ok], n_perm=500)
    elif k == "antipodal":
        r = G.antipodal_score(b, h["a"], h["b"], h["axis_genes"], h["axis_sign"])
    return r, int(ok.sum())


print("\n=== per-hypothesis coverage + baseline behaviour (coexpr vs esm2) ===", flush=True)
print(f"  {'hypothesis':<24}{'kind':<9}{'n_cov':>6}   coexpr        esm2", flush=True)
out = {}
for name, h in S.H.items():
    row = {}
    ncov = None
    for b in REF:
        r, n = score(h, b); row[b] = r; ncov = n
    def fmt(r):
        if r is None or "err" in r:
            return (r or {}).get("err", "None")
        if "cos" in r:
            return f"cos{r['cos']:+.2f} z{r.get('z',0):+.1f}"
        return f"{r['stat']:.2f}(p{r['p']:.3f})"
    print(f"  {name:<24}{h['kind']:<9}{ncov:>6}   {fmt(row['coexpr']):<13} {fmt(row['esm2'])}", flush=True)
    out[name] = dict(kind=h["kind"], role=h["role"], n_cov=ncov,
                     coexpr=row["coexpr"], esm2=row["esm2"])

json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "validate_local.json"), "w"), indent=1)
print("\n[done] -> results/validate_local.json", flush=True)
