"""Save the per-cell-type projection of the context main effect onto the nuclear/surface functional axis, so the
per-context loadings quoted in §4.4 (CD8 T +4.1 ... alveolar type-2 -5.3) are traceable to a result file.
Out: results/ctx_celltype_loadings.json"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
NUC = ["GO:0005634", "GO:0000785", "GO:0003677"]; SURF = ["GO:0005886", "GO:0005576", "GO:0005615"]

ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
z = np.load(os.path.join(RES, "ctx_maxtoki_L04.npz"), allow_pickle=True)
M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
genes = z["genes"].astype(str); ctxs = z["contexts"].astype(str); syms = [ens2sym.get(g) for g in genes]
full = (counts == cap).all(0); d = M.shape[-1]
flat = M[:, full]; mu = flat.reshape(-1, d).mean(0); sd = flat.reshape(-1, d).std(0) + 1e-6
Mz = (M - mu) / sd
a = np.full((len(genes), d), np.nan, np.float32)
for gi in range(len(genes)):
    cs = np.where(full[:, gi])[0]
    if len(cs): a[gi] = Mz[:, cs, gi].mean((0, 1))
ok = np.isfinite(a[:, 0])
ia = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(NUC)]
ib = [i for i, s in enumerate(syms) if ok[i] and s in g2g and g2g[s] & set(SURF)]
both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
u = a[ia].mean(0) - a[ib].mean(0); u /= np.linalg.norm(u)
q = np.tensordot(Mz.mean(0), u, axes=([2], [0]))          # (n_ctx, n_gene)
load = {ctxs[ci]: float(np.nanmean(np.where(full[ci], q[ci], np.nan))) for ci in range(len(ctxs))}
load = dict(sorted(load.items(), key=lambda kv: -kv[1]))
json.dump({"axis": "nuclear(+)/surface(-)", "per_context_loading": load,
           "note": "context main effect projected on the frozen nuclear/surface axis (§4.4)"},
          open(os.path.join(RES, "ctx_celltype_loadings.json"), "w"), indent=1)
for c, v in load.items():
    print(f"  {v:+.2f}  {c}")
print("[done] -> results/ctx_celltype_loadings.json")
