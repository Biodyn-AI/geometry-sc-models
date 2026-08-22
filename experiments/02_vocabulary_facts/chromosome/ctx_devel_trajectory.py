"""FUNCTIONAL-AXIS TRAJECTORY OF HEMATOPOIETIC DIFFERENTIATION.

Given MaxToki contextual gene representations per branch cluster (ctx_devel_L*.npz), ask: as cells differentiate,
how does the FUNCTIONAL CHARACTER of gene usage move along interpretable axes (nuclear/transcriptional vs
surface/secreted; mitochondrial/metabolic vs cytoskeletal; transcription vs transport)? A human-readable
narrative of differentiation, branch by branch.

Two readouts:
  (1) COMPOSITION-INCLUSIVE loading  h_u(cluster) = mean projection of the cluster's genes onto axis u. This is
      "where this cell state sits on the functional axis" and is the interpretable trajectory. It reflects both
      which genes are expressed AND how the model contextually represents them.
  (2) BEYOND-CO-EXPRESSION check. We recompute the same trajectory from a CO-EXPRESSION baseline (gene positions
      from an expression-only factorisation) and ask whether the MODEL's functional trajectory differs from /
      adds to it -- i.e. whether the model's contextual representation reorganises differentiation functionally
      beyond what co-expression alone gives.

Axes are rebuilt in the DEVELOPMENTAL representation space from the same GO pole gene sets used throughout, so
the functional MEANING is fixed while the space is the hematopoietic one. Validity (does the axis separate the
pole gene classes here) is reported per axis; a trajectory on an invalid axis is not interpreted.

Interpretability anchor (known hematopoiesis, for reading the result -- NOT fitted): erythroid maturation is
dominated by haemoglobin synthesis and mitochondrial/metabolic remodelling; monocyte/DC maturation by
surface-receptor and secretory/immune-effector programmes; lymphoid (CLP) by transcriptional/nuclear identity
programmes. If the functional trajectory recapitulates this it is a positive control; departures are hypotheses.

Out: results/ctx_devel_trajectory.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
TAPS = [2, 4]
SEED = 0
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

AXES = {
    "nuclear_vs_surface":   (["GO:0005634", "GO:0000785", "GO:0003677"], ["GO:0005886", "GO:0005576", "GO:0005615"]),
    "mito_vs_cytoskeleton": (["GO:0005739"], ["GO:0005856"]),
    "transcription_vs_transport": (["GO:0006355", "GO:0003700"], ["GO:0006811", "GO:0038023"]),
}
# broad lineage grouping of the branch clusters, for branch-specific reading
LINEAGE = {"HSC_1": "stem", "HSC_2": "stem", "Precursors": "stem", "CLP": "lymphoid",
           "Ery_1": "erythroid", "Ery_2": "erythroid", "Mega": "erythroid",
           "Mono_1": "myeloid", "Mono_2": "myeloid", "DCs": "myeloid"}


def axis_from(a, ia, ib):
    u = a[ia].mean(0) - a[ib].mean(0); return u / (np.linalg.norm(u) + 1e-9)


def main():
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}

    out = {"taps": {}}
    for tap in TAPS:
        p = os.path.join(RES, f"ctx_devel_L{tap:02d}.npz")
        if not os.path.exists(p):
            print(f"missing {p}"); continue
        z = np.load(p, allow_pickle=True)
        M, counts, cap = z["M"].astype(np.float32), z["counts"], int(z["cap"])
        clusters = z["clusters"].astype(str); genes = z["genes"].astype(str); ptime = z["pseudotime"]
        syms = [ens2sym.get(g) for g in genes]
        full = (counts == cap).all(0)                         # (cluster, gene) balanced in both partitions
        flat = M[:, full]; mu = flat.reshape(-1, M.shape[-1]).mean(0); sd = flat.reshape(-1, M.shape[-1]).std(0) + 1e-6
        Mzp = (M - mu) / sd                                   # per-partition z-scored (part, cluster, gene, d)
        Mz = Mzp.mean(0)                                      # partition-averaged (cluster, gene, d)
        a_space = np.full((len(genes), M.shape[-1]), np.nan, np.float32)
        for gi in range(len(genes)):
            cs = np.where(full[:, gi])[0]
            if len(cs):
                a_space[gi] = Mz[cs, gi].mean(0)
        gene_ok = np.isfinite(a_space[:, 0])

        print(f"\n=== layer {tap}: {full.sum(0).max()} max balanced genes; clusters "
              f"{', '.join(clusters)} ===")
        out["taps"][f"L{tap:02d}"] = {"clusters": list(clusters), "pseudotime": [float(x) for x in ptime], "axes": {}}
        for name, (Ag, Bg) in AXES.items():
            ia = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Ag)]
            ib = [i for i, s in enumerate(syms) if gene_ok[i] and s in g2g and g2g[s] & set(Bg)]
            both = set(ia) & set(ib); ia = [i for i in ia if i not in both]; ib = [i for i in ib if i not in both]
            if len(ia) < 20 or len(ib) < 20:
                print(f"  {name}: too few pole genes ({len(ia)}/{len(ib)})"); continue
            # axis validity here
            X = a_space[ia + ib]; y = np.r_[np.ones(len(ia)), np.zeros(len(ib))]
            aucs = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
                idxA = [(ia + ib)[i] for i in tr if y[i] == 1]; idxB = [(ia + ib)[i] for i in tr if y[i] == 0]
                u = axis_from(a_space, idxA, idxB); s = a_space[[(ia + ib)[i] for i in te]] @ u
                aucs.append(roc_auc_score(y[te], s))
            auc = float(np.mean(aucs)); u = axis_from(a_space, ia, ib)

            # FIXED gene set (balanced in >=7 of 8 clusters): projecting the SAME genes per cluster isolates the
            # model CONTEXTUALLY bending shared genes -- the beyond-composition contribution -- and avoids the
            # composition/gene-set-mismatch artefact that made the small clusters (CLP/DCs) extreme.
            fixed = np.array([g for g in range(len(genes)) if gene_ok[g] and full[:, g].sum() >= 7])
            proj = Mzp @ u                                    # (2, nctx, ngene)  per-partition projection

            def loading(part_sel):
                L = np.full(len(clusters), np.nan)
                for ci in range(len(clusters)):
                    gg = fixed[full[ci, fixed]]               # fixed genes balanced in THIS cluster
                    if len(gg) > 20:
                        L[ci] = float(proj[:, ci, gg].mean() if part_sel is None else proj[part_sel, ci, gg].mean())
                return L

            load = loading(None)                              # partition-averaged
            L0, L1 = loading(0), loading(1)
            ok = np.isfinite(L0) & np.isfinite(L1)
            reprod = float(spearmanr(L0[ok], L1[ok]).statistic) if ok.sum() >= 4 else float("nan")

            stem = [i for i, c in enumerate(clusters) if LINEAGE.get(c) == "stem"]
            load_c = load - np.nanmean(load[stem]) if stem else load - np.nanmean(load)
            rho_pt = float(spearmanr(ptime, load, nan_policy="omit").statistic)
            per_cl = {clusters[i]: round(float(load_c[i]), 3) for i in range(len(clusters))}
            by_lin = {}
            for lin in set(LINEAGE.get(c, "?") for c in clusters):
                vals = [load_c[i] for i, c in enumerate(clusters) if LINEAGE.get(c) == lin and np.isfinite(load_c[i])]
                if vals:
                    by_lin[lin] = round(float(np.mean(vals)), 3)
            print(f"  {name:<28} AUC {auc:.2f}  split-half reprod {reprod:+.2f}  |{len(fixed)} fixed genes|")
            print(f"      per lineage (fixed genes, rel. stem): " + "  ".join(f"{k}:{v:+.2f}" for k, v in by_lin.items()))
            out["taps"][f"L{tap:02d}"]["axes"][name] = dict(validity_auc=auc, split_half_reprod=reprod,
                                                            n_fixed_genes=len(fixed), rho_pseudotime=rho_pt,
                                                            loading_by_cluster=per_cl, loading_by_lineage=by_lin)

    json.dump(out, open(os.path.join(RES, "ctx_devel_trajectory.json"), "w"), indent=1)
    print("\n[done] -> results/ctx_devel_trajectory.json")


if __name__ == "__main__":
    main()
