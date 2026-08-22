"""Build the Ainciburu (GSE180298) young+elderly CD34+ HSPC data into the Setty h5ad schema.

Writes X (raw counts, CSR), var/index (gene symbols), obs/clusters (+__categories), obs/palantir_pseudotime,
obs/donor, obs/age_group -- the same on-disk layout as setty19_cd34_bm.h5ad, so
route_steering/extract_scgpt_binned.py runs on it unchanged via BP_H5AD / BP_OUT.

Ground truth (each donor's committed-lineage composition) is computed from the FULL metadata (all ~75k cells,
free) and saved alongside; only a balanced SUBSAMPLE is embedded (<=500 HSC + <=200 of each other type per
donor) so the scGPT extraction is tractable.

Pseudotime is computed (scanpy DPT rooted in HSC) and stored for completeness, but the PRIMARY readout does not
use it -- fate directions are HSC-centroid -> committed-centroid and need no pseudotime. Flagged because DPT on
merged un-integrated donors is donor-driven and should not be trusted here.

Run (conda bio_mech_interp -- needs scanpy):
  ~/anaconda3/envs/bio_mech_interp/bin/python preprocess_aging.py
Out: data/aging/aging_setty_schema.h5ad  +  data/aging/ground_truth.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, json
import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, vstack

ROOT = f"{_DATA}"
AG = f"{ROOT}/data/aging"
OUT = f"{AG}/aging_setty_schema.h5ad"
OUT_MDS = f"{AG}/agingmds_setty_schema.h5ad"
GT = f"{AG}/ground_truth.json"

GROUPS = ("young", "elderly", "mds") if os.environ.get("INCLUDE_MDS") else ("young", "elderly")
FILES = {  # donor -> 10x h5 in the extracted RAW tar (young1=GSM5460406 ... elderly3=GSM5460413)
    **{f"young{i}": f"GSM{5460405+i}_young{i}_filtered_feature_bc_matrix.h5" for i in range(1, 6)},
    **{f"elderly{i}": f"GSM{5460410+i}_elderly{i}_filtered_feature_bc_matrix.h5" for i in range(1, 4)},
    **{f"mds{i}": f"GSM{6946064+i}_mds{i}_filtered_feature_bc_matrix.h5" for i in range(1, 5)},
}
LIN = {"HSC": "stem", "LMPP": "stem",
       "MEP": "ery", "Erythroid_early": "ery", "Erythroid_late": "ery",
       "GMP": "myeloid", "GMP_Granulocytes": "myeloid", "Monocytes": "myeloid", "Basophils": "myeloid",
       "CLP": "lymphoid", "ProB": "lymphoid", "T_NK": "lymphoid", "pDC": "lymphoid",
       "Megakaryocytes": "mega"}
CAP_HSC, CAP_OTHER, SEED = 500, 200, 0


def main():
    meta = []
    for grp in GROUPS:
        m = pd.read_csv(f"{AG}/GSE180298_{grp}_metadata.txt.gz", sep="\t", index_col=0)
        m["donor"] = [i.split("_")[-1] for i in m.index]
        m["age_group"] = grp
        m["barcode"] = [i.split("_")[0] for i in m.index]
        meta.append(m)
    meta = pd.concat(meta)
    meta["lineage"] = meta.CellType.map(LIN)
    print(f"[meta] {len(meta)} annotated cells, {meta.donor.nunique()} donors")

    # ---- GROUND TRUTH from the FULL data: committed-lineage composition per donor
    comm = meta[~meta.lineage.isin(["stem"]) & meta.lineage.notna()]
    truth = (comm.groupby("donor").lineage.value_counts(normalize=True).unstack().fillna(0.0))
    gt = {d: {f: float(truth.loc[d, f]) for f in truth.columns} for d in truth.index}
    for d in gt:
        gt[d]["age_group"] = "young" if d.startswith("young") else ("elderly" if d.startswith("elderly") else "mds")
        gt[d]["n_committed"] = int((comm.donor == d).sum())
    json.dump(gt, open(GT if not os.environ.get("INCLUDE_MDS") else GT.replace("ground_truth","ground_truth_mds"), "w"), indent=1)
    print("[truth] committed-output composition per donor written to ground_truth.json")
    print(truth.round(3).to_string())

    # ---- balanced subsample for embedding
    rng = np.random.default_rng(SEED)
    keep = []
    for d, sub in meta.groupby("donor"):
        for ct, s2 in sub.groupby("CellType"):
            cap = CAP_HSC if ct == "HSC" else CAP_OTHER
            idx = s2.index.values
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.append(pd.Series(idx))
    keep = pd.Index(pd.concat(keep).values)
    sel = meta.loc[keep]
    print(f"[subsample] {len(sel)} cells for embedding "
          f"({(sel.CellType == 'HSC').sum()} HSC across {sel.donor.nunique()} donors)")

    # ---- load the 10x matrices, restricted to the kept barcodes.
    # The MDS samples were processed against a DIFFERENT CellRanger gene reference than the healthy ones, so we
    # align on the INTERSECTION of gene symbols (order taken from the first sample) rather than assuming an
    # identical var order. Gene identity, not position, is what matters to scGPT (no positional encoding).
    use = [d for d in FILES if d in set(sel.donor)]
    ads = {}
    for d in use:
        a = sc.read_10x_h5(f"{AG}/raw/{FILES[d]}")
        a.var_names_make_unique()
        a.obs_names = [b.split("-")[0] for b in a.obs_names]   # 10x barcodes carry a "-1" suffix; metadata does not
        ads[d] = a
    common = None
    for d in use:
        s = set(ads[d].var_names)
        common = s if common is None else (common & s)
    genes_ref = pd.Index([g for g in ads[use[0]].var_names if g in common])
    print(f"[genes] {len(genes_ref)} symbols common to all {len(use)} samples "
          f"(per-sample: {[ads[d].n_vars for d in use]})")

    blocks, obs = [], []
    for d in use:
        a = ads[d][:, genes_ref].copy()                        # reindex every sample onto the common gene order
        want = sel[sel.donor == d]
        bc = pd.Index(want.barcode.values)
        present = bc.intersection(a.obs_names)
        a = a[present].copy()
        blocks.append(csr_matrix(a.X))
        o = want.set_index("barcode").loc[present]
        o["cell"] = [f"{b}_{d}" for b in present]
        obs.append(o)
        print(f"  [{d}] {a.n_obs}/{len(bc)} barcodes matched")
    X = vstack(blocks, format="csr").astype(np.float32)
    obs = pd.concat(obs)
    print(f"[matrix] X = {X.shape} raw counts")

    # ---- pseudotime (stored for completeness; NOT used by the primary readout -- see docstring)
    A = ad.AnnData(X=X, obs=obs.reset_index(drop=True), var=pd.DataFrame(index=genes_ref))
    try:
        B = A.copy()
        sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
        sc.pp.highly_variable_genes(B, n_top_genes=2000); B = B[:, B.var.highly_variable].copy()
        sc.pp.scale(B, max_value=10); sc.tl.pca(B, n_comps=30)
        sc.pp.neighbors(B, n_neighbors=15); sc.tl.diffmap(B)
        hsc = np.where(B.obs.CellType.values == "HSC")[0]
        dc = B.obsm["X_diffmap"][:, 1]
        ery = np.where(B.obs.lineage.values == "ery")[0]
        sign = 1 if dc[hsc].mean() < dc[ery].mean() else -1
        B.uns["iroot"] = int(hsc[np.argmin(sign * dc[hsc])])
        sc.tl.dpt(B)
        pt = np.nan_to_num(B.obs["dpt_pseudotime"].values.astype(np.float64),
                           nan=0.5, posinf=1.0, neginf=0.0)
        print("[dpt] mean pseudotime by lineage:",
              {l: round(float(pt[A.obs.lineage.values == l].mean()), 3)
               for l in ["stem", "ery", "myeloid", "lymphoid", "mega"]})
    except Exception as e:
        print(f"[dpt] FAILED ({e}) -> storing zeros; primary readout does not use pseudotime")
        pt = np.zeros(A.n_obs)

    # ---- write the Setty-schema h5ad
    cats = sorted(obs.CellType.unique())
    codes = np.array([cats.index(c) for c in obs.CellType], np.int8)
    sdt = h5py.string_dtype("utf-8")
    out_path = OUT_MDS if os.environ.get("INCLUDE_MDS") else OUT
    with h5py.File(out_path, "w") as f:
        g = f.create_group("X")
        g.create_dataset("data", data=X.data.astype(np.float32))
        g.create_dataset("indices", data=X.indices.astype(np.int32))
        g.create_dataset("indptr", data=X.indptr.astype(np.int32))
        g.attrs["encoding-type"] = "csr_matrix"; g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array([X.shape[0], X.shape[1]], dtype=np.int64)
        f.create_group("var").create_dataset("index", data=np.array(list(genes_ref), dtype=object), dtype=sdt)
        o = f.create_group("obs")
        o.create_dataset("index", data=np.array(list(obs.cell), dtype=object), dtype=sdt)
        o.create_dataset("clusters", data=codes)
        o.create_dataset("palantir_pseudotime", data=pt)
        o.create_dataset("donor", data=np.array(list(obs.donor), dtype=object), dtype=sdt)
        o.create_dataset("age_group", data=np.array(list(obs.age_group), dtype=object), dtype=sdt)
        o.create_group("__categories").create_dataset("clusters", data=np.array(cats, dtype=object), dtype=sdt)
    print(f"[done] wrote {out_path}  X={X.shape}  celltypes={cats}")


if __name__ == "__main__":
    main()
