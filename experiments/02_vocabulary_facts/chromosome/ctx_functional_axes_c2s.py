"""ctx_functional_axes_c2s — HEADLINE #2: do genes move along FUNCTIONALLY meaningful directions in context?
Port of ctx_functional_axes.py.

For each GO-defined axis (poleA vs poleB), build u = unit(mean a(poleA) - mean a(poleB)) in gene-main-effect
space, then:
  VALIDITY  5-fold CV AUC that <a(g),u> separates held-out poleA from poleB genes (is the axis real).
  POWER     reproducible along-axis interaction power = mean_{(g,c) balanced} I0(g,c)*I1(g,c), where I_p is the
            double-centred projection q_p=<Mz,u>; cross-partition so noise cancels. Raw and rank-residualised.
  NULL      N_NULL random gene-set partitions of identical pole sizes -> z_raw, z_rank_controlled.
Verdict POSITIVE iff best VALID axis (AUC>0.65) has z_rank_controlled > 3.  Out: results/ctx_functional_axes_c2s.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, argparse, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx_lib_c2s as L

G2G_PATH = os.environ.get("C2S_GENE2GO",
    f"{_DATA}/perturb/gene2go_all.pkl")
AXES = {
    "mito_vs_cytoskeleton": (["GO:0005739"], ["GO:0005856"]),
    "nuclear_vs_surface": (["GO:0005634", "GO:0000785", "GO:0003677"],
                           ["GO:0005886", "GO:0005576", "GO:0005615"]),
    "cellcycle_vs_diff": (["GO:0007049"], ["GO:0030154"]),
    "transcription_vs_transport": (["GO:0006355", "GO:0003700"], ["GO:0006811", "GO:0038023"]),
}
N_NULL = 300
SEED = 0


def pole_rows(genes, g2g, A, B):
    A, B = set(A), set(B)
    ia = [i for i, s in enumerate(genes) if s in g2g and (set(g2g[s]) & A)]
    ib = [i for i, s in enumerate(genes) if s in g2g and (set(g2g[s]) & B)]
    common = set(ia) & set(ib)
    ia = np.array([i for i in ia if i not in common]); ib = np.array([i for i in ib if i not in common])
    return ia, ib


def for_tap(tap, bases, g2g):
    d = L.load(tap, bases)
    full = L.balanced(d["counts"], d["cap"])
    Mz, _ = L.zscore_dims(d["M"], full)
    A = L.a_space(Mz, full)                              # (nG, H) gene main effect
    rank = L.rank_mean(d["rank_tok"])
    genes = d["genes"]
    valid_gene = np.isfinite(A).all(1)                   # genes with a defined main effect
    rng = np.random.default_rng(SEED)
    pool = np.where(valid_gene)[0]
    res = {}
    for name, (Ag, Bg) in AXES.items():
        ia, ib = pole_rows(genes, g2g, Ag, Bg)
        ia = ia[np.isin(ia, pool)]; ib = ib[np.isin(ib, pool)]
        if len(ia) < 6 or len(ib) < 6:
            res[name] = dict(skipped=f"poleA={len(ia)} poleB={len(ib)}"); continue
        u = L.axis_from_poles(A, None, ia, ib)
        # validity: 5-fold CV AUC (rebuild axis on train, score test)
        idx = np.concatenate([ia, ib]); y = np.r_[np.ones(len(ia)), np.zeros(len(ib))]
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y):
            utr = L.axis_from_poles(A, None, idx[tr][y[tr] == 1], idx[tr][y[tr] == 0])
            s = A[idx[te]] @ utr
            if len(set(y[te])) == 2:
                aucs.append(roc_auc_score(y[te], s))
        auc = float(np.mean(aucs)) if aucs else np.nan

        def power(vec, resid):
            I = L.along_axis_interaction(Mz, full, vec, rank if resid else None)
            return L.axis_power(I)
        p_raw, p_res = power(u, False), power(u, True)
        null_raw, null_res = [], []
        for _ in range(N_NULL):
            ra = rng.choice(pool, len(ia), replace=False); rb = rng.choice(pool, len(ib), replace=False)
            v = L.axis_from_poles(A, None, ra, rb)
            null_raw.append(power(v, False)); null_res.append(power(v, True))
        null_raw, null_res = np.array(null_raw), np.array(null_res)
        z_raw = float((p_raw - null_raw.mean()) / (null_raw.std() + 1e-12))
        z_res = float((p_res - null_res.mean()) / (null_res.std() + 1e-12))
        res[name] = dict(validity_auc=auc, power=p_raw, power_resid=p_res, z_raw=z_raw,
                         z_rank_controlled=z_res, null_mean=float(null_raw.mean()),
                         null_resid_mean=float(null_res.mean()), poleA=len(ia), poleB=len(ib))
        print(f"  L{tap:02d} {name:<26} AUC={auc:.3f} z_raw={z_raw:+5.1f} || z_RANK-CTRL={z_res:+5.1f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="ctx_bases")
    ap.add_argument("--taps", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results/ctx_functional_axes_c2s.json")
    a = ap.parse_args()
    import glob
    g2g = pickle.load(open(G2G_PATH, "rb"))
    taps = a.taps or sorted(int(os.path.basename(p).split("_L")[1].split(".")[0]) for p in glob.glob(os.path.join(a.bases, "ctx_c2s_L*.npz")))
    res = {}
    for t in taps:
        res[f"L{t:02d}"] = for_tap(t, a.bases, g2g)
    # verdict: best valid (AUC>0.65) axis z_rank_controlled across all taps
    best = None
    for tk, tv in res.items():
        for ax, dd in tv.items():
            if isinstance(dd, dict) and dd.get("validity_auc", 0) and dd["validity_auc"] > 0.65 and "z_rank_controlled" in dd:
                if best is None or dd["z_rank_controlled"] > best[2]["z_rank_controlled"]:
                    best = (tk, ax, dd)
    verdict = "no valid axis"
    if best:
        verdict = (f"POSITIVE beyond random ({best[1]} @ {best[0]}, z_rank={best[2]['z_rank_controlled']:+.1f})"
                   if best[2]["z_rank_controlled"] > 3 else
                   f"organised movement NOT beyond random-partition null (best z_rank={best[2]['z_rank_controlled']:+.1f})")
        print(f"\nVERDICT: {verdict}", flush=True)
    res["_verdict"] = verdict
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
