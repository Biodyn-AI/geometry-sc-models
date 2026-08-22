"""Re-test the antipodal hypothesis PROPERLY: in a subspace, not in the full space.

Ihor's objection, and it is correct: run_contextfree.py measured cos(W[GATA1], W[SPI1]) over ALL dimensions.
If two TFs share a big common component -- "I am a transcription factor", "I am a blood gene", plus whatever
generic axes (token frequency, embedding norm) dominate a gene table -- that shared mass swamps any
anti-alignment living in a small lineage subspace. The ESM2 numbers demonstrate exactly this failure: +0.66 to
+0.89 for every antagonistic pair, because shared DNA-binding domains dominate the whole vector. That measured
the confound, not the hypothesis.

Three progressively fairer tests:

  (1) GLOBAL-PC ABLATION SWEEP. Remove the gene table's mean and its top-k global PCs (the generic axes every
      gene shares), then re-measure the pair's cosine. Sweep k = 0, 1, 5, 20, 50. If the pair is antipodal
      inside a lineage subspace, stripping the shared mass should drive the cosine DOWN, ideally negative.

  (2) LINEAGE-SUBSPACE COSINE. Project both TFs onto the subspace spanned by their own lineage marker genes
      (centred), then measure the cosine there. This asks the hypothesis's actual question: within the space
      where this lineage lives, do the two master regulators point in opposite directions?

  (3) MARKER-AXIS PROJECTION. Build the axis from the MARKERS alone (mean of pole-A markers minus mean of
      pole-B markers) -- never touching the two TFs -- and ask where the TFs fall on it. If GATA1 and SPI1 are
      the poles of the erythroid/myeloid axis, they should sit at opposite ends of an axis defined by everyone
      else. This is the strongest version and is completely independent of the pair's shared mass.

Every test runs on the model bases AND on coexpr (data) and esm2 (sequence), as always: a model result must
beat both.

Run: ../../.venv/bin/python -u antipodal_subspace.py
Out: results/antipodal_subspace.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S

BASES = ["scgpt_we", "maxtoki_we", "maxtoki_lmhead", "coexpr", "esm2"]
KS = [0, 1, 5, 20, 50]
N_NULL = 2000


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def global_pc_ablate(M, k):
    """Remove the table mean and its top-k global PCs -- the generic axes every gene shares."""
    X = M - M.mean(0)
    if k == 0:
        return X
    from sklearn.utils.extmath import randomized_svd
    _, _, Vt = randomized_svd(X, n_components=k, n_iter=4, random_state=0)
    return X - (X @ Vt.T) @ Vt


def main():
    res = {}
    for name, h in S.by_kind("antipodal").items():
        a, b = h["a"], h["b"]
        rec = {"a": a, "b": b, "by_basis": {}}
        print(f"\n=== {name}  ({a} vs {b}) ===", flush=True)
        for base in BASES:
            M, syms = G.basis(base)
            pos = {s: i for i, s in enumerate(syms)}
            if a not in pos or b not in pos:
                print(f"  {base:<15} {a}/{b} not in basis", flush=True); continue
            out = {}

            # (1) global-PC ablation sweep
            sweep = {}
            for k in KS:
                X = global_pc_ablate(M, k)
                sweep[k] = float(_unit(X[pos[a]]) @ _unit(X[pos[b]]))
            out["pc_ablation"] = sweep

            # (2) lineage-subspace cosine
            Mg, ok = G.subset(base, h["axis_genes"])
            if ok.sum() >= 5:
                B = Mg - Mg.mean(0)
                Q, _ = np.linalg.qr(B.T)                       # orthonormal basis of the lineage subspace
                pa, pb = M[pos[a]] @ Q, M[pos[b]] @ Q
                out["lineage_subspace_cos"] = float(_unit(pa) @ _unit(pb))

            # (3) marker-axis projection -- axis built WITHOUT the two TFs
            sign = np.asarray(h["axis_sign"])[ok]
            if (sign > 0).sum() >= 2 and (sign < 0).sum() >= 2:
                axis = _unit(Mg[sign > 0].mean(0) - Mg[sign < 0].mean(0))
                ctr = M.mean(0)
                za, zb = (M[pos[a]] - ctr) @ axis, (M[pos[b]] - ctr) @ axis
                # null: random gene pairs projected on the same axis
                rng = np.random.default_rng(0)
                i1, i2 = rng.integers(0, len(M), N_NULL), rng.integers(0, len(M), N_NULL)
                nd = ((M[i1] - ctr) @ axis) - ((M[i2] - ctr) @ axis)
                out["marker_axis"] = dict(
                    z_a=float(za), z_b=float(zb), opposite_sign=bool(za * zb < 0),
                    separation=float(za - zb),
                    p_null=float((np.abs(nd) >= abs(za - zb)).mean()))
            rec["by_basis"][base] = out
            pcs = " ".join(f"k{k}:{sweep[k]:+.2f}" for k in KS)
            ls = out.get("lineage_subspace_cos")
            ma = out.get("marker_axis", {})
            print(f"  {base:<15} PCablate[{pcs}]  subspace_cos:"
                  f"{'n/a' if ls is None else format(ls, '+.2f')}  "
                  f"marker_axis {a}:{ma.get('z_a', float('nan')):+.2f} {b}:{ma.get('z_b', float('nan')):+.2f}"
                  f" {'OPPOSITE' if ma.get('opposite_sign') else 'same side'}"
                  f" (p={ma.get('p_null', float('nan')):.3f})", flush=True)
        res[name] = rec

    # ---- summary
    print("\n" + "=" * 96)
    print("Does ANY basis put an antagonistic pair on OPPOSITE sides of a marker-defined axis?")
    print("=" * 96)
    for name, rec in res.items():
        hits = [b for b, o in rec["by_basis"].items()
                if o.get("marker_axis", {}).get("opposite_sign") and o["marker_axis"]["p_null"] < 0.05]
        neg = [b for b, o in rec["by_basis"].items()
               if o.get("lineage_subspace_cos") is not None and o["lineage_subspace_cos"] < 0]
        print(f"  {name:<24} opposite&sig: {hits or '-'}   |  negative subspace cos: {neg or '-'}")
    json.dump(res, open(os.path.join(HERE, "results", "antipodal_subspace.json"), "w"), indent=1)
    print("\n[done] -> results/antipodal_subspace.json")


if __name__ == "__main__":
    main()
