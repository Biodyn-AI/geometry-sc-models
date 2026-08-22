"""Do the SUPERVISED-PROBE revivals survive a paired bootstrap? (the test that killed the PCA battery)

run_probe.py revived 4 hypotheses by swapping the unsupervised PCA-max statistic for a cross-validated ridge
probe. But its verdict rule compares POINT ESTIMATES across bases -- the exact error that manufactured four
false positives in run_contextfree.py and that verify_hits.py then destroyed. Same discipline applies to my own
good news: a bigger number is not a win.

Paired bootstrap over GENES. Each resample re-runs the ENTIRE pipeline on both bases -- refit folds, refit
RidgeCV, recompute the out-of-fold statistic -- so the margin's CI includes the instability of fitting a probe
to a few dozen genes, which was the dominant term last time.

A hypothesis is REAL only if the margin CI excludes 0 against EVERY reference basis that has the genes:
    coexpr / coexpr_k562 (the data) and esm2 (protein sequence).

Run: ../../.venv/bin/python -u verify_probe.py
Out: results/verify_probe.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
from run_probe import score, _folds, MODEL_BASES, REFS

N_BOOT = 300
SEED = 0


def coord_of(h, ok, idx=None):
    if h["kind"] == "grid":
        c = (np.asarray(h["coord"][0], float)[ok], np.asarray(h["coord"][1], float)[ok])
        return (c[0][idx], c[1][idx]) if idx is not None else c
    c = np.asarray(h["coord"], float)[ok]
    return c[idx] if idx is not None else c


def main():
    probe = json.load(open(os.path.join(HERE, "results", "probe.json")))
    todo = [k for k, v in probe.items() if v.get("verdict") == "MODEL"]
    print(f"Verifying {len(todo)} probe revivals: {todo}\n", flush=True)
    res = {}
    for name in todo:
        h = S.H[name]
        mb = probe[name]["best_model_basis"]
        rec = {"best_model_basis": mb, "kind": h["kind"], "vs": {}}
        print(f"=== {name}  (model basis {mb}, kind={h['kind']}) ===", flush=True)
        for ref in REFS:
            Mm, okm = G.subset(mb, h["genes"])
            Mr, okr = G.subset(ref, h["genes"])
            both = okm & okr
            if both.sum() < 8:
                print(f"  vs {ref:<12} SKIP ({int(both.sum())} shared genes)", flush=True); continue
            gsub = np.asarray(h["genes"])[both]
            Mm2, _ = G.subset(mb, gsub); Mr2, _ = G.subset(ref, gsub)
            c = coord_of(h, both)
            f = _folds(len(gsub))
            obs = score(h["kind"], Mm2, c, f) - score(h["kind"], Mr2, c, f)
            rng = np.random.default_rng(SEED)
            d = np.empty(N_BOOT)
            for b in range(N_BOOT):
                i = rng.integers(0, len(gsub), len(gsub))
                fb = _folds(len(i))
                cb = coord_of(h, both, i)
                try:
                    d[b] = score(h["kind"], Mm2[i], cb, fb) - score(h["kind"], Mr2[i], cb, fb)
                except Exception:
                    d[b] = np.nan
            d = d[np.isfinite(d)]
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            rec["vs"][ref] = dict(margin=float(obs), ci_lo=lo, ci_hi=hi, n=int(both.sum()),
                                  frac_le0=float((d <= 0).mean()))
            print(f"  vs {ref:<12} n={int(both.sum()):<3} margin {obs:+.3f} CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"{'SURVIVES' if lo > 0 else 'dies'}", flush=True)
        ok_refs = [k for k, v in rec["vs"].items() if "ci_lo" in v]
        rec["survives_all"] = bool(ok_refs) and all(rec["vs"][k]["ci_lo"] > 0 for k in ok_refs)
        res[name] = rec
        print()

    print("=" * 88)
    print("SURVIVORS OF THE PAIRED BOOTSTRAP")
    print("=" * 88)
    for k, v in res.items():
        fails = [r for r in v["vs"] if v["vs"][r].get("ci_lo", -1) <= 0]
        print(f"  {k:<22} {'** SURVIVES **' if v['survives_all'] else 'dies'}   fails={fails or '-'}")
    json.dump(res, open(os.path.join(HERE, "results", "verify_probe.json"), "w"), indent=1)
    print("\n[done] -> results/verify_probe.json")


if __name__ == "__main__":
    main()
