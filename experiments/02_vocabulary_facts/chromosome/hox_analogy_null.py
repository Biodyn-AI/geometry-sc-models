"""ANALOGY: the MISSING NULL + the per-cluster breakdown. (Ihor, 2026-07-17)

WHY THIS EXISTS. `hox_analogy.py` promises, in its own docstring, "a RANDOM null (shuffle which gene sits at
each grid cell) to calibrate what the grid's own structure gives for free" -- and never implements it. There is
no null anywhere in that file. So `median_rank = 2 / 39` has never been compared against what you would get
from the HOX island's geometry alone. That null is the whole ballgame: the 39 HOX genes are a tight, highly
non-random clump, and a query built out of three of them lands inside the clump BY CONSTRUCTION.

TWO NULLS, AND THE CHOICE DECIDES THE VERDICT -- so both are run.

  `joint`   -- permute (cluster, paralog) JOINTLY: which gene sits in which cell. Holds the island's geometry,
               the gene set, the abundance prior and the occupancy pattern fixed. The obvious reading of the
               docstring hox_analogy.py never implemented.
  `within`  -- permute paralog WITHIN cluster. Holds all of the above AND the cluster block structure, so it
               asks the sharper question: given that the four clusters are separated, is the PARALOG assignment
               inside them what makes the analogy work?

`within` is the STRICTER null and it is the one `hox_within.py`'s T3 already uses for the test whose docstring
calls it "the supervised twin of hox_analogy.py". Using `joint` here and `within` there would be incoherent.
It matters: `joint` destroys cluster structure, which under-credits the null ONLY for a basis that HAS cluster
structure (MaxToki), while for a basis with none (esm2) the two nulls coincide. So `joint` mechanically
flatters MaxToki. Report both; let the verdict rule read the strict one.

ALSO: the per-cluster breakdown, grouped by the TARGET cluster c2 (where D lives). `hox_shape.py` TEST 3 called
HOXB the exception on a PC1-only test. If HOXB really lacks a paralog direction, analogies that must LAND in
HOXB should fail. This checks that directly, on the same quadruples.

Reproduction of hox_analogy.py's published numbers is ASSERTED in `_assert_reproduces()` before any null runs
(an earlier revision of this docstring advertised that guard without implementing it -- exactly the genre of
defect this file exists to correct).

Run: ../../.venv/bin/python -u hox_analogy_null.py
Out: results/hox_analogy_null.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S

BASES = ["maxtoki_lmhead", "maxtoki_we", "scgpt_we", "coexpr", "coexpr_devel", "esm2"]
N_NULL = 200
SEED = 0
CN = "ABCD"

h = S.H["hox_grid"]
GENES = np.array(h["genes"]); PAR = np.asarray(h["coord"][0], int); CLU = np.asarray(h["coord"][1], int)


def quads_idx(par, clu):
    """(A,B,C,D,c1,c2) as indices into the present-gene arrays. A,B in cluster c1 at paralogs p1<p2;
    C,D the same two paralogs in cluster c2. Mirrors hox_analogy.quads (one gene per (cluster,paralog) cell,
    so 'first match' and 'the match' coincide)."""
    n = len(par)
    out = []
    for c1 in range(4):
        for c2 in range(4):
            if c1 == c2:
                continue
            for i in range(n):
                for j in range(n):
                    if clu[i] != c1 or clu[j] != c1 or par[i] >= par[j]:
                        continue
                    ci = np.where((clu == c2) & (par == par[i]))[0]
                    cj = np.where((clu == c2) & (par == par[j]))[0]
                    if len(ci) and len(cj):
                        out.append((i, j, int(ci[0]), int(cj[0]), c1, c2))
    return out


def evaluate(M, Mn, rows, par, clu, cand, per_cluster=False):
    """3CosAdd over `cand` (row indices into M). rows[i] = the model row of present-gene i.
    Excludes A,B,C from the pool. Rank = 1 + #candidates strictly more similar than the true D."""
    Q = quads_idx(par, clu)
    if len(Q) < 20:
        return None
    A = np.array([rows[q[0]] for q in Q]); B = np.array([rows[q[1]] for q in Q])
    C = np.array([rows[q[2]] for q in Q]); D = np.array([rows[q[3]] for q in Q])
    qv = M[B] - M[A] + M[C]
    qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    sims = Mn[cand] @ qv.T                                  # (n_cand, n_quad)
    kpos = {int(ci): k for k, ci in enumerate(cand)}
    nq = len(Q)
    for j in range(nq):
        for x in (A[j], B[j], C[j]):
            k = kpos.get(int(x))
            if k is not None:
                sims[k, j] = -np.inf
    tgt = np.array([kpos[int(d)] for d in D])
    tsim = sims[tgt, np.arange(nq)]
    ranks = 1 + (sims > tsim[None, :]).sum(0)
    out = dict(n=nq, top1=float((ranks == 1).mean()), top5=float((ranks <= 5).mean()),
               median_rank=float(np.median(ranks)), pool_size=int(len(cand)))
    if per_cluster:
        by = defaultdict(list)
        for j, q in enumerate(Q):
            by[CN[q[5]]].append(ranks[j])
        out["by_target_cluster"] = {k: dict(n=len(v), top1=float(np.mean(np.array(v) == 1)),
                                            top5=float(np.mean(np.array(v) <= 5)),
                                            median_rank=float(np.median(v)))
                                    for k, v in sorted(by.items())}
    return out


def _permute(par, clu, rng, mode):
    """`joint`  -> permute which gene sits in which (cluster, paralog) cell.
       `within` -> permute paralog WITHIN cluster, preserving the cluster block structure (the stricter null)."""
    if mode == "joint":
        p = rng.permutation(len(par))
        return par[p], clu[p]
    par2 = par.copy()
    for c in np.unique(clu):
        m = clu == c
        par2[m] = rng.permutation(par2[m])
    return par2, clu


def _assert_reproduces(M, Mn, rows, par, clu):
    """hox_analogy.py's published numbers, which this file's vectorised rewrite must match before it is trusted
    to null them. Values from hox_analogy.py's own output (recovered from the 2026-07-17 session transcript)."""
    hox = evaluate(M, Mn, rows, par, clu, rows)
    voc = evaluate(M, Mn, rows, par, clu, np.arange(len(M)))
    for got, exp, tag in [(hox, dict(n=258, top1=0.353, top5=0.849, median_rank=2), "hox"),
                          (voc, dict(n=258, top1=0.333, top5=0.822, median_rank=2), "vocab")]:
        assert got["n"] == exp["n"], f"{tag}: n {got['n']} != {exp['n']}"
        for k in ("top1", "top5"):
            assert abs(got[k] - exp[k]) < 1e-3, f"{tag}: {k} {got[k]:.4f} != {exp[k]}"
        assert got["median_rank"] == exp["median_rank"], f"{tag}: median_rank {got['median_rank']}"
    print("[assert] reproduces hox_analogy.py exactly: hox 0.353/0.849/rank2, vocab 0.333/0.822/rank2 OK\n")


def main():
    res = {}
    # the reproduction guard runs on the basis hox_analogy.py's published numbers came from
    _M, _s = G.basis("maxtoki_lmhead")
    _pos = {s: i for i, s in enumerate(_s)}
    _ok = np.array([g in _pos for g in GENES])
    _assert_reproduces(_M, _M / (np.linalg.norm(_M, axis=1, keepdims=True) + 1e-12),
                       np.array([_pos[g] for g in GENES[_ok]]), PAR[_ok], CLU[_ok])

    for b in BASES:
        M, syms = G.basis(b)
        pos = {s: i for i, s in enumerate(syms)}
        ok = np.array([g in pos for g in GENES])
        rows = np.array([pos[g] for g in GENES[ok]])
        par, clu = PAR[ok], CLU[ok]
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        print(f"\n{'=' * 92}\n=== {b}   (n={int(ok.sum())}/39 HOX genes present, d={M.shape[1]})\n{'=' * 92}",
              flush=True)
        rec = {"n_present": int(ok.sum())}

        for pool, cand in [("hox", rows), ("vocab", np.arange(len(M)))]:
            obs = evaluate(M, Mn, rows, par, clu, cand, per_cluster=(pool == "hox"))
            if obs is None:
                print(f"  [{pool}] too few quadruples"); continue
            print(f"  [{pool:<5}] n={obs['n']:<4} top1={obs['top1']:.3f}  top5={obs['top5']:.3f}  "
                  f"median_rank={obs['median_rank']:.0f} / {obs['pool_size']}", flush=True)
            rec[pool] = obs

            # ---- THE MISSING NULLS. `within` is the strict one and carries the verdict; `joint` is the
            #      loose one and is reported only so the gap between them is visible.
            for mode in ("joint", "within"):
                rng = np.random.default_rng(SEED)
                n_t1, n_mr = np.empty(N_NULL), np.empty(N_NULL)
                for t in range(N_NULL):
                    pp, cc = _permute(par, clu, rng, mode)
                    r = evaluate(M, Mn, rows, pp, cc, cand)
                    n_t1[t] = r["top1"] if r else np.nan
                    n_mr[t] = r["median_rank"] if r else np.nan
                n_t1, n_mr = n_t1[np.isfinite(n_t1)], n_mr[np.isfinite(n_mr)]
                p_t1 = float(((n_t1 >= obs["top1"]).sum() + 1) / (len(n_t1) + 1))
                p_mr = float(((n_mr <= obs["median_rank"]).sum() + 1) / (len(n_mr) + 1))
                z = float((obs["top1"] - n_t1.mean()) / (n_t1.std() + 1e-12))
                mult = float(obs["top1"] / (n_t1.mean() + 1e-12))
                tag = "joint (loose)" if mode == "joint" else "within (STRICT)"
                print(f"          NULL {tag:<16} n={len(n_t1)}: top1 {n_t1.mean():.3f}+-{n_t1.std():.3f} "
                      f"rank {np.median(n_mr):.0f} -> {mult:.1f}x  z={z:+.1f}  p_top1={p_t1:.4f} "
                      f"p_rank={p_mr:.4f}", flush=True)
                rec[pool][f"null_{mode}"] = dict(n=len(n_t1), top1_mean=float(n_t1.mean()),
                                                 top1_sd=float(n_t1.std()), rank_median=float(np.median(n_mr)),
                                                 multiple=mult, p_top1=p_t1, p_rank=p_mr, z=z)

        if "hox" in rec and "by_target_cluster" in rec["hox"]:
            print("  per TARGET cluster (where D must land) -- is HOXB the exception hox_shape.py claimed?")
            for k, v in rec["hox"]["by_target_cluster"].items():
                print(f"    -> HOX{k}: n={v['n']:<4} top1={v['top1']:.3f}  top5={v['top5']:.3f}  "
                      f"median_rank={v['median_rank']:.0f}")
        res[b] = rec

    # ---- the project's verdict rule (gm_lib.py:20, run_probe.py:23), read off the STRICT null.
    print(f"\n{'=' * 92}\nVERDICT -- gm_lib.py:20: 'a hypothesis only counts if a model basis beats BOTH'")
    print(f"{'=' * 92}")
    MODELS = ["maxtoki_lmhead", "maxtoki_we", "scgpt_we"]
    REFS = ["coexpr", "coexpr_devel", "esm2"]
    for mode in ("joint", "within"):
        got = {b: res[b]["hox"][f"null_{mode}"]["multiple"] for b in res
               if "hox" in res.get(b, {}) and f"null_{mode}" in res[b]["hox"]}
        mods = {b: v for b, v in got.items() if b in MODELS}
        refs = {b: v for b, v in got.items() if b in REFS}
        if not mods or not refs:
            continue
        mb = max(mods, key=mods.get)
        beaten = {r: mods[mb] > v for r, v in refs.items()}
        ok = all(beaten.values())
        rstr = "  ".join(f"{r.split('_')[0][:6]}:{v:.1f}x" for r, v in refs.items())
        print(f"  [{mode:<6}] best model {mb} {mods[mb]:.1f}x   refs[{rstr}]   "
              f"-> {'** MODEL **' if ok else 'NO'}")
        if not ok:
            print(f"            loses to {', '.join(r for r, w in beaten.items() if not w)}")
        res[f"_verdict_{mode}"] = dict(best_model=mb, multiple=mods[mb], refs=refs, passes=bool(ok))
    print("\n  The two nulls disagree -> the STRICT (`within`) one governs. See module docstring for why.")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "hox_analogy_null.json"), "w"), indent=1)
    print("\n[done] -> results/hox_analogy_null.json")


if __name__ == "__main__":
    main()
