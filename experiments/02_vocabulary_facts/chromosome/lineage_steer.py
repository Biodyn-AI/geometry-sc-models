"""CAUSAL STEERING OF LINEAGE BISTABLE SWITCHES — are the antagonistic-TF axes movable, and RECIPROCAL? (Ihor)

route_genemanifold shelved the antipodal lineage axes (RORC/FOXP3 Th17-Treg, GATA1/SPI1 ery-myeloid, ...) as
"decodable, model-specific directions" (§8, antipodal_subspace.py) but NEVER steered them. The bigger-picture
reframe (2026-07-18): stop gating on novelty-vs-baselines; ask instead whether a direction is one the model's
computation USES — i.e. can you move along it and change what the model predicts? This is that test, on the
lineage switches, built on the validated steering tool (`steer_lib.py`, `STEERING_TOOL.md`).

THE BISTABLE SIGNATURE. A fate switch is not just "two gene sets". Its defining property is RECIPROCAL
REPRESSION: pushing toward pole A should RAISE A's target program AND LOWER B's. So the readout is the
DIFFERENTIAL R = mean_logit(A_targets) - mean_logit(B_targets) at held-out positions, and the test is:
    switch = ΔR(steer toward A) - ΔR(steer toward B)
A true switch gives switch >> 0 (A-push favours A, B-push favours B); a norm-matched random push gives ~0.

NON-CIRCULARITY (the §5 trap). The steering direction is built from the two master TFs + a DIR-half of each
pole's markers, in INPUT (embed_tokens) space; the readout is the model's own next-gene logits on the HELD-OUT
READ-half of the markers, at DIFFERENT positions than the push. Direction genes and readout genes are disjoint,
and the readout is the model's native logits (no fitted probe) — nothing to be circular with.

TISSUE GATING (expected, per §6B / §12). Cells are setty CD34+ bone marrow. GATA1/SPI1 (erythroid vs myeloid)
is the on-tissue axis and should move most; SOX2/CDX2 (embryonic) should be ~null here. Per-axis reporting makes
the gating visible rather than averaging it away.

Run: ../../.venv_state/bin/python -u lineage_steer.py [n_cells=48] [seed=0]
Out: results/lineage_steer.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if os.environ.get("FORCE_CPU"):
    import torch as _torch
    _torch.backends.mps.is_available = lambda: False      # force the adapter's _auto_device() onto CPU
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S
import steer_lib as SL
from steer_propagation import load_cells

ALPHAS = [2.0, 4.0]          # units of mean gene-embedding norm (matches the validated chromosome test)
N_RAND = 3
MIN_HALF = 2                 # each pole needs >= MIN_HALF genes in BOTH the direction-half and the readout-half
SEED = 0


def sym2tok():
    tokmap = json.load(open(SL.TOKMAP))                       # ensembl -> token id
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    out = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s is not None and isinstance(tid, int):
            out.setdefault(s, tid)
    return out


def split_pole(genes, s2t, rng):
    """token ids of genes present in the vocab, split into (direction-half, readout-half)."""
    toks = [s2t[g] for g in genes if g in s2t]
    if len(toks) < 2 * MIN_HALF:
        return None, None
    rng.shuffle(toks)
    h = len(toks) // 2
    return toks[:h], toks[h:]                                 # dir-half, read-half


def main(n_cells=48, seed=SEED):
    st = SL.Steerer()
    s2t = sym2tok()
    EMB = st.model.model.embed_tokens.weight.detach().cpu().numpy()
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    rng = np.random.default_rng(seed)

    # build per-axis directions (toward pole A) + disjoint readout token sets
    axes = {}
    for k, v in S.ANTIPODAL.items():
        ag, sg = v["axis_genes"], v["axis_sign"]
        poleA = [v["a"]] + [g for g, s in zip(ag, sg) if s > 0]
        poleB = [v["b"]] + [g for g, s in zip(ag, sg) if s < 0]
        dA, rA = split_pole(poleA, s2t, rng)
        dB, rB = split_pole(poleB, s2t, rng)
        if dA is None or dB is None:
            print(f"[skip] {k}: too few genes in vocab"); continue
        # direction toward pole A = mean(A_dir) - mean(B_dir); centroid_direction(a,b)=mean(b)-mean(a)
        dirA = SL.centroid_direction(st.xt, dB, dA, name=f"{k}:towardA", which="embed")
        axes[k] = dict(dir=dirA, readA=rA, readB=rB, a=v["a"], b=v["b"],
                       nA=len(rA), nB=len(rB))
        print(f"[axis] {k:<12} dir from {len(dA)}+{len(dB)} genes, readout on {len(rA)}(A)+{len(rB)}(B) held out",
              flush=True)
    rand_dirs = [SL.random_direction(st.xt, seed=2000 + k) for k in range(N_RAND)]

    seqs = load_cells(st, n_cells, seed)
    print(f"[cells] {len(seqs)} setty CD34 cells, mean {np.mean([len(s) for s in seqs]):.0f} tokens\n", flush=True)

    def R(logits, read_pos, tA, tB):
        """differential readout R = mean_logit(A_targets) - mean_logit(B_targets), averaged over read positions."""
        a = np.mean([SL.mean_logit(logits[p], tA) for p in read_pos])
        b = np.mean([SL.mean_logit(logits[p], tB) for p in read_pos])
        return float(a - b)

    # acc[axis][alpha] = list over cells of dict(switch_steer, switch_rand, dRA, dRB)
    acc = {k: {a: [] for a in ALPHAS} for k in axes}
    res_path = os.path.join(HERE, "results", "lineage_steer.json")

    for ci, s in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], s, [st.tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        if len(gp) < 8:
            continue
        sh = rng.permutation(len(gp)); half = len(gp) // 2
        push_mask = np.zeros(len(ids), bool); push_mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        base = st.logits(ids)
        for k, ax in axes.items():
            tA, tB = ax["readA"], ax["readB"]
            R0 = R(base, read_pos, tA, tB)
            for a in ALPHAS:
                push = a * mean_norm

                def dR(direction, sign):
                    with st.steering(direction, alpha=sign * push, positions=push_mask, site="embed"):
                        lg = st.logits(ids)
                    return R(lg, read_pos, tA, tB) - R0

                dRA = dR(ax["dir"], +1)          # steer toward A
                dRB = dR(ax["dir"], -1)          # steer toward B
                switch = dRA - dRB               # bistable: >0
                sr = []
                for rd in rand_dirs:
                    sr.append(dR(rd, +1) - dR(rd, -1))
                acc[k][a].append(dict(switch=switch, rand=float(np.mean(sr)), dRA=dRA, dRB=dRB))
        if (ci + 1) % 8 == 0:
            print(f"  {ci + 1}/{len(seqs)} cells", flush=True)
            try:
                import torch as _t
                if _t.backends.mps.is_available():
                    _t.mps.empty_cache()          # guard the MPS memory buildup that wedged a prior run
            except Exception:
                pass

    # ---- aggregate
    out = dict(seed=seed, n_cells=len(seqs), alphas=ALPHAS, n_rand=N_RAND, mean_norm=mean_norm, per_axis={})
    print("\n=== per-axis bistable steering (switch = ΔR_towardA - ΔR_towardB; specific = switch - random) ===")
    print(f"  {'axis':<13} {'alpha':<6} {'switch':<9} {'random':<9} {'specific':<10} {'dRA(A up?)':<11} {'dRB(B up?)':<10} n")
    for k, ax in axes.items():
        out["per_axis"][k] = dict(a=ax["a"], b=ax["b"], nA=ax["nA"], nB=ax["nB"], by_alpha={})
        for a in ALPHAS:
            rows = acc[k][a]
            if not rows:
                continue
            sw = np.array([r["switch"] for r in rows]); rn = np.array([r["rand"] for r in rows])
            dra = np.array([r["dRA"] for r in rows]); drb = np.array([r["dRB"] for r in rows])
            spec = sw - rn
            rng2 = np.random.default_rng(seed)
            bs = np.array([spec[rng2.integers(0, len(spec), len(spec))].mean() for _ in range(4000)])
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            out["per_axis"][k]["by_alpha"][a] = dict(
                n=len(rows), switch=float(sw.mean()), random=float(rn.mean()), specific=float(spec.mean()),
                specific_ci=ci, dRA=float(dra.mean()), dRB=float(drb.mean()))
            star = "*" if ci[0] > 0 else " "
            print(f"  {k:<13} {a:<6} {sw.mean():<+9.4f} {rn.mean():<+9.4f} {spec.mean():<+10.4f}{star} "
                  f"{dra.mean():<+11.4f} {drb.mean():<+10.4f} {len(rows)}")
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        json.dump(out, open(res_path, "w"), indent=1)

    # overall: paired bootstrap over axes at the top alpha
    top = ALPHAS[-1]
    specs = [out["per_axis"][k]["by_alpha"].get(top, {}).get("specific") for k in axes]
    specs = np.array([x for x in specs if x is not None])
    if len(specs):
        rng3 = np.random.default_rng(seed)
        bs = np.array([specs[rng3.integers(0, len(specs), len(specs))].mean() for _ in range(5000)])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        pos = int((specs > 0).sum())
        print(f"\n  OVERALL @alpha={top}: mean specific over {len(specs)} axes = {specs.mean():+.4f}  "
              f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  ({pos}/{len(specs)} axes positive)")
        out["overall"] = dict(alpha=top, mean_specific=float(specs.mean()), ci=ci,
                              n_axes=len(specs), n_positive=pos)
        json.dump(out, open(res_path, "w"), indent=1)
    print(f"\n[done] -> {res_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    main(n, sd)
