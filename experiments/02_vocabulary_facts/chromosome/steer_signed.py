"""IS THE CHROMOSOME VARIABLE A SIGNED QUANTITY (dosage-like) OR A CATEGORY LABEL (identity-like)?

THE GAP THIS CLOSES. Every steering run in this project has used POSITIVE alpha only
(`steer_propagation.ALPHAS = [0, 1, 2, 4]`, and the loops literally filter `x > 0`). So we know that pushing a
cell's genes TOWARD chromosome C raises C-mass at untouched positions. We have never once pushed AWAY. That
matters because the deletion case -- the direction every clinical application actually needs -- lives at
negative alpha, and because the sign structure discriminates two accounts of what the variable IS:

  DOSAGE-LIKE (a signed quantity, "how much chromosome C"). Response should be roughly ANTISYMMETRIC:
  R(-a) ~ -R(+a). Pushing away suppresses C-mass about as much as pushing toward raises it.

  IDENTITY-LIKE (a categorical label, "is/is not on C"). "Not-C" is not a coherent direction -- it is 21 other
  categories -- so the negative push has no reason to produce a clean mirrored suppression. Expect a large
  SYMMETRIC component: both signs perturb C-mass in the same direction, or the negative side is flat/noisy.

WHY THIS AND NOT THE PROPOSED K562 DOSAGE REGRESSION. The ranked agenda's top experiment was to regress
steering response on true K562 copy number across the 22 autosomes. That design is PRE-REFUTED IN THIS REPO:
`cnv_mechanism_test.py` ran the per-chromosome regression and `cnv_gene_level_test.py` records the verdict --
"correlates 21 numbers against 21 numbers and is hopeless: every bootstrap CI spans roughly +-0.5. It cannot
support or refute anything." A properly powered version needs SEGMENT-level copy number (n ~ thousands), which
is not on disk; K562's copy number is segmental rather than per-chromosome, so no clean chromosome label exists
even in principle. This test needs no external ground truth at all and answers the prior question anyway: if the
representation is not even a signed quantity, no dosage regression can succeed and the CNV family closes.

Protocol is deliberately IDENTICAL to steer_propagation.run() -- same split-half push/read, same train/test
token split, same centroid directions, same norm-matched random control -- so numbers are comparable to the
established +0.170 (1B) result. The only change is that alpha ranges over both signs.

Run: ../../.venv_state/bin/python -u steer_signed.py [n_cells] [seed] [model]
Out: results/steer_signed_<model>_seed<seed>.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import steer_lib as SL
from steer_propagation import SPECS, load_cells, MIN_CAT_TOKENS, N_RAND

ALPHAS = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
if os.environ.get("STEER_ALPHAS"):                 # e.g. STEER_ALPHAS="-0.25,-0.1,0.05,0.1,0.25"
    ALPHAS = [float(x) for x in os.environ["STEER_ALPHAS"].split(",")]


def main(n_cells=32, seed=0, model="1b"):
    torch.manual_seed(seed)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    tok2cat, pname = SPECS["chromosome"](st)
    tids = np.array(sorted(tok2cat)); tcat = np.array([tok2cat[t] for t in tids])

    rng = np.random.default_rng(seed)
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    cats, dirs, read_idx = [], {}, {}
    for c in sorted(set(tcat)):
        m = (tcat == c) & is_tr
        if m.sum() < MIN_CAT_TOKENS:
            continue
        dirs[c] = SL.Direction(vec=EMB[tids[m]].mean(0) - gcen, name=f"{pname}:{c}", basis="embed_tokens")
        read_idx[c] = np.array([t for t in tids[(tcat == c) & (~is_tr)]], dtype=np.int64)
        cats.append(c)
    rand_dirs = [SL.random_direction(st.xt, seed=1000 + k) for k in range(N_RAND)]
    print(f"[signed sweep] model={model} hidden={st.hidden_size} | {len(cats)} chromosomes | "
          f"alphas={ALPHAS} x mean-norm({mean_norm:.2f}) | seed={seed}", flush=True)

    seqs = load_cells(st, n_cells, seed)
    print(f"[cells] {len(seqs)} cells, mean {np.mean([len(s) for s in seqs]):.0f} tokens\n", flush=True)

    def cat_mass(lr, c):
        p = torch.softmax(lr, dim=-1)
        return float(p[:, torch.as_tensor(read_idx[c], dtype=torch.long)].sum(-1).mean())

    acc = {c: {a: {"steer": [], "rand": []} for a in ALPHAS} for c in cats}
    prng = np.random.default_rng(seed + 7)
    for ci, s in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], s, [st.tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        push_pos, read_pos = gp[sh[:half]], gp[sh[half:]]
        push_mask = np.zeros(len(ids), bool); push_mask[push_pos] = True
        base = st.logits(ids)[read_pos]
        base_mass = {c: cat_mass(base, c) for c in cats}
        for a in ALPHAS:
            push = a * mean_norm
            rm = {c: [] for c in cats}
            for rd in rand_dirs:
                with st.steering(rd, alpha=push, positions=push_mask, site="embed"):
                    lr = st.logits(ids)[read_pos]
                for c in cats:
                    rm[c].append(cat_mass(lr, c))
            for c in cats:
                acc[c][a]["rand"].append(np.mean(rm[c]) - base_mass[c])
            for c in cats:
                with st.steering(dirs[c], alpha=push, positions=push_mask, site="embed"):
                    lr = st.logits(ids)[read_pos]
                acc[c][a]["steer"].append(cat_mass(lr, c) - base_mass[c])
        if (ci + 1) % 4 == 0:
            print(f"  {ci + 1}/{len(seqs)} cells", flush=True)

    curve, bs = {}, np.random.default_rng(0)
    for a in ALPHAS:
        per = np.array([np.mean(acc[c][a]["steer"]) - np.mean(acc[c][a]["rand"]) for c in cats])
        b = [per[bs.integers(0, len(per), len(per))].mean() for _ in range(2000)]
        curve[a] = {"specific": float(per.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    "n_pos": int((per > 0).sum()), "n_cat": len(per)}
        print(f"  alpha {a:+.3g}: specific {per.mean():+.5f}  CI[{curve[a]['ci'][0]:+.5f},{curve[a]['ci'][1]:+.5f}]"
              f"  {curve[a]['n_pos']}/{len(per)} chr positive", flush=True)

    # decompose into symmetric and antisymmetric parts at each |alpha|
    decomp = {}
    for a in [x for x in ALPHAS if x > 0]:
        rp, rn = curve[a]["specific"], curve[-a]["specific"]
        anti, sym = (rp - rn) / 2, (rp + rn) / 2
        decomp[str(a)] = {"R_plus": rp, "R_minus": rn, "antisymmetric": anti, "symmetric": sym,
                          "anti_fraction": float(abs(anti) / (abs(anti) + abs(sym) + 1e-12))}
        print(f"  |alpha|={a}: R(+)={rp:+.5f} R(-)={rn:+.5f} -> anti {anti:+.5f}, sym {sym:+.5f}, "
              f"anti-fraction {decomp[str(a)]['anti_fraction']:.2f}", flush=True)

    out = dict(model=model, seed=seed, n_cells=len(seqs), alphas=ALPHAS, mean_norm=mean_norm,
               curve={str(k): v for k, v in curve.items()}, decomposition=decomp,
               per_chrom={c: {str(a): {"steer": float(np.mean(acc[c][a]["steer"])),
                                       "rand": float(np.mean(acc[c][a]["rand"]))} for a in ALPHAS} for c in cats})
    p = os.path.join(HERE, "results", f"steer_signed_{model}_seed{seed}.json")
    json.dump(out, open(p, "w"), indent=1)

    af = float(np.mean([d["anti_fraction"] for d in decomp.values()]))
    negs = [a for a in ALPHAS if a < 0]
    neg_sig = bool(negs) and all(curve[a]["ci"][1] < 0 for a in negs)
    print("\n=== VERDICT ===")
    print(f"  mean antisymmetric fraction: {af:.2f}")
    print(f"  negative-alpha effect significantly NEGATIVE at |a|>=1: {neg_sig}")
    if af > 0.7 and neg_sig:
        print("  -> SIGNED QUANTITY (dosage-like). Pushing away suppresses as pushing toward raises.")
        print("     A segment-level copy-number regression becomes worth building.")
    elif af < 0.4:
        print("  -> CATEGORY LABEL (identity-like): response is largely symmetric in the sign of the push.")
        print("     No dosage representation -> the CNV/clinical application family closes.")
    else:
        print("  -> INTERMEDIATE: partially signed. Report the curve; do not claim dosage.")
    print(f"\n[done] -> {p}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0,
         sys.argv[3] if len(sys.argv) > 3 else "1b")
