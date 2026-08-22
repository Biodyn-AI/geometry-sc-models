"""DEPTH PROFILE, DONE PROPERLY: a chromosome direction derived NATIVELY AT EACH LAYER (Ihor, 2026-07-20).

WHY steer_layers.py IS NOT ENOUGH. That script injects the EMBED-SPACE chromosome direction
(centroid of chr-C's embed_tokens rows minus the global centroid) at every depth, and finds the effect only at
the embedding site (+0.145 vs +0.0007 at layer 0, ~0 beyond -- and that is WITH the push rescaled to the local
residual norm, which was itself a necessary fix). But that result is AMBIGUOUS between two very different
readings:
    (a) the model consumes the chromosome variable at the input and later layers have moved on, or
    (b) the embed-space direction is simply not the direction along which chromosome lives deeper in the stack.
A transformer is free to rotate a feature into a different subspace at every layer, so (b) is entirely likely
and it would produce exactly the same curve.

THE FIX. Derive the chromosome direction AT EACH SITE from the model's own representations there:
  * run N cells, capture the residual stream at each site, and accumulate a per-GENE mean vector at that site
    (a gene's layer-L representation is contextual, so this is its average over the cells it appears in --
    the same construction gm_lib.build_ctx uses for scGPT);
  * chromosome direction at site s = centroid(chr-C's TRAIN genes at s) - centroid(all genes at s);
  * steer at site s with THAT site's direction, push scaled to the local residual norm.
Now every site is asked the same question in its own basis, and a decline with depth means the variable really
is consumed early rather than merely rotated away from the embedding basis.

Retains the built-in control: injecting at the LAST layer must give ~0 at the read positions, because the
readout is at the unsteered half and no attention remains to carry it there.

Run: ../../.venv_state/bin/python -u steer_layers2.py [n_dir_cells] [n_test_cells] [alpha] [model]
Out: results/steer_layers2_<model>.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import steer_lib as SL
import steer_classifier as SC
from genome_wide import coords, AUTOSOMES

SEED = 0
N_RAND = 2
MIN_OBS = 5          # a gene needs this many observations for a stable per-site mean vector


def main(n_dir_cells=150, n_test_cells=10, alpha=0.5, model="1b"):
    rng = np.random.default_rng(SEED)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    vocab, hid = EMB.shape
    nL = st.n_layers
    sites = ["embed"] + sorted(set([0, nL // 4, nL // 2, (3 * nL) // 4, nL - 1]))

    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok2chr = {int(t): str(C.loc[s, "chromosome"]) for ens, t in tokmap.items()
               if (s := ens2sym.get(ens)) in C.index and C.loc[s, "chromosome"] in AUTOSOMES and int(t) < vocab}
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    row = {int(t): i for i, t in enumerate(tids)}
    is_tr = rng.random(len(tids)) < 0.5

    # ---- pass 1: per-gene mean representation at every site
    print(f"[pass 1] building per-gene representations at {len(sites)} sites from {n_dir_cells} cells "
          f"({len(tids)} autosomal genes, hidden={hid})", flush=True)
    seqs_d, _, tok = SC.load_cells(n_dir_cells, seed=SEED + 1)
    S = {s: np.zeros((len(tids), hid), np.float32) for s in sites}
    Ncount = np.zeros(len(tids), np.int64)
    norms = {s: [] for s in sites}
    with torch.no_grad():
        for i, sq in enumerate(seqs_d):
            ids = np.concatenate([[tok.BOS], sq, [tok.EOS]]).astype(np.int64)
            gp = np.arange(1, 1 + len(sq))
            out = st.model(input_ids=torch.as_tensor(ids, dtype=torch.long,
                                                     device=st.device).reshape(1, -1),
                           output_hidden_states=True)
            hs = out.hidden_states
            keep = [(k, row[int(t)]) for k, t in zip(gp, ids[gp]) if int(t) in row]
            if not keep:
                continue
            pos = np.array([k for k, _ in keep]); rws = np.array([r for _, r in keep])
            for s in sites:
                h = (hs[0] if s == "embed" else hs[int(s) + 1])[0].detach().cpu().float().numpy()
                S[s][rws] += h[pos]          # genes are unique within a cell -> no duplicate rows
                norms[s].append(float(np.linalg.norm(h[pos], axis=-1).mean()))
            Ncount[rws] += 1
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(seqs_d)}", flush=True)
    ok = Ncount >= MIN_OBS
    print(f"[pass 1] {int(ok.sum())}/{len(tids)} genes with >= {MIN_OBS} observations")
    site_norm = {s: float(np.mean(v)) for s, v in norms.items()}
    print("[norms] " + "  ".join(f"{s}={site_norm[s]:.1f}" for s in sites) + "\n")

    # ---- site-native chromosome directions
    dirs = {s: {} for s in sites}
    read_idx, chroms = {}, []
    for s in sites:
        M = S[s] / np.maximum(Ncount, 1)[:, None]
        gc = M[ok & is_tr].mean(0)
        for c in sorted(set(tchr)):
            m = ok & is_tr & (tchr == c)
            if m.sum() < 20:
                continue
            dirs[s][c] = SL.Direction(vec=(M[m].mean(0) - gc).astype(np.float64),
                                      name=f"{s}:chr{c}", basis=f"resid@{s}")
            if s == sites[0]:
                read_idx[c] = np.array(tids[ok & (~is_tr) & (tchr == c)], dtype=np.int64)
                chroms.append(c)
    chroms = [c for c in chroms if all(c in dirs[s] for s in sites)]
    print(f"[dirs] {len(chroms)} chromosomes have a direction at every site")
    rand_dirs = [SL.random_direction(st.xt, seed=4000 + k) for k in range(N_RAND)]

    # ---- pass 2: steer at each site with that site's own direction
    seqs, labels, tok = SC.load_cells(n_test_cells, seed=SEED + 500)
    ridx = {c: torch.as_tensor(read_idx[c]) for c in chroms}
    acc = {s: {c: {"steer": [], "rand": []} for c in chroms} for s in sites}
    prng = np.random.default_rng(SEED + 9)
    print(f"[pass 2] steering {len(seqs)} cells at each site (alpha={alpha} x local residual norm)\n")
    for i, sq in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], sq, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(sq))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        p0 = torch.softmax(st.logits(ids)[read_pos], -1)
        bm = {c: float(p0[:, ridx[c]].sum(-1).mean()) for c in chroms}
        for s in sites:
            push = alpha * site_norm[s]
            rm = {c: [] for c in chroms}
            for rd in rand_dirs:
                with st.steering(rd, alpha=push, positions=mask, site=s):
                    pr = torch.softmax(st.logits(ids)[read_pos], -1)
                for c in chroms:
                    rm[c].append(float(pr[:, ridx[c]].sum(-1).mean()))
            for c in chroms:
                acc[s][c]["rand"].append(np.mean(rm[c]) - bm[c])
                with st.steering(dirs[s][c], alpha=push, positions=mask, site=s):
                    ps = torch.softmax(st.logits(ids)[read_pos], -1)
                acc[s][c]["steer"].append(float(ps[:, ridx[c]].sum(-1).mean()) - bm[c])
        if (i + 1) % 2 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    print(f"\n=== DEPTH PROFILE with SITE-NATIVE directions ===")
    print(f"  {'site':<8} {'SPECIFIC':>11} {'95% CI':>24} {'chr +':>8}")
    prof = []
    for s in sites:
        sp = np.array([np.mean(acc[s][c]["steer"]) - np.mean(acc[s][c]["rand"]) for c in chroms])
        bs = np.array([sp[rng.integers(0, len(sp), len(sp))].mean() for _ in range(5000)])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        prof.append(dict(site=str(s), specific=float(sp.mean()), ci=ci, n_pos=int((sp > 0).sum()), n=len(sp)))
        print(f"  {str(s):<8} {sp.mean():>11.5f} [{ci[0]:>+10.5f},{ci[1]:>+10.5f}] {int((sp>0).sum()):>4}/{len(sp)}")

    peak = max(prof, key=lambda p: p["specific"])
    last = prof[-1]
    ctrl = abs(last["specific"]) < 0.2 * peak["specific"]
    print(f"\n  peak site: {peak['site']} ({peak['specific']:+.5f})")
    print(f"  last-layer control: {last['specific']:+.5f} -> {'PASSES' if ctrl else 'FAILS'}")
    deep = [p for p in prof if p["site"] not in ("embed", "0")]
    any_deep = any(p["ci"][0] > 0 for p in deep)
    print(f"  any DEEP site with a significant effect in its own basis? {'YES' if any_deep else 'NO'}")
    print("  -> " + ("the variable is usable at depth too (it was a BASIS problem before, not a depth problem)"
                     if any_deep else
                     "even in its OWN basis the variable is only usable at/near the input: the model reads "
                     "chromosome off the token embedding early and does not maintain a steerable chromosome "
                     "axis deeper in the stack"))

    json.dump(dict(model=model, alpha=alpha, sites=[str(s) for s in sites], n_dir_cells=len(seqs_d),
                   n_test_cells=len(seqs), profile=prof, peak=peak["site"], last_layer_ok=bool(ctrl),
                   any_deep_significant=bool(any_deep), site_norm=site_norm),
              open(os.path.join(HERE, "results", f"steer_layers2_{model}.json"), "w"), indent=1)
    print(f"\n[done] -> results/steer_layers2_{model}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150,
         int(sys.argv[2]) if len(sys.argv) > 2 else 10,
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
         sys.argv[4] if len(sys.argv) > 4 else "1b")
