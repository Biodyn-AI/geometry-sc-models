"""WHERE do the classifier's predictions GO when you push the chromosome variable? (Ihor, 2026-07-18)

steer_classifier.py measured HOW FAR each head's predicted distribution moves (total variation). Ihor's
question: that says a distance, not a destination. What LITERALLY happens to the predictions?

Three qualitatively different things could produce the same TV, and they mean opposite things:
  (1) DIFFUSE DEGRADATION -- the distribution just flattens; the model becomes uncertain. Entropy rises, the
      argmax often doesn't even change, and where it does change it's arbitrary. => steering is damage/noise.
  (2) GENERIC RELABELLING -- cells move confidently to ONE sink class, the same one no matter which chromosome
      you push. => steering shoves the embedding to a generic corner; still not chromosome-semantic.
  (3) CHROMOSOME-SPECIFIC RELABELLING -- the destination DEPENDS ON WHICH CHROMOSOME you steer toward.
      => the chromosome variable is genuinely wired into the identity representation in a structured way.

Only (3) makes "chromosome rides on co-regulated expression programs" a positive, interpretable claim rather
than a hedge. This script distinguishes them.

MEASURED, per head, at one alpha, against the SHAM control (the fair one -- see STEERING_TOOL.md):
  * entropy of the predicted distribution: base vs sham vs chr        -> (1) diffuse?
  * fraction of cells whose ARGMAX class changes                      -> is anything actually relabelled?
  * confidence of the NEW label when it changes                       -> confident move vs mush
  * destination distribution, and CRUCIALLY the destination BROKEN DOWN BY CHROMOSOME
      - if every chromosome sends cells to the same modal class -> (2)
      - if destinations differ by chromosome -> (3).  Quantified by "destination agreement": the mean pairwise
        overlap of the destination distributions of different chromosomes (1.0 = identical sink, 0 = disjoint).

Run: ../../.venv_state/bin/python -u steer_where.py [n_cells] [alpha] [head]
Out: results/steer_where.json
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


def entropy(P):
    return float((-(P * np.log(P + 1e-12)).sum(1)).mean())


def main(n_cells=24, alpha=0.5, focus="cell_type", model="217m"):
    st = SL.Steerer(model_dir=SL.MODELS[model])
    with open(SC.heads_path(model), "rb") as f:
        heads = pickle.load(f)

    # --- chromosome + sham directions, exactly as steer_classifier builds them
    EMB = SL._embed_matrix(st.xt, "embed")
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    vocab = EMB.shape[0]
    tok2chr = {int(t): str(C.loc[s, "chromosome"]) for ens, t in tokmap.items()
               if (s := ens2sym.get(ens)) in C.index and C.loc[s, "chromosome"] in AUTOSOMES and int(t) < vocab}
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    rng = np.random.default_rng(SEED)
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    dirs = {}
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() >= 20:
            dirs[c] = SL.Direction(vec=EMB[tids[m]].mean(0) - gcen, name=f"chr:{c}", basis="embed_tokens")
    chroms = sorted(dirs)
    srng = np.random.default_rng(SEED + 31)
    tr_tok = tids[is_tr]
    shams = {}
    for k, c in enumerate(chroms):
        n_c = int(((tchr == c) & is_tr).sum())
        grp = srng.choice(tr_tok, size=n_c, replace=False)
        shams[f"sham{k}"] = SL.Direction(vec=EMB[grp].mean(0) - gcen, name=f"sham{k}", basis="embed_tokens")

    push = alpha * mean_norm
    seqs, labels, tok = SC.load_cells(n_cells, seed=SEED + 500)
    print(f"[setup] head='{focus}', alpha={alpha} (push {push:.2f}), {len(seqs)} cells, "
          f"{len(chroms)} chromosomes\n")

    H = heads[focus]
    classes = np.array(H["classes"])
    y = np.asarray(labels[focus]).astype(str)
    keep = np.array([v in set(classes) for v in y])

    def predict(Z):
        return H["clf"].predict_proba(H["scaler"].transform(Z))

    # --- collect pooled embeddings for base / each chromosome / each sham
    base_Z, chr_Z, sham_Z = [], {c: [] for c in chroms}, {k: [] for k in shams}
    prng = np.random.default_rng(SEED + 9)
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        pmask = np.zeros(len(ids), bool); pmask[gp[sh[:half]]] = True
        base_Z.append(SL.Steerer.pool(st.hidden(ids), gp))
        for c in chroms:
            with st.steering(dirs[c], alpha=push, positions=pmask, site="embed"):
                chr_Z[c].append(SL.Steerer.pool(st.hidden(ids), gp))
        for k, sd in shams.items():
            with st.steering(sd, alpha=push, positions=pmask, site="embed"):
                sham_Z[k].append(SL.Steerer.pool(st.hidden(ids), gp))
        if (i + 1) % 4 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    P_base = predict(np.stack(base_Z))[keep]
    P_chr = {c: predict(np.stack(chr_Z[c]))[keep] for c in chroms}
    P_sham = {k: predict(np.stack(sham_Z[k]))[keep] for k in shams}
    a_base = P_base.argmax(1)

    def summarize(Pd, tag):
        ent = float(np.mean([entropy(P) for P in Pd.values()]))
        changed, conf_new, dests = [], [], []
        for P in Pd.values():
            a = P.argmax(1)
            ch = a != a_base
            changed.append(float(ch.mean()))
            if ch.any():
                conf_new.append(float(P[np.arange(len(a)), a][ch].mean()))
            dests.append(np.bincount(a, minlength=len(classes)) / len(a))
        return dict(entropy=ent, frac_changed=float(np.mean(changed)),
                    conf_new_label=float(np.mean(conf_new)) if conf_new else float("nan"),
                    dest=np.stack(dests))

    base_ent = entropy(P_base)
    S_chr = summarize(P_chr, "chr")
    S_sham = summarize(P_sham, "sham")

    print(f"\n=== WHAT HAPPENS TO '{focus}' PREDICTIONS (n={int(keep.sum())} cells) ===")
    print(f"  {'':<26} {'base':<10} {'sham':<10} {'chromosome'}")
    print(f"  {'entropy of prediction':<26} {base_ent:<10.3f} {S_sham['entropy']:<10.3f} {S_chr['entropy']:.3f}")
    print(f"  {'frac cells relabelled':<26} {'-':<10} {S_sham['frac_changed']:<10.3f} "
          f"{S_chr['frac_changed']:.3f}")
    print(f"  {'confidence in NEW label':<26} {'-':<10} {S_sham['conf_new_label']:<10.3f} "
          f"{S_chr['conf_new_label']:.3f}")

    # --- (2) vs (3): does the DESTINATION depend on which chromosome you push?
    def agreement(D):
        """mean pairwise overlap (sum of elementwise min) of destination distributions."""
        n = len(D); v = []
        for i in range(n):
            for j in range(i + 1, n):
                v.append(float(np.minimum(D[i], D[j]).sum()))
        return float(np.mean(v)) if v else float("nan")

    ag_chr, ag_sham = agreement(S_chr["dest"]), agreement(S_sham["dest"])
    print(f"\n  destination agreement across directions (1.0 = same sink for all, 0 = disjoint):")
    print(f"    chromosomes: {ag_chr:.3f}     shams: {ag_sham:.3f}")

    print(f"\n  modal destination per chromosome (top class its cells land in):")
    for i, c in enumerate(chroms):
        d = S_chr["dest"][i]
        top = np.argsort(-d)[:2]
        print(f"    chr{c:<3} -> {classes[top[0]]:<28} {d[top[0]]:.2f}   (2nd {classes[top[1]]} {d[top[1]]:.2f})")
    base_dist = np.bincount(a_base, minlength=len(classes)) / len(a_base)
    tb = np.argsort(-base_dist)[:3]
    print(f"\n  for reference, UNSTEERED distribution: " +
          ", ".join(f"{classes[t]} {base_dist[t]:.2f}" for t in tb))

    out = dict(head=focus, alpha=alpha, n_cells=int(keep.sum()), n_chrom=len(chroms),
               base_entropy=base_ent,
               chr=dict(entropy=S_chr["entropy"], frac_changed=S_chr["frac_changed"],
                        conf_new=S_chr["conf_new_label"], agreement=ag_chr),
               sham=dict(entropy=S_sham["entropy"], frac_changed=S_sham["frac_changed"],
                         conf_new=S_sham["conf_new_label"], agreement=ag_sham),
               modal_destination={c: str(classes[S_chr["dest"][i].argmax()]) for i, c in enumerate(chroms)},
               classes=list(map(str, classes)),
               # FULL destination distributions (chrom x class) -- needed to test the destination against
               # real-data expression enrichment (steer_mechanism.py). Modal class alone throws this away.
               chroms=list(map(str, chroms)),
               dest_chr=S_chr["dest"].tolist(), dest_sham=S_sham["dest"].tolist(),
               base_dist=(np.bincount(a_base, minlength=len(classes)) / len(a_base)).tolist())
    json.dump(out, open(os.path.join(HERE, "results", f"steer_where_{model}.json"), "w"), indent=1)
    print("\n[done] -> results/steer_where.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5,
         sys.argv[3] if len(sys.argv) > 3 else "cell_type",
         sys.argv[4] if len(sys.argv) > 4 else "217m")
