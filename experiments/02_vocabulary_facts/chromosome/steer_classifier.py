"""steer_classifier — the SHARED MULTI-HEAD readout, and the SPECIFICITY test (Ihor, 2026-07-18).

WHY A SHARED CLASSIFIER (Ihor's design). One lightweight classifier on top of the model's pooled final cell
embedding, with MANY heads predicting MANY properties at once. Its job is NOT to re-measure what the native
logit readout already measures. Its job is **SPECIFICITY**:

    steering feature F should move F's readout AND LEAVE UNRELATED PROPERTIES ALONE,
    beyond what an equal-norm RANDOM push does.

A single-property probe cannot answer that; a shared multi-head one gets it for free. This is the second
readout in STEERING_TOOL.md's design; the non-circular anchor it is checked against is the native chr-mass
readout (steer_propagation.py: SPECIFIC +0.057, replicated over 2 seeds and an alpha-sweep).

NON-CIRCULARITY (the rule from steer_lib):
  * heads are trained ONLY on UNSTEERED embeddings -- the classifier never sees steered data;
  * steering is injected at the INPUT embedding, the classifier reads the FINAL hidden state, so the readout is
    separated from the intervention by all 11 transformer layers;
  * every head's response is reported against the RANDOM-push control on the SAME cells, so "the perturbation
    degraded everything" is subtracted off. What survives is specificity.

HEADS (fetal gut -- the section-6B dataset, genuinely fine-grained labels):
    cell_type        21 classes  -- fine biological identity
    cell_type_group   4 classes  -- coarse lineage (epithelium / immune / mesenchymal / vasculature)
    tissue            3 classes  -- colon / ileum / duodeno-jejunal junction
    cyc               2 classes  -- cycling vs not (marker-score median split; computed, no annotation needed)

THE PREDICTION. Chromosome is a *positional/organisational* variable, not a cell-identity variable. If the
model keeps it in a subspace largely separate from identity, chromosome steering should move the native chr
readout strongly while these identity heads move NO MORE than a random push of the same norm. If instead the
identity heads collapse under chr-steering specifically, chromosome is entangled with identity and the clean
"chromosome is its own computational variable" story weakens. Either result is informative.

Run:  ../../.venv_state/bin/python -u steer_classifier.py train  [n_cells]      (extract + fit heads; caches)
      ../../.venv_state/bin/python -u steer_classifier.py probe  [n_cells] [alpha]
Out:  results/steer_classifier_heads.json , results/steer_specificity.json
      data cache: ../../data/genemanifold/cellemb_fetalgut_<n>.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import steer_lib as SL
from genome_wide import coords, AUTOSOMES
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

NAME_ID_PKL = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "genemanifold")


def heads_path(model="217m"):
    """Heads are model-specific: they read a pooled hidden state whose WIDTH differs (217M 1232, 1B 2304),
    so a head trained on one model is meaningless on the other."""
    return os.path.join(CACHE_DIR, f"steer_heads_{model}.pkl")


def emb_cache(n_cells, model="217m"):
    return os.path.join(CACHE_DIR, f"cellemb_fetalgut_{model}_{n_cells}.npz")
MAX_LEN = 512
HEADS = ["cell_type", "cell_type_group", "tissue", "cyc"]
CYC_MARKERS = ["MKI67", "TOP2A", "CCNB1", "CDK1", "PCNA", "AURKB", "BIRC5", "TYMS", "RRM2", "UBE2C"]
SEED = 0


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def _cat(f, key):
    """Read an obs categorical column as a string array."""
    g = f["obs"][key]
    if isinstance(g, h5py.Group) and "categories" in g:
        cats = _dec(g["categories"][:]).astype(str)
        return cats[g["codes"][:]]
    return _dec(g[:]).astype(str)


def load_cells(n_cells, seed=SEED):
    """fetal_gut cells -> (token sequences, label dict). Same tokenisation path as hox_causal_locus.py."""
    tok = SL.MaxTokiTokenizer(model_input_size=MAX_LEN)
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))
    with h5py.File(G.FETAL_GUT, "r") as f:
        fn = f["var"]["feature_name"]
        syms = _dec(fn["categories"][:]).astype(str)[fn["codes"][:]] if isinstance(fn, h5py.Group) \
            else _dec(fn[:]).astype(str)
        labels_all = {h: _cat(f, h) for h in ("cell_type", "cell_type_group", "tissue")}
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(shape[0], min(n_cells, shape[0]), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            E[i, idx[s:e]] = data[s:e]
    labels = {h: labels_all[h][sel] for h in labels_all}

    # cycling score from markers (computed -> no annotation needed), median split
    up = np.char.upper(syms.astype(str))
    mk = np.array([i for i, s in enumerate(up) if s in set(CYC_MARKERS)])
    tot = E.sum(1); tot[tot == 0] = 1
    cyc_score = (E[:, mk].sum(1) / tot) if len(mk) else np.zeros(len(E))
    labels["cyc"] = np.where(cyc_score > np.median(cyc_score), "cycling", "quiescent")

    var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in up])
    seqs = []
    for i in range(len(E)):
        rs = E[i].sum() or 1.0
        en = np.log1p(E[i] / rs * 1e4)[var_idx]
        nz = en > 0
        nm = np.zeros_like(en); nm[nz] = en[nz] / medians[nz]
        kept = np.nonzero(nz)[0][np.argsort(-nm[nz])][: MAX_LEN - 2]
        seqs.append(token_ids[kept])
    keep = [i for i, s in enumerate(seqs) if len(s) >= 8]
    return [seqs[i] for i in keep], {h: labels[h][keep] for h in labels}, tok


def embed_cells(st, seqs, tok):
    """Pooled final hidden state per cell (mean over gene positions) -> (n, hidden)."""
    out = []
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        h = st.hidden(ids)
        out.append(SL.Steerer.pool(h, gp))
        if (i + 1) % 100 == 0:
            print(f"    embedded {i + 1}/{len(seqs)}", flush=True)
    return np.stack(out)


def fit_heads(Z, labels):
    """Fit one multinomial head per property on the pooled embedding; report 5-fold CV accuracy vs chance.
    A head is only usable as a readout if it is clearly above chance."""
    heads, report = {}, {}
    for h in HEADS:
        y = np.asarray(labels[h]).astype(str)
        vals, cnt = np.unique(y, return_counts=True)
        ok = set(vals[cnt >= 10])                      # need enough per class to CV
        m = np.array([v in ok for v in y])
        if m.sum() < 40 or len(ok) < 2:
            print(f"  [{h}] skipped (too few labelled cells/classes)"); continue
        Zh, yh = Z[m], y[m]
        sc = StandardScaler().fit(Zh)
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Zh, yh):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(Zh[tr]), yh[tr])
            accs.append(float((clf.predict(sc.transform(Zh[te])) == yh[te]).mean()))
        chance = float(np.max(np.unique(yh, return_counts=True)[1]) / len(yh))   # majority-class baseline
        full = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Zh), yh)
        heads[h] = dict(scaler=sc, clf=full, classes=list(full.classes_))
        report[h] = dict(cv_acc=float(np.mean(accs)), majority=chance, n=int(m.sum()),
                         n_classes=int(len(ok)), usable=bool(np.mean(accs) > chance + 0.02))
        print(f"  [{h}] n={m.sum():<5} classes={len(ok):<3} CV acc={np.mean(accs):.3f}  "
              f"majority={chance:.3f}  {'USABLE' if report[h]['usable'] else 'NOT usable'}")
    return heads, report


def head_probs(heads, Z, labels):
    """Per-head: full predicted distribution, mean p(true class), accuracy. Returns P for TV comparisons."""
    out = {}
    for h, H in heads.items():
        y = np.asarray(labels[h]).astype(str)
        m = np.array([v in set(H["classes"]) for v in y])
        if m.sum() == 0:
            continue
        P = H["clf"].predict_proba(H["scaler"].transform(Z[m]))
        cls = {c: i for i, c in enumerate(H["classes"])}
        p_true = np.array([P[i, cls[v]] for i, v in enumerate(y[m])])
        acc = float((H["clf"].predict(H["scaler"].transform(Z[m])) == y[m]).mean())
        out[h] = dict(P=P, p_true=float(p_true.mean()), acc=acc, n=int(m.sum()),
                      chance=float(1.0 / len(H["classes"])))
    return out


def tv(P, Q):
    """Mean total-variation distance between two sets of predicted distributions (same cells, same head).
    Unlike p_true this does not bottom out at a floor, so it stays informative when a push is strong."""
    n = min(len(P), len(Q))
    return float(0.5 * np.abs(P[:n] - Q[:n]).sum(1).mean())


# ---------------------------------------------------------------- modes
def train(n_cells=1000, model="217m"):
    st = SL.Steerer(model_dir=SL.MODELS[model])
    cache = emb_cache(n_cells, model)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        Z = z["Z"]; labels = {h: z[f"y_{h}"].astype(str) for h in HEADS if f"y_{h}" in z}
        print(f"[cache] {cache}  Z={Z.shape}")
    else:
        seqs, labels, tok = load_cells(n_cells)
        print(f"[cells] {len(seqs)} fetal-gut cells, mean {np.mean([len(s) for s in seqs]):.0f} tokens")
        Z = embed_cells(st, seqs, tok)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, Z=Z, **{f"y_{h}": labels[h] for h in labels})
        print(f"[cache] wrote {cache}  Z={Z.shape}")
    print("\nfitting shared multi-head classifier on the pooled final embedding:")
    heads, report = fit_heads(Z, labels)
    json.dump(report, open(os.path.join(HERE, "results", f"steer_classifier_heads_{model}.json"), "w"), indent=1)
    with open(heads_path(model), "wb") as f:
        pickle.dump(heads, f)
    print("\n[done] -> results/steer_classifier_heads.json  (+ heads pickled)")
    return heads, report


def probe(n_cells=24, alphas=(0.5, 1.0, 2.0, 4.0), model="217m"):
    """THE SPECIFICITY TEST. Steer chromosome; read every head; compare against a random push of equal norm.

    ALPHA MATTERS AND MUST BE SWEPT. At alpha=4 (4x the mean gene-embedding norm) a push in ANY direction --
    random included -- obliterates the pooled cell embedding: measured, cell_type_group p_true fell 0.999 ->
    0.05, BELOW its 0.25 chance level, under the RANDOM control. At that point both arms are on the floor and
    "excess ~ 0" is a saturation artifact, not specificity. So sweep alpha down to where the heads retain
    dynamic range, and report a FLOOR FLAG per head/alpha so a saturated cell is never read as evidence.

    Two metrics per head:
      d_p_true : change in mean p(true class)   -- interpretable, but floors at 0
      TV       : mean total-variation distance from the UNSTEERED prediction -- pure "how much did this head's
                 output move", no floor. SPECIFICITY = TV(chr-steer) - TV(random).
    """
    st = SL.Steerer(model_dir=SL.MODELS[model])
    with open(heads_path(model), "rb") as f:
        heads = pickle.load(f)
    print(f"[heads] model={model} (hidden={st.hidden_size}): {list(heads)}")

    # chromosome directions in INPUT space (train half of tokens), + held-out readout tokens (native anchor)
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
    dirs, read_idx = {}, {}
    for c in sorted(set(tchr)):
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        dirs[c] = SL.Direction(vec=EMB[tids[m]].mean(0) - gcen, name=f"chr:{c}", basis="embed_tokens")
        read_idx[c] = np.array(tids[(tchr == c) & (~is_tr)], dtype=np.int64)
    chroms = sorted(dirs)
    rand_dirs = [SL.random_direction(st.xt, seed=2000 + k) for k in range(3)]

    # SHAM directions -- the control the random one cannot be. A random vector in 1232-d is nearly orthogonal
    # to the model's whole embedding manifold, while a chromosome centroid direction lies INSIDE it by
    # construction; so an in-manifold direction perturbs the heads more than an out-of-manifold one for reasons
    # that have nothing to do with chromosome. Sham = the IDENTICAL construction (centroid of a token group
    # minus the global centroid) on RANDOM gene groupings with matched group sizes. Holds construction, norm
    # and in-manifold-ness fixed; varies only whether the grouping means "chromosome".
    sham_dirs = {}
    srng = np.random.default_rng(SEED + 31)
    tr_tok = tids[is_tr]
    for k, c in enumerate(sorted(dirs)):
        n_c = int(((tchr == c) & is_tr).sum())
        grp = srng.choice(tr_tok, size=n_c, replace=False)
        sham_dirs[f"sham{k}"] = SL.Direction(vec=EMB[grp].mean(0) - gcen, name=f"sham{k}", basis="embed_tokens",
                                             meta=dict(n=n_c))
    alphas = list(alphas)
    print(f"[setup] {len(chroms)} chromosome directions, alphas={alphas} x mean-norm({mean_norm:.2f}); "
          f"{n_cells} held-out cells\n")

    # held-out cells: use a different seed than train() so these cells are unseen by the heads
    seqs, labels, tok = load_cells(n_cells, seed=SEED + 500)

    ridx = {c: torch.as_tensor(read_idx[c]) for c in chroms}

    def both(ids, gp, read_pos):
        """One forward -> (pooled cell embedding for the heads, per-chromosome native mass at read positions)."""
        h, lg = st.forward_both(ids)
        p = torch.softmax(lg[read_pos], -1)
        return SL.Steerer.pool(h, gp), {c: float(p[:, ridx[c]].sum(-1).mean()) for c in chroms}

    base_Z = []
    steer_Z = {a: {c: [] for c in chroms} for a in alphas}
    sham_Z = {a: {k: [] for k in sham_dirs} for a in alphas}
    sham_native = {a: [] for a in alphas}
    rand_Z = {a: [] for a in alphas}
    native = {a: {c: {"steer": [], "rand": []} for c in chroms} for a in alphas}
    native_base = {c: [] for c in chroms}
    prng = np.random.default_rng(SEED + 9)
    for i, s in enumerate(seqs):
        ids = np.concatenate([[tok.BOS], s, [tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        push_pos, read_pos = gp[sh[:half]], gp[sh[half:]]
        pmask = np.zeros(len(ids), bool); pmask[push_pos] = True

        z, nb = both(ids, gp, read_pos)
        base_Z.append(z)
        for c in chroms:
            native_base[c].append(nb[c])

        for a in alphas:
            push = a * mean_norm
            rz, rn = [], {c: [] for c in chroms}
            for rd in rand_dirs:
                with st.steering(rd, alpha=push, positions=pmask, site="embed"):
                    z, nm = both(ids, gp, read_pos)
                rz.append(z)
                for c in chroms:
                    rn[c].append(nm[c])
            rand_Z[a].append(np.mean(rz, 0))
            for c in chroms:
                native[a][c]["rand"].append(float(np.mean(rn[c])))
            for c in chroms:
                with st.steering(dirs[c], alpha=push, positions=pmask, site="embed"):
                    z, nm = both(ids, gp, read_pos)
                steer_Z[a][c].append(z)
                native[a][c]["steer"].append(nm[c])
            # sham arm: same construction, random gene groupings. Its native chr-mass effect should be ~0
            # (no chromosome semantics), while its HEAD perturbation should match chromosome's if the head
            # movement is just "an in-manifold push".
            sh_nat = []
            for k, sd in sham_dirs.items():
                with st.steering(sd, alpha=push, positions=pmask, site="embed"):
                    z, nm = both(ids, gp, read_pos)
                sham_Z[a][k].append(z)
                sh_nat.append(float(np.mean([nm[c] for c in chroms])))
            sham_native[a].append(float(np.mean(sh_nat)))
        if (i + 1) % 4 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)

    base_Z = np.stack(base_Z)
    s_base = head_probs(heads, base_Z, labels)
    nb_all = float(np.mean([np.mean(native_base[c]) for c in chroms]))

    sweep = []
    for a in alphas:
        s_rand = head_probs(heads, np.stack(rand_Z[a]), labels)
        per_chr = {c: head_probs(heads, np.stack(steer_Z[a][c]), labels) for c in chroms}
        per_sham = {k: head_probs(heads, np.stack(sham_Z[a][k]), labels) for k in sham_dirs}
        ns = float(np.mean([np.mean(native[a][c]["steer"]) for c in chroms]))
        nr = float(np.mean([np.mean(native[a][c]["rand"]) for c in chroms]))
        nsh = float(np.mean(sham_native[a]))
        print(f"\n=== alpha={a}  (push = {a * mean_norm:.2f}) ===")
        print(f"  NATIVE anchor chr-mass: base {nb_all:.4f}  Δrand {nr - nb_all:+.5f}  "
              f"Δsham {nsh - nb_all:+.5f}  Δchr {ns - nb_all:+.5f}   "
              f"chr−sham = {ns - nsh:+.5f}  (sham has no chromosome semantics -> should be ~0)")
        print(f"  {'head':<17} {'base p':<9} {'TV rand':<9} {'TV sham':<9} {'TV chr':<9} "
              f"{'chr−sham':<10} {'chr−rand':<10} floor?")
        rows = {}
        for h in s_base:
            b = s_base[h]["p_true"]
            dr = s_rand[h]["p_true"] - b
            ds = float(np.mean([per_chr[c][h]["p_true"] for c in chroms])) - b
            tv_r = tv(s_base[h]["P"], s_rand[h]["P"])
            tv_s = float(np.mean([tv(s_base[h]["P"], per_chr[c][h]["P"]) for c in chroms]))
            tv_sh = float(np.mean([tv(s_base[h]["P"], per_sham[k][h]["P"]) for k in sham_dirs]))
            floor = bool((b + dr) <= s_base[h]["chance"] * 1.1)
            rows[h] = dict(base_p=b, d_p_rand=dr, d_p_steer=ds, tv_rand=tv_r, tv_sham=tv_sh, tv_steer=tv_s,
                           tv_excess_vs_sham=tv_s - tv_sh, tv_excess_vs_rand=tv_s - tv_r,
                           chance=s_base[h]["chance"], floored=floor)
            print(f"  {h:<17} {b:<9.4f} {tv_r:<9.4f} {tv_sh:<9.4f} {tv_s:<9.4f} {tv_s - tv_sh:<+10.4f} "
                  f"{tv_s - tv_r:<+10.4f} {'FLOORED' if floor else '-'}")
        sweep.append(dict(alpha=a, native=dict(base=nb_all, d_rand=nr - nb_all, d_sham=nsh - nb_all,
                                               d_steer=ns - nb_all, chr_minus_sham=ns - nsh,
                                               specific=(ns - nb_all) - (nr - nb_all)), heads=rows))

    # ---- data-driven verdict (never assert a conclusion the numbers do not support)
    usable = [s for s in sweep if not any(r["floored"] for r in s["heads"].values())]
    if usable:
        s = usable[-1]
        nat = s["native"]["chr_minus_sham"]
        ex = {h: r["tv_excess_vs_sham"] for h, r in s["heads"].items()}
        worst = max(ex, key=ex.get)
        print(f"\n  VERDICT at alpha={s['alpha']} (largest unfloored):")
        print(f"    native chr readout, chr−sham = {nat:+.5f}  -> the chromosome effect IS semantically "
              f"specific (a same-construction push with no chromosome meaning gives ~0).")
        print(f"    identity heads, chr−sham = " + ", ".join(f"{h} {v:+.3f}" for h, v in ex.items()))
        if max(ex.values()) > 0.02:
            print(f"    -> NOT orthogonal to identity: chromosome steering disturbs the identity heads MORE")
            print(f"       than a matched meaningless in-manifold push (worst: {worst} {ex[worst]:+.3f}).")
            print("       Consistent with chromosome being carried by co-regulated expression programs rather")
            print("       than an identity-independent coordinate. CAVEAT: sham uses RANDOM gene groupings, so")
            print("       it is in-manifold but semantically UNSTRUCTURED; a fairer next control is a")
            print("       MEANINGFUL non-chromosome grouping (expression decile / pathway / GO term).")
        else:
            print("    -> chromosome steering disturbs identity no more than a matched in-manifold push:")
            print("       the chromosome effect is SPECIFIC and identity-orthogonal.")
    out = dict(alphas=alphas, n_cells=len(seqs), n_chrom=len(chroms), sweep=sweep)
    json.dump(out, open(os.path.join(HERE, "results", f"steer_specificity_{model}.json"), "w"), indent=1)
    print("\n[done] -> results/steer_specificity.json")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    mdl = sys.argv[-1] if sys.argv[-1] in SL.MODELS else "217m"
    if mode == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1000, mdl)
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        al = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and "," in sys.argv[3] else (0.25, 0.5, 1.0, 2.0, 4.0)
        probe(n, al, mdl)
