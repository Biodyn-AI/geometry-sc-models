"""steer_propagation — the split-half causal-propagation test, for ANY categorical gene property (Ihor).

Generalises `genome_causal.py` (chromosome-specific, single alpha, one seed) into a reusable experiment on top
of `steer_lib`, and closes the two controls that script left owed: an ALPHA-SWEEP (dose-response) and a SECOND
SEED. Any property that partitions genes into categories and gives a per-category token set can be dropped in
via a PropertySpec — chromosome today, cell-cycle phase / pathway / tissue program next.

THE DESIGN (why it is not circular — see steer_lib's validity rule):
  * Build each category's steering direction d_c in INPUT (embed_tokens) space, from a TRAIN half of that
    category's tokens.
  * In each cell, split the gene positions into a PUSH half and a READ half. Add alpha*d_c to the PUSH
    positions only.
  * Read the model's softmax mass on category-c tokens at the READ positions -- and only on the HELD-OUT
    (test-half) tokens. For that mass to rise, category identity must PROPAGATE PUSH->READ THROUGH ATTENTION;
    it cannot be input->output table pass-through (different positions, different tokens, direction in embed
    space, readout in lm_head space through all layers).
  * CONTROL = norm-matched RANDOM input pushes (kills generic input->output energy leakage).
  * The statistic is SPECIFIC = (mass increase under steer-toward-c) - (mass increase under random push),
    averaged over categories, with a paired bootstrap CI over categories. Dose-response: SPECIFIC should climb
    with alpha, random should stay flat.

Run: ../../.venv_state/bin/python -u steer_propagation.py [property] [n_cells] [seed]
     property in {chromosome} (more to come);  defaults: chromosome 48 0
Out: results/steer_propagation_<property>_seed<seed>.json
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

SETTY = (f"{_DATA}/"
         "hematopoiesis/setty19_cd34_bm.h5ad")
NAME_ID_PKL = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
MAX_LEN = 512
ALPHAS = [0.0, 1.0, 2.0, 4.0]      # in units of the mean gene-embedding norm (dose-response)
N_RAND = 3
MIN_CAT_TOKENS = 20                # a category needs >= this many TRAIN tokens to build a direction


# ---------------------------------------------------------------- property specs
def chromosome_spec(st):
    """Categories = autosomes. token->chromosome from species_chrom.csv (the canonical coordinate table)."""
    C = coords()
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    vocab = st.model.lm_head.weight.shape[0]
    tok2cat = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        tid = int(tid)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and tid < vocab:
            tok2cat[tid] = str(C.loc[s, "chromosome"])
    return tok2cat, "chromosome"


SPECS = {"chromosome": chromosome_spec}


# ---------------------------------------------------------------- cells
def load_cells(st, n_cells, seed):
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x
                       for x in f["var"]["index"][:]]).astype(str)
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(shape[0], n_cells, replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            E[i, idx[s:e]] = data[s:e]
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))
    var_idx, token_ids, medians = st.tok.make_var_mapping([name_id.get(s) for s in gn])
    seqs = []
    for i in range(len(E)):
        rs = E[i].sum() or 1.0
        en = np.log1p(E[i] / rs * 1e4)[var_idx]
        nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        order = np.argsort(-norm[nz])
        kept = np.nonzero(nz)[0][order][: MAX_LEN - 2]
        seqs.append(token_ids[kept])
    return [s for s in seqs if len(s) >= 8]


# ---------------------------------------------------------------- experiment
def run(prop="chromosome", n_cells=48, seed=0, model="217m"):
    torch.manual_seed(seed)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")            # (vocab, hidden) INPUT table
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    tok2cat, pname = SPECS[prop](st)
    tids = np.array(sorted(tok2cat)); tcat = np.array([tok2cat[t] for t in tids])

    # train/test token split: directions from TRAIN, readout on TEST only
    rng = np.random.default_rng(seed)
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    cats, dirs, read_idx = [], {}, {}
    for c in sorted(set(tcat)):
        m = (tcat == c) & is_tr
        if m.sum() < MIN_CAT_TOKENS:
            continue
        v = EMB[tids[m]].mean(0) - gcen                # centroid(cat-c TRAIN) - global, INPUT space
        dirs[c] = SL.Direction(vec=v, name=f"{pname}:{c}", basis="embed_tokens")
        read_idx[c] = np.array([t for t in tids[(tcat == c) & (~is_tr)]], dtype=np.int64)
        cats.append(c)
    rand_dirs = [SL.random_direction(st.xt, seed=1000 + k) for k in range(N_RAND)]
    print(f"[{pname}] model={model} (hidden={st.hidden_size}, layers={st.n_layers}); {len(cats)} categories; "
          f"alphas={ALPHAS} x mean-norm({mean_norm:.2f}); seed={seed}", flush=True)

    seqs = load_cells(st, n_cells, seed)
    print(f"[cells] {len(seqs)} cells (>=8 genes), mean {np.mean([len(s) for s in seqs]):.0f} tokens\n",
          flush=True)

    def cat_mass(logits_read, c):
        """softmax mass on cat-c held-out tokens, summed per read position, meaned over positions."""
        p = torch.softmax(logits_read, dim=-1)
        idx = torch.as_tensor(read_idx[c], dtype=torch.long)
        return float(p[:, idx].sum(-1).mean())

    # accumulate mass increases (relative to alpha=0 base) per category, per alpha
    acc = {c: {a: {"steer": [], "rand": []} for a in ALPHAS if a > 0} for c in cats}
    prng = np.random.default_rng(seed + 7)

    for ci, s in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], s, [st.tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))                  # positions holding a gene token
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        push_pos = gp[sh[:half]]; read_pos = gp[sh[half:]]
        push_mask = np.zeros(len(ids), bool); push_mask[push_pos] = True

        base = st.logits(ids)[read_pos]                # (Pread, vocab), no steering
        base_mass = {c: cat_mass(base, c) for c in cats}

        # random pushes: shared across categories (compute once per alpha)
        for a in [x for x in ALPHAS if x > 0]:
            push = a * mean_norm
            rand_masses = {c: [] for c in cats}
            for rd in rand_dirs:
                with st.steering(rd, alpha=push, positions=push_mask, site="embed"):
                    lr = st.logits(ids)[read_pos]
                for c in cats:
                    rand_masses[c].append(cat_mass(lr, c))
            for c in cats:
                acc[c][a]["rand"].append(np.mean(rand_masses[c]) - base_mass[c])
            # steer toward each category
            for c in cats:
                with st.steering(dirs[c], alpha=push, positions=push_mask, site="embed"):
                    lr = st.logits(ids)[read_pos]
                acc[c][a]["steer"].append(cat_mass(lr, c) - base_mass[c])
        if (ci + 1) % 8 == 0:
            print(f"  {ci + 1}/{len(seqs)} cells", flush=True)

    # aggregate: per alpha, SPECIFIC = mean_c(steer_c - rand_c), paired bootstrap CI over categories
    sweep = []
    for a in [x for x in ALPHAS if x > 0]:
        per_cat = []
        for c in cats:
            ds = float(np.mean(acc[c][a]["steer"]))
            dr = float(np.mean(acc[c][a]["rand"]))
            per_cat.append(dict(cat=c, d_steer=ds, d_rand=dr, specific=ds - dr))
        sp = np.array([x["specific"] for x in per_cat])
        bs = np.array([sp[rng.integers(0, len(sp), len(sp))].mean() for _ in range(5000)])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        n_pos = int((sp > 0).sum())
        sweep.append(dict(alpha=a, mean_steer=float(np.mean([x["d_steer"] for x in per_cat])),
                          mean_rand=float(np.mean([x["d_rand"] for x in per_cat])),
                          mean_specific=float(sp.mean()), ci=ci, n_cat_positive=n_pos, n_cat=len(sp),
                          per_cat=per_cat))
        print(f"  alpha={a}: steer {np.mean([x['d_steer'] for x in per_cat]):+.5f}  "
              f"rand {np.mean([x['d_rand'] for x in per_cat]):+.5f}  "
              f"SPECIFIC {sp.mean():+.5f} CI[{ci[0]:+.5f},{ci[1]:+.5f}]  {n_pos}/{len(sp)} cats +", flush=True)

    mono = all(sweep[i]["mean_specific"] <= sweep[i + 1]["mean_specific"] + 1e-9 for i in range(len(sweep) - 1))
    print(f"\n  dose-response monotone in alpha? {mono}")
    top = sweep[-1]
    print(f"  at max alpha={top['alpha']}: SPECIFIC {top['mean_specific']:+.5f} CI{top['ci']}  -> "
          f"{'USED (propagates through attention)' if top['ci'][0] > 0 else 'not significant'}")

    out = dict(property=pname, model=model, hidden=st.hidden_size, n_layers=st.n_layers,
               seed=seed, n_cells=len(seqs), alphas=ALPHAS, n_rand=N_RAND,
               mean_norm=mean_norm, monotone=bool(mono), sweep=sweep)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", f"steer_propagation_{pname}_{model}_seed{seed}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[done] -> {p}")
    return out


if __name__ == "__main__":
    prop = sys.argv[1] if len(sys.argv) > 1 else "chromosome"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    model = sys.argv[4] if len(sys.argv) > 4 else "217m"
    run(prop, n, seed, model)
