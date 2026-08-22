"""IS CHROMOSOME *USED* BY THE MODEL, OR ONLY READABLE IN THE TABLE? — the causal steering test (Ihor).

RESULTS.md section 12 shows the INPUT embedding table linearly encodes a gene's chromosome. That is a fact
about the static table. It does NOT show the model's COMPUTATION uses chromosome -- exactly the distinction
section 6 draws for the secretory axis (a readable correlate is not a used manifold). This is the causal test.

THE TAUTOLOGY TRAP (section 6 / causal_ablate.py). MaxToki's readout is linear: logits = h @ lm_head.T. Embed
and lm_head both encode chromosome (section 3: 1B embed 0.926 ~ lm_head 0.931), so "steer a gene's own input
toward chr-C and its OWN output readout moves toward chr-C" is mostly the two correlated tables, not the layers.
AVOIDED two ways:
  * the steering direction d_C is built in INPUT (embed_tokens) space; the effect is read in OUTPUT (lm_head)
    space, through the whole transformer;
  * the readout is at DIFFERENT positions than the push. We add d_C to a random HALF of a cell's gene tokens and
    read the next-gene prediction at the OTHER half. For chr-C mass to rise at unsteered positions, chromosome
    identity must PROPAGATE THROUGH ATTENTION -- it cannot be table pass-through.
  * control = norm-matched RANDOM input directions (kills any generic input->output leakage).

THE CLAIM UNDER TEST. If the model represents "this cell's context is enriched for chromosome C" as a
computational variable, then pushing context genes toward chr-C should raise the model's predicted probability
of chr-C genes at the held-out positions, MORE than a random push does. If not -- if delta ~ 0 -- chromosome is
a readable table feature the computation does not use, a section-6-style honest negative. Either result stands.

Run: ../../.venv_state/bin/python -u genome_causal.py     (needs transformers; ../../.venv lacks it)
Out: results/genome_causal.json
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
MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MAXTOKI_SETUP)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402

SETTY = (f"{_DATA}/"
         "hematopoiesis/setty19_cd34_bm.h5ad")
NAME_ID_PKL = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
MDIR = f"{MAXTOKI_SETUP}/MaxToki-217M-HF"
N_CELLS, MAX_LEN, BATCH = 80, 512, 4
N_RAND = 5               # random control directions (shared across chromosomes; not chr-specific)
ALPHA = 4.0             # push strength in units of the mean gene-embedding norm
SEED = 0


def load_setty(n):
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]]).astype(str)
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], n, replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            E[i, idx[s:e]] = data[s:e]
    return gn, E


def main():
    torch.manual_seed(SEED)
    C = coords()

    # ---- token id -> chromosome, and the INPUT embedding table (embed_tokens), indexed by token id
    R = G.ST_Reader(f"{MDIR}/model.safetensors")
    EMB = R.get("model.embed_tokens.weight")                       # (vocab, d) INPUT table
    tokmap = json.load(open(f"{MAXTOKI_SETUP}/token_dictionary.json"))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID_PKL, "rb")).items()}
    tok2chr = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and tid < EMB.shape[0]:
            tok2chr[int(tid)] = str(C.loc[s, "chromosome"])
    tids = np.array(sorted(tok2chr))
    tchr = np.array([tok2chr[t] for t in tids])
    print(f"[setup] {len(tids)} MaxToki tokens carry an autosome label", flush=True)

    # ---- chromosome INPUT directions on a TRAIN half; a per-chromosome token mask for OUTPUT readout
    rng = np.random.default_rng(SEED)
    is_tr = rng.random(len(tids)) < 0.5
    mean_norm = float(np.linalg.norm(EMB[tids], axis=1).mean())
    dC = {}                                                        # chromosome -> unit input direction
    out_mask = {}                                                  # chromosome -> bool over full vocab (readout)
    gcen = EMB[tids[is_tr]].mean(0)
    for c in AUTOSOMES:
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        v = EMB[tids[m]].mean(0) - gcen                            # chr-c centroid minus global, INPUT space
        dC[c] = v / (np.linalg.norm(v) + 1e-12)
        ov = np.zeros(EMB.shape[0], bool); ov[tids[(tchr == c)]] = True
        out_mask[c] = ov
    chroms = sorted(dC)
    te_tok = set(tids[~is_tr].tolist())                           # readout only on HELD-OUT tokens
    print(f"[setup] {len(chroms)} chromosome directions built; push alpha={ALPHA}x mean-norm "
          f"({ALPHA * mean_norm:.2f})", flush=True)

    # ---- model + tokenised cells
    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32)
    dev = xt.device
    embed = xt.model.model.embed_tokens                          # to build inputs_embeds we can steer
    LM = xt.model.lm_head
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    gn, E = load_setty(N_CELLS)
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))
    var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in gn])
    seqs = []
    for i in range(len(E)):
        rs = E[i].sum() or 1.0
        en = np.log1p(E[i] / rs * 1e4)[var_idx]
        nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        order = np.argsort(-norm[nz])
        seqs.append(np.nonzero(nz)[0][order][: MAX_LEN - 2])
    print(f"[cells] {len(seqs)} cells, mean {np.mean([len(s) for s in seqs]):.0f} gene tokens\n", flush=True)

    dC_t = {c: torch.tensor(dC[c], dtype=torch.float32, device=dev) for c in chroms}
    RD = torch.randn(N_RAND, EMB.shape[1], device=dev); RD = RD / RD.norm(dim=1, keepdim=True)
    push = ALPHA * mean_norm

    # precompute held-out token index tensors per chromosome (readout is only on held-out tokens)
    te_idx = {c: torch.tensor([t for t in np.where(out_mask[c])[0] if t in te_tok], device=dev, dtype=torch.long)
              for c in chroms}

    def masses(logits, c):
        p = torch.softmax(logits, -1)                            # (P, vocab)
        return p[:, te_idx[c]].sum(-1)                           # (P,)

    acc = {c: {"base": [], "steer": [], "rand": []} for c in chroms}
    rng2 = np.random.default_rng(SEED)
    for a in range(0, len(seqs), BATCH):
        ch = seqs[a:a + BATCH]
        L = max(len(s) for s in ch) + 2
        ids = np.full((len(ch), L), tok.EOS, np.int64); am = np.zeros((len(ch), L), np.int64)
        gene_pos = []
        for j, s in enumerate(ch):
            sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]])
            ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
            gene_pos.append(np.arange(1, 1 + len(s)))            # positions holding a gene token
        ids_t = torch.from_numpy(ids).to(dev); am_t = torch.from_numpy(am).to(dev)
        with torch.no_grad():
            base_emb = embed(ids_t)                              # (B,L,d) input embeddings
            # split each cell's gene positions into PUSH half and READ half
            push_mask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
            read_mask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
            for j in range(len(ch)):
                gp = gene_pos[j]
                if len(gp) < 8:
                    continue
                sh = rng2.permutation(len(gp)); half = len(gp) // 2
                push_mask[j, gp[sh[:half]]] = True
                read_mask[j, gp[sh[half:]]] = True

            def run(delta):
                e = base_emb.clone()
                if delta is not None:
                    e = e + push_mask.unsqueeze(-1) * delta      # add push to PUSH positions only
                out = xt.model(inputs_embeds=e, attention_mask=am_t)
                return out.logits                                # (B,L,vocab)

            rm = read_mask.reshape(-1)
            base_logits = run(None).reshape(-1, EMB.shape[0])[rm]
            if base_logits.shape[0] == 0:
                print(f"  cells {a + len(ch)}/{len(seqs)} (no read positions)", flush=True); continue
            # random pushes are NOT chromosome-specific: compute once, read every chromosome's mass from them
            rand_logits = torch.stack([run(push * RD[k]).reshape(-1, EMB.shape[0])[rm]
                                       for k in range(N_RAND)]).mean(0)
            for c in chroms:
                st_logits = run(push * dC_t[c]).reshape(-1, EMB.shape[0])[rm]
                acc[c]["base"].append(masses(base_logits, c).cpu())
                acc[c]["steer"].append(masses(st_logits, c).cpu())
                acc[c]["rand"].append(masses(rand_logits, c).cpu())
        print(f"  cells {a + len(ch)}/{len(seqs)}", flush=True)

    # ---- aggregate: steering toward C vs random, on chr-C mass at UNSTEERED positions
    rows = []
    for c in chroms:
        if not acc[c]["base"]:
            continue
        b = torch.cat(acc[c]["base"]).numpy()
        s = torch.cat(acc[c]["steer"]).numpy()
        r = torch.cat(acc[c]["rand"]).numpy()
        rows.append(dict(chrom=c, n=int(len(b)), base=float(b.mean()),
                         d_steer=float((s - b).mean()), d_rand=float((r - b).mean()),
                         specific=float((s - b).mean() - (r - b).mean())))
    ds = np.array([x["d_steer"] for x in rows]); dr = np.array([x["d_rand"] for x in rows])
    sp = np.array([x["specific"] for x in rows])
    # paired bootstrap over chromosomes
    rng3 = np.random.default_rng(SEED)
    bs = np.array([sp[rng3.integers(0, len(sp), len(sp))].mean() for _ in range(5000)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print("\n=== does steering context toward chr-C raise chr-C mass at UNSTEERED positions? ===")
    print(f"  mean chr-C mass increase, STEER toward C : {ds.mean():+.5f}")
    print(f"  mean chr-C mass increase, RANDOM push    : {dr.mean():+.5f}")
    print(f"  SPECIFIC (steer - random), mean over {len(rows)} chr: {sp.mean():+.5f}  CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    used = ci[0] > 0
    print(f"  -> {'USED: chromosome propagates through the computation' if used else 'NOT USED: readable table feature, not a computational variable (a section-6-style negative)'}")

    res = dict(alpha=ALPHA, n_cells=len(seqs), n_chrom=len(rows), push_norm=push,
               mean_d_steer=float(ds.mean()), mean_d_rand=float(dr.mean()),
               mean_specific=float(sp.mean()), specific_ci=ci, used=bool(used), per_chrom=rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "genome_causal.json"), "w"), indent=1)
    print("\n[done] -> results/genome_causal.json")


if __name__ == "__main__":
    main()
