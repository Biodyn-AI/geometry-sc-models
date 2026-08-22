"""GATE G2 — what is the steering effect worth, in units the model itself defines?

The causal test reports a raw increase in chromosome-C probability mass at unsteered genes (+0.055). That number
is uninterpretable on its own. Two things make it interpretable, and both are measured here.

  (1) BASELINE-RELATIVE. How large is +0.055 against the chr-C mass that was already there? (Baseline mass is
      recorded in results/genome_causal.json, so this needs no new compute; it is recomputed here for the record.)

  (2) SUBSTITUTION CEILING. What would a *genuine* chromosome-C-enriched context achieve? Instead of adding a
      steering vector to the push half of a cell's genes, REPLACE those tokens with real chr-C genes (drawn from
      the train half, so the read-out tokens stay held out) and measure the same quantity. That is the effect an
      actual intervention of this kind can produce, and steering should be reported as a fraction of it.

  (3) REALLOCATION. Softmax mass lives on a simplex, so any gain must come from somewhere. For each steer toward
      chr-C we record the mass on ALL 22 chromosomes, not just C, which shows whether the model is genuinely
      re-weighting toward C or merely shuffling mass between large and small categories. (A naive check that sums
      deltas across the 22 independent steering runs is meaningless; this is the valid version.)

Run: ../../.venv_state/bin/python -u steer_units.py
Out: results/steer_units.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch, h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
import gm_lib as G
from genome_wide import coords, AUTOSOMES
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa

MDIR = f"{MSETUP}/MaxToki-217M-HF"
SETTY = (f"{_DATA}/"
         "hematopoiesis/setty19_cd34_bm.h5ad")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
N_CELLS, MAX_LEN, BATCH, ALPHA, SEED = 80, 512, 4, 4.0, 0


def load_cells(n):
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]]).astype(str)
        X = f["X"]; sh = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED); sel = np.sort(rng.choice(sh[0], n, replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), sh[1]), np.float32)
        for i, r in enumerate(sel):
            a, b = int(indptr[r]), int(indptr[r + 1]); E[i, idx[a:b]] = data[a:b]
    return gn, E


def main():
    torch.manual_seed(SEED)
    C = coords()
    R = G.ST_Reader(f"{MDIR}/model.safetensors"); EMB = R.get("model.embed_tokens.weight")
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    e2s = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    tok2chr = {}
    for ens, tid in tokmap.items():
        s = e2s.get(ens)
        if s in C.index and C.chromosome[s] in AUTOSOMES and tid < EMB.shape[0]:
            tok2chr[int(tid)] = str(C.chromosome[s])
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])

    rng = np.random.default_rng(SEED)
    is_tr = rng.random(len(tids)) < 0.5
    mean_norm = float(np.linalg.norm(EMB[tids], axis=1).mean()); push = ALPHA * mean_norm
    gcen = EMB[tids[is_tr]].mean(0)
    dC, read_idx, subs_pool = {}, {}, {}
    for c in AUTOSOMES:
        m_tr = (tchr == c) & is_tr; m_te = (tchr == c) & (~is_tr)
        if m_tr.sum() < 20 or m_te.sum() < 20:
            continue
        v = EMB[tids[m_tr]].mean(0) - gcen
        dC[c] = v / (np.linalg.norm(v) + 1e-12)
        read_idx[c] = tids[m_te]              # held-out chr-c tokens: the read-out
        subs_pool[c] = tids[m_tr]             # train-half chr-c tokens: what we substitute in
    chroms = sorted(dC)
    print(f"[setup] {len(chroms)} chromosomes | push alpha={ALPHA} ({push:.2f})", flush=True)

    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32); dev = xt.device
    embed = xt.model.model.embed_tokens
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    gn, E = load_cells(N_CELLS)
    name_id = pickle.load(open(NAME_ID, "rb"))
    var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in gn])
    seqs = []
    for i in range(len(E)):
        rs = E[i].sum() or 1.0
        en = np.log1p(E[i] / rs * 1e4)[var_idx]; nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        seqs.append(np.nonzero(nz)[0][np.argsort(-norm[nz])][: MAX_LEN - 2])
    print(f"[cells] {len(seqs)} cells\n", flush=True)

    dC_t = {c: torch.tensor(dC[c], dtype=torch.float32, device=dev) for c in chroms}
    ridx = {c: torch.tensor(read_idx[c], device=dev, dtype=torch.long) for c in chroms}
    acc = {c: {k: [] for k in ("base", "steer", "subst")} for c in chroms}
    alloc = {c: {"base": [], "steer": []} for c in chroms}     # mass on ALL chromosomes
    rng2 = np.random.default_rng(SEED)

    for a in range(0, len(seqs), BATCH):
        ch = seqs[a:a + BATCH]; L = max(len(s) for s in ch) + 2
        ids = np.full((len(ch), L), tok.EOS, np.int64); am = np.zeros((len(ch), L), np.int64); gp = []
        for j, s in enumerate(ch):
            sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]])
            ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1; gp.append(np.arange(1, 1 + len(s)))
        idt = torch.from_numpy(ids).to(dev); amt = torch.from_numpy(am).to(dev)
        pmask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
        rmask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
        push_pos = []
        for j in range(len(ch)):
            g = gp[j]
            if len(g) < 8:
                push_pos.append(np.array([], int)); continue
            sh = rng2.permutation(len(g)); half = len(g) // 2
            pmask[j, g[sh[:half]]] = True; rmask[j, g[sh[half:]]] = True
            push_pos.append(g[sh[:half]])
        with torch.no_grad():
            base_emb = embed(idt)
            def masses(logits, c):
                p = torch.softmax(logits, -1); return p[:, ridx[c]].sum(-1)
            def allmass(logits):
                p = torch.softmax(logits, -1)
                return {c: float(p[:, ridx[c]].sum(-1).mean()) for c in chroms}
            base_logits = xt.model(inputs_embeds=base_emb, attention_mask=amt).logits
            rm = rmask.reshape(-1)
            bl = base_logits.reshape(-1, base_logits.shape[-1])[rm]
            if bl.shape[0] == 0:
                continue
            ab = allmass(bl)
            for c in chroms:
                acc[c]["base"].append(masses(bl, c).cpu()); alloc[c]["base"].append(ab)
                # --- steering
                e = base_emb + pmask.unsqueeze(-1) * (push * dC_t[c])
                sl_full = xt.model(inputs_embeds=e, attention_mask=amt).logits
                sl = sl_full.reshape(-1, sl_full.shape[-1])[rm]
                acc[c]["steer"].append(masses(sl, c).cpu()); alloc[c]["steer"].append(allmass(sl))
                # --- substitution ceiling: replace push tokens with REAL chr-c genes (train half)
                ids2 = ids.copy()
                for j in range(len(ch)):
                    if len(push_pos[j]):
                        ids2[j, push_pos[j]] = rng2.choice(subs_pool[c], len(push_pos[j]))
                ul = xt.model(input_ids=torch.from_numpy(ids2).to(dev), attention_mask=amt).logits
                acc[c]["subst"].append(masses(ul.reshape(-1, ul.shape[-1])[rm], c).cpu())
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
        print(f"  cells {a+len(ch)}/{len(seqs)}", flush=True)

    rows = []
    for c in chroms:
        b = torch.cat(acc[c]["base"]).numpy().mean()
        s = torch.cat(acc[c]["steer"]).numpy().mean()
        u = torch.cat(acc[c]["subst"]).numpy().mean()
        # reallocation: change on OTHER chromosomes when steering toward c
        ob = np.mean([np.mean([v[k] for k in chroms if k != c]) for v in alloc[c]["base"]])
        os_ = np.mean([np.mean([v[k] for k in chroms if k != c]) for v in alloc[c]["steer"]])
        rows.append(dict(chrom=c, base=float(b), steer=float(s), subst=float(u),
                         effect=float(s - b), ceiling=float(u - b),
                         frac_of_ceiling=float((s - b) / (u - b)) if (u - b) > 1e-9 else float("nan"),
                         other_base=float(ob), other_steer=float(os_), other_delta=float(os_ - ob)))
    eff = np.array([r["effect"] for r in rows]); ceil = np.array([r["ceiling"] for r in rows])
    fr = np.array([r["frac_of_ceiling"] for r in rows], float)
    print("\n=== STEERING IN INTERPRETABLE UNITS ===")
    print(f"  baseline chr-C mass            : {np.mean([r['base'] for r in rows]):.4f}")
    print(f"  steering effect                : {eff.mean():+.4f}")
    print(f"  SUBSTITUTION CEILING (real genes): {ceil.mean():+.4f}")
    print(f"  steering as % of ceiling       : mean {100*np.nanmean(fr):.0f}%  median {100*np.nanmedian(fr):.0f}%")
    print(f"  reallocation: mean change on OTHER chromosomes when steering toward C: "
          f"{np.mean([r['other_delta'] for r in rows]):+.5f}")
    json.dump(dict(alpha=ALPHA, n_cells=len(seqs), rows=rows,
                   mean_effect=float(eff.mean()), mean_ceiling=float(ceil.mean()),
                   frac_mean=float(np.nanmean(fr)), frac_median=float(np.nanmedian(fr))),
              open(os.path.join(HERE, "results", "steer_units.json"), "w"), indent=1)
    print("\n[done] -> results/steer_units.json")


if __name__ == "__main__":
    main()
