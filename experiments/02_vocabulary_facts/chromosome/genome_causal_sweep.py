"""ROBUSTNESS for the causal steering result (genome_causal.py): intensity sweep + multiple seeds (Ihor).

genome_causal.py showed one point: at a push of ALPHA=4x the mean token norm, seed 0, steering a cell's context
toward chromosome C raises chr-C mass at UNSTEERED positions by +0.055 (vs random -0.0004; 18/22 chr positive).
PAPER_chromosome_variable.md flags two things it does not yet establish:
  (b) does the effect hold at NATURAL magnitudes, or only under a strong 4x shove?  -> ALPHA sweep
  (b) is it a one-seed fluke?                                                        -> multiple SEEDs

This reruns the identical, tautology-avoided protocol (direction built in INPUT/embed_tokens space; read in
OUTPUT/lm_head space at positions DIFFERENT from the push, so the signal must cross attention; norm-matched
random push as control) across a grid of ALPHA x SEED. A seed re-randomises: cell sample, train/test gene split
(directions + readout tokens), push/read position split, and the random control directions. Natural magnitude ~
ALPHA=1 (one token's worth of norm).

Verdict we want: the SPECIFIC effect (steer - random) stays > 0 with a CI excluding 0 across seeds, and rises
monotonically-ish with ALPHA from a small-but-positive value near ALPHA=1 (dose-response, not a 4x-only artifact).

Incremental: saves after every (seed, alpha) cell, and skips cells already in the JSON -> crash-resumable.

Run: ../../.venv_state/bin/python -u genome_causal_sweep.py       (needs transformers)
Out: results/genome_causal_sweep.json
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
N_CELLS, MAX_LEN, BATCH = 60, 512, 4
N_RAND = 3                       # random control directions (mean is very stable; 3 is enough for the sweep)
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]     # x mean gene-embedding norm; ALPHA=1 ~ natural magnitude
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, "results", "genome_causal_sweep.json")


def load_setty(n, seed):
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]]).astype(str)
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(shape[0], n, replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1])
            E[i, idx[s:e]] = data[s:e]
    return gn, E


def build_seed(seed, C, EMB, tokmap, ens2sym):
    """Everything that depends on the seed: chromosome directions (train half), readout token indices, cells."""
    tok2chr = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and tid < EMB.shape[0]:
            tok2chr[int(tid)] = str(C.loc[s, "chromosome"])
    tids = np.array(sorted(tok2chr)); tchr = np.array([tok2chr[t] for t in tids])
    rng = np.random.default_rng(seed)
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    mean_norm = float(np.linalg.norm(EMB[tids], axis=1).mean())
    dC, te_idx = {}, {}
    te_tok = set(tids[~is_tr].tolist())
    for c in AUTOSOMES:
        m = (tchr == c) & is_tr
        if m.sum() < 20:
            continue
        v = EMB[tids[m]].mean(0) - gcen
        dC[c] = v / (np.linalg.norm(v) + 1e-12)
        te_idx[c] = np.array([t for t in tids[(tchr == c)] if t in te_tok], dtype=np.int64)
    return sorted(dC), dC, te_idx, mean_norm


def main():
    torch.manual_seed(0)
    C = coords()
    R = G.ST_Reader(f"{MDIR}/model.safetensors")
    EMB = R.get("model.embed_tokens.weight")
    tokmap = json.load(open(f"{MAXTOKI_SETUP}/token_dictionary.json"))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID_PKL, "rb")).items()}

    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32)
    dev = xt.device
    embed = xt.model.model.embed_tokens
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))

    res = json.load(open(RES)) if os.path.exists(RES) else {"alphas": ALPHAS, "seeds": SEEDS, "cells": {}}
    RD_all = torch.randn(N_RAND, EMB.shape[1])                       # shared unit random dirs (norm-matched)
    RD_all = (RD_all / RD_all.norm(dim=1, keepdim=True)).to(dev)

    for seed in SEEDS:
        chroms, dC, te_idx, mean_norm = build_seed(seed, C, EMB, tokmap, ens2sym)
        dC_t = {c: torch.tensor(dC[c], dtype=torch.float32, device=dev) for c in chroms}
        te_idx_t = {c: torch.tensor(te_idx[c], device=dev) for c in chroms}
        gn, E = load_setty(N_CELLS, seed)
        var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in gn])
        seqs = []
        for i in range(len(E)):
            rs = E[i].sum() or 1.0
            en = np.log1p(E[i] / rs * 1e4)[var_idx]; nz = en > 0
            norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
            seqs.append(np.nonzero(nz)[0][np.argsort(-norm[nz])][: MAX_LEN - 2])
        print(f"\n=== seed {seed}: {len(chroms)} chr dirs, {len(seqs)} cells, mean_norm={mean_norm:.3f} ===",
              flush=True)

        def masses(logits, c):
            return torch.softmax(logits, -1)[:, te_idx_t[c]].sum(-1)

        # accumulators: per alpha -> per chrom -> lists of (base, steer, rand) mass tensors
        acc = {a: {c: {"base": [], "steer": [], "rand": []} for c in chroms} for a in ALPHAS}
        rng2 = np.random.default_rng(seed)
        for bstart in range(0, len(seqs), BATCH):
            ch = seqs[bstart:bstart + BATCH]
            L = max(len(s) for s in ch) + 2
            ids = np.full((len(ch), L), tok.EOS, np.int64); am = np.zeros((len(ch), L), np.int64)
            gene_pos = []
            for j, s in enumerate(ch):
                sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]])
                ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
                gene_pos.append(np.arange(1, 1 + len(s)))
            ids_t = torch.from_numpy(ids).to(dev); am_t = torch.from_numpy(am).to(dev)
            push_mask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
            read_mask = torch.zeros(len(ch), L, dtype=torch.bool, device=dev)
            for j in range(len(ch)):
                gp = gene_pos[j]
                if len(gp) < 8:
                    continue
                sh = rng2.permutation(len(gp)); half = len(gp) // 2
                push_mask[j, gp[sh[:half]]] = True
                read_mask[j, gp[sh[half:]]] = True
            rm = read_mask.reshape(-1)
            with torch.no_grad():
                base_emb = embed(ids_t)

                def run(delta):
                    e = base_emb if delta is None else base_emb + push_mask.unsqueeze(-1) * delta
                    return xt.model(inputs_embeds=e, attention_mask=am_t).logits.reshape(-1, EMB.shape[0])[rm]

                base_logits = run(None)
                if base_logits.shape[0] == 0:
                    continue
                for a in ALPHAS:
                    push = a * mean_norm
                    rand_logits = torch.stack([run(push * RD_all[k]) for k in range(N_RAND)]).mean(0)
                    for c in chroms:
                        st = run(push * dC_t[c])
                        acc[a][c]["base"].append(masses(base_logits, c).cpu())
                        acc[a][c]["steer"].append(masses(st, c).cpu())
                        acc[a][c]["rand"].append(masses(rand_logits, c).cpu())
            print(f"  seed {seed} cells {bstart + len(ch)}/{len(seqs)}", flush=True)

        # aggregate per alpha, save incrementally
        for a in ALPHAS:
            key = f"seed{seed}_alpha{a}"
            per = []
            for c in chroms:
                if not acc[a][c]["base"]:
                    continue
                b = torch.cat(acc[a][c]["base"]).numpy()
                s = torch.cat(acc[a][c]["steer"]).numpy()
                r = torch.cat(acc[a][c]["rand"]).numpy()
                per.append(float((s - b).mean() - (r - b).mean()))
            per = np.array(per)
            rng3 = np.random.default_rng(0)
            bs = np.array([per[rng3.integers(0, len(per), len(per))].mean() for _ in range(5000)])
            res["cells"][key] = dict(seed=seed, alpha=a, n_chrom=int(len(per)),
                                     specific_mean=float(per.mean()),
                                     ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     n_pos=int((per > 0).sum()))
            os.makedirs(os.path.dirname(RES), exist_ok=True)
            json.dump(res, open(RES, "w"), indent=1)
            rr = res["cells"][key]
            print(f"    alpha={a}: specific={rr['specific_mean']:+.4f} CI[{rr['ci'][0]:+.4f},{rr['ci'][1]:+.4f}]"
                  f" {rr['n_pos']}/{rr['n_chrom']} pos", flush=True)

    # ---- summary table: alpha x seed
    print(f"\n{'='*70}\nSPECIFIC EFFECT (steer - random), by alpha x seed:")
    print(f"  {'alpha':<8}" + "".join(f"seed{s:<8}" for s in SEEDS))
    for a in ALPHAS:
        row = f"  {a:<8}"
        for s in SEEDS:
            k = f"seed{s}_alpha{a}"
            v = res["cells"].get(k)
            row += (f"{v['specific_mean']:+.4f}{'*' if v['ci'][0] > 0 else ' '}  " if v else "  --      ")
        print(row)
    print("  (* = 95% bootstrap CI over chromosomes excludes 0)")
    print(f"\n[done] -> {RES}")


if __name__ == "__main__":
    main()
