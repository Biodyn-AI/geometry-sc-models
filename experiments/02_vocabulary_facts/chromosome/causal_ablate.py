"""CAUSAL USE of the secretory axis — the only test that can make the survivor a "manifold" in Goodfire's sense.

WHY THIS EXISTS. Ihor's decisive point: Engels et al. never claimed the days-of-the-week circle was
statistically significant over 7 points. Their evidence is CAUSAL -- intervene on the circular subspace and the
model's modular-arithmetic answers change -- so the n is thousands of prompts, not 7 tokens. Our whole battery
was a geometric-fit-plus-permutation machine, which correctly reported that it has no power at n=20-46 genes.
The power was never supposed to come from the gene count. It comes from the intervention.

THE TAUTOLOGY TRAP, and how this avoids it. MaxToki's readout is linear: logits = h @ lm_head.T. So "steer h
along an axis defined in lm_head space and predictions move along that axis" is arithmetic, not evidence. Any
such steering result would be circular. The non-trivial question is whether the model's OWN computation USES
that axis, which is an ABLATION question:

    h' = h - (h.v_hat) v_hat        remove the axis from the real residual
    does next-gene prediction get WORSE, and worse SPECIFICALLY for secretory-compartment genes,
    than when an equal-norm RANDOM direction is removed?

Removing any direction hurts. The test is (a) does v hurt MORE than random directions, and (b) is the damage
CONCENTRATED on the compartment the axis encodes. (b) is the real evidence: a generic "this direction carries
variance" effect damages everything equally; a used secretory axis damages secretory-gene prediction
selectively.

NON-CIRCULARITY. v is fit on a TRAIN half of the route genes and every metric is computed on the held-out TEST
half, so "the axis orders the genes it was fit on" cannot leak into the result.

Run: ../../.venv/bin/python -u causal_ablate.py
Out: results/causal_ablate.json
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
sys.path.insert(0, os.path.join(HERE, "..", "route_a", "scgpt", "weightpapers"))
MAXTOKI_SETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MAXTOKI_SETUP)
import gm_lib as G
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa: E402

SETTY = (f"{_DATA}/"
         "hematopoiesis/setty19_cd34_bm.h5ad")
NAME_ID_PKL = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
MDIR = f"{MAXTOKI_SETUP}/MaxToki-217M-HF"
ROUTE = [("endoplasmic reticulum", 1), ("plasma membrane", 2), ("extracellular space", 3)]
N_CELLS, MAX_LEN, BATCH = 160, 1024, 4
N_RAND = 12          # random control directions, each equal-norm to v
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


def compartments():
    import wp_lib as L
    cc = L.load_dbs()["GO_CC"]
    sets = {t: cc[t] for t, _ in ROUTE}
    out = {}
    for t, i in ROUTE:
        others = set().union(*[sets[u] for u, _ in ROUTE if u != t])
        for g in sets[t] - others:
            out[g.upper()] = i
    return out


def main():
    torch.manual_seed(SEED)
    comp = compartments()
    print(f"[setup] {len(comp)} genes with a unique GO-CC route label", flush=True)

    # ---- lm_head, indexed by TOKEN ID (not symbol) -- we need the model's own readout matrix
    R = G.ST_Reader(f"{MDIR}/model.safetensors")
    LM = R.get("lm_head.weight")                                   # (vocab, d)
    tokmap = json.load(open(f"{MAXTOKI_SETUP}/token_dictionary.json"))     # ensembl -> token id
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID_PKL, "rb")).items()}
    tok2comp = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s in comp and tid < LM.shape[0]:
            tok2comp[int(tid)] = comp[s]
    tids = np.array(sorted(tok2comp))
    cvals = np.array([tok2comp[t] for t in tids], float)
    print(f"[setup] {len(tids)} route genes are MaxToki tokens "
          + ", ".join(f"c{int(i)}={int((cvals == i).sum())}" for i in (1, 2, 3)), flush=True)

    # ---- fit the secretory axis v on a TRAIN half of the route tokens, in lm_head space
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tids))
    tr, te = perm[:len(perm) // 2], perm[len(perm) // 2:]
    from sklearn.linear_model import Ridge
    v = Ridge(alpha=1.0).fit(LM[tids[tr]], cvals[tr]).coef_        # direction in d_model space
    v = v / (np.linalg.norm(v) + 1e-12)
    from scipy.stats import spearmanr
    rho_te = spearmanr(LM[tids[te]] @ v, cvals[te]).statistic
    print(f"[axis] fit on {len(tr)} train tokens; HELD-OUT Spearman(lm_head.v, compartment) = {rho_te:+.3f}",
          flush=True)

    # ---- model + tokenized cells
    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32)
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
    print(f"[cells] {len(seqs)} setty cells, mean {np.mean([len(s) for s in seqs]):.0f} gene tokens",
          flush=True)

    LMt = torch.from_numpy(LM).float()
    vt = torch.from_numpy(v).float()
    # random controls: equal-norm directions (v is unit, so unit randoms)
    RD = torch.randn(N_RAND, LM.shape[1]); RD = RD / RD.norm(dim=1, keepdim=True)
    te_tok = torch.from_numpy(tids[te]).long()                     # evaluate ONLY on held-out route tokens
    te_comp = torch.from_numpy(cvals[te]).float()

    def losses(h, tgt):
        """CE of the next-gene prediction, and the same restricted to each compartment class (held-out tokens)."""
        lg = h @ LMt.T
        ce = torch.nn.functional.cross_entropy(lg, tgt, reduction="none")
        return ce

    acc = {"baseline": [], "ablate_v": [], "ablate_rand": [], "comp": []}
    for a in range(0, len(seqs), BATCH):
        ch = seqs[a:a + BATCH]
        L = max(len(s) for s in ch) + 2
        ids = np.full((len(ch), L), tok.EOS, np.int64); am = np.zeros((len(ch), L), np.int64)
        for j, s in enumerate(ch):
            sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]])
            ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
        with torch.no_grad():
            _, hid = xt.forward_with_hidden_states(torch.from_numpy(ids), torch.from_numpy(am))
            h = hid[-1].float().cpu()                              # final residual (B,L,d)
            tgt = torch.from_numpy(ids)[:, 1:]                     # next-token targets
            hh = h[:, :-1]
            m = torch.from_numpy(am)[:, 1:].bool()
            # keep only positions whose TARGET is a held-out route gene
            keep = m & torch.isin(tgt, te_tok)
            if not keep.any():
                continue
            hv, tv = hh[keep], tgt[keep]
            base = losses(hv, tv)
            hv_ab = hv - (hv @ vt).unsqueeze(-1) * vt
            abl = losses(hv_ab, tv)
            rnd = torch.stack([losses(hv - (hv @ r).unsqueeze(-1) * r, tv) for r in RD]).mean(0)
            c = torch.tensor([float(cvals[np.searchsorted(tids, int(t))]) for t in tv])
            acc["baseline"].append(base); acc["ablate_v"].append(abl)
            acc["ablate_rand"].append(rnd); acc["comp"].append(c)
        print(f"  cells {a + len(ch)}/{len(seqs)}", flush=True)

    B = torch.cat(acc["baseline"]).numpy(); A = torch.cat(acc["ablate_v"]).numpy()
    Rn = torch.cat(acc["ablate_rand"]).numpy(); C = torch.cat(acc["comp"]).numpy()
    print(f"\n[eval] {len(B)} held-out route-gene prediction positions", flush=True)

    def boot(d, n=2000):
        rng = np.random.default_rng(SEED)
        s = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
        return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

    res = {"held_out_axis_rho": float(rho_te), "n_positions": int(len(B)),
           "baseline_ce": float(B.mean())}
    print("\n=== (a) does ablating v hurt MORE than an equal-norm random direction? ===")
    dv, lo, hi = boot(A - B); dr, rlo, rhi = boot(Rn - B)
    dd, dlo, dhi = boot((A - B) - (Rn - B))
    res["delta_v"] = dict(mean=dv, ci=[lo, hi]); res["delta_rand"] = dict(mean=dr, ci=[rlo, rhi])
    res["v_minus_rand"] = dict(mean=dd, ci=[dlo, dhi])
    print(f"  CE increase, ablate v      : {dv:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  CE increase, ablate random : {dr:+.4f}  CI [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"  v MINUS random             : {dd:+.4f}  CI [{dlo:+.4f}, {dhi:+.4f}]  "
          f"{'SPECIFIC' if dlo > 0 else 'not distinguishable from a generic direction'}")

    print("\n=== (b) is the damage CONCENTRATED on the compartment the axis encodes? ===")
    res["by_compartment"] = {}
    for i, nm in [(1, "ER"), (2, "plasma membrane"), (3, "extracellular")]:
        s = C == i
        if s.sum() < 20:
            print(f"  {nm:<18} n={int(s.sum())} -- too few"); continue
        m, l, hgh = boot((A[s] - B[s]) - (Rn[s] - B[s]))
        res["by_compartment"][nm] = dict(n=int(s.sum()), mean=m, ci=[l, hgh])
        print(f"  {nm:<18} n={int(s.sum()):<5} (v - random) CE increase {m:+.4f} CI [{l:+.4f}, {hgh:+.4f}]")

    json.dump(res, open(os.path.join(HERE, "results", "causal_ablate.json"), "w"), indent=1)
    print("\n[done] -> results/causal_ablate.json")


if __name__ == "__main__":
    main()
