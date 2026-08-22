"""DOES THE GENOMIC COORDINATE LIVE ONLY IN THE STATIC TABLE, OR ALSO IN THE INTERNAL, CONTEXTUAL LAYERS? (Ihor)

Everything so far decodes chromosome/position from the STATIC per-gene tables (context-free: one vector per gene).
This tests the CONTEXTUAL internal representation: a gene token's hidden state at each transformer layer, given
the rest of the cell as context. We run cells through MaxToki-217M with output_hidden_states, average each gene's
hidden state over all its occurrences at each layer, and decode chromosome (22-class balanced acc) and
within-chromosome position (leakage-clean) at every layer. Trajectory across layers vs the static embed_tokens
(input) and lm_head (output) tables tells us whether the coordinate strengthens, persists, or washes out as the
model computes.

Needs the transformers venv: ../../.venv_state/bin/python -u maxtoki_layers.py
Out: results/maxtoki_layers.json (+ figures/fig6_layers.pdf)
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
from genome_position_geometry import dedup
from maxtoki_adapter import MaxTokiTokenizer, MaxTokiAttentionExtractor  # noqa
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

MDIR = f"{MSETUP}/MaxToki-217M-HF"
SETTY = (f"{_DATA}/"
         "hematopoiesis/setty19_cd34_bm.h5ad")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
N_CELLS, MAX_LEN, BATCH, MINCOUNT = int(os.environ.get("NCELLS", 600)), 512, 4, 10
SEED = 0


def load_cells(n):
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]]).astype(str)
        X = f["X"]; shape = tuple(int(v) for v in X.attrs["shape"])
        rng = np.random.default_rng(SEED); sel = np.sort(rng.choice(shape[0], min(n, shape[0]), replace=False))
        indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
        E = np.zeros((len(sel), shape[1]), np.float32)
        for i, r in enumerate(sel):
            s, e = int(indptr[r]), int(indptr[r + 1]); E[i, idx[s:e]] = data[s:e]
    return gn, E


def chrom_bal(M, chrom):
    y = chrom; skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=object)
    for tr, te in skf.split(M, y):
        sc = StandardScaler().fit(M[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1).fit(sc.transform(M[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(M[te]))
    return float(balanced_accuracy_score(y, pred.astype(str)))


def pos_rho(M, syms, C, pos_i_ok):
    rr = []
    for c in AUTOSOMES:
        g = [s for s in syms if C.chromosome.get(s) == c]
        if len(g) < 150:
            continue
        loc = [pos_i_ok[s] for s in g]; X = M[loc]; start = C.start.loc[g].values.astype(float)
        keep = dedup(X, start)
        if keep.sum() < 80:
            continue
        Xk, y = X[keep], start[keep]
        P = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xk):
            sc = StandardScaler().fit(Xk[tr]); P[te] = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(
                sc.transform(Xk[tr]), y[tr]).predict(sc.transform(Xk[te]))
        r = spearmanr(P, y).statistic
        rr.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(rr)) if rr else float("nan")


def main():
    torch.manual_seed(SEED)
    C = coords()
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    tid2sym = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s in C.index and C.chromosome[s] in AUTOSOMES:
            tid2sym[int(tid)] = s

    # tokenise cells
    tok = MaxTokiTokenizer(model_input_size=MAX_LEN)
    gn, E = load_cells(N_CELLS)
    name_id = pickle.load(open(NAME_ID, "rb"))
    var_idx, token_ids, medians = tok.make_var_mapping([name_id.get(s) for s in gn])
    seqs = []
    for i in range(len(E)):
        rs = E[i].sum() or 1.0
        en = np.log1p(E[i] / rs * 1e4)[var_idx]; nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        order = np.argsort(-norm[nz]); seqs.append(np.nonzero(nz)[0][order][: MAX_LEN - 2])
    seq_tids = [token_ids[s] for s in seqs]

    # which gene tokens appear >= MINCOUNT and carry an autosome
    from collections import Counter
    cnt = Counter(int(t) for sq in seq_tids for t in sq if int(t) in tid2sym)
    keep_tids = [t for t, c in cnt.items() if c >= MINCOUNT]
    loc = {t: i for i, t in enumerate(keep_tids)}
    K = len(keep_tids); print(f"[setup] {len(seqs)} cells, {K} gene tokens with >={MINCOUNT} occurrences", flush=True)

    xt = MaxTokiAttentionExtractor(model_dir=MDIR, dtype=torch.float32); dev = xt.device
    n_hid = xt.model.config.num_hidden_layers + 1                      # embeddings + each layer
    acc = np.zeros((n_hid, K, xt.model.config.hidden_size), np.float32); counts = np.zeros(K, np.int64)

    for a in range(0, len(seqs), BATCH):
        chunk = seqs[a:a + BATCH]; L = max(len(s) for s in chunk) + 2
        ids = np.full((len(chunk), L), tok.EOS, np.int64); am = np.zeros((len(chunk), L), np.int64); gp = []
        for j, s in enumerate(chunk):
            sq = np.concatenate([[tok.BOS], token_ids[s], [tok.EOS]]); ids[j, :len(sq)] = sq; am[j, :len(sq)] = 1
            gp.append(np.arange(1, 1 + len(s)))
        idt = torch.from_numpy(ids).to(dev); amt = torch.from_numpy(am).to(dev)
        with torch.no_grad():
            out = xt.model(input_ids=idt, attention_mask=amt, output_hidden_states=True)
            hs = [h.to("cpu", torch.float32).numpy() for h in out.hidden_states]      # n_hid x (B,L,d)
        # vectorised accumulation: gather all valid gene positions in the batch, then np.add.at per layer
        jj, pp, li = [], [], []
        for j in range(len(chunk)):
            for p in gp[j]:
                t = int(ids[j, p])
                if t in loc:
                    jj.append(j); pp.append(p); li.append(loc[t])
        if li:
            jj, pp, li = np.array(jj), np.array(pp), np.array(li)
            np.add.at(counts, li, 1)
            for l in range(n_hid):
                np.add.at(acc[l], li, hs[l][jj, pp])
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        print(f"  cells {a + len(chunk)}/{len(seqs)}", flush=True)

    syms = np.array([tid2sym[t] for t in keep_tids])
    chrom = np.array([C.chromosome[s] for s in syms])
    pos_i_ok = {s: i for i, s in enumerate(syms)}
    layer_bal, layer_pos = [], []
    for l in range(n_hid):
        M = acc[l] / counts[:, None]
        layer_bal.append(chrom_bal(M, chrom))
        layer_pos.append(pos_rho(M, syms, C, pos_i_ok))
        print(f"  layer {l:>2}: chromosome bal_acc {layer_bal[-1]:.3f} | position rho {layer_pos[-1]:+.3f}", flush=True)

    # static reference on the SAME gene set
    def static(name):
        Ms, ss = G.basis(name); pi = {s: i for i, s in enumerate(ss)}
        idx = [pi[s] for s in syms if s in pi]; sub = [s for s in syms if s in pi]
        Msub = Ms[idx]; ch = np.array([C.chromosome[s] for s in sub])
        return chrom_bal(Msub, ch), pos_rho(Msub, np.array(sub), C, {s: i for i, s in enumerate(sub)})
    emb_bal, emb_pos = static("maxtoki_we"); lm_bal, lm_pos = static("maxtoki_lmhead")

    res = dict(n_cells=len(seqs), n_genes=K, n_hidden=n_hid,
               layer_bal=layer_bal, layer_pos=layer_pos,
               static_embed=dict(bal=emb_bal, pos=emb_pos), static_lmhead=dict(bal=lm_bal, pos=lm_pos),
               chance_bal=1/22)
    print(f"\nSTATIC embed_tokens: bal {emb_bal:.3f} pos {emb_pos:+.3f} | lm_head: bal {lm_bal:.3f} pos {lm_pos:+.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "maxtoki_layers.json"), "w"), indent=1)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
        xs = list(range(n_hid))
        a1.plot(xs, layer_bal, "-o", color="#2166AC", lw=1.8, ms=4)
        a1.axhline(1/22, ls="--", lw=1, color="#B0B4BA"); a1.text(0, 1/22+0.005, "chance", fontsize=7.5, color="#6b7079")
        a1.axhline(lm_bal, ls=":", lw=1.2, color="#D6743C"); a1.text(n_hid-1, lm_bal+0.005, "static lm_head", ha="right", fontsize=7.5, color="#D6743C")
        a1.set_xlabel("layer (0 = input embedding)"); a1.set_ylabel("chromosome balanced acc"); a1.set_ylim(0, max(0.6, max(layer_bal)+0.05))
        a1.set_title("a  Chromosome across layers", loc="left", fontweight="bold")
        a2.plot(xs, layer_pos, "-o", color="#4A9B8E", lw=1.8, ms=4)
        a2.axhline(lm_pos, ls=":", lw=1.2, color="#D6743C"); a2.text(n_hid-1, lm_pos+0.01, "static lm_head", ha="right", fontsize=7.5, color="#D6743C")
        a2.set_xlabel("layer (0 = input embedding)"); a2.set_ylabel("within-chr position ρ"); a2.set_ylim(0, max(0.5, max(layer_pos)+0.05))
        a2.set_title("b  Position across layers", loc="left", fontweight="bold")
        fig.tight_layout()
        for ext in ("pdf", "png"): fig.savefig(os.path.join(HERE, "figures", f"fig6_layers.{ext}"), bbox_inches="tight", dpi=200)
        print("[fig] figures/fig6_layers.pdf")
    except Exception as e:
        print("fig skipped:", e)
    print("[done] -> results/maxtoki_layers.json")


if __name__ == "__main__":
    main()
