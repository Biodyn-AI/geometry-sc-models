"""Invert the fate direction into a ranked TF list -- does scGPT nominate real reprogramming factors?

THE QUESTION. NATIVE_DECODE_RESULTS.md established that steering a progenitor along an on-manifold fate
direction produces that fate's gene program (4/4, scGPT's own head, p=0.042). That is *navigation*: we move the
embedding and read what it means. The useful inverse is: **which PERTURBATION would move the cell that way?**
If forcing TF g's expression shifts the cell embedding along the erythroid direction, scGPT is nominating g as
an erythroid driver. Score the nominations against known lineage-driver TFs.

METHOD (in-silico activation, Geneformer-style, but in scGPT's binned input space).
  baseline:  cell -> tokens(top-1200 expressed genes, 51-bin values) -> scGPT -> cell_emb (mean-pooled L11)
  perturbed: force TF g to the TOP expression bin (50); if g is not expressed, append its token
  score:     Delta_z = whiten(emb_pert) - whiten(emb_base)   [same 20-PC space the steering was validated in]
             align(g, fate) = mean over source cells of cos(Delta_z_i, d_i),  d_i = the ON-MANIFOLD fate
             direction at cell i (project_constant field -- the exact direction the validated steer follows)
  Rank the 896 TFs by align(g, fate). Ask whether the known drivers of that fate rise to the top.

WHY scGPT AND NOT STATE/MaxToki (deviation from the GENE_DECODE next-step note, stated up front). The fate
directions are validated in *scGPT's* embedding space on Setty. STATE/MaxToki live in different spaces trained
on different data (K562 CRISPRi), so starting there needs a cross-space mapping that could sink the result for
reasons unrelated to biology. scGPT keeps perturbation and direction in ONE validated space. STATE is the
cross-model check afterwards (state_setty.npz exists). Caveat kept honest: scGPT is not perturbation-TRAINED,
so this is an in-silico expression intervention, not a predicted experimental outcome.

*** THE CONTROL THAT DECIDES THIS (co-expression confound; CLAUDE.md lists it as a standing failure mode). ***
A gene that is merely a MARKER of the target fate will align trivially: pushing it moves the cell toward cells
that express it. So "known drivers rank high" is NOT sufficient -- differential expression alone would do that.
The real question is whether scGPT's nomination beats a pure marker ranking:
  BASELINE   de(g, fate) = mean log-expr in the fate cluster - mean log-expr in the source progenitors
  TESTS      (1) driver AUROC under align   vs  (2) driver AUROC under de   vs
             (3) driver AUROC under align RESIDUALISED on de (regress align ~ de across TFs, keep residual)
  If (1) ~ (2) and (3) ~ 0.5, the model is a co-expression readout and adds nothing -- an honest negative.
  Only if align beats de, and survives residualisation, has scGPT nominated regulators beyond marker-ness.
Plus: a random on-manifold direction (driver AUROC must be ~0.5), and the fate x driver-set DIAGONAL.

POSITIVE SETS -- pre-registered here BEFORE running, standard hematopoietic lineage drivers, intersected with
the TF universe. Ery/Mega deliberately share GATA1/TAL1/NFE2/GATA2 (real shared MEP program) -- that makes the
ery-vs-mega diagonal genuinely hard, and it is reported as such rather than hidden.

TF UNIVERSE: DoRothEA u TRRUST union (896 TFs), an EXTERNAL database -- not a list we chose.

Run:  ~/anaconda3/envs/bio_mech_interp/bin/python perturb_invert.py [n_src=64]
Out:  results/perturb_invert_scgptbin_setty.json  (+ printed report)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tangent_diagnostic import zscore, unit, SEED, DIM  # noqa: E402
from local_steering import project_constant  # noqa: E402

MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_REPO = os.path.join(MI, "external", "scGPT")
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
TF_DB = (f"{_DATA}/biodyn-work/network_inference/data/"
         "dorothea_trrust_union_immune.tsv")
ROOT = f"{_DATA}"
SETTY = os.path.join(ROOT, "data/hematopoiesis/setty19_cd34_bm.h5ad")
EMB = os.path.join(ROOT, "data/branchpoint/scgptbin_setty.npz")
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ, LAYER = 512, 12, 8, 1200, 11
N_BINS, PAD_VALUE, TOP_BIN = 51, -2, 50
BATCH = 32

FATE_CLUSTERS = {"ery": ["Ery_1", "Ery_2"], "myeloid": ["Mono_1", "Mono_2"],
                 "lymphoid": ["CLP"], "mega": ["Mega"]}

# pre-registered known drivers (standard hematopoietic literature), intersected with the TF universe at runtime
DRIVERS = {
    "ery":      ["GATA1", "KLF1", "TAL1", "GATA2", "LMO2", "NFE2", "ZFPM1"],
    "myeloid":  ["SPI1", "CEBPA", "CEBPB", "CEBPE", "IRF8", "RUNX1", "KLF4", "EGR1"],
    "lymphoid": ["EBF1", "PAX5", "TCF3", "IKZF1", "FOXO1", "GATA3", "LEF1", "TCF7"],
    "mega":     ["FLI1", "RUNX1", "GATA1", "NFE2", "MEIS1", "TAL1", "GATA2"],
}


# ------------------------------------------------------------------ data / model
def load_setty():
    with h5py.File(SETTY, "r") as f:
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["index"][:]])
        G = int(f["X"].attrs["shape"][1])
    return gn, G


def sparse_rows(cell_idx, G):
    """Raw counts for the requested h5ad rows."""
    out = np.zeros((len(cell_idx), G), np.float32)
    with h5py.File(SETTY, "r") as f:
        X = f["X"]; ip = X["indptr"][:]; ind = X["indices"]; dat = X["data"]
        for r, i in enumerate(cell_idx):
            s, e = int(ip[i]), int(ip[i + 1])
            out[r, ind[s:e]] = dat[s:e]
    return out


def _digitize(x, bins, rng):
    left = np.digitize(x, bins); right = np.digitize(x, bins, right=True)
    return np.ceil(rng.random(len(x)) * (right - left) + left).astype(np.int64)


def tokenize(expr, gene_names, vocab, rng):
    """Baseline tokens: top-1200 expressed genes, 51-bin quantile values (extract_scgpt_binned.py convention)."""
    nz = np.where(expr > 0)[0]
    ids, val = [], []
    for idx in nz:
        g = gene_names[idx]
        if g in vocab:
            ids.append(vocab[g]); val.append(expr[idx])
    ids = np.array(ids, np.int64); val = np.array(val, np.float32)
    o = np.argsort(-val)[:MAX_SEQ]
    ids, val = ids[o], val[o]
    bins = np.quantile(val, np.linspace(0, 1, N_BINS - 1))
    return ids, _digitize(val, bins, rng).astype(np.float32)


def build_model(vocab):
    sys.path.insert(0, SCGPT_REPO)
    import scgpt  # noqa
    from scgpt.model.model import TransformerModel
    m = TransformerModel(ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS, d_hid=D_MODEL, nlayers=N_LAYERS,
                         vocab=vocab, dropout=0.2, pad_token="<pad>", pad_value=PAD_VALUE,
                         input_emb_style="continuous", use_fast_transformer=False, do_mvc=False, do_dab=False,
                         use_batch_labels=False, cell_emb_style="avg-pool", n_cls=1)
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.eval()


class Embedder:
    """Batched scGPT L11 token states -> (sum over non-pad tokens, the state at one chosen token, n_tokens).

    Returning the SUM plus one token's state (rather than just the mean) is what lets us build the two scores:
      DIRECT   pool over ALL tokens                      = sum / n
      INDIRECT pool over all tokens EXCEPT the perturbed = (sum - h[p]) / (n - 1)
    The indirect pool is the one that matters -- see the note at the screen loop.
    """

    def __init__(self, model, pad_id, dev):
        self.m, self.pad_id, self.dev = model, pad_id, dev
        self.buf = {}
        self.h = model.transformer_encoder.layers[LAYER].register_forward_hook(
            lambda mod, inp, out: self.buf.__setitem__("h", out.detach()))

    def __call__(self, token_sets, excl_idx=None):
        """token_sets: list of (ids[n], vals[n]); excl_idx[j] = token position to also return (or -1).
        Returns (S[B,512] sum over non-pad, Hx[B,512] state at excl_idx, N[B] token counts)."""
        B = len(token_sets)
        S = np.zeros((B, D_MODEL), np.float32)
        Hx = np.zeros((B, D_MODEL), np.float32)
        N = np.zeros(B, np.int64)
        if excl_idx is None:
            excl_idx = [-1] * B
        for a in range(0, B, BATCH):
            chunk = token_sets[a:a + BATCH]
            L = max(len(t[0]) for t in chunk)
            gi = np.full((len(chunk), L), self.pad_id, np.int64)
            gv = np.full((len(chunk), L), PAD_VALUE, np.float32)
            mk = np.ones((len(chunk), L), bool)
            for j, (ids, vals) in enumerate(chunk):
                n = len(ids)
                gi[j, :n] = ids; gv[j, :n] = vals; mk[j, :n] = False
                N[a + j] = n
            with torch.no_grad():
                self.buf.clear()
                self.m._encode(src=torch.tensor(gi).to(self.dev),
                               values=torch.tensor(gv).to(self.dev),
                               src_key_padding_mask=torch.tensor(mk).to(self.dev))
                h = self.buf["h"].float()                                  # (B, L, 512)
                w = torch.tensor(~mk).to(self.dev).float().unsqueeze(-1)   # (B, L, 1) non-pad
                s = (h * w).sum(1)
                # gather the excluded token's state ON DEVICE -- copying the full (B,L,512) states to CPU here
                # was a ~10x slowdown in the pilot
                p = torch.tensor([max(0, excl_idx[a + j]) for j in range(len(chunk))]).to(self.dev)
                hx = h[torch.arange(len(chunk), device=self.dev), p]
            S[a:a + len(chunk)] = s.cpu().numpy()
            Hx[a:a + len(chunk)] = hx.cpu().numpy()
        return S, Hx, N


# ------------------------------------------------------------------ metrics
def auroc(scores, pos_mask):
    """Rank-based AUROC of positives vs the rest (ties averaged)."""
    s = np.asarray(scores, float); y = np.asarray(pos_mask, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auroc_perm_p(scores, pos_mask, n=2000, seed=SEED):
    obs = auroc(scores, pos_mask)
    rng = np.random.default_rng(seed)
    y = np.asarray(pos_mask, bool)
    null = [auroc(scores, rng.permutation(y)) for _ in range(n)]
    return obs, float((np.sum(np.array(null) >= obs) + 1) / (n + 1))


def main(n_src=64):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    rng = np.random.default_rng(SEED)
    vocab = json.load(open(SCGPT_VOCAB)); pad_id = vocab["<pad>"]
    genes, G = load_setty()
    gsym = {g: i for i, g in enumerate(genes)}

    # ---- TF universe (external DB), restricted to scGPT vocab AND the Setty gene panel
    tfs_all = sorted({l.split("\t")[0] for l in open(TF_DB).read().splitlines()[1:] if l.strip()})
    tfs = [t for t in tfs_all if t in vocab and t in gsym]
    tf_ids = np.array([vocab[t] for t in tfs], np.int64)
    drivers = {f: [d for d in v if d in tfs] for f, v in DRIVERS.items()}
    print(f"TF universe: {len(tfs_all)} in DB -> {len(tfs)} in scGPT vocab & Setty panel")
    print("drivers kept: " + " | ".join(f"{f}:{len(v)}" for f, v in drivers.items()))

    # ---- embedding space + fate directions (identical construction to native_decode.py)
    z = np.load(EMB, allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int); clusters = z["clusters"]
    ok = np.isfinite(y); emb, y, ci, clusters = emb[ok], y[ok], ci[ok], clusters[ok]
    mean = emb.mean(0)
    pca = PCA(min(DIM, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    wsc = StandardScaler().fit(pca.transform(emb - mean))
    to_z = lambda E: wsc.transform(pca.transform(E - mean))          # noqa: E731  raw emb -> whitened 20-PC
    Xz = to_z(emb)
    yz = zscore(y)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    clu_tr = clusters[tr]

    src_pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]              # held-out progenitors
    src_sel = src_pool[rng.choice(len(src_pool), min(n_src, len(src_pool)), replace=False)]
    src_c = Xz[src_sel].mean(0)
    print(f"source progenitors: {len(src_sel)} (of {len(src_pool)} held-out)")

    # on-manifold fate direction AT EACH SOURCE CELL (the exact field the validated steer follows)
    dirs = {}
    for f, cls in FATE_CLUSTERS.items():
        m = np.isin(clu_tr, cls)
        if m.sum() < 20:
            continue
        w_f = unit(Xz[tr][m].mean(0) - src_c)
        D = project_constant(Xz[tr], w_f)(Xz[src_sel])
        dirs[f] = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    # *** COMMON-MODE REMOVAL (found necessary in the n=4 pilot; same nuisance as in native_decode.py) ***
    # All four fate directions share a large component -- "generic forward differentiation" -- so cos(shift, d_f)
    # is dominated by that shared axis, not by the branch. Symptom in the pilot: the top-10 nominated TFs were
    # near-IDENTICAL across fates and lymphoid drivers won under EVERY direction (diagonal 1/4).
    # The branch-SPECIFIC direction is d_f orthogonalised against the mean of the four -- literally the
    # bifurcation contrast ("toward ery RATHER THAN the average fate"). The identical correction is applied to
    # the DE baseline below, so model and baseline are judged on the same footing.
    fl0 = list(dirs)
    Dmean = np.mean([dirs[f] for f in fl0], axis=0)
    for f in fl0:
        S = dirs[f] - Dmean
        dirs[f + "|spec"] = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)

    w_rand = unit(rng.standard_normal(Xz.shape[1]))                  # negative control direction
    Dr = project_constant(Xz[tr], w_rand)(Xz[src_sel])
    dirs["_random"] = Dr / (np.linalg.norm(Dr, axis=1, keepdims=True) + 1e-12)

    # ---- baseline tokens for the source cells
    expr = sparse_rows(ci[src_sel], G)
    base_tok = [tokenize(expr[i], genes, vocab, np.random.default_rng(1000 + i)) for i in range(len(src_sel))]

    model = build_model(vocab).to(dev)
    embed = Embedder(model, pad_id, dev)

    # baseline: keep the FULL per-token L11 states so we can re-pool while excluding any chosen token
    base_H = [None] * len(base_tok)                    # per cell, (n_i, 512) token states (64 cells: cheap)
    for i, (ids, vals) in enumerate(base_tok):
        embed.buf.clear()
        with torch.no_grad():
            n = len(ids)
            model._encode(src=torch.tensor(ids)[None].to(dev),
                          values=torch.tensor(vals)[None].to(dev),
                          src_key_padding_mask=torch.zeros(1, n, dtype=torch.bool).to(dev))
            base_H[i] = embed.buf["h"][0, :n].float().cpu().numpy()
    base_S = np.stack([H.sum(0) for H in base_H])
    base_N = np.array([len(H) for H in base_H])
    base_emb = base_S / base_N[:, None]
    base_z = to_z(base_emb.astype(np.float64))

    r = np.corrcoef(base_emb.ravel(), emb[src_sel].ravel())[0, 1]
    print(f"[sanity] batched embedder vs cached scgptbin embeddings: r={r:.4f} "
          f"(binning is stochastic per-cell, so <1 is expected; must be >0.95)")

    # ---- the screen -------------------------------------------------------------------------------------
    # TWO scores per TF, from the SAME forward passes:
    #   DIRECT   = shift of the full mean-pooled cell embedding. But scGPT's cell embedding is a mean over
    #              ~1200 gene tokens, so a large part of this shift is just the PERTURBED TOKEN ITSELF moving
    #              => cos(shift, fate_dir) largely asks "is gene g's own token marker-like for this fate?"
    #              That is the co-expression confound wired into the mechanism, not merely into the statistics.
    #   INDIRECT = shift of the pool over EVERY TOKEN EXCEPT g (same token set before and after). Nothing of g's
    #              own representation enters. What remains is what the model PROPAGATED: how forcing g up changed
    #              the representation of the OTHER genes. That is regulation, not co-expression -- the real test.
    # Costs no extra forwards: both come from the same hidden states.
    print(f"\nscreening {len(tfs)} TFs x {len(src_sel)} cells = {len(tfs)*len(src_sel):,} forward passes...")
    align = {f: np.zeros(len(tfs)) for f in dirs}
    align_ind = {f: np.zeros(len(tfs)) for f in dirs}
    shift_norm = np.zeros(len(tfs)); shift_norm_ind = np.zeros(len(tfs))
    t0 = time.time()
    for k, (tf, tid) in enumerate(zip(tfs, tf_ids)):
        pert, ex, base_excl = [], [], np.zeros((len(base_tok), D_MODEL), np.float32)
        for i, (ids, vals) in enumerate(base_tok):
            hit = np.where(ids == tid)[0]
            if len(hit):                                  # expressed -> force to the top bin, same length
                p = int(hit[0])
                v2 = vals.copy(); v2[p] = TOP_BIN
                pert.append((ids, v2)); ex.append(p)
                base_excl[i] = (base_S[i] - base_H[i][p]) / (base_N[i] - 1)
            else:                                         # not expressed -> append the token at the top bin
                p = len(ids)
                pert.append((np.append(ids, tid), np.append(vals, TOP_BIN).astype(np.float32))); ex.append(p)
                base_excl[i] = base_S[i] / base_N[i]      # baseline has no g: its pool already excludes it
        S, Hx, N = embed(pert, ex)
        pert_all = S / N[:, None]
        pert_excl = (S - Hx) / (N - 1)[:, None]           # same token set as base_excl, by construction

        for tag, dzr in (("d", to_z(pert_all.astype(np.float64)) - base_z),
                         ("i", to_z(pert_excl.astype(np.float64)) - to_z(base_excl.astype(np.float64)))):
            nn = np.linalg.norm(dzr, axis=1, keepdims=True) + 1e-12
            if tag == "d":
                shift_norm[k] = float(nn.mean())
            else:
                shift_norm_ind[k] = float(nn.mean())
            for f, D in dirs.items():
                v = float(((dzr / nn) * D).sum(1).mean())
                (align if tag == "d" else align_ind)[f][k] = v
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(tfs)} TFs | {el/60:.1f} min | eta {(len(tfs)-k-1)*el/(k+1)/60:.1f} min", flush=True)

    # ---- co-expression baseline: differential expression, fate cluster vs source progenitors
    E_all = sparse_rows(ci, G)
    tot = E_all.sum(1, keepdims=True); tot[tot == 0] = 1
    E_all = np.log1p(E_all / tot * 1e4)
    src_mean = E_all[src_sel].mean(0)
    de = {}
    for f in fl0:
        m = np.isin(clusters, FATE_CLUSTERS[f])
        de[f] = np.array([E_all[m][:, gsym[t]].mean() - src_mean[gsym[t]] for t in tfs])
    de_mean = np.mean([de[f] for f in fl0], axis=0)                  # SAME common-mode removal as the directions
    for f in fl0:
        de[f + "|spec"] = de[f] - de_mean

    # ---- evaluate
    out = dict(model="scgptbin", dataset="setty", n_src=int(len(src_sel)), n_tfs=len(tfs),
               tf_db="dorothea_trrust_union_immune", sanity_r=float(r), drivers=drivers, fates={})

    def block(title, note, sc, keys, store):
        print("\n" + "=" * 110)
        print(title); print(note)
        print("=" * 110)
        print(f"  {'fate':<14} {'AUROC align':>12} {'p':>7} | {'AUROC de':>9} | {'align|de resid':>15} {'p':>7} | "
              f"{'rand-dir':>9} | {'r(align,de)':>11}")
        for f in keys:
            base = f.split("|")[0]
            pos = np.array([t in drivers[base] for t in tfs])
            a_al, p_al = auroc_perm_p(sc[f], pos)
            a_de = auroc(de[f], pos)
            a_rd = auroc(sc["_random"], pos)
            res = sc[f] - LinearRegression().fit(de[f][:, None], sc[f]).predict(de[f][:, None])
            a_rs, p_rs = auroc_perm_p(res, pos)
            rr = float(np.corrcoef(sc[f], de[f])[0, 1])
            order = np.argsort(-sc[f])
            # (i) does the model ADD anything on top of DE?  (ii) can it find drivers where DE is UNINFORMATIVE?
            zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)               # noqa: E731
            a_comb = auroc(zs(de[f]) + zs(sc[f]), pos)
            lo = np.abs(de[f]) <= np.median(np.abs(de[f]))                  # DE-uninformative stratum
            a_lo_m = auroc(sc[f][lo], pos[lo]); a_lo_d = auroc(de[f][lo], pos[lo])
            store[f] = dict(auroc_align=a_al, p_align=p_al, auroc_de=a_de, auroc_align_resid_de=a_rs,
                            p_align_resid_de=p_rs, auroc_random_dir=a_rd, r_align_de=rr,
                            auroc_de_plus_align=a_comb, n_drivers_in_lowDE=int(pos[lo].sum()),
                            auroc_lowDE_align=a_lo_m, auroc_lowDE_de=a_lo_d,
                            top20=[tfs[i] for i in order[:20]],
                            driver_ranks={t: int(np.where(order == tfs.index(t))[0][0]) + 1
                                          for t in drivers[base]})
            print(f"  {f:<14} {a_al:>12.3f} {p_al:>7.3f} | {a_de:>9.3f} | {a_rs:>15.3f} {p_rs:>7.3f} | "
                  f"{a_rd:>9.3f} | {rr:>11.3f}  || DE+model={a_comb:.3f}  "
                  f"lowDE(n_drv={int(pos[lo].sum())}): model={a_lo_m:.3f} de={a_lo_d:.3f}")
        for f in keys:
            base = f.split("|")[0]
            print(f"    ->{f:<14} " + ", ".join(f"**{t}**" if t in drivers[base] else t
                                                for t in store[f]["top20"][:10]))
            print("       known drivers at ranks: "
                  + ", ".join(f"{t}={rk}" for t, rk in sorted(store[f]["driver_ranks"].items(),
                                                              key=lambda kv: kv[1])))

    out["direct"], out["indirect"] = {}, {}
    spec = [f + "|spec" for f in fl0]
    block("A. DIRECT score, RAW fate direction — contaminated twice over (perturbed token's own shift; "
          "shared differentiation axis)",
          "   shown only for transparency.", align, fl0, out["direct"])
    block("B. DIRECT score, FATE-SPECIFIC direction — common mode removed from BOTH model and DE baseline",
          "   still includes gene g's own token => still largely a marker readout by construction.",
          align, spec, out["direct"])
    block("C. INDIRECT score, FATE-SPECIFIC direction — *** THE REAL TEST ***",
          "   pool EXCLUDES the perturbed token: only what the model PROPAGATED to other genes. "
          "Regulation, not co-expression.", align_ind, spec, out["indirect"])

    # ---- fate x driver-set diagonal
    def diagonal(sc, keys, tag):
        A = np.array([[auroc(sc[f], np.array([t in drivers[g] for t in tfs])) for g in fl0] for f in keys])
        hits = sum(fl0[int(np.argmax(A[i]))] == fl0[i] for i in range(len(keys)))
        print(f"\n  fate-direction x driver-set AUROC [{tag}] (row=direction aimed at, col=whose drivers):")
        print("     dir            " + "".join(f"{g:>10}" for g in fl0))
        for i, f in enumerate(keys):
            star = fl0[int(np.argmax(A[i]))]
            print(f"     ->{f:<12} " + "".join(f"{A[i, j]:>10.3f}" for j in range(len(fl0)))
                  + f"   top={star}{'  ✓' if star == fl0[i] else ''}")
        print(f"     own drivers score highest in {hits}/{len(keys)} fates")
        return dict(matrix=A.tolist(), fates=fl0, hits=f"{hits}/{len(keys)}")

    out["diagonal_direct_spec"] = diagonal(align, spec, "DIRECT, fate-specific")
    out["diagonal_indirect_spec"] = diagonal(align_ind, spec, "INDIRECT, fate-specific — the real test")
    print("     [NB ery/mega share GATA1/TAL1/NFE2/GATA2 by real biology (shared MEP program) -- expect bleed]")

    json.dump(out, open(os.path.join(RESULTS, "perturb_invert_scgptbin_setty.json"), "w"), indent=1)
    print(f"\n[done] {(time.time()-t0)/60:.1f} min -> results/perturb_invert_scgptbin_setty.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
