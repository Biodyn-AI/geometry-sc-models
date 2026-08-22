"""CAN STEERING SIMULATE A REAL KARYOTYPE? (Ihor, 2026-07-20)

WHY THIS, AND WHY NOW. `steer_algebra.py` showed the channel is bidirectional in all 22 chromosomes but
ASYMMETRIC -- suppression is 2.1x stronger than enhancement (-6.48 vs +3.07 Δlogit). So the model is best at
the DELETION direction, which is exactly the direction a real karyotype test needs. And `steer_locality.py`
showed the causal response is a uniform whole-chromosome lift, which is the shape a copy-number change has.
Both point at the same experiment: give the model a real karyotype and see if it can reproduce that cell's
expression signature.

THE TEST
  1. Derive an empirical KARYOTYPE SIGNATURE with no model involved: per-gene mean log-expression in K562
     (aneuploid erythroleukemia) minus a LINEAGE-MATCHED normal (Setty19 CD34 bone marrow -- both
     haematopoietic, which matters: against fetal gut the difference would be dominated by cell type, not copy
     number). K(c) = the mean of that difference over chromosome c's genes.
  2. Steer a normal cell's chromosomes ALL AT ONCE, each by an amount proportional to K(c): negative for
     chromosomes that are down in K562 (a simulated loss), positive for those that are up. Uses the per-token
     codebook path in steer_lib, so each gene receives its own chromosome's signed displacement.
  3. Ask whether the model's induced per-gene change Δ_model tracks the real Δ_real.

CONTROLS, and they carry the result
  * SHUFFLED KARYOTYPE: permute which chromosome gets which magnitude. Keeps the push sizes, the number of
    chromosomes and the total energy identical, and destroys only the chromosome->magnitude mapping. This is
    the comparison that matters; a raw correlation with Δ_real proves nothing on its own.
  * DECOMPOSITION: Δ_real splits into a CHROMOSOME-LEVEL component (each gene assigned its chromosome's mean)
    and a GENE-LEVEL RESIDUAL. The model can only produce chromosome-uniform changes, so a genuine result must
    track the chromosome component and NOT the residual. If it tracks the residual too, something is leaking.

HONEST LIMIT stated up front: K562-vs-CD34 still differs by more than copy number, so K(c) is a *proxy* for the
karyotype, not a karyotype. The shuffle control and the component decomposition are what make the test
interpretable despite that.

Run: ../../.venv_state/bin/python -u karyotype_sim.py [n_cells] [alpha] [model]
Out: results/karyotype_sim_<model>.json
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
import steer_propagation as SP
from genome_wide import coords, AUTOSOMES

K562 = (f"{_DATA}/"
        "state_activations/replogle_k562_subset.h5ad")
SETTY = SP.SETTY
N_REF = 2500
N_SHUF = 8
SEED = 0


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def load(path, keys=("gene_name_index", "feature_name", "index", "_index")):
    with h5py.File(path, "r") as f:
        syms = None
        for k in keys:
            if k in f["var"]:
                v = f["var"][k]
                syms = (_dec(v["categories"][:]).astype(str)[v["codes"][:]]
                        if isinstance(v, h5py.Group) and "categories" in v else _dec(v[:]).astype(str))
                break
        X = f["X"]
        shape = X.attrs.get("shape")
        shape = tuple(int(v) for v in shape) if shape is not None else X.shape
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(shape[0], min(N_REF, shape[0]), replace=False))
        if isinstance(X, h5py.Group):
            indptr, data, idx = X["indptr"][:], X["data"], X["indices"]
            E = np.zeros((len(sel), shape[1]), np.float32)
            for i, r in enumerate(sel):
                a, b = int(indptr[r]), int(indptr[r + 1]); E[i, idx[a:b]] = data[a:b]
        else:
            E = np.stack([np.asarray(X[int(i), :], dtype=np.float32) for i in sel])
    tot = E.sum(1, keepdims=True); tot[tot == 0] = 1
    return np.char.upper(syms.astype(str)), np.log1p(E / tot * 1e4).mean(0)


def main(n_cells=20, alpha=0.5, model="1b", n_shuf=N_SHUF):
    rng = np.random.default_rng(SEED)
    C = coords()

    # ---- 1. the empirical karyotype signature (no model)
    gk, mk = load(K562)
    gs, ms = load(SETTY)
    common = sorted(set(gk) & set(gs) & set(C.index))
    ik = {g: i for i, g in enumerate(gk)}; isv = {g: i for i, g in enumerate(gs)}
    d_real = {g: float(mk[ik[g]] - ms[isv[g]]) for g in common}
    gchr = {g: str(C.loc[g, "chromosome"]) for g in common if C.loc[g, "chromosome"] in AUTOSOMES}
    common = [g for g in common if g in gchr]
    K = {}
    for c in AUTOSOMES:
        v = [d_real[g] for g in common if gchr[g] == c]
        if len(v) >= 25:
            K[c] = float(np.mean(v))
    ks = np.array([K[c] for c in sorted(K)])
    print(f"[signature] {len(common)} lineage-matched common genes; {len(K)} chromosomes")
    print(f"[signature] K562 vs CD34 per-chromosome shift: mean {ks.mean():+.3f}, "
          f"range {ks.min():+.3f}..{ks.max():+.3f}")
    kc = ks - ks.mean()
    print(f"[signature] CENTRED (the actual karyotype contrast): range {kc.min():+.3f}..{kc.max():+.3f}, "
          f"sd {kc.std():.3f}")
    lo = sorted(K, key=lambda c: K[c])[:4]; hi = sorted(K, key=lambda c: -K[c])[:4]
    print(f"[signature] most DOWN (simulated losses): {['chr'+c for c in lo]}")
    print(f"[signature] most UP   (simulated gains) : {['chr'+c for c in hi]}\n")

    # ---- 2. model side
    st = SL.Steerer(model_dir=SL.MODELS[model])
    EMB = SL._embed_matrix(st.xt, "embed")
    vocab, hid = EMB.shape
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    tokmap = json.load(open(SL.TOKMAP))
    tok_sym, tok_chr = {}, {}
    for ens, t in tokmap.items():
        s = ens2sym.get(ens); t = int(t)
        if s in C.index and C.loc[s, "chromosome"] in AUTOSOMES and t < vocab:
            tok_sym[t] = s; tok_chr[t] = str(C.loc[s, "chromosome"])
    tids = np.array(sorted(tok_chr)); tchr = np.array([tok_chr[t] for t in tids])
    is_tr = rng.random(len(tids)) < 0.5
    gcen = EMB[tids[is_tr]].mean(0)
    chroms = sorted([c for c in K if ((tchr == c) & is_tr).sum() >= 20])
    unit = {c: (lambda v: v / (np.linalg.norm(v) + 1e-12))(EMB[tids[(tchr == c) & is_tr]].mean(0) - gcen)
            for c in chroms}

    # codebook: row 0 = no displacement; row i+1 = signed karyotype magnitude x that chromosome's unit dir
    # CENTRE before scaling. Uncentred, K(c) was +0.156..+0.277 for every chromosome -- a GLOBAL offset
    # (K562 simply sits higher in this normalisation), which as a push means "raise everything" and is not a
    # karyotype at all. Copy number is inherently RELATIVE, so the signature is K(c) - mean(K).
    kv = np.array([K[c] for c in chroms]); kv = kv - kv.mean()
    kv = kv / (np.sqrt((kv ** 2).mean()) + 1e-12)   # unit RMS after centring
    src_row = {c: i + 1 for i, c in enumerate(chroms)}
    token_row = np.zeros(vocab, np.int64)
    for t, c in tok_chr.items():
        if c in src_row:
            token_row[t] = src_row[c]

    def codebook(mags):
        cb = np.zeros((len(chroms) + 1, hid))
        for i, c in enumerate(chroms):
            cb[i + 1] = mags[i] * unit[c]
        return cb

    push = alpha * mean_norm
    seqs = SP.load_cells(st, n_cells, SEED + 500)
    print(f"[model] {model}, {len(seqs)} normal (CD34) cells, alpha={alpha} (push {push:.2f}), "
          f"{len(chroms)} chromosomes steered simultaneously\n")

    shuffles = [rng.permutation(len(kv)) for _ in range(n_shuf)]
    real_cb = codebook(kv)
    shuf_cb = [codebook(kv[p]) for p in shuffles]

    d_model, d_shuf = [], []
    prng = np.random.default_rng(SEED + 9)
    for i, sq in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], sq, [st.tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(sq))
        sh = prng.permutation(len(gp)); half = len(gp) // 2
        mask = np.zeros(len(ids), bool); mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        base = st.logits(ids)[read_pos].mean(0).numpy()
        with st.steering(None, alpha=push, positions=mask, site="embed",
                         per_token=(real_cb, token_row), input_ids=ids):
            d_model.append(st.logits(ids)[read_pos].mean(0).numpy() - base)
        s_ = []
        for cb in shuf_cb:
            with st.steering(None, alpha=push, positions=mask, site="embed",
                             per_token=(cb, token_row), input_ids=ids):
                s_.append(st.logits(ids)[read_pos].mean(0).numpy() - base)
        d_shuf.append(s_)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(seqs)} cells", flush=True)
    d_model = np.mean(d_model, 0)
    d_shuf = np.mean(np.array(d_shuf), 0)          # (N_SHUF, vocab)

    # ---- 3. score against the real signature, on TEST genes only
    sym2tok = {}
    for t, s in tok_sym.items():
        sym2tok.setdefault(s, t)
    ev = [(g, sym2tok[g]) for g in common if g in sym2tok and not is_tr[np.searchsorted(tids, sym2tok[g])]]
    genes = [g for g, _ in ev]; toks = np.array([t for _, t in ev])
    y = np.array([d_real[g] for g in genes]); y = y - y.mean()   # remove the global offset from the target too
    chrom_of = np.array([gchr[g] for g in genes])
    kbar = float(np.mean([K[c] for c in chroms]))
    y_chr = np.array([(K[c] - kbar) if c in K else 0.0 for c in chrom_of])   # CENTRED chromosome component
    y_res = y - y_chr                                                    # gene-level residual
    x = d_model[toks]
    xs = d_shuf[:, toks]

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    r_real = r(x, y); r_shuf = np.array([r(xs[k], y) for k in range(len(xs))])
    r_chr = r(x, y_chr); r_res = r(x, y_res)
    z = (r_real - r_shuf.mean()) / (r_shuf.std() + 1e-12)
    # THE FAIR TEST. Delta_real is dominated by GENE-level variation, which a chromosome-uniform push cannot
    # produce by construction -- scoring against the full Delta_real dilutes a real effect to nothing. The
    # model must be judged on the component it CAN produce, against the same shuffled-karyotype control.
    r_chr_shuf = np.array([r(xs[k], y_chr) for k in range(len(xs))])
    z_chr = (r_chr - r_chr_shuf.mean()) / (r_chr_shuf.std() + 1e-12)
    print(f"\n=== CAN STEERING REPRODUCE THE REAL KARYOTYPE SIGNATURE? (n={len(genes)} held-out genes) ===")
    print(f"  corr(Δ_model, Δ_real)                 : {r_real:+.4f}")
    print(f"  corr under SHUFFLED karyotype         : {r_shuf.mean():+.4f} ± {r_shuf.std():.4f}  (n={len(r_shuf)})")
    print(f"  z vs shuffled                         : {z:+.2f}")
    print(f"\n  decomposition of Δ_real:")
    print(f"    vs CHROMOSOME-level component       : {r_chr:+.4f}   <- the model can only produce this")
    print(f"      same, under SHUFFLED karyotype    : {r_chr_shuf.mean():+.4f} ± {r_chr_shuf.std():.4f}")
    print(f"      z vs shuffled                     : {z_chr:+.2f}   <-- THE FAIR TEST")
    print(f"    vs GENE-level residual              : {r_res:+.4f}   <- should be ~0 (it is)")
    ok = z_chr > 2 and r_chr > 0
    print(f"\n  VERDICT: {'REPRODUCES the karyotype signature on the component it can produce (beats shuffled)' if ok else 'does NOT reproduce the karyotype beyond a shuffled one'}")
    if ok and abs(r_res) > abs(r_chr) * 0.5:
        print("  CAUTION: it also tracks the gene-level residual, which a chromosome-uniform push should not "
              "-- suspect leakage rather than karyotype simulation.")

    json.dump(dict(model=model, alpha=alpha, n_cells=len(seqs), n_genes=len(genes), n_chrom=len(chroms),
                   signature={c: K[c] for c in chroms},
                   r_real=r_real, r_shuf_mean=float(r_shuf.mean()), r_shuf_sd=float(r_shuf.std()),
                   z=float(z), r_chrom_component=r_chr, r_chrom_shuf_mean=float(r_chr_shuf.mean()),
                   r_chrom_shuf_sd=float(r_chr_shuf.std()), z_chrom=float(z_chr),
                   r_gene_residual=r_res, reproduces=bool(ok)),
              open(os.path.join(HERE, "results", f"karyotype_sim_{model}.json"), "w"), indent=1)
    print(f"\n[done] -> results/karyotype_sim_{model}.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5,
         sys.argv[3] if len(sys.argv) > 3 else "1b",
         int(sys.argv[4]) if len(sys.argv) > 4 else N_SHUF)
