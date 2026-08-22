"""THE "scGPT-ONLY" FIX: MaxToki has its own pretrained gene decoder, and nobody used it.

NATIVE_DECODE_RESULTS.md lists as an honest limit: "One model, one lineage. Geneformer has no comparable
pretrained cell-embedding->expression head, so this circularity-free check is currently scGPT-only." That is the
single biggest caveat on the headline — the whole fate-steering result rests on ONE model's readout.

It is also not true. MaxToki-217M is a LlamaForCausalLM with

    tie_word_embeddings : false                 <- the output head is a SEPARATE, independently-trained tensor
    lm_head.weight      : (20275, 1232)         <- a pretrained hidden-state -> gene-token decoder
    model.norm.weight   : (1232,)               <- the final RMSNorm it consumes

Not one of those parameters was fit on Setty. That is exactly the property that made scGPT's MVC head the right
instrument, in a completely different architecture (causal LM over rank-ordered gene tokens vs masked-expression
transformer). If the fate diagonal reproduces here, the claim stops being "scGPT asserts it" and becomes "two
independently pretrained models, two architectures, each read out through its OWN head, independently assert the
same fate diagonal."

THE READOUT, and the one place it differs from scGPT's:
  scGPT's MVC head is trained to predict a gene's EXPRESSION VALUE from the CELL embedding — so applying it to a
  cell embedding is exactly what it was built for. MaxToki's lm_head is trained to score the NEXT GENE TOKEN from
  a POSITIONAL hidden state. MaxToki tokenizes a cell Geneformer-style — genes rank-ordered by normalised
  expression, descending — so a gene's logit is a monotone proxy for "this gene sits high in this cell's rank
  ordering", i.e. for expression. But pushing a MEAN-POOLED cell state through a positional head is an
  approximation, and it might simply not work.

  That is what the GATE is for, and it is the same gate scGPT had to pass (and initially FAILED, which is how the
  raw-counts/51-bin input-convention bug was caught). We read out at hidden_states[11] — the FINAL block, which
  is what model.norm + lm_head actually consume — not the L8 tap the rest of the project uses for geometry.
  Those per-layer embeddings already exist (route_benchmark's depth sweep, maxtoki217L11_setty.npz).

  If the gate fails we say so and report nothing else. A readout we cannot trust on real cells cannot be trusted
  on steered ones.

Run:  ../../.venv/bin/python maxtoki_native.py [layer]      (default 11 = final block)
Out:  results/maxtoki_native_setty.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, struct
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
from itertools import permutations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from common import (RESULTS, FATE_CLUSTERS, ROOT_CLUSTERS, load_expression, panel_index, N_HVG)  # noqa: E402
from tangent_diagnostic import zscore, unit, steer, SEED, BUDGETS, T_STAR                        # noqa: E402
from local_steering import project_constant                                                      # noqa: E402
from native_decode import build_spaces, program_z                                                # noqa: E402
from fate_matrix import dominance, N_BOOT                                                        # noqa: E402

DATA = f"{_DATA}/branchpoint"
SETUP = f"{_DATA}/maxtoki/setup"
CKPT = os.path.join(SETUP, "MaxToki-217M-HF", "model.safetensors")
TOKEN_DICT = os.path.join(SETUP, "token_dictionary.json")
NAME_ID_PKL = (f"{_DATA}/biodyn-nmi-paper/src/02_cssi_method/"
               "crispri_validation/data/gene_name_id_dict_gc104M.pkl")
TISSUE = "setty"


def read_safetensors(path, names):
    """Pull named tensors out of a safetensors file without loading the model (or torch)."""
    DT = {"F32": np.float32, "F16": np.float16, "BF16": None}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k in names:
            meta = hdr[k]
            dt = DT[meta["dtype"]]
            if dt is None:
                raise SystemExit(f"bf16 tensor {k} — not handled")
            s, e = meta["data_offsets"]
            f.seek(base + s)
            out[k] = np.frombuffer(f.read(e - s), dtype=dt).reshape(meta["shape"]).astype(np.float64)
    return out


def build_maxtoki_head(genes):
    """MaxToki's OWN pretrained decoder as a matrix. Returns (M, keep, rms_w) with

           score(cell) = RMSNorm(cell_emb) @ M.T ,     M[g] = lm_head.weight[token_of(g)]

    Every weight comes from MaxToki pretraining; nothing is fit on Setty. `keep` indexes `genes`.
    """
    W = read_safetensors(CKPT, ["lm_head.weight", "model.norm.weight"])
    lm, rms_w = W["lm_head.weight"], W["model.norm.weight"]         # (20275, 1232), (1232,)
    tok = json.load(open(TOKEN_DICT))                               # ensembl -> token id
    name_id = pickle.load(open(NAME_ID_PKL, "rb"))                  # symbol  -> ensembl

    keep, rows = [], []
    for i, g in enumerate(genes):
        ens = name_id.get(g)
        t = tok.get(ens) if ens else None
        if t is not None and t < lm.shape[0]:
            keep.append(i); rows.append(t)
    keep = np.array(keep)
    return lm[np.array(rows)], keep, rms_w


def rmsnorm(H, w, eps=1e-5):
    return H / np.sqrt((H ** 2).mean(1, keepdims=True) + eps) * w[None, :]


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    p = os.path.join(DATA, f"maxtoki217L{layer:02d}_{TISSUE}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} — needs route_benchmark's depth-sweep embeddings")
    z = np.load(p, allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64)
    ci = z["cell_idx"].astype(int); clu = z["clusters"]
    ok = np.isfinite(y); emb, y, ci, clu = emb[ok], y[ok], ci[ok], clu[ok]

    E_full, genes, _ = load_expression(TISSUE, ci)
    M, keep, rms_w = build_maxtoki_head(genes)
    genes_k, E = genes[keep], E_full[:, keep]
    pi = panel_index(genes_k, TISSUE)
    fates = [f for f, cls in FATE_CLUSTERS[TISSUE].items() if np.isin(clu, cls).sum() >= 20 and f in pi]

    print(f"\n{'='*100}\n=== MaxToki-217M's OWN lm_head as the readout | {TISSUE} | hidden_states[{layer}] "
          f"(final block) | n={len(y)}")
    print(f"  genes decodable by MaxToki's head: {len(keep)}/{len(genes)}   | fates: {', '.join(fates)}")
    print(f"  markers kept: " + ", ".join(f"{k}={len(v)}" for k, v in pi.items()))

    native = lambda ce: rmsnorm(ce, rms_w) @ M.T           # noqa: E731
    P_real = native(emb)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    clu_tr = clu[tr]

    # ---------------------------------------------------------------- THE GATE (identical to NATIVE_DECODE's)
    hvg = np.argsort(-E[tr].var(0))[:N_HVG]
    marker_all = np.unique(np.concatenate(list(pi.values())))
    sub = te[rng.choice(len(te), min(300, len(te)), replace=False)]
    within = float(np.mean([spearmanr(P_real[i, hvg], E[i, hvg]).statistic for i in sub]))
    across = np.nan_to_num(np.array([np.corrcoef(P_real[te, j], E[te, j])[0, 1] for j in marker_all]))
    frac_pos = float((across > 0).mean())
    gate = (within > 0.15) and (frac_pos > 0.6)

    print(f"\n  [gate: can MaxToki's own head read MaxToki's own cell embeddings?]")
    print(f"    within-cell gene ranking (Spearman, HVG, real held-out cells) = {within:+.3f}   (bar > 0.15)")
    print(f"    across-cell marker tracking, fraction in the right direction  = {frac_pos*100:.0f}%   (bar > 60%)")
    print(f"    across-cell marker tracking, median Pearson r                 = {float(np.median(across)):+.3f}")
    out = dict(model="maxtoki217", layer=layer, dataset=TISSUE, decoder="maxtoki_native_lm_head",
               n=int(len(y)), n_genes_decodable=int(len(keep)),
               gate=dict(within_cell_spearman_hvg=within, marker_frac_pos=frac_pos,
                         marker_r_median=float(np.median(across)), gate_passed=bool(gate)))
    if not gate:
        print(f"\n  *** GATE FAILED -> {'PASSED' if gate else 'refusing to report a fate diagonal'}.")
        print("      MaxToki's lm_head is a POSITIONAL next-token head; a mean-pooled cell state is off-convention")
        print("      for it. This is the honest outcome to report, not a readout we cannot trust.")
        json.dump(out, open(os.path.join(RESULTS, f"maxtoki_native_{TISSUE}.json"), "w"), indent=1)
        return out
    print(f"    -> GATE PASSED. MaxToki's pretrained head reads its own embeddings. Proceeding.\n")

    # ---------------------------------------------------------------- the fate diagonal, MaxToki's own head
    Xz, to_raw = build_spaces(emb)
    yz = zscore(y)
    decode = lambda pz: native(to_raw(pz))                 # noqa: E731
    mu, sd = P_real[tr].mean(0), P_real[tr].std(0) + 1e-9
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))

    src_pool = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = Xz[src_pool]
    if len(src) > 200:
        src = src[rng.choice(len(src), 200, replace=False)]
    roots = np.isin(clu_tr, ROOT_CLUSTERS[TISSUE])
    src_c = Xz[tr][roots].mean(0) if roots.sum() >= 20 else src.mean(0)

    per_cell = {}
    for f in fates:
        w_f = unit(Xz[tr][np.isin(clu_tr, FATE_CLUSTERS[TISSUE][f])].mean(0) - src_c)
        pth = steer(src, project_constant(Xz[tr], w_f), 0.25 * d0, BUDGETS)
        p0, p8 = decode(pth[0]), decode(pth[T_STAR])
        per_cell[f] = np.stack([(((p8[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)
                                 - ((p0[:, pi[k]] - mu[pi[k]]) / sd[pi[k]]).mean(1)) for k in fates], axis=1)

    A = np.array([per_cell[f].mean(0) for f in fates])
    Ac = A - A.mean(0, keepdims=True)
    D_obs = dominance(Ac)
    perms = [dominance(Ac[list(pm)]) for pm in permutations(range(len(fates)))]
    p_exact = float(np.mean([d >= D_obs - 1e-12 for d in perms]))
    hits_raw = sum(int(np.argmax(A[i]) == i) for i in range(len(fates)))
    hits_ctr = sum(int(np.argmax(Ac[i]) == i) for i in range(len(fates)))

    boot = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(src), len(src))
        Ab = np.array([per_cell[f][b].mean(0) for f in fates])
        boot.append(dominance(Ab - Ab.mean(0, keepdims=True)))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"  column-centered Δprogram (T0->T8), MaxToki's OWN lm_head:")
    print("     target      " + "".join(f"{c:>11}" for c in fates))
    for i, t in enumerate(fates):
        s = int(np.argmax(Ac[i]))
        print(f"     ->{t:<10}" + "".join(f"{Ac[i, j]:>+11.2f}" for j in range(len(fates)))
              + f"   top={fates[s]}{'  ✓' if s == i else '  ✗'}")
    print(f"\n  targeted fate is top riser: raw {hits_raw}/{len(fates)}, centered {hits_ctr}/{len(fates)}")
    print(f"  diagonal dominance D = {D_obs:+.3f}   bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  exact permutation p = {p_exact:.4f} over {len(perms)} assignments (floor {1/len(perms):.4f})")
    print(f"\n  scGPT's own MVC head, same 5 fates, same steer: 5/5, D=+0.759, p=0.0083.")
    print(f"  MaxToki's own lm_head, independent architecture, independent pretraining: "
          f"{hits_ctr}/{len(fates)}, D={D_obs:+.3f}, p={p_exact:.4f}.")

    out.update(fates=fates, n_fates=len(fates), matrix_raw=A.tolist(), matrix_centered=Ac.tolist(),
               hits_raw=hits_raw, hits_centered=hits_ctr, D=D_obs,
               boot_ci=[float(lo), float(hi)], p_exact=p_exact, n_perms=len(perms),
               p_floor=1.0 / len(perms), n_src=int(len(src)))
    json.dump(out, open(os.path.join(RESULTS, f"maxtoki_native_{TISSUE}.json"), "w"), indent=1)
    print(f"\n[done] -> results/maxtoki_native_{TISSUE}.json")
    return out


if __name__ == "__main__":
    main()
