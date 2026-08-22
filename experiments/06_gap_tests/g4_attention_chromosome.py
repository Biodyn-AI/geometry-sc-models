"""G4 - is chromosome information actually carried by ATTENTION?

The chromosome result is the corpus's strongest positive, and its mechanism claim is that the
variable is "carried transiently through attention": steering half a cell's gene tokens raises
chromosome-C probability at the OTHER half, and the effect is present at the input embedding, ~26x
weaker one layer in, and gone by layer 5. But that is entirely INFERRED from a behavioural
split-half design. No attention map or head has ever been measured for chromosome. The coverage
grid lists chromosome x attention as empty.

This measures it directly, two ways.

(A) STATIC. In ordinary forward passes, is attention between two gene tokens higher when they sit on
    the same chromosome? Reported per layer and per head.

    The confound that would fake this: Llama uses RoPE, so attention depends on |i-j| in sequence
    position, and genes are ordered by expression rank. Same-chromosome genes co-express, so they
    land at similar ranks, so they would attend more WITHOUT any chromosome representation. The test
    therefore matches on |rank_i - rank_j|: same-chr and different-chr pairs are compared only
    within the same positional-distance bin. A label-permutation null over the genes actually
    present in each cell holds the attention matrix and the gene set fixed.

(B) CAUSAL. Under a chromosome-C push applied to a random half of the tokens, does attention FROM
    the untouched read half TO the pushed half change specifically for chromosome-C read genes,
    relative to a norm-matched random push? This is the attention-level version of the behavioural
    experiment, on the same cells and the same directions.

Reads the model with eager attention (SDPA silently drops attention weights).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import json, os, pickle, sys
import numpy as np
import torch
import h5py

GM = f"{_DATA}/codebase/route_genemanifold"
sys.path.insert(0, GM)
import gm_lib as G           # noqa: E402
import steer_lib as SL       # noqa: E402
from genome_wide import coords, AUTOSOMES  # noqa: E402
from steer_propagation import load_cells, chromosome_spec  # noqa: E402

OUT = (f"{_DATA}/manifolds/gaps/results")
N_CELLS, SEED, N_RAND = 24, 0, 3
ALPHA = 2.0
DIST_BINS = [(1, 2), (3, 8), (9, 32), (33, 128), (129, 512)]


def attn_maps(st, ids, delta=None, push_mask=None):
    """Forward with eager attention. Optionally add `delta` to the pushed input embeddings.
    Returns attention (n_layers, n_heads, T, T) averaged over nothing -- full maps."""
    x = torch.tensor(ids, dtype=torch.long, device=st.device)[None, :]
    hooks = []
    if delta is not None:
        emb = st.model.get_input_embeddings()

        def hook(mod, inp, out):
            m = torch.tensor(push_mask, device=out.device)[None, :, None]
            return out + m * torch.tensor(delta, device=out.device, dtype=out.dtype)[None, None, :]
        hooks.append(emb.register_forward_hook(hook))
    try:
        with torch.no_grad():
            o = st.model(x, output_attentions=True, use_cache=False)
        A = torch.stack([a[0] for a in o.attentions]).float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return A                                        # (L, H, T, T)


def static_enrichment(A, chrom, rng, n_perm=20):
    """Same-chr minus different-chr attention, matched on |i-j|, per layer and head.

    chrom: array of length T; '' for tokens with no chromosome label.
    """
    L, H, T, _ = A.shape
    i, j = np.triu_indices(T, 1)
    lab = chrom[i], chrom[j]
    ok = (lab[0] != "") & (lab[1] != "")
    i, j = i[ok], j[ok]
    same = chrom[i] == chrom[j]
    dist = np.abs(i - j)
    # symmetrise: attention i->j and j->i both count
    def stat(sm):
        per_lh = np.zeros((L, H))
        for lo, hi in DIST_BINS:
            b = (dist >= lo) & (dist <= hi)
            if b.sum() < 40 or sm[b].sum() < 10 or (~sm[b]).sum() < 10:
                continue
            a = A[:, :, i[b], j[b]] + A[:, :, j[b], i[b]]
            per_lh += a[:, :, sm[b]].mean(-1) - a[:, :, ~sm[b]].mean(-1)
        return per_lh / max(1, len(DIST_BINS))
    real = stat(same)
    null = np.stack([stat(rng.permutation(same)) for _ in range(n_perm)])
    return real, null.mean(0), null.std(0)


def main(model="217m", n_cells=N_CELLS):
    os.makedirs(OUT, exist_ok=True)
    st = SL.Steerer(model_dir=SL.MODELS[model])
    tok2cat, _ = chromosome_spec(st)
    print(f"model {model}: {st.model.config.num_hidden_layers} layers x "
          f"{st.model.config.num_attention_heads} heads, {len(tok2cat)} tokens with a chromosome")

    # force eager attention so the maps are actually returned
    st.model.config._attn_implementation = "eager"
    for m in st.model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"

    seqs = load_cells(st, n_cells, SEED)
    print(f"cells: {len(seqs)}, mean {np.mean([len(s) for s in seqs]):.0f} tokens\n")
    rng = np.random.default_rng(SEED)

    L = st.model.config.num_hidden_layers
    H = st.model.config.num_attention_heads
    acc_real = np.zeros((L, H)); acc_null = np.zeros((L, H)); acc_sd = np.zeros((L, H)); n_ok = 0

    print("(A) STATIC: same-chromosome attention enrichment, matched on |rank distance|")
    for ci, ids in enumerate(seqs):
        ids = np.asarray(ids)[: 320]                       # cap length: attention is T^2
        chrom = np.array([tok2cat.get(int(t), "") for t in ids])
        if (chrom != "").sum() < 40:
            continue
        A = attn_maps(st, ids)
        r, nm, ns = static_enrichment(A, chrom, rng)
        acc_real += r; acc_null += nm; acc_sd += ns; n_ok += 1
        if (ci + 1) % 6 == 0:
            print(f"  {ci+1}/{len(seqs)} cells", flush=True)
    acc_real /= n_ok; acc_null /= n_ok; acc_sd /= max(n_ok, 1)
    z = (acc_real - acc_null) / np.maximum(acc_sd, 1e-12)

    print(f"\n  scored {n_ok} cells")
    print("  layer  mean_excess      max_head_z   n_heads |z|>3")
    for l in range(L):
        print(f"  {l:5d}  {acc_real[l].mean() - acc_null[l].mean():+.3e}   "
              f"{z[l][np.argmax(np.abs(z[l]))]:+8.2f}   {(np.abs(z[l]) > 3).sum():d}/{H}")
    best = np.unravel_index(np.argmax(np.abs(z)), z.shape)
    print(f"\n  strongest head: L{best[0]}H{best[1]}  excess {acc_real[best]:+.3e}  z {z[best]:+.2f}")

    out = {"model": model, "n_cells_scored": int(n_ok), "n_layers": L, "n_heads": H,
           "static": {"excess": acc_real.tolist(), "null_mean": acc_null.tolist(),
                      "null_sd": acc_sd.tolist(), "z": z.tolist(),
                      "best_head": [int(best[0]), int(best[1])],
                      "best_z": float(z[best]),
                      "n_heads_z_gt3": int((np.abs(z) > 3).sum()),
                      "n_heads_total": int(L * H)}}
    json.dump(out, open(f"{OUT}/g4_attention_chromosome.json", "w"), indent=1)
    print(f"\nwrote {OUT}/g4_attention_chromosome.json")


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else "217m",
         n_cells=int(sys.argv[2]) if len(sys.argv) > 2 else N_CELLS)
