"""Rerun the gene-decode test with scGPT's OWN expression head -- removing decoder circularity.

GENE_DECODE_RESULTS.md established that the on-manifold steered path decodes to ordered, fate-specific
hematopoiesis. But the decoder there was a Ridge model FIT ON REAL SETTY CELLS. Even though the load-bearing
content (program order, random-direction control, on-vs-off contrast, fate diagonal) does not follow from the
decoder being data-fit, a fitted decoder is a residual circularity: we taught the readout what genes look like.

The strongest version of the test uses a decoder that NEVER SAW OUR DATA. scGPT's pretrained checkpoint carries
one: the MVC / GEPC head (`mvc_decoder`), trained during pretraining to predict a gene's expression FROM THE
CELL EMBEDDING. Reading external/scGPT/scgpt/model/model.py:972-991 (MVCDecoder, arch_style="inner product"):

    query_vecs = sigmoid(gene2query(gene_embs))          # gene_embs = GeneEncoder(token_id) = LN(embedding[g])
    pred_value = W(query_vecs) @ cell_emb                 # (batch, seq_len)

which is LINEAR in cell_emb with pretraining weights, so the whole head collapses to a fixed matrix

    M[g, :] = W.weight @ sigmoid(gene2query(LN(embedding[g])))        M: (n_genes, 512)
    pred_expr(cell) = M @ cell_emb

and our cached `emb` IS scGPT's cell embedding (mean-pooled L11 residual over non-pad gene tokens; matches
model.py:213 avg-pool + extract_scgpt.py). So we can decode any point of a steered path with the model's own
readout, with zero parameters fit on Setty.

WHAT THIS CHANGES: the claim stops being "a decoder we fit says the path sweeps hematopoiesis" and becomes
"scGPT ITSELF says the path sweeps hematopoiesis". The fate diagonal, if it survives, is the model's own
assertion about which genes define each fate.

GATE (checked first, honestly): extract_scgpt.py fed RAW COUNTS into the value encoder, not scGPT's binned
pretraining convention, so the cell embeddings are somewhat off-distribution for the pretrained head. If the
native head cannot decode REAL cells sensibly, it cannot be trusted on steered paths either, and we say so
rather than shipping a broken readout. Stage 1 measures exactly this and refuses to proceed if it fails.

Run:  ../../.venv/bin/python native_decode.py [setty]
Out:  results/native_decode_scgpt_<dataset>.json  (+ printed report)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tangent_diagnostic import path as npz_path, zscore, unit, steer, SEED, DIM, BUDGETS, T_STAR  # noqa: E402
from local_steering import project_constant  # noqa: E402
from gene_decode import PANEL, N_HVG, load_expression  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")

FATE_CLUSTERS = {"ery": ["Ery_1", "Ery_2"], "myeloid": ["Mono_1", "Mono_2"],
                 "lymphoid": ["CLP"], "mega": ["Mega"]}


# ---------------------------------------------------------------- scGPT's own MVC head, as a matrix
def build_native_head(genes):
    """Collapse scGPT's pretrained MVC decoder to M (n_kept, 512) s.t. pred_expr = M @ cell_emb.

    Returns (M, keep) where keep indexes `genes` (the h5ad var order) for genes present in scGPT's vocab.
    Every weight here comes from best_model.pt; nothing is fit on Setty.
    """
    import torch
    import torch.nn.functional as F
    vocab = json.load(open(SCGPT_VOCAB))
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck

    keep = np.array([i for i, g in enumerate(genes) if g in vocab])
    ids = torch.tensor([vocab[genes[i]] for i in keep], dtype=torch.long)

    with torch.no_grad():
        e = sd["encoder.embedding.weight"][ids]                                   # (n, 512) gene token embedding
        e = F.layer_norm(e, (e.shape[1],), sd["encoder.enc_norm.weight"], sd["encoder.enc_norm.bias"])
        q = torch.sigmoid(e @ sd["mvc_decoder.gene2query.weight"].T + sd["mvc_decoder.gene2query.bias"])
        M = q @ sd["mvc_decoder.W.weight"].T                                      # (n, 512)
    return M.double().numpy(), keep


def build_spaces(emb, dim=DIM):
    """Steering space (dim-PC whitened) + map back to the RAW embedding (where the native head lives)."""
    mean = emb.mean(0)
    pca = PCA(min(dim, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    wsc = StandardScaler().fit(pca.transform(emb - mean))
    Xz = wsc.transform(pca.transform(emb - mean))

    def to_raw(pz):                                        # whitened dim-PC point -> raw 512-d cell embedding
        return pca.inverse_transform(wsc.inverse_transform(pz)) + mean
    return Xz, to_raw


def program_z(P, panel_idx, mu, sd):
    """Mean z-scored program score, where z is against the SAME decoder's output on real train cells."""
    return {k: float((((P[:, idx] - mu[idx]) / sd[idx]).mean(1)).mean()) for k, idx in panel_idx.items()}


def run(model="scgptbin", sfx="setty"):
    z = np.load(npz_path(model, sfx), allow_pickle=True)
    emb = z["emb"].astype(np.float64); y = z["pseudotime"].astype(np.float64); ci = z["cell_idx"].astype(int)
    ok = np.isfinite(y); emb, y, ci = emb[ok], y[ok], ci[ok]
    E_full, genes, clusters = load_expression(ci)          # log1p CP10k, h5ad var order

    M, keep = build_native_head(genes)                     # scGPT's own head; keep = genes in scGPT vocab
    genes_k = genes[keep]
    E = E_full[:, keep]                                    # true expression, restricted to decodable genes
    gsym = {s: i for i, s in enumerate(genes_k)}
    panel_idx = {k: np.array([gsym[s] for s in v if s in gsym]) for k, v in PANEL.items()}
    panel_idx = {k: v for k, v in panel_idx.items() if len(v)}

    Xz, to_raw = build_spaces(emb)
    yz = zscore(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]

    native = lambda cell_emb: cell_emb @ M.T               # noqa: E731   raw cell emb -> per-gene prediction
    decode = lambda pz: native(to_raw(pz))                 # noqa: E731   whitened path point -> prediction

    print(f"\n===== scGPT NATIVE (MVC) decode / emb={model} / {sfx} =====")
    print(f"  genes decodable by scGPT's head: {len(keep)}/{len(genes)} | markers kept: "
          + ", ".join(f"{k}={len(v)}" for k, v in panel_idx.items()))

    # ---------------------------------------------------------------- STAGE 1: does the native head work at all?
    # Gate. The head is pretrained; our embeddings came from a raw-count value convention. If the head cannot
    # rank genes within a real cell, it is not a usable readout and we stop.
    P_real = native(emb)                                                        # (n_cells, n_kept)
    var = E[tr].var(0)
    hvg = np.argsort(-var)[:N_HVG]
    marker_all = np.unique(np.concatenate(list(panel_idx.values())))

    # (a) within-cell: does the head rank genes correctly inside a real cell? (the natural MVC check)
    sub = te[rng.choice(len(te), min(300, len(te)), replace=False)]
    within = [spearmanr(P_real[i, hvg], E[i, hvg]).statistic for i in sub]
    within_all = [spearmanr(P_real[i], E[i]).statistic for i in sub]
    # (b) across-cell, per gene: does the head track a gene's variation across cells? (what steering needs)
    across = np.array([np.corrcoef(P_real[te, j], E[te, j])[0, 1] for j in marker_all])
    across_hvg = np.array([np.corrcoef(P_real[te, j], E[te, j])[0, 1] for j in hvg])
    across = np.nan_to_num(across); across_hvg = np.nan_to_num(across_hvg)

    # baseline: the FITTED Ridge decoder, same split -- for an honest side-by-side
    esc = StandardScaler().fit(emb)
    dec = Ridge(alpha=1000.0).fit(esc.transform(emb)[tr], E[tr])
    R = dec.predict(esc.transform(emb)[te])
    ridge_across = float(np.median([r2_score(E[te, j], R[:, j]) for j in marker_all]))

    g1 = dict(within_cell_spearman_hvg=float(np.mean(within)),
              within_cell_spearman_all=float(np.mean(within_all)),
              across_cell_r_markers_median=float(np.median(across)),
              across_cell_r_markers_frac_pos=float((across > 0).mean()),
              across_cell_r_hvg_median=float(np.median(across_hvg)),
              ridge_r2_markers_median=ridge_across)
    print(f"\n  [stage 1: is scGPT's own head a usable readout of THESE embeddings?]")
    print(f"    within-cell gene ranking (Spearman, real cells): HVG={g1['within_cell_spearman_hvg']:+.3f}  "
          f"all-genes={g1['within_cell_spearman_all']:+.3f}")
    print(f"    across-cell per-gene tracking (Pearson r): markers median={g1['across_cell_r_markers_median']:+.3f} "
          f"({g1['across_cell_r_markers_frac_pos']*100:.0f}% positive) | HVG median={g1['across_cell_r_hvg_median']:+.3f}")
    print(f"    [ref] fitted-Ridge marker R2 on same split: {ridge_across:+.3f}")

    # The gate: the head must (i) rank genes within a cell well above chance, and (ii) track markers across cells
    # in the right direction for most markers. (ii) is what steering actually depends on.
    gate_ok = (g1["within_cell_spearman_hvg"] > 0.15) and (g1["across_cell_r_markers_frac_pos"] > 0.6)
    g1["gate_passed"] = bool(gate_ok)
    if not gate_ok:
        print("\n  *** GATE FAILED: scGPT's pretrained head does not decode these embeddings usefully.")
        print("      Most likely cause: extract_scgpt.py fed RAW COUNTS to the value encoder, while scGPT was")
        print("      pretrained on binned values -- the cell embeddings are off-distribution for the head.")
        print("      Reporting this rather than a readout we cannot trust. The Ridge-decoder result in")
        print("      GENE_DECODE_RESULTS.md stands with its stated caveat; removing the circularity needs a")
        print("      re-extraction under scGPT's binning convention.")
        json.dump(dict(model=model, dataset=sfx, stage1=g1, gate_passed=False),
                  open(os.path.join(RESULTS, f"native_decode_{model}_{sfx}.json"), "w"), indent=1)
        return dict(gate_passed=False, stage1=g1)
    print("    -> GATE PASSED: the native head is a usable readout. Proceeding to the steered paths.\n")

    # ---------------------------------------------------------------- STAGE 2: the steered paths, native readout
    # everything below z-scores against the NATIVE head's output on TRAIN REAL cells, so units are self-consistent
    mu, sd = P_real[tr].mean(0), P_real[tr].std(0) + 1e-9
    lo, hi = np.quantile(yz[tr], 1 / 3), np.quantile(yz[tr], 2 / 3)
    # real early->late direction, measured in the NATIVE readout (internally consistent) and in TRUE expression
    real_dir_nat = P_real[tr][yz[tr] >= hi].mean(0) - P_real[tr][yz[tr] <= lo].mean(0)
    real_dir_true = E[tr][yz[tr] >= hi].mean(0) - E[tr][yz[tr] <= lo].mean(0)
    sd_true = E[tr].std(0) + 1e-9
    order = np.argsort(yz[tr]); pts_sorted = yz[tr][order]
    P_sorted, E_sorted = P_real[tr][order], E[tr][order]

    def real_at(pt, A):                                   # nearest-in-pseudotime real mean (window of 60 cells)
        j = np.searchsorted(pts_sorted, pt); a, b = max(0, j - 30), min(len(order), j + 30)
        return A[a:b].mean(0)

    judge = KNeighborsRegressor(15).fit(Xz[tr], yz[tr])
    nn1 = NearestNeighbors(n_neighbors=1).fit(Xz[tr])
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))
    w_lin = unit(Ridge(alpha=10.0).fit(Xz[tr], yz[tr]).coef_)
    clu_tr = clusters[tr]

    src_mask = te[yz[te] <= np.quantile(yz[te], 1 / 3)]
    src = Xz[src_mask]
    if len(src) > 200:
        src = src[rng.choice(len(src), 200, replace=False)]

    # ---- control design (2x2). The old single "random_dir" control changed TWO things at once (random
    # direction AND no projection), so it could not isolate direction-specificity. Cross the two factors:
    #                          projected onto manifold     not projected
    #   developmental dir  ->  on_manifold  (the method)   off_manifold   [isolates the PROJECTION]
    #   random dir         ->  random_proj  [isolates the DIRECTION]      random_dir  [old, confounded]
    # random_proj is the control that matters: same manifold constraint, no developmental information.
    # Averaged over N_RAND draws so the verdict does not hinge on one lucky/unlucky random vector.
    N_RAND = 5
    rdirs = [unit(rng.standard_normal(Xz.shape[1])) for _ in range(N_RAND)]
    proj_tr = lambda w: project_constant(Xz[tr], w)                          # noqa: E731
    fields = {"on_manifold": [proj_tr(w_lin)],
              "off_manifold": [lambda x: np.tile(w_lin, (len(x), 1))],
              "random_proj": [proj_tr(r) for r in rdirs],
              "random_dir": [(lambda r: (lambda x: np.tile(r, (len(x), 1))))(r) for r in rdirs]}

    out = dict(model=model, dataset=sfx, decoder="scgpt_native_mvc", n=int(len(y)), n_src=int(len(src)),
               n_genes_decodable=int(len(keep)), n_rand=N_RAND, gate_passed=True, stage1=g1, rules={})

    print("  [rules: on_manifold = dev dir + projection (the method); off_manifold = dev dir, no projection;")
    print(f"          random_proj = RANDOM dir + projection (isolates direction; mean of {N_RAND}); "
          "random_dir = random, no projection]\n")

    def eval_rule(fn):
        """Program trajectory + geometry for one direction field."""
        pth = steer(src, fn, 0.25 * d0, BUDGETS)
        p0 = decode(pth[0])
        progs = {k: [] for k in panel_idx}
        fid, offr = [], []
        for T in BUDGETS:
            pT = decode(pth[T])
            pz = program_z(pT, panel_idx, mu, sd)
            for k in panel_idx:
                progs[k].append(pz[k])
            ptT = judge.predict(pth[T])
            r_at = np.array([real_at(v, P_sorted) for v in ptT])
            fid.append(float(np.nanmean([np.corrcoef(pT[i, hvg], r_at[i, hvg])[0, 1] for i in range(len(pT))])))
            offr.append(float(nn1.kneighbors(pth[T])[0][:, 0].mean() / d0))
        chg = decode(pth[T_STAR]) - p0
        cs = (chg / sd[None, :])[:, hvg]; rt = (real_dir_true / sd_true)[hvg]
        ct = float(np.mean([(c @ rt) / (np.linalg.norm(c) * np.linalg.norm(rt) + 1e-9) for c in cs]))
        return dict(programs=progs, real_fidelity=fid, off_ratio=offr, dev_axis_cosine_vs_true=ct)

    for name, fns in fields.items():
        reps = [eval_rule(f) for f in fns]
        rec = dict(programs={k: [float(np.mean([r["programs"][k][t] for r in reps])) for t in range(len(BUDGETS))]
                             for k in panel_idx},
                   real_fidelity_T8=float(np.mean([r["real_fidelity"][T_STAR] for r in reps])),
                   off_ratio_T8=float(np.mean([r["off_ratio"][T_STAR] for r in reps])),
                   dev_axis_cosine_vs_true=float(np.mean([r["dev_axis_cosine_vs_true"] for r in reps])))
        # differentiation contrast: does a COMMITTED program rise while stem falls? Robust to the common-mode
        # deflation that any large embedding move induces in the native head.
        P = rec["programs"]
        dlt = {k: P[k][T_STAR] - P[k][0] for k in panel_idx}
        committed = [k for k in panel_idx if k != "stem"]
        rec["diff_contrast"] = float(max(dlt[k] for k in committed) - dlt["stem"])
        rec["top_riser"] = max(committed, key=lambda k: dlt[k])
        rec["deltas"] = dlt
        out["rules"][name] = rec
        print(f"  [{name:12s}] dev-axis cos vs TRUE={rec['dev_axis_cosine_vs_true']:+.2f} | "
              f"native fidelity@T8={rec['real_fidelity_T8']:+.2f} | off-manifold ratio@T8={rec['off_ratio_T8']:.2f}")
        print(f"     Δprogram T0->T8:  " + "  ".join(f"{k}={dlt[k]:+.2f}" for k in panel_idx)
              + f"   | top riser={rec['top_riser']}, diff-contrast={rec['diff_contrast']:+.2f}")

    # ---------------------------------------------------------------- STAGE 3: the fate diagonal, native readout
    # Any large move in the embedding deflates ALL programs under the native head (common mode, visible above).
    # So the raw diagonal is read alongside a COLUMN-CENTERED one: subtract each program's mean response across
    # the fate targets, leaving the contrast that actually matters -- does program F respond most to target F,
    # RELATIVE to how it responds to the other targets? Significance by exact permutation over target->fate
    # assignments (4! = 24), so a diagonal cannot be talked into existence.
    fate_mat, src_c = {}, src.mean(0)
    for fate, cls in FATE_CLUSTERS.items():
        tgt = np.isin(clu_tr, cls)
        if tgt.sum() < 20:
            continue
        w_f = unit(Xz[tr][tgt].mean(0) - src_c)
        pth = steer(src, project_constant(Xz[tr], w_f), 0.25 * d0, BUDGETS)
        a = program_z(decode(pth[0]), panel_idx, mu, sd)
        b = program_z(decode(pth[T_STAR]), panel_idx, mu, sd)
        fate_mat[fate] = {k: b[k] - a[k] for k in panel_idx}
    out["fate_steering"] = fate_mat

    if len(fate_mat) >= 2:
        from itertools import permutations
        tg = list(fate_mat)                                   # target rows
        pr = [c for c in tg if c in panel_idx]                # program columns (same fate names)
        A = np.array([[fate_mat[t][c] for c in pr] for t in tg])          # rows=target, cols=program
        Ac = A - A.mean(0, keepdims=True)                                  # column-centre: drop common mode

        def dominance(Mx):        # mean over F of [ response of program F to target F  -  to the other targets ]
            n = len(tg)
            return float(np.mean([Mx[i, i] - (Mx[:, i].sum() - Mx[i, i]) / (n - 1) for i in range(n)]))

        D_obs = dominance(A)
        perms = [dominance(A[list(p)]) for p in permutations(range(len(tg)))]
        p_exact = float(np.mean([d >= D_obs - 1e-12 for d in perms]))
        hits_raw = sum(max(pr, key=lambda c: A[i, pr.index(c)]) == t for i, t in enumerate(tg))
        hits_ctr = sum(max(pr, key=lambda c: Ac[i, pr.index(c)]) == t for i, t in enumerate(tg))

        print(f"\n  ---- fate-biased steering with scGPT's OWN head (Δprogram T0->T8) ----")
        print("     RAW (common-mode deflation still in):")
        print("     target    " + "".join(f"{c:>10}" for c in pr))
        for i, t in enumerate(tg):
            star = max(pr, key=lambda c: A[i, pr.index(c)])
            print(f"     ->{t:<8} " + "".join(f"{A[i, j]:>+10.2f}" for j in range(len(pr)))
                  + f"   top={star}{'  ✓' if star == t else ''}")
        print("     COLUMN-CENTERED (common mode removed -- the contrast that matters):")
        print("     target    " + "".join(f"{c:>10}" for c in pr))
        for i, t in enumerate(tg):
            star = max(pr, key=lambda c: Ac[i, pr.index(c)])
            print(f"     ->{t:<8} " + "".join(f"{Ac[i, j]:>+10.2f}" for j in range(len(pr)))
                  + f"   top={star}{'  ✓' if star == t else ''}")
        out.update(fate_hits_raw=f"{hits_raw}/{len(tg)}", fate_hits_centered=f"{hits_ctr}/{len(tg)}",
                   diagonal_dominance=D_obs, diagonal_p_exact=p_exact, n_perms=len(perms))
        print(f"\n     targeted fate is top riser:  raw {hits_raw}/{len(tg)}   centered {hits_ctr}/{len(tg)}")
        print(f"     diagonal dominance D={D_obs:+.3f}   exact permutation p={p_exact:.3f} "
              f"({len(perms)} target->fate assignments; min attainable p={1/len(perms):.3f})")

    json.dump(out, open(os.path.join(RESULTS, f"native_decode_{model}_{sfx}.json"), "w"), indent=1)
    onm, rp = out["rules"]["on_manifold"], out["rules"]["random_proj"]
    print(f"\n  ---- summary (scGPT's own head, no parameters fit on Setty) ----")
    print(f"    THE control (random dir, SAME manifold projection): diff-contrast "
          f"{rp['diff_contrast']:+.2f}  vs  on-manifold {onm['diff_contrast']:+.2f}")
    print(f"    fate diagonal: raw {out.get('fate_hits_raw','n/a')}, centered "
          f"{out.get('fate_hits_centered','n/a')}, p={out.get('diagonal_p_exact', float('nan')):.3f}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "scgptbin",
        sys.argv[2] if len(sys.argv) > 2 else "setty")
    print("\n[done] -> results/native_decode_*.json")
