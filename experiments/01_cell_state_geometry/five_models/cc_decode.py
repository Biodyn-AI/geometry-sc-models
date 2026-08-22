"""Does the steered path sweep the real cell cycle -- according to scGPT itself?

The steering result (cc_steering.py) is judged by a kNN regressor in embedding space. That could in principle
be geometric bookkeeping with no gene-level content. Here we decode every point of the path back to gene
expression through scGPT's OWN pretrained MVC head -- zero parameters fit on K562 -- and ask whether the cell
actually cycles.

The head (external/scGPT/scgpt/model/model.py:972-991, arch_style="inner product") is LINEAR in the cell
embedding, so it collapses to a fixed matrix (native_decode.py:62-82):

    M[g, :] = W.weight @ sigmoid(gene2query(LN(embedding[g])))          M: (n_genes, 512)
    pred_expr(cell) = M @ cell_emb

Every weight comes from best_model.pt. Our cached `emb` IS scGPT's cell embedding (mean-pooled L11 residual).

HEADLINE, SO IT IS NOT BURIED: **the native head FAILS on this substrate.** It passes native_decode.py's gate
(within-cell gene ranking, Spearman +0.314) but cannot read a cell-cycle CLOCK: decoded phase vs true phase on
held-out REAL cells is only circular r = +0.191. It knows which genes are high WITHIN a cell; it does not track
a given gene ACROSS cells well enough to tell the time (decoded S-score vs true S-score: r = +0.121). So the
circularity-free readout is UNDERPOWERED here and can neither confirm nor refute the steering result at gene
level. We report that rather than dressing up noise. That is what GATE 2 below exists to catch -- and it is a
gate the program should run before any future native-decode claim, since gate 1 alone does not imply it.

TWO DECODERS, therefore, both reported:
  native_mvc    scGPT's own head, zero params fit on K562.  Fails gate 2 -> its numbers are UNINFORMATIVE.
  fitted_ridge  Ridge fit on real TRAIN cells (gene_decode.py's convention). Residual circularity, stated --
                but it reads the clock on held-out real cells at circular r = +0.909, so it is a real
                instrument, and gene_decode.py:21-23 already made the argument for why it is admissible: the
                SAME decoder is applied to every arm, so it cannot know which arm is which and cannot
                manufacture a difference BETWEEN them.

THE CO-EXPRESSION CAVEAT, STATED PLAINLY (PREREGISTRATION.md). Cell-cycle genes are the most tightly
co-expressed module in transcriptomics, so "push along the S direction, S genes go up" is nearly tautological
and is NOT the claim. The claim is the CONTRAST between arms that move the same path length through the same
space with the same projection and the same retraction, and differ in ONE thing -- the direction rule. A
co-expressed module would act identically in all three arms; it cannot explain a difference between them.
Result: local_proj 4.07 laps, fixed_proj 0.22, random_proj 0.00.

ON THE NULL. The obvious null for a circular correlation -- rotating one series by a constant angle -- is a
NO-OP: circ_corr uses sin(a - circmean(a)), so a constant offset cancels and null_r comes back exactly equal to
real_r. (We hit this; it fails silently, looking like p ~ 0.9 rather than crashing.) The null that bites for a
path is a shift along the BUDGET axis: circ_corr_timeshift_p in cc_common. It stays continuous -- unlike the
fate experiment's 4! = 24 assignment permutation, whose p FLOOR is 0.042.

Caveat on reading the table: circ_corr is degenerate for arms that barely move (a near-constant series
correlates trivially with anything), so random_proj can post r = +0.80 at p = 0.0005 while travelling zero laps.
The discriminative statistic is the decoded WINDING, not the correlation.

Run:  ../../.venv/bin/python cc_decode.py [scgptcc]
Out:  results/cc_decode_<model>.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
import h5py
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import (wrap, circ_mean, circ_corr, circ_corr_timeshift_p, unit, prep, emb_path, to_lognorm,
                       build_substrate, REPLOGLE, _load_categorical, S_GENES, G2M_GENES, WAVE, WAVE_ORDER,
                       DIM)  # noqa: E402
from cc_steering import (CircJudge, local_phase_field, project_constant, retraction, steer_path,
                         loop_circumference, K_LOCAL, M_TANGENT, SEED, N_SRC, TAU)  # noqa: E402

RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
MI = f"{_DATA}/biodyn-work/single_cell_mechinterp"
SCGPT_CKPT = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "best_model.pt")
SCGPT_VOCAB = os.path.join(MI, "external", "scGPT_checkpoints", "whole-human", "vocab.json")
N_HVG, N_RAND, TWO_PI = 1500, 5, 2.0 * np.pi


def build_native_head(genes):
    """scGPT's pretrained MVC decoder, collapsed to a matrix. Verbatim native_decode.py:62-82."""
    import torch
    import torch.nn.functional as F
    vocab = json.load(open(SCGPT_VOCAB))
    ck = torch.load(SCGPT_CKPT, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    keep = np.array([i for i, g in enumerate(genes) if g in vocab])
    ids = torch.tensor([vocab[genes[i]] for i in keep], dtype=torch.long)
    with torch.no_grad():
        e = sd["encoder.embedding.weight"][ids]
        e = F.layer_norm(e, (e.shape[1],), sd["encoder.enc_norm.weight"], sd["encoder.enc_norm.bias"])
        q = torch.sigmoid(e @ sd["mvc_decoder.gene2query.weight"].T + sd["mvc_decoder.gene2query.bias"])
        M = q @ sd["mvc_decoder.W.weight"].T
    return M.double().numpy(), keep


def load_expression(cell_idx):
    """True log1p(CP10k) expression. to_lognorm() refuses to re-normalise replogle_concat's already-log X --
    double-normalising it corrupts per-gene variance, hence the HVG set, and silently fails the MVC gate
    (+0.326 -> +0.094 within-cell Spearman). See cc_common.to_lognorm."""
    with h5py.File(REPLOGLE, "r") as f:
        genes = _load_categorical(f["var"], "gene_name_index").astype(str)
        E = np.stack([np.asarray(f["X"][int(i), :], dtype=np.float32) for i in cell_idx])
    return to_lognorm(E), genes


def build_spaces(emb, dim=DIM):
    """20-PC whitened steering space + the inverse map back to the raw 512-D embedding where the head lives."""
    mean = emb.mean(0)
    pca = PCA(min(dim, emb.shape[1], len(emb) - 1), random_state=SEED).fit(emb - mean)
    wsc = StandardScaler().fit(pca.transform(emb - mean))
    Xz = wsc.transform(pca.transform(emb - mean))
    return Xz, (lambda pz: pca.inverse_transform(wsc.inverse_transform(pz)) + mean)


class DecodedPhase:
    """Read a cell-cycle phase OUT of decoded expression, using the same Tirosh recipe as the ground truth.

    Fit on the decoder's output for REAL TRAIN cells: z-score the S+G2M marker columns, PCA(2), atan2. The only
    things fit on K562 here are an unsupervised centring/rotation (mu, sd, PCA basis) -- the decoder M itself
    has zero parameters fit on our data. Handedness and origin are pinned against the true phi on train, since
    atan2 fixes neither.
    """

    def __init__(self, P_train, gsym, phi_train):
        idx = [gsym[g] for g in (S_GENES + G2M_GENES) if g in gsym]
        self.idx = np.array(idx)
        Z = P_train[:, self.idx]
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Zz = (Z - self.mu) / self.sd
        self.pca = PCA(2, random_state=0).fit(Zz)
        raw = self._raw(P_train)
        self.flip = 1.0 if circ_corr(raw, phi_train) >= circ_corr(-raw, phi_train) else -1.0
        self.off = circ_mean(wrap(self.flip * raw - phi_train))

    def _raw(self, P):
        Q = self.pca.transform((P[:, self.idx] - self.mu) / self.sd)
        return np.arctan2(Q[:, 1], Q[:, 0])

    def __call__(self, P):
        return wrap(self.flip * self._raw(P) - self.off)


def program_z(P, panel_idx, mu, sd):
    return {k: ((P[:, i] - mu[i]) / sd[i]).mean(1) for k, i in panel_idx.items()}


def run(model="scgptcc"):
    z = np.load(emb_path(model), allow_pickle=True)
    emb, phi, ci = z["emb"].astype(np.float64), z["phi"].astype(np.float64), z["cell_idx"].astype(int)
    E_full, genes = load_expression(ci)

    M, keep = build_native_head(genes)
    genes_k = genes[keep]
    E = E_full[:, keep]
    gsym = {g: i for i, g in enumerate(genes_k)}
    panel_idx = {k: np.array([gsym[g] for g in v if g in gsym]) for k, v in WAVE.items()}
    panel_idx = {k: v for k, v in panel_idx.items() if len(v) >= 3}

    Xz, to_raw = build_spaces(emb)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(phi)); tr, te = perm[: len(phi) // 2], perm[len(phi) // 2:]

    native = lambda ce: ce @ M.T                                    # noqa: E731
    decode = lambda pz: native(to_raw(pz))                          # noqa: E731

    print(f"\n===== scGPT NATIVE (MVC) cell-cycle decode / {model} =====")
    print(f"  genes decodable by scGPT's head: {len(keep)}/{len(genes)} | wave modules: "
          + ", ".join(f"{k}={len(v)}" for k, v in panel_idx.items()))

    # ---------------------------------------------------------------- GATE (native_decode.py:127-177)
    P_real = native(emb)
    hvg = np.argsort(-E[tr].var(0))[:N_HVG]
    marker_all = np.unique(np.concatenate(list(panel_idx.values())))
    sub = te[rng.choice(len(te), min(300, len(te)), replace=False)]
    within = float(np.mean([spearmanr(P_real[i, hvg], E[i, hvg]).statistic for i in sub]))
    across = np.nan_to_num(np.array([np.corrcoef(P_real[te, j], E[te, j])[0, 1] for j in marker_all]))
    g1 = dict(within_cell_spearman_hvg=within, across_cell_r_markers_median=float(np.median(across)),
              across_cell_r_markers_frac_pos=float((across > 0).mean()))
    gate = (within > 0.15) and (g1["across_cell_r_markers_frac_pos"] > 0.6)
    g1["gate_passed"] = bool(gate)
    print(f"\n  [gate] within-cell gene ranking (Spearman, HVG) = {within:+.3f}   (need > 0.15)")
    print(f"  [gate] across-cell marker tracking: median r = {g1['across_cell_r_markers_median']:+.3f}, "
          f"{g1['across_cell_r_markers_frac_pos']*100:.0f}% positive   (need > 60%)")
    if not gate:
        print("\n  *** GATE FAILED -- scGPT's own head cannot read these embeddings. Reporting nothing else.")
        json.dump(dict(model=model, gate=g1, gate_passed=False),
                  open(os.path.join(RESULTS, f"cc_decode_{model}.json"), "w"), indent=1)
        return
    print("  -> GATE PASSED. The native head is a usable readout.\n")

    # ---------------------------------------------------------------- GATE 2: can it read the CLOCK?
    # The gate above (within-cell gene ranking) is necessary but NOT sufficient. It asks "does the head know
    # which genes are high in this cell?" -- not "does it track a given gene ACROSS cells well enough to tell
    # the time?". Steering a cell cycle needs the latter, and they come apart badly here.
    dphase = DecodedPhase(P_real[tr], gsym, phi[tr])
    agree = circ_corr(dphase(P_real[te]), phi[te])
    clock_ok = abs(agree) > 0.40
    print(f"\n  [gate 2] decoded phase vs TRUE phase on held-out REAL cells: circular r = {agree:+.3f}"
          f"   (need > 0.40)")
    if not clock_ok:
        print(f"  *** GATE 2 FAILED. scGPT's own head cannot read the cell-cycle clock, even on real cells.")
        print(f"      It ranks genes within a cell (rho = {within:+.3f}) but tracks markers across cells only")
        print(f"      weakly (median r = {g1['across_cell_r_markers_median']:+.3f}), and that is what a clock")
        print(f"      needs. So the circularity-free readout is UNDERPOWERED on this substrate -- it cannot")
        print(f"      confirm OR refute the steering result at gene level, and we do not pretend otherwise.")
        print(f"      Falling back to a FITTED decoder below, with the residual circularity stated.\n")
    out_gate2 = dict(decoded_phase_vs_true_real=float(agree), clock_gate_passed=bool(clock_ok))

    judge = CircJudge(Xz[tr], phi[tr])
    d0 = float(np.median(NearestNeighbors(n_neighbors=2).fit(Xz[tr]).kneighbors(Xz[tr])[0][:, 1]))
    h = 0.25 * d0
    T_max = int(np.clip(np.ceil(2.5 * loop_circumference(Xz, phi) / h), 40, 400))

    src_m = te[np.abs(wrap(phi[te])) < np.pi / 4]
    src = Xz[src_m]
    if len(src) > N_SRC:
        src = src[rng.choice(len(src), N_SRC, replace=False)]
    w0 = unit(local_phase_field(Xz[tr], phi[tr])(src.mean(0)[None, :])[0])

    # ------------------------- the SECOND decoder: fitted on real train cells (gene_decode.py's convention)
    # The native head fails gate 2, so on its own it leaves the gene-level question blank. gene_decode.py:21-23
    # already argued the honest fallback: a decoder FIT on real train cells carries a residual circularity, but
    # the load-bearing content is the ARM CONTRAST -- the same decoder is applied to every arm, so it cannot
    # know which arm is which, and it cannot manufacture a difference between them. We report both decoders and
    # label each.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as _SS
    esc = _SS().fit(emb)
    ridge = Ridge(alpha=1e4).fit(esc.transform(emb)[tr], E[tr])
    fitted = lambda ce: ridge.predict(esc.transform(ce))                       # noqa: E731
    R_real = fitted(emb)
    dphase_fit = DecodedPhase(R_real[tr], gsym, phi[tr])
    agree_fit = circ_corr(dphase_fit(R_real[te]), phi[te])
    print(f"  [ref] FITTED decoder (Ridge on train cells; residual circularity, stated):"
          f" decoded phase vs true, held-out real cells: circular r = {agree_fit:+.3f}")

    DECODERS = {
        "native_mvc": dict(pred=lambda pz: decode(pz), dph=dphase, real=P_real,
                           agree=agree, note="scGPT's own pretrained head; ZERO params fit on K562"),
        "fitted_ridge": dict(pred=lambda pz: fitted(to_raw(pz)), dph=dphase_fit, real=R_real,
                             agree=agree_fit, note="Ridge fit on real TRAIN cells; residual circularity"),
    }

    # All three arms get IDENTICAL geometric treatment -- same 20-D space, same step size, same tangent
    # projection, same label-free retraction, same number of steps, same source cells. They differ in ONE
    # thing: the direction rule. That is what makes this a rebuttal to the co-expression objection: whatever a
    # tightly co-expressed cell-cycle module would "trivially" do to the readout, it would do equally in all
    # three arms. Only the operator can explain a difference between them.
    retr = retraction(Xz[tr])
    rdirs = [unit(rng.standard_normal(Xz.shape[1])) for _ in range(N_RAND)]
    arms = {"local_proj":  [local_phase_field(Xz[tr], phi[tr], project_m=M_TANGENT)],   # the method
            "fixed_proj":  [project_constant(Xz[tr], w0)],                              # the program's operator
            "random_proj": [project_constant(Xz[tr], r) for r in rdirs]}                # no phase information

    out = dict(model=model, gate=g1, gate2=out_gate2, gate_passed=True, n_genes_decodable=int(len(keep)),
               fitted_decoder_phase_vs_true_real=float(agree_fit), T_max=T_max, n_src=int(len(src)),
               decoders={})

    paths = {name: [steer_path(src, fn, h, T_max, retract=retr)[0] for fn in fns]
             for name, fns in arms.items()}

    for dname, D in DECODERS.items():
        print(f"\n  ===== readout: {dname}  ({D['note']}) =====")
        print(f"        can it read the clock on REAL held-out cells? circular r = {D['agree']:+.3f}"
              f"  {'OK' if abs(D['agree']) > 0.40 else '<-- TOO WEAK; treat everything below as uninformative'}")
        mu, sd = D["real"][tr].mean(0), D["real"][tr].std(0) + 1e-9
        rec_all = {}
        for name, pth_list in paths.items():
            reps = []
            for pts in pth_list:
                emb_phi = np.array([judge(p) for p in pts])
                dec_phi = np.array([D["dph"](D["pred"](p)) for p in pts])
                cum = lambda P: np.vstack([np.zeros((1, len(src))),                       # noqa: E731
                                           np.cumsum(wrap(np.diff(P, axis=0)), axis=0)])
                progs = {k: np.array([program_z(D["pred"](p), panel_idx, mu, sd)[k].mean() for p in pts])
                         for k in panel_idx}
                reps.append(dict(ae=cum(emb_phi).mean(1), ad=cum(dec_phi).mean(1),
                                 dec_phi=dec_phi, emb_phi=emb_phi, progs=progs))

            ae = np.mean([r["ae"] for r in reps], 0)
            ad = np.mean([r["ad"] for r in reps], 0)
            pr = {k: np.mean([r["progs"][k] for r in reps], 0) for k in panel_idx}
            r_obs, p_ts, r_null = circ_corr_timeshift_p(
                np.hstack([r["emb_phi"] for r in reps]), np.hstack([r["dec_phi"] for r in reps]),
                n=2000, seed=SEED)
            peaks = {k: int(np.argmax(pr[k])) for k in panel_idx}
            seq = sorted(peaks, key=lambda k: peaks[k])
            order_ok = (set(seq) == set(WAVE_ORDER)
                        and any(seq == WAVE_ORDER[i:] + WAVE_ORDER[:i] for i in range(len(WAVE_ORDER))))

            rec_all[name] = dict(
                winding_embedding=float(ae.max() / TWO_PI), winding_decoded=float(ad.max() / TWO_PI),
                advance_decoded=[float(v) for v in ad],
                circ_corr_decoded_vs_steered=r_obs, p_timeshift_null=p_ts, null_mean_r=r_null,
                programs={k: [float(v) for v in pr[k]] for k in panel_idx},
                peak_budget=peaks, peak_order=seq, wave_order_correct=bool(order_ok))
            print(f"    [{name:12s}] embedding says {rec_all[name]['winding_embedding']:+.2f} laps | "
                  f"this readout says {rec_all[name]['winding_decoded']:+.2f} laps"
                  f" | circ-r={r_obs:+.3f} (time-shift null p={p_ts:.4f}, null r={r_null:.3f})")
            print(f"                   marker peaks: " + ", ".join(f"{k}@T{peaks[k]}" for k in seq)
                  + f"  order {'CORRECT' if order_ok else 'WRONG'} (want {'->'.join(WAVE_ORDER)})")
        out["decoders"][dname] = dict(note=D["note"], phase_vs_true_real=float(D["agree"]),
                                      usable=bool(abs(D["agree"]) > 0.40), arms=rec_all)

    print(f"\n  ---- the contrast that IS the result (co-expression cannot explain it) ----")
    for dname in DECODERS:
        A = out["decoders"][dname]["arms"]
        tag = "" if out["decoders"][dname]["usable"] else "   [UNINFORMATIVE: readout cannot read the clock]"
        print(f"    {dname:<13} laps travelled: local_proj {A['local_proj']['winding_decoded']:+.2f}"
              f" | fixed_proj {A['fixed_proj']['winding_decoded']:+.2f}"
              f" | random_proj {A['random_proj']['winding_decoded']:+.2f}{tag}")
    print(f"    All arms move the same path length through the same space. Only the direction rule differs.")

    f = os.path.join(RESULTS, f"cc_decode_{model}.json")
    json.dump(out, open(f, "w"), indent=1)
    print(f"\n[done] -> {f}")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "scgptcc")
