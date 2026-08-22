"""continuity_test — is the cell-cycle manifold CONTINUOUS, or a few DISCRETE attractor states?

Three tests, increasing in decisiveness.

(1) OCCUPANCY DENSITY. If the model snaps cells onto a few discrete states, the density of decoded phase is
    multimodal with gaps between attractors. Compared against the density of the TRUE phase (itself non-uniform,
    because cells dwell unequally in the phases), so the statistic is *excess* clumping added by the model.

(2) METRIC PROFILE. Mean representation distance as a function of |Δ true phase|. A continuous manifold gives a
    smooth monotone rise; discrete states give a STEP function (flat within a state, jump between states).
    Also reports the local "speed" d(distance)/d(phase) around the loop — non-uniform speed is compatible with
    continuity (it just means the metric is stretched somewhere), whereas a *plateau at zero* is not.

(3) BEHAVIOURAL INTERPOLATION — the decisive one. Linearly interpolate two cells' states,
    h(λ) = (1-λ)h_A + λ h_B, inject h(λ) in place of the last-token residual, and read the MODEL'S OWN
    marker-gene logits. A linear decoder would report intermediate phases *by construction*, so it is useless
    here; the model's nonlinear output is not.
      continuous  -> readout phase moves gradually through intermediate phases as λ sweeps
      discrete    -> readout SNAPS: it stays on A's programme, then flips to B's, with a jump in between
    Statistics: fraction of the total readout movement occurring in the single largest λ-step (1.0 = pure snap,
    ~1/n_steps = perfectly gradual), and the readout entropy at mid-interpolation (a genuine intermediate state
    should not be more committed than the endpoints).

Out: results/continuity_test.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--n-lambda", type=int, default=11)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/continuity_test.json")
    a = ap.parse_args()

    import anndata, torch, scipy.sparse as sp
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    ad = anndata.read_h5ad(a.h5ad)
    theta, _, _, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    n = len(H)

    # out-of-fold decoded phase (honest)
    sc = StandardScaler().fit(H)
    Y = np.column_stack([np.cos(t), np.sin(t)]); P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(H):
        m = Ridge(alpha=1e3).fit(sc.transform(H[tr]), Y[tr]); P[te] = m.predict(sc.transform(H[te]))
    P = P - P.mean(0)
    dec_phase = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)

    res = {}
    # ---- (1) occupancy density -------------------------------------------------
    nb = 36
    hd, _ = np.histogram(dec_phase, bins=nb, range=(0, 2 * np.pi))
    ht, _ = np.histogram(np.mod(t, 2 * np.pi), bins=nb, range=(0, 2 * np.pi))
    ent = lambda h: float(-(h / h.sum() * np.log(h / h.sum() + 1e-12)).sum() / np.log(len(h)))
    def n_modes(h):
        s = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, mode="valid")
        return int(sum(1 for i in range(len(s)) if s[i] > s[i - 1] and s[i] > s[(i + 1) % len(s)]))
    res["occupancy"] = dict(decoded_entropy=ent(hd), true_entropy=ent(ht),
                            decoded_modes=n_modes(hd), true_modes=n_modes(ht),
                            decoded_hist=hd.tolist(), true_hist=ht.tolist(),
                            empty_bins_decoded=int((hd == 0).sum()), empty_bins_true=int((ht == 0).sum()))
    o = res["occupancy"]
    print(f"(1) OCCUPANCY: decoded entropy {o['decoded_entropy']:.3f} ({o['decoded_modes']} modes, "
          f"{o['empty_bins_decoded']} empty bins) vs TRUE {o['true_entropy']:.3f} "
          f"({o['true_modes']} modes, {o['empty_bins_true']} empty)", flush=True)

    # ---- (2) metric profile ----------------------------------------------------
    rng = np.random.default_rng(0)
    i1 = rng.integers(0, n, 200000); i2 = rng.integers(0, n, 200000)
    ok = i1 != i2; i1, i2 = i1[ok], i2[ok]
    dp = circ_dist(t[i1], t[i2])
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    cosd = 1.0 - np.sum(Hn[i1] * Hn[i2], axis=1)
    edges = np.linspace(0, np.pi, 13)
    prof = [float(cosd[(dp >= edges[k]) & (dp < edges[k + 1])].mean()) for k in range(12)]
    res["metric_profile"] = dict(phase_bin_centers=[float(x) for x in (edges[:-1] + edges[1:]) / 2],
                                 mean_cos_distance=prof)
    dif = np.diff(prof)
    res["metric_profile"]["monotone_frac"] = float((dif > 0).mean())
    res["metric_profile"]["largest_step_share"] = float(np.max(np.abs(dif)) / (np.sum(np.abs(dif)) + 1e-12))
    print(f"(2) METRIC: distance-vs-Δphase monotone in {res['metric_profile']['monotone_frac']:.0%} of bins; "
          f"largest single step = {res['metric_profile']['largest_step_share']:.2f} of total rise "
          f"(step function -> ~1.0, smooth -> ~1/12=0.08)", flush=True)

    # ---- (3) behavioural interpolation -----------------------------------------
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    var = np.char.upper(np.asarray(ad.var_names).astype(str)); gm = X.mean(0)
    edges_b = np.linspace(0, 2 * np.pi, a.n_bins + 1)
    b_all = np.clip(np.digitize(np.mod(theta, 2 * np.pi), edges_b) - 1, 0, a.n_bins - 1)
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    mark = {}
    for k in range(a.n_bins):
        m = b_all == k
        ii = []
        if m.sum() >= 20:
            for g in [str(var[i]) for i in np.argsort(-(X[m].mean(0) - gm))[:25]]:
                z = tok(" " + g, add_special_tokens=False)["input_ids"]
                if z:
                    ii.append(z[0])
        mark[k] = np.array(ii, int)
    centers = (edges_b[:-1] + edges_b[1:]) / 2

    lams = np.linspace(0, 1, a.n_lambda)
    pairs = []
    for _ in range(a.n_pairs * 4):
        i, j = rng.integers(0, n, 2)
        if circ_dist(t[i], t[j]) > 2.0:                       # well-separated phases
            pairs.append((int(i), int(j)))
        if len(pairs) >= a.n_pairs:
            break
    rows = []
    for (i, j) in pairs:
        ci = int(ids[i])
        genes = anndata_to_ranked_genes(ad, ci, max_genes=a.max_genes)
        if len(genes) < 20:
            continue
        cs = build_cell_sentence(genes)
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
        pos = int(enc["input_ids"].shape[1]) - 1
        ph, ents = [], []
        for lam in lams:
            h = (1 - lam) * H[i] + lam * H[j]
            ht = torch.tensor(h, dtype=torch.float32, device=dev)
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                o = o.clone(); o[0, pos, :] = ht.to(o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            hd_ = model.model.layers[a.layer].register_forward_hook(hook)
            with torch.no_grad():
                lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0, pos].float().cpu().numpy()
            hd_.remove()
            s = np.array([lg[mark[k]].mean() if len(mark[k]) else -np.inf for k in range(a.n_bins)])
            e = np.exp(s - np.nanmax(s[np.isfinite(s)])); e[~np.isfinite(s)] = 0; p = e / (e.sum() + 1e-12)
            ph.append(float(np.angle((p * np.exp(1j * centers)).sum())))
            ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        d = np.abs(np.diff(np.unwrap(ph)))
        rows.append(dict(i=i, j=j, dphase_true=float(circ_dist(t[i], t[j])),
                         readout_phase=ph, entropy=ents,
                         largest_step_share=float(d.max() / (d.sum() + 1e-12)),
                         total_move=float(d.sum()),
                         mid_entropy_excess=float(ents[len(ents) // 2] - 0.5 * (ents[0] + ents[-1]))))
    ls = np.array([r["largest_step_share"] for r in rows])
    me = np.array([r["mid_entropy_excess"] for r in rows])
    res["interpolation"] = dict(n_pairs=len(rows), n_lambda=int(a.n_lambda),
                                largest_step_share_mean=float(ls.mean()), largest_step_share_sd=float(ls.std()),
                                snap_baseline=1.0, gradual_baseline=float(1.0 / (a.n_lambda - 1)),
                                mid_entropy_excess_mean=float(me.mean()), rows=rows)
    print(f"(3) INTERPOLATION (n={len(rows)} pairs): largest-step share = {ls.mean():.3f} ± {ls.std():.3f}",
          flush=True)
    print(f"    [pure SNAP = 1.00 | perfectly GRADUAL = {1.0/(a.n_lambda-1):.3f}]", flush=True)
    print(f"    mid-interpolation entropy excess = {me.mean():+.4f} "
          f"(>0 = genuine intermediate mixture, <0 = committed to an attractor)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
