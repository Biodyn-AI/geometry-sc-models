"""lap_walk — ITERATIVE steering around the cell-cycle loop: does a FIXED direction stall at ~pi/2?

THE THEOREM (route_cellcycle/RESULTS.md). For any CLOSED curve and any FIXED direction w, the net work per lap
is zero: ∮ w·dx = w·∮ dx = 0. An iterative steerer using a constant w therefore advances while w·t̂ > 0,
retreats once w·t̂ < 0, and converges to a STALL where w ⊥ t̂ — about a quarter-lap from the start — *even on a
perfectly flat circle*. Making the direction LOCAL (the tangent at the current position) removes the obstruction
and the walker laps indefinitely. Verified there on scGPT: fixed 0.34 laps vs local-tangent 4.53 laps, and
reproduced on a synthetic zero-curvature circle (stall at 1.49 rad vs predicted pi/2 = 1.571).

This is the test the single-shot dose sweep in manifold_steer.py cannot reach: a dose sweep from source to
target lands on the target by construction, so it can never expose the stall.

ARMS (all matched on step size)
  fixed            h += a * ŵ,  w frozen = the tangent at the STARTING position (locally correct at t=0)
  fixed_proj       w projected onto the local tangent, then unit-stepped -> stays ON the manifold and still
                   stalls; this is the arm the theorem indicts
  local_tangent    h += a * t̂(h) recomputed every step  -> should traverse
  random           norm-matched random direction, fresh each step (control)
  *_retract        variants that snap back onto the knot path after each step (retraction does NOT rescue fixed)

READOUTS
  geometric  decoded phase of the walked state (ridge cos/sin decoder fitted on REAL cells), UNWRAPPED -> laps.
             Unwrapped advance is mandatory: circular correlation is degenerate here (route_cellcycle saw a
             random arm score r = +0.80 while travelling 0.00 laps).
  behavioural every `--behav-every` steps the walked state REPLACES the last-token residual in a real forward
             pass and the model's own marker-gene logits are read -> does the model's output lap too?
Out: results/lap_walk.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def nearest_knot(h, K):
    return int(np.argmin(np.linalg.norm(K - h, axis=1)))


def tangent_at(h, K):
    """Local forward tangent of the closed knot path at the nearest knot (phase-increasing orientation)."""
    nb = len(K); i = nearest_knot(h, K)
    return unit(K[(i + 1) % nb] - K[(i - 1) % nb])


def retract(h, K):
    """Snap onto the polyline through the knots (nearest point on the nearest segment)."""
    nb = len(K); best, bd = h, np.inf
    for i in range(nb):
        a, b = K[i], K[(i + 1) % nb]
        ab = b - a; t = np.clip(np.dot(h - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        p = a + t * ab; d = np.linalg.norm(h - p)
        if d < bd:
            bd, best = d, p
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--step-frac", type=float, default=0.25, help="step size as a fraction of median knot spacing")
    ap.add_argument("--behav-every", type=int, default=20, help="0 disables the behavioural readout")
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/lap_walk.json")
    a = ap.parse_args()

    import anndata
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    theta, _, _, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sc = StandardScaler().fit(H)
    dec = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
    P0 = dec.predict(sc.transform(H)); C0 = P0.mean(0)

    def phase_of(h):
        p = dec.predict(sc.transform(h.reshape(1, -1)))[0] - C0
        return float(np.arctan2(p[1], p[0]))

    # knots = phase-bin means (the discrete closed manifold)
    edges = np.linspace(0, 2 * np.pi, a.n_bins + 1)
    b = np.clip(np.digitize(t, edges) - 1, 0, a.n_bins - 1)
    K = np.stack([H[b == k].mean(0) for k in range(a.n_bins)])
    step_len = a.step_frac * float(np.median(np.linalg.norm(np.diff(np.vstack([K, K[:1]]), axis=0), axis=1)))
    print(f"L{a.layer:02d}: {a.n_bins} knots, median spacing "
          f"{np.median(np.linalg.norm(np.diff(np.vstack([K, K[:1]]), axis=0), axis=1)):.2f}, "
          f"step {step_len:.2f}, {a.steps} steps", flush=True)

    # optional behavioural readout
    model = tok = None; mark_ids = {}; centers = (edges[:-1] + edges[1:]) / 2
    if a.behav_every:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import scipy.sparse as sp
        X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
        var = np.char.upper(np.asarray(ad.var_names).astype(str)); gm = X.mean(0)
        tok = AutoTokenizer.from_pretrained(a.model)
        try:
            model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        for k in range(a.n_bins):
            m = b == k
            if m.sum() < 20:
                mark_ids[k] = np.array([], int); continue
            top = [str(var[i]) for i in np.argsort(-(X[m].mean(0) - gm))[:25]]
            ii = []
            for g in top:
                z = tok(" " + g, add_special_tokens=False)["input_ids"]
                if z:
                    ii.append(z[0])
            mark_ids[k] = np.array(ii, int)

    ARMS = ["fixed", "fixed_proj", "local_tangent", "random", "fixed_proj_retract", "local_tangent_retract"]
    rng = np.random.default_rng(0)
    starts = rng.choice(len(H), a.n_cells, replace=False)
    rows = []
    for si, s0 in enumerate(starts):
        h0 = H[s0].copy()
        w_fixed = tangent_at(h0, K)                       # locally CORRECT at t=0
        cell = dict(cell=int(ids[s0]), arms={})
        for arm in ARMS:
            h = h0.copy(); ph = [phase_of(h)]; offman = []
            for it in range(a.steps):
                tg = tangent_at(h, K)
                if arm.startswith("fixed_proj"):
                    d = unit(np.dot(w_fixed, tg) * tg)     # fixed direction projected on the local tangent
                elif arm == "fixed":
                    d = w_fixed
                elif arm.startswith("local_tangent"):
                    d = tg
                else:
                    d = unit(rng.standard_normal(len(h)))
                h = h + step_len * d
                if arm.endswith("retract"):
                    h = retract(h, K)
                ph.append(phase_of(h))
                offman.append(float(np.linalg.norm(h - retract(h, K)) / (step_len + 1e-12)))
            adv = float(np.sum(np.diff(np.unwrap(ph))))
            cell["arms"][arm] = dict(laps=adv / (2 * np.pi), advance_rad=adv,
                                     peak_laps=float(np.max(np.abs(np.unwrap(ph) - ph[0])) / (2 * np.pi)),
                                     offmanifold_mean=float(np.mean(offman)),
                                     phase=[float(x) for x in ph[:: max(1, a.steps // 40)]])
        rows.append(cell)
        if (si + 1) % 4 == 0:
            print(f"  {si+1}/{a.n_cells} cells", flush=True)

    print("\nGEOMETRIC LAPS (mean over cells):", flush=True)
    summ = {}
    for arm in ARMS:
        L_ = np.array([c["arms"][arm]["laps"] for c in rows])
        O_ = np.array([c["arms"][arm]["offmanifold_mean"] for c in rows])
        summ[arm] = dict(laps_mean=float(L_.mean()), laps_sd=float(L_.std()), offmanifold=float(O_.mean()))
        print(f"  {arm:<24} laps = {L_.mean():+7.3f} ± {L_.std():.3f}   off-manifold = {O_.mean():.2f}", flush=True)

    # ---- behavioural: does the MODEL's own output lap too? (local_tangent vs fixed_proj) ----
    behav = {}
    if a.behav_every and model is not None:
        import torch
        dev = next(model.parameters()).device
        max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
        for arm in ["local_tangent", "fixed_proj"]:
            adv_all = []
            for si, s0 in enumerate(starts[: min(6, len(starts))]):
                ci = int(ids[s0])
                genes = anndata_to_ranked_genes(ad, ci, max_genes=a.max_genes)
                if len(genes) < 20:
                    continue
                cs = build_cell_sentence(genes)
                enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
                pos = int(enc["input_ids"].shape[1]) - 1
                h = H[s0].copy(); w_fixed = tangent_at(h, K); readout = []
                for it in range(a.steps):
                    tg = tangent_at(h, K)
                    d = unit(np.dot(w_fixed, tg) * tg) if arm == "fixed_proj" else tg
                    h = h + step_len * d
                    if it % a.behav_every == 0 or it == a.steps - 1:
                        ht = torch.tensor(h, dtype=torch.float32, device=dev)
                        def hook(mod, inp, out):
                            o = out[0] if isinstance(out, tuple) else out
                            o = o.clone(); o[0, pos, :] = ht.to(o.dtype)   # REPLACE the state with the walked one
                            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                        hd = model.model.layers[a.layer].register_forward_hook(hook)
                        with torch.no_grad():
                            lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0, pos].float().cpu().numpy()
                        hd.remove()
                        sc_ = np.array([lg[mark_ids[k]].mean() if len(mark_ids[k]) else -np.inf
                                        for k in range(a.n_bins)])
                        e = np.exp(sc_ - np.nanmax(sc_[np.isfinite(sc_)])); e[~np.isfinite(sc_)] = 0
                        p = e / (e.sum() + 1e-12)
                        readout.append(float(np.angle((p * np.exp(1j * centers)).sum())))
                adv_all.append(float(np.sum(np.diff(np.unwrap(readout)))) / (2 * np.pi))
            behav[arm] = dict(behavioural_laps_mean=float(np.mean(adv_all)), n=len(adv_all),
                              per_cell=[float(x) for x in adv_all])
            print(f"  BEHAVIOURAL {arm:<16} laps = {np.mean(adv_all):+.3f} (n={len(adv_all)})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summ, behavioural=behav, layer=a.layer, steps=a.steps,
                   step_frac=a.step_frac, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
