"""plateau_test — ACTIVATION PLATEAUS on the cell-cycle manifold (Nostalgebraist-style / LessWrong
"Activation plateaus: where and how they emerge").

THEIR DEFINITION. Interpolating an activation between two inputs produces minimal downstream change for most of
the path, then a sudden phase change near a boundary. Operationalised as the RELATIVE OUTPUT DISTANCE

    d(lam) = ||y(lam) - y_A|| / ( ||y(lam) - y_A|| + ||y(lam) - y_B|| )

where y(.) are the model's output logits. A pure PLATEAU gives a step/sigmoid in d(lam); a smooth manifold gives
d(lam) ~ lam. Plateaus are created by MLPs (low Jacobian inside, high at the edge) and SHARPEN WITH DEPTH.

WHY IT MATTERS HERE. Our manifold is CONTINUOUS in representation (interpolation moves the readout gradually,
occupancy is smooth) but its METRIC IS STRETCHED at G1->S. Plateaus are the behavioural counterpart: the model
could hold a continuous phase code yet still respond in quasi-discrete steps. If the plateau EDGES line up with
the real cell-cycle transitions, that is a clean synthesis — continuous representation, quasi-discrete behaviour.

TWO MEASUREMENTS
 A. PAIRWISE (faithful to the post): interpolate cell A -> cell B, sweep lam, record d(lam) at several layers.
    Statistics: max|d(lam) - lam| (deviation from linear = plateau strength), and TRANSITION WIDTH = the
    fraction of the lam range over which d goes 0.25 -> 0.75 (narrow = sharp boundary).
    Pairs are stratified: WITHIN one phase vs ACROSS the G1/S boundary vs ACROSS the G2/M boundary.
 B. PHASE-RESOLVED: walk the fitted manifold in equal-arc steps all the way round and record the output step
    size ||delta logits|| at each position -> a sensitivity profile over phase. Plateaus = low-sensitivity
    stretches; boundaries = peaks. Then ask WHERE the peaks sit relative to G1/S and G2/M.
Out: results/plateau_test.json
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
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 15, 21])
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--n-pairs", type=int, default=8, help="pairs PER stratum")
    ap.add_argument("--n-lambda", type=int, default=21)
    ap.add_argument("--walk-steps", type=int, default=60)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/plateau_test.json")
    a = ap.parse_args()

    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ad = anndata.read_h5ad(a.h5ad)
    theta, _, _, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    Hs = {L: np.load(os.path.join(a.states, f"layer_{L:02d}_activations.npy")).astype(np.float64)
          for L in a.layers}
    nmin = min(len(v) for v in Hs.values())
    t = theta[ids][:nmin]
    Hs = {L: v[:nmin] for L, v in Hs.items()}
    lab = np.asarray(ad.obs["clusters"]).astype(str)[ids][:nmin] if "clusters" in ad.obs else None

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    def logits_with(state, enc, pos, layer):
        ht = torch.tensor(state, dtype=torch.float32, device=dev)
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o.clone(); o[0, pos, :] = ht.to(o.dtype)
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        h = model.model.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0, pos].float().cpu().numpy()
        h.remove()
        return lg

    def encode(ci):
        genes = anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes)
        if len(genes) < 20:
            return None, None
        cs = build_cell_sentence(genes)
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
        return enc, int(enc["input_ids"].shape[1]) - 1

    # ---------- A. PAIRWISE plateaus, stratified by whether the pair crosses a real transition ----------
    rng = np.random.default_rng(0)
    deg = np.degrees(np.mod(t, 2 * np.pi))
    # k562 bins: G2M ~0-90, G1 ~90-180, S ~180-330 (validated in cc_phase)
    strata = {"within_S": ((deg > 200) & (deg < 260), (deg > 200) & (deg < 260)),
              "across_G1S": ((deg > 100) & (deg < 160), (deg > 200) & (deg < 260)),
              "across_G2M": ((deg > 260) & (deg < 340), (deg > 10) & (deg < 70))}
    lams = np.linspace(0, 1, a.n_lambda)
    pair_res = {}
    for sname, (mA, mB) in strata.items():
        iA, iB = np.where(mA)[0], np.where(mB)[0]
        if len(iA) < 2 or len(iB) < 2:
            continue
        recs = []
        for _ in range(a.n_pairs):
            i, j = int(rng.choice(iA)), int(rng.choice(iB))
            if i == j:
                continue
            enc, pos = encode(ids[i])
            if enc is None:
                continue
            per_layer = {}
            for L in a.layers:
                yA = logits_with(Hs[L][i], enc, pos, L)
                yB = logits_with(Hs[L][j], enc, pos, L)
                ds, steps, prev = [], [], None
                for lam in lams:
                    y = logits_with((1 - lam) * Hs[L][i] + lam * Hs[L][j], enc, pos, L)
                    dA, dB = np.linalg.norm(y - yA), np.linalg.norm(y - yB)
                    ds.append(float(dA / (dA + dB + 1e-12)))
                    if prev is not None:
                        steps.append(float(np.linalg.norm(y - prev)))
                    prev = y
                ds = np.array(ds); steps = np.array(steps)
                # plateau strength = deviation from the linear reference; transition width = 0.25->0.75 span
                dev_lin = float(np.max(np.abs(ds - lams)))
                lo = np.argmax(ds >= 0.25) if (ds >= 0.25).any() else 0
                hi = np.argmax(ds >= 0.75) if (ds >= 0.75).any() else len(ds) - 1
                width = float(abs(lams[hi] - lams[lo]))
                per_layer[f"L{L:02d}"] = dict(dev_from_linear=dev_lin, transition_width=width,
                                              largest_step_share=float(steps.max() / (steps.sum() + 1e-12)),
                                              d_curve=[float(x) for x in ds])
            recs.append(per_layer)
        pair_res[sname] = recs
        for L in a.layers:
            k = f"L{L:02d}"
            dv = np.mean([r[k]["dev_from_linear"] for r in recs])
            wd = np.mean([r[k]["transition_width"] for r in recs])
            ls = np.mean([r[k]["largest_step_share"] for r in recs])
            print(f"  A {sname:<12} {k}: dev_from_linear={dv:.3f}  transition_width={wd:.3f}  "
                  f"largest_step_share={ls:.3f}", flush=True)

    # ---------- B. PHASE-RESOLVED output sensitivity around the loop ----------
    edges = np.linspace(0, 2 * np.pi, a.n_bins + 1)
    b = np.clip(np.digitize(np.mod(t, 2 * np.pi), edges) - 1, 0, a.n_bins - 1)
    walk = {}
    for L in a.layers:
        K = np.stack([Hs[L][b == k].mean(0) for k in range(a.n_bins)])
        # equal-arc walk around the closed knot polyline
        pts = np.vstack([K, K[:1]])
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seglen)]; total = cum[-1]
        s = np.linspace(0, total, a.walk_steps, endpoint=False)
        traj = []
        for si in s:
            k = int(np.clip(np.searchsorted(cum, si) - 1, 0, len(seglen) - 1))
            w = (si - cum[k]) / max(seglen[k], 1e-12)
            traj.append(pts[k] * (1 - w) + pts[k + 1] * w)
        traj = np.array(traj)
        # use one representative cell's sentence as the carrier
        enc, pos = encode(ids[int(np.argmin(np.abs(np.mod(t, 2*np.pi) - np.pi)))])
        Y = np.array([logits_with(x, enc, pos, L) for x in traj])
        step = np.linalg.norm(np.diff(np.vstack([Y, Y[:1]]), axis=0), axis=1)
        phase_of_step = (s / total) * 360.0
        walk[f"L{L:02d}"] = dict(phase_deg=[float(x) for x in phase_of_step],
                                 output_step=[float(x) for x in step],
                                 cv=float(step.std() / (step.mean() + 1e-12)),
                                 peak_phase_deg=float(phase_of_step[int(np.argmax(step))]),
                                 top3_peak_phases=[float(phase_of_step[i]) for i in np.argsort(-step)[:3]])
        w = walk[f"L{L:02d}"]
        print(f"  B L{L:02d}: output-sensitivity CV={w['cv']:.3f}  peak at {w['peak_phase_deg']:.0f}deg "
              f"(top3 {np.round(w['top3_peak_phases'],0)})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(pairwise=pair_res, walk=walk, layers=a.layers, n_lambda=int(a.n_lambda)),
              open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
