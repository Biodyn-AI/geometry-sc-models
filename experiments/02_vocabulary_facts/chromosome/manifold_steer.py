"""manifold_steer — MANIFOLD vs LINEAR steering along a fitted cell-state manifold (Goodfire-style).

Given knots K[bin] (mean cell-state per bin of the ordering variable, from manifold_fit.py) at layer L, move a
cell from a SOURCE bin to a TARGET bin two ways, and read the model's OWN next-gene logits:

  LINEAR    h <- h + a * (K[tgt] - K[src])                       a in [0,1]; intermediate a = the CHORD,
                                                                 which for a curved manifold is OFF-manifold.
  MANIFOLD  h <- h + (path(a) - K[src])                          path(a) = point at arc-fraction a ALONG the
                                                                 knots from src to tgt (short way round).

READOUT (no fitted probe -> non-circular): per-bin marker genes are derived from expression (genes enriched in
that bin). Readout distribution over bins = softmax of mean next-token logit over each bin's marker first
subtokens. From it:
  readout_phase(a)  circular (or linear) mean of the bin distribution -> where the model "thinks" the cell is
  peakedness(a)     max bin probability (and entropy) -> is the output ON-distribution or mush
  adjacency         fraction of consecutive a-steps where argmax bin moves by <=1 bin (SMOOTH) vs jumps (TELEPORT)
  max_jump          largest single-step move of the argmax bin, in bins

Goodfire's claim, made quantitative: MANIFOLD steering -> high adjacency, small max_jump, peakedness preserved;
LINEAR steering -> teleports (argmax jumps), peakedness collapses mid-sweep (off-manifold mush).
Controls: a norm-matched RANDOM displacement of the same length as the manifold move.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence


def bin_markers(adata, bins, n_bins, top=25, min_cells=20):
    """Per-bin marker genes: highest (mean-in-bin - mean-elsewhere), on the panel's own expression."""
    import scipy.sparse as sp
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    out = {}
    gm = X.mean(0)
    for k in range(n_bins):
        m = bins == k
        if m.sum() < min_cells:
            out[k] = []; continue
        d = X[m].mean(0) - gm
        out[k] = [str(var[i]) for i in np.argsort(-d)[:top]]
    return out


def circ_mean_of(weights, centers):
    v = (weights * np.exp(1j * centers)).sum()
    return float(np.mod(np.angle(v), 2 * np.pi)), float(np.abs(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knots", required=True, help="npz from manifold_fit.py")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--kind", choices=["cyclic", "ordered"], default="cyclic")
    ap.add_argument("--var-key", default=None)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-cells", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=9, help="alpha grid 0..1")
    ap.add_argument("--half-turn", action="store_true", help="target = opposite bin (default: +half the ring)")
    ap.add_argument("--max-genes", type=int, default=384)
    ap.add_argument("--top-markers", type=int, default=25)
    ap.add_argument("--out", default="results/manifold_steer.json")
    a = ap.parse_args()

    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    z = np.load(a.knots, allow_pickle=True)
    K = z[f"L{a.layer:02d}"].astype(np.float64)             # (n_bins, d)
    centers = z["centers"].astype(float)
    n_bins = int(z["n_bins"])
    ad = anndata.read_h5ad(a.h5ad)
    if a.var_key:
        t_all = np.asarray(ad.obs[a.var_key]).astype(float)
        if a.kind == "cyclic":
            t_all = np.mod(t_all, 2 * np.pi)
    else:
        from cc_phase import phase_angle
        t_all, _, _, _ = phase_angle(ad)
    if a.kind == "cyclic":
        edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        bins_all = np.clip(np.digitize(t_all, edges) - 1, 0, n_bins - 1)
    else:
        q = np.quantile(t_all, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
        bins_all = np.clip(np.digitize(t_all, q) - 1, 0, n_bins - 1)
    markers = bin_markers(ad, bins_all, n_bins, top=a.top_markers)

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    first_tok = {}
    def ftok(sym):
        if sym not in first_tok:
            ids = tok(" " + sym, add_special_tokens=False)["input_ids"]
            first_tok[sym] = ids[0] if ids else None
        return first_tok[sym]
    mark_ids = {k: np.array([x for x in (ftok(g) for g in v) if x is not None], dtype=int)
                for k, v in markers.items()}
    usable = [k for k in range(n_bins) if len(mark_ids.get(k, [])) >= 5 and np.isfinite(K[k]).all()]
    print(f"layer L{a.layer:02d} | {n_bins} bins | usable bins {usable}", flush=True)

    # --- path helpers -------------------------------------------------------
    def arc_point(src, tgt, frac):
        """Point at arc-fraction `frac` along the knots from src to tgt (short way round if cyclic)."""
        if a.kind == "cyclic":
            fwd = (tgt - src) % n_bins; bwd = (src - tgt) % n_bins
            step = 1 if fwd <= bwd else -1
            nsteps = min(fwd, bwd)
            seq = [(src + step * i) % n_bins for i in range(nsteps + 1)]
        else:
            step = 1 if tgt >= src else -1
            seq = list(range(src, tgt + step, step))
        seq = [s for s in seq if np.isfinite(K[s]).all()]
        if len(seq) < 2:
            return K[src].copy()
        pts = K[seq]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        target_len = frac * d[-1]
        j = int(np.clip(np.searchsorted(d, target_len) - 1, 0, len(seq) - 2))
        w = (target_len - d[j]) / max(d[j + 1] - d[j], 1e-12)
        return pts[j] * (1 - w) + pts[j + 1] * w

    def readout(logits_row):
        sc = np.array([logits_row[mark_ids[k]].mean() if len(mark_ids.get(k, [])) else -np.inf for k in range(n_bins)])
        sc = np.where(np.isfinite(sc), sc, -np.inf)
        e = np.exp(sc - np.nanmax(sc[np.isfinite(sc)]))
        e[~np.isfinite(sc)] = 0.0
        return e / (e.sum() + 1e-12)

    rng = np.random.default_rng(0)
    alphas = np.linspace(0, 1, a.n_steps)
    modes = ["manifold", "linear", "random"]
    rows = []
    # pick cells from well-populated source bins
    order = rng.permutation(ad.n_obs)
    picked, per_src = [], {}
    for ci in order:
        s = int(bins_all[ci])
        if s not in usable:
            continue
        if per_src.get(s, 0) >= max(1, a.n_cells // max(1, len(usable))):
            continue
        picked.append(ci); per_src[s] = per_src.get(s, 0) + 1
        if len(picked) >= a.n_cells:
            break

    for ci in picked:
        genes = anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes)
        if len(genes) < 20:
            continue
        cs = build_cell_sentence(genes)
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
        pos = int(enc["input_ids"].shape[1]) - 1                 # last real token = the cell-state position
        src = int(bins_all[ci])
        tgt = (src + n_bins // 2) % n_bins if a.kind == "cyclic" else (n_bins - 1 if src < n_bins // 2 else 0)
        if tgt not in usable:
            cand = [u for u in usable if u != src]
            if not cand:
                continue
            tgt = max(cand, key=lambda u: min((u - src) % n_bins, (src - u) % n_bins)) if a.kind == "cyclic" else cand[-1]
        lin_disp = K[tgt] - K[src]
        rand_disp = rng.standard_normal(K.shape[1]); rand_disp *= np.linalg.norm(lin_disp) / np.linalg.norm(rand_disp)

        cell = dict(cell=int(ci), src=src, tgt=tgt, traj={m: [] for m in modes})
        for mode in modes:
            for al in alphas:
                if al == 0:
                    disp = np.zeros(K.shape[1])
                elif mode == "linear":
                    disp = al * lin_disp
                elif mode == "random":
                    disp = al * rand_disp
                else:
                    disp = arc_point(src, tgt, al) - K[src]
                d_t = torch.tensor(disp, dtype=torch.float32, device=dev)

                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    h[0, pos, :] = h[0, pos, :] + d_t.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hd = model.model.layers[a.layer].register_forward_hook(hook)
                with torch.no_grad():
                    lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0, pos].float().cpu().numpy()
                hd.remove()
                p = readout(lg)
                ph, conc = circ_mean_of(p, centers) if a.kind == "cyclic" else (float((p * centers).sum()), 0.0)
                cell["traj"][mode].append(dict(alpha=float(al), argmax=int(np.argmax(p)),
                                               peak=float(p.max()), entropy=float(-(p * np.log(p + 1e-12)).sum()),
                                               phase=ph, conc=conc))
        rows.append(cell)
        if len(rows) % 5 == 0:
            print(f"  {len(rows)}/{len(picked)} cells", flush=True)

    # --- INSTRUMENT GATE (pre-registered): can the readout decode phase on UNSTEERED real cells? ----------
    # This is exactly what killed the earlier scGPT cell-cycle attempt (its readout scored circ r = 0.19).
    # alpha=0 is the unsteered forward pass, so the gate is free.
    def _circ_corr(x, y):
        sx = np.sin(x - np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x))))
        sy = np.sin(y - np.arctan2(np.mean(np.sin(y)), np.mean(np.cos(y))))
        return float(np.sum(sx * sy) / (np.sqrt(np.sum(sx ** 2) * np.sum(sy ** 2)) + 1e-12))
    base_read = np.array([c["traj"]["manifold"][0]["phase"] for c in rows])
    base_true = np.array([centers[c["src"]] for c in rows])
    gate_val = abs(_circ_corr(base_read, base_true)) if a.kind == "cyclic" else float(
        np.corrcoef(base_read, base_true)[0, 1])
    gate_argmax = float(np.mean([c["traj"]["manifold"][0]["argmax"] == c["src"] for c in rows]))
    # AMPLITUDE half of the gate: correlation alone is NOT enough — a readout can track the variable
    # perfectly while moving in the 3rd decimal, which makes any steering test unfalsifiable
    # (cf. route_cellcycle: "random scored r=+0.80 while travelling 0.00 laps"). Require BOTH.
    var_range = 2 * np.pi if a.kind == "cyclic" else float(centers.max() - centers.min())
    spread = float(base_read.max() - base_read.min()) / (var_range + 1e-12)
    instrument = dict(readout_vs_true_corr=gate_val, argmax_hits_src_bin=gate_argmax,
                      readout_spread_frac_of_range=spread, n=len(rows),
                      gate_pass=bool(gate_val > 0.40 and spread > 0.15))
    if instrument["gate_pass"]:
        verdict_txt = "PASS"
    elif gate_val <= 0.40:
        verdict_txt = "FAIL (correlation: readout cannot see the variable)"
    else:
        verdict_txt = ("FAIL (AMPLITUDE: readout spans only %.1f%% of the variable range "
                       "-> cannot register steering)" % (100 * spread))
    print(f"\nINSTRUMENT GATE: readout-vs-true corr={gate_val:+.3f}, dynamic range={spread:.1%} "
          f"(argmax==src {gate_argmax:.2f}) -> {verdict_txt}", flush=True)

    # --- aggregate: smooth vs teleport -------------------------------------
    def bin_gap(i, j):
        d = abs(i - j)
        return min(d, n_bins - d) if a.kind == "cyclic" else d

    summary = {"_instrument": instrument}
    for mode in modes:
        jumps, adj, peaks, reach, ent = [], [], [], [], []
        for c in rows:
            tr = c["traj"][mode]
            am = [x["argmax"] for x in tr]
            steps = [bin_gap(am[i], am[i + 1]) for i in range(len(am) - 1)]
            jumps.append(max(steps) if steps else 0)
            adj.append(float(np.mean([s <= 1 for s in steps])) if steps else np.nan)
            peaks.append(float(np.mean([x["peak"] for x in tr])))
            ent.append(float(np.mean([x["entropy"] for x in tr])))
            reach.append(1.0 - bin_gap(am[-1], c["tgt"]) / (n_bins / 2))       # 1 = landed on target
        summary[mode] = dict(mean_max_jump=float(np.mean(jumps)), adjacency=float(np.nanmean(adj)),
                             mean_peak=float(np.mean(peaks)), mean_entropy=float(np.mean(ent)),
                             target_reach=float(np.mean(reach)), n=len(rows))
        print(f"  {mode:<9} adjacency={summary[mode]['adjacency']:.2f} max_jump={summary[mode]['mean_max_jump']:.2f} "
              f"peak={summary[mode]['mean_peak']:.3f} entropy={summary[mode]['mean_entropy']:.3f} "
              f"reach={summary[mode]['target_reach']:+.2f}", flush=True)
    mf, ln = summary["manifold"], summary["linear"]
    summary["_verdict"] = ("MANIFOLD > LINEAR (smoother + more on-distribution)"
                           if (mf["adjacency"] > ln["adjacency"] and mf["mean_max_jump"] <= ln["mean_max_jump"])
                           else "no manifold advantage")
    print(f"\nVERDICT: {summary['_verdict']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, n_bins=n_bins, layer=a.layer, markers={str(k): v for k, v in markers.items()},
                   rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
