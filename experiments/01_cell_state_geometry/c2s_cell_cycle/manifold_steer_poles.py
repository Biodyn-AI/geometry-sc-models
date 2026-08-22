"""manifold_steer_poles — manifold vs linear steering on an ORDERED manifold, with a 2-POLE readout.

WHY THIS EXISTS. The 12-bin marker readout in manifold_steer.py works for the cell cycle (phases have sharply
distinct, highly expressed programs) but FAILS its instrument gate on differentiation pseudotime: adjacent bins
share most genes, so the softmax over bins is pinned and cannot register steering (all-lineage: corr 0.62 but
only 3.2% dynamic range; erythroid: corr 0.26 / 12.7%). This is the readout that DID work in this project's
earlier steering (Threads A/B): a signed contrast between two well-separated marker poles.

  readout(cell) = mean logit(END-pole markers) - mean logit(START-pole markers)

Poles are derived by contrasting the EXTREME bins of the ordering variable (maximally distinct), not each bin
against the global mean. Same three arms (manifold arc / linear chord / norm-matched random) and the same
instrument gate (correlation AND dynamic range) before any claim is allowed.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence


def pole_markers(adata, bins, n_bins, top=30, df_max=0.5):
    """Markers of the LAST bin vs the FIRST bin (the two extremes of the ordering variable).

    STOPWORD FILTER (essential): the raw DE contrast on a differentiation branch is swamped by MT-*/MALAT1/
    ribosomal genes, because late erythroid cells shift library composition. Those genes sit at the top of
    nearly every cell sentence, so they carry no phase information in logit space and destroy the readout
    (observed: END pole = HBB, MT-CO3, MT-CO2, MT-ND4, MALAT1 -> instrument corr -0.19). Same trap, and same
    fix, as ctx_extract_c2s.is_stopword in Thread B.
    """
    import re
    import scipy.sparse as sp
    STOP_RE = re.compile(r"^(MT-|MTRNR|RPL|RPS|MRPL|MRPS|HIST|LINC|AC[0-9]{6}|AL[0-9]{6}|RP[0-9]+-)")
    STOP_SET = {"MALAT1", "NEAT1", "RACK1", "EEF1A1", "EEF2", "ACTB", "ACTG1", "B2M", "TMSB4X", "TMSB10",
                "FTL", "FTH1", "TPT1", "GAPDH", "XIST"}
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    df = (X > 0).mean(0)                                   # document frequency across cells
    ok = np.array([not (STOP_RE.match(s) or s in STOP_SET) for s in var]) & (df <= df_max)
    lo, hi = bins == 0, bins == (n_bins - 1)
    d = X[hi].mean(0) - X[lo].mean(0)
    d = np.where(ok, d, 0.0)
    end = [str(var[i]) for i in np.argsort(-d)[:top]]
    start = [str(var[i]) for i in np.argsort(d)[:top]]
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knots", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--var-key", required=True)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-cells", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=9)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="results/manifold_steer_poles.json")
    a = ap.parse_args()

    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    z = np.load(a.knots, allow_pickle=True)
    K = z[f"L{a.layer:02d}"].astype(np.float64); n_bins = int(z["n_bins"])
    ad = anndata.read_h5ad(a.h5ad)
    t_all = np.asarray(ad.obs[a.var_key]).astype(float)
    q = np.quantile(t_all, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
    bins_all = np.clip(np.digitize(t_all, q) - 1, 0, n_bins - 1)
    start_g, end_g = pole_markers(ad, bins_all, n_bins, a.top)
    print(f"START pole: {', '.join(start_g[:8])}\nEND   pole: {', '.join(end_g[:8])}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    ft = {}
    def ids_of(gs):
        out = []
        for g in gs:
            if g not in ft:
                i = tok(" " + g, add_special_tokens=False)["input_ids"]; ft[g] = i[0] if i else None
            if ft[g] is not None:
                out.append(ft[g])
        return np.array(out, dtype=int)
    SI, EI = ids_of(start_g), ids_of(end_g)

    def score(lg):
        return float(lg[EI].mean() - lg[SI].mean())

    def arc_point(src, tgt, frac):
        step = 1 if tgt >= src else -1
        seq = [s for s in range(src, tgt + step, step) if np.isfinite(K[s]).all()]
        if len(seq) < 2:
            return K[src].copy()
        pts = K[seq]; d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        tl = frac * d[-1]; j = int(np.clip(np.searchsorted(d, tl) - 1, 0, len(seq) - 2))
        w = (tl - d[j]) / max(d[j + 1] - d[j], 1e-12)
        return pts[j] * (1 - w) + pts[j + 1] * w

    rng = np.random.default_rng(0)
    alphas = np.linspace(0, 1, a.n_steps)
    # sources: cells in the FIRST third, target = last bin (push forward along differentiation)
    cand = np.where(bins_all <= max(1, n_bins // 4))[0]
    rng.shuffle(cand)
    rows = []
    for ci in cand[: a.n_cells * 3]:
        if len(rows) >= a.n_cells:
            break
        genes = anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes)
        if len(genes) < 20:
            continue
        cs = build_cell_sentence(genes)
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=max_len)
        pos = int(enc["input_ids"].shape[1]) - 1
        src = int(bins_all[ci]); tgt = n_bins - 1
        lin = K[tgt] - K[src]
        rnd = rng.standard_normal(K.shape[1]); rnd *= np.linalg.norm(lin) / np.linalg.norm(rnd)
        cell = dict(cell=int(ci), src=src, tgt=tgt, true_t=float(t_all[ci]), traj={})
        for mode in ["manifold", "linear", "random"]:
            vals = []
            for al in alphas:
                disp = (np.zeros(K.shape[1]) if al == 0 else
                        al * lin if mode == "linear" else
                        al * rnd if mode == "random" else
                        arc_point(src, tgt, al) - K[src])
                dt = torch.tensor(disp, dtype=torch.float32, device=dev)
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone(); h[0, pos, :] = h[0, pos, :] + dt.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hd = model.model.layers[a.layer].register_forward_hook(hook)
                with torch.no_grad():
                    lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0, pos].float().cpu().numpy()
                hd.remove()
                vals.append(score(lg))
            cell["traj"][mode] = vals
        rows.append(cell)
        if len(rows) % 6 == 0:
            print(f"  {len(rows)}/{a.n_cells}", flush=True)

    base = np.array([c["traj"]["manifold"][0] for c in rows])
    true = np.array([c["true_t"] for c in rows])
    corr = float(np.corrcoef(base, true)[0, 1]) if len(set(np.round(base, 9))) > 1 else 0.0
    # dynamic range measured in READOUT units, relative to the across-cell spread it must resolve
    dyn = float(base.std())
    d_m = np.array([c["traj"]["manifold"][-1] - c["traj"]["manifold"][0] for c in rows])
    d_l = np.array([c["traj"]["linear"][-1] - c["traj"]["linear"][0] for c in rows])
    d_r = np.array([c["traj"]["random"][-1] - c["traj"]["random"][0] for c in rows])
    snr = float(np.abs(d_m).mean() / (np.abs(d_r).mean() + 1e-12))
    gate = bool(abs(corr) > 0.40 and snr > 3.0)
    print(f"\nINSTRUMENT: readout-vs-true corr={corr:+.3f} (across-cell sd={dyn:.3f}) | "
          f"steer/random SNR={snr:.1f}x -> {'PASS' if gate else 'FAIL'}", flush=True)
    print(f"  Δreadout at alpha=1: manifold {d_m.mean():+.3f} | linear {d_l.mean():+.3f} | random {d_r.mean():+.3f}",
          flush=True)
    for m in ["manifold", "linear", "random"]:
        tr = np.array([c["traj"][m] for c in rows]).mean(0)
        print(f"  {m:<9} dose curve: {np.round(tr - tr[0], 3)}", flush=True)
    pathm = np.array([sum(abs(np.diff(c["traj"]["manifold"]))) for c in rows])
    pathl = np.array([sum(abs(np.diff(c["traj"]["linear"]))) for c in rows])
    out = dict(instrument=dict(corr=corr, across_cell_sd=dyn, snr_vs_random=snr, gate_pass=gate, n=len(rows)),
               delta=dict(manifold=float(d_m.mean()), linear=float(d_l.mean()), random=float(d_r.mean())),
               path=dict(manifold=float(pathm.mean()), linear=float(pathl.mean()),
                         ratio=float(pathm.sum() / (pathl.sum() + 1e-12))),
               start_pole=start_g, end_pole=end_g, rows=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
