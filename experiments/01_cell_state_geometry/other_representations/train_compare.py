"""Train linear SAE (PolySAE deg1) + bilinear AE (PolySAE deg2) on UCE-100M layer-2 activations,
matched budget, then run the pre-registered comparison (R2_x, R2_SAE, novel-nonlinear, fidelity gate).
Self-contained; imports model/data classes from route_b read-only. Adapted from route_state/train_compare.py.
Outputs results/uce_L2_comparison.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RB = HERE.parent / "route_b"
sys.path.insert(0, str(RB))
sys.path.insert(0, str(RB / "poly"))
sys.path.insert(0, str(HERE))
import config as C                       # route_uce config
from data import make_split, ActivationStream, data_mean   # route_b, dep-light
from poly_sae import PolySAE                                 # route_b/poly

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 4096
STEPS = int(os.environ.get("UCE_STEPS", "3000"))
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
TAG = C.TARGET["name"]                    # "uce_L2"


def train_poly(degree, masks, mean):
    run = f"{TAG}_poly{degree}_k{C.K}"
    rd = C.RUNS / run; rd.mkdir(exist_ok=True)
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    stream = ActivationStream(C.TARGET["acts"], masks["train"], BATCH, seed=C.SEED, device=DEV, preload=True)
    val_x = torch.from_numpy(ActivationStream(C.TARGET["acts"], masks["val"], 4096)
                             .sample(20000, seed=C.SEED)[0].astype(np.float32)).to(DEV)
    model = PolySAE(C.D_MODEL, C.N_FEATURES, C.K, degree=degree).to(DEV)
    model.set_mean(mean, freeze=True)
    twod = [p for p in model.parameters() if p.ndim == 2]
    oned = [p for p in model.parameters() if p.ndim == 1 and p.requires_grad]
    opt2 = torch.optim.AdamW(twod, lr=5e-4, betas=(0.9, 0.999)); opt1 = torch.optim.AdamW(oned, lr=3e-4) if oned else None
    t0 = time.time(); step = 0; ep = 0; done = False; nan_hits = 0
    while not done:
        for x in stream.epoch(ep):
            x = x.float()
            loss, parts = model.loss(x)
            if not np.isfinite(float(loss)):
                nan_hits += 1
                if nan_hits > 20:
                    raise RuntimeError(f"[deg{degree}] persistent non-finite loss at step {step}")
                continue
            opt2.zero_grad(set_to_none=True)
            if opt1: opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(twod + oned, 1.0)
            opt2.step()
            if opt1: opt1.step()
            with torch.no_grad():
                model.W_dec.weight.data = torch.nn.functional.normalize(model.W_dec.weight.data, dim=0)
            step += 1
            if step % 200 == 0:
                print(f"  [deg{degree}] step {step} nmse {parts['nmse']:.4f} ve {1-parts['nmse']:.4f}", flush=True)
            if step >= STEPS: done = True; break
        ep += 1
    with torch.no_grad():
        recon, xc, z, pre = model.forward(val_x)
        ve = float(1 - ((recon - xc).pow(2).sum(-1) / xc.pow(2).sum(-1).clamp_min(1e-6)).mean())
    torch.save(dict(model=model.state_dict(), degree=degree), rd / "latest.pt")
    print(f"[deg{degree}] DONE val VE={ve:.4f} ({(time.time()-t0)/60:.1f} min)")
    return model, ve


def ridge_r2(Phi_fit, Z_fit, Phi_eval, Z_eval, alphas, device="cpu"):
    Pf = torch.from_numpy(Phi_fit).to(device); Zf = torch.from_numpy(Z_fit).to(device)
    Pe = torch.from_numpy(Phi_eval).to(device); Ze = torch.from_numpy(Z_eval).to(device)
    pmu = Pf.mean(0, keepdim=True); zmu = Zf.mean(0, keepdim=True)
    Pfc = Pf - pmu; Zfc = Zf - zmu; Pec = Pe - pmu; Zec = Ze - zmu
    p = Pfc.shape[1]; I = torch.eye(p, device=device)
    trace_scale = (Pfc.T @ Pfc).diagonal().mean().item() + 1e-8
    n = Pfc.shape[0]; half = n // 2
    def fe(Pt, Zt, Pv, Zv, a):
        W = torch.linalg.solve(Pt.T @ Pt + a * I, Pt.T @ Zt); pred = Pv @ W
        ssr = (Zv - pred).pow(2).sum(0); sst = (Zv - Zv.mean(0)).pow(2).sum(0).clamp_min(1e-12)
        return 1 - ssr / sst
    best_a, best = alphas[0], -1e9
    for af in alphas:
        a = af * trace_scale
        s = float(((fe(Pfc[:half], Zfc[:half], Pfc[half:], Zfc[half:], a) +
                    fe(Pfc[half:], Zfc[half:], Pfc[:half], Zfc[:half], a)) / 2).mean())
        if s > best: best, best_a = s, a
    W = torch.linalg.solve(Pfc.T @ Pfc + best_a * I, Pfc.T @ Zfc)
    pred = Pec @ W
    ssr = (Zec - pred).pow(2).sum(0); sst = (Zec - Zec.mean(0)).pow(2).sum(0).clamp_min(1e-12)
    return (1 - ssr / sst).cpu().numpy()


@torch.no_grad()
def encode(model, X, bs=16384):
    out = np.empty((X.shape[0], model.n_features), np.float32)
    for s in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[s:s+bs].astype(np.float32)).to(DEV)
        out[s:s+bs] = model.encode(xb).float().cpu().numpy()
    return out


def main():
    sf = C.RUNS / "split_masks.npz"
    masks, _ = make_split(C.TARGET["cell_ids"], C.SEED, C.SPLIT)
    np.savez(sf, **masks)
    mean = torch.tensor(data_mean(C.TARGET["acts"], masks["train"]), dtype=torch.float32, device=DEV)

    print(f"[train] device={DEV} steps={STEPS} n_features={C.N_FEATURES} k={C.K}")
    lin, ve_lin = train_poly(1, masks, mean)      # linear TopK SAE
    bil, ve_bil = train_poly(2, masks, mean)      # bilinear

    Xtr, _ = ActivationStream(C.TARGET["acts"], masks["train"], BATCH).sample(60000, seed=1)
    Xte, _ = ActivationStream(C.TARGET["acts"], masks["test"], BATCH).sample(60000, seed=2)
    Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
    Zb_tr, Zb_te = encode(bil, Xtr), encode(bil, Xte)
    Hl_tr, Hl_te = encode(lin, Xtr), encode(lin, Xte)

    thr = np.quantile(np.abs(Zb_te), 0.999, axis=0) * 1e-6
    fire = (np.abs(Zb_te) > np.maximum(thr, 1e-9)).mean(0)
    alive = fire > 0.001
    ali = np.nonzero(alive)[0]
    print(f"[compare] alive bilinear latents: {alive.sum()}/{C.N_FEATURES}")

    sel = ali if len(ali) <= 1500 else np.random.default_rng(0).choice(ali, 1500, replace=False)
    r2_x = ridge_r2(Xtr, Zb_tr[:, sel], Xte, Zb_te[:, sel], ALPHAS, device=DEV)
    r2_sae = ridge_r2(Hl_tr, Zb_tr[:, sel], Hl_te, Zb_te[:, sel], ALPHAS, device=DEV)
    r2_x = np.clip(r2_x, -1, 1); r2_sae = np.clip(r2_sae, -1, 1)

    nonlinear = r2_x < 0.25
    novel = r2_sae < 0.50
    headline = nonlinear & novel
    out = dict(
        target=TAG, model="UCE-100M (Rosen et al. 2026)", n_features=C.N_FEATURES, k=C.K, steps=STEPS,
        fidelity=dict(linear_ve=ve_lin, bilinear_ve=ve_bil, gate_within_0p05=abs(ve_lin - ve_bil) < 0.05),
        n_alive=int(alive.sum()), n_evaluated=int(len(sel)),
        r2_x=dict(median=float(np.median(r2_x)), mean=float(r2_x.mean())),
        r2_sae=dict(median=float(np.median(r2_sae)), mean=float(r2_sae.mean())),
        redundancy_frac=float((r2_sae >= 0.50).mean()),
        genuinely_nonlinear_frac=float(nonlinear.mean()),
        novel_nonlinear_frac=float(headline.mean()),
        novel_nonlinear_ids=[int(sel[i]) for i in np.nonzero(headline)[0]],
    )
    (C.RESULTS / f"{TAG}_comparison.json").write_text(json.dumps(out, indent=2))
    np.savez(C.RESULTS / f"{TAG}_per_latent.npz", latent_ids=sel, r2_x=r2_x, r2_sae=r2_sae)
    print(f"\n==== UCE-100M {TAG} pre-registered comparison ====")
    print(f"  fidelity: linear VE={ve_lin:.3f}  bilinear VE={ve_bil:.3f}  gate={'PASS' if out['fidelity']['gate_within_0p05'] else 'FAIL'}")
    print(f"  R2_x median={out['r2_x']['median']:.3f}  (nonlinear frac R2x<0.25 = {out['genuinely_nonlinear_frac']:.3f})")
    print(f"  R2_SAE median={out['r2_sae']['median']:.3f}  (redundancy R2sae>=0.50 = {out['redundancy_frac']:.3f})")
    print(f"  NOVEL-NONLINEAR fraction = {out['novel_nonlinear_frac']:.3f}  ({len(out['novel_nonlinear_ids'])} latents)")


if __name__ == "__main__":
    main()
