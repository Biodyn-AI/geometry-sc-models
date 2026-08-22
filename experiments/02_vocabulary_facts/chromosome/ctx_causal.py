"""IS THE FUNCTIONAL-CONTEXT DIRECTION CAUSALLY USED? (upgrades Level 1 from 'represented' to 'used')

Level 1 established that genes move along functional axes (nuclear/transcriptional vs surface/secreted) with cell
context. That is a REPRESENTATION fact. This asks whether the model's COMPUTATION uses that direction: if we push
some of a cell's genes along the functional axis, does the model raise its predicted probability of
functionally-matching genes at the OTHER, untouched positions?

DESIGN (the chromosome-steering playbook, per STEERING_TOOL.md):
  - Direction = the functional-context axis at layer 4 (mean raw hidden state of nuclear-pole genes minus
    surface-pole genes; the same axis Level 1 is about), injected after layer 3 (= the hidden_states[4] tap).
  - Push it into a RANDOM HALF of a cell's gene positions; READ the model's own next-gene logits at the OTHER
    half. Because read positions are disjoint from steer positions, any effect travels through ATTENTION, not
    local pass-through.
  - Readout = the MODEL'S OWN logits (no fitted probe -> non-circular): mean logit of nuclear-pole gene tokens
    (target) vs surface-pole gene tokens (control). specificity = d_target - d_control.
  - Dose-response over alpha; norm-matched RANDOM direction as the control (must stay flat).
NON-CIRCULARITY: the direction is built from layer-4 representations; the readout is the model's output logits
for GO-defined gene sets; read positions never overlap steer positions. Nothing is read in the basis it was
defined in.

A positive result -- target logits rise with alpha, control flat, random flat, at disjoint positions -- means
the functional-context direction is a channel the model's computation actually uses.

Out: results/ctx_causal.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  *[_os.pardir] * 3, 'src'))
from geomsc.paths import DATA as _DATA, MODELS as _MODELS  # noqa: E402
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, h5py, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import steer_lib as SL
MSETUP = f"{_DATA}/maxtoki/setup"
sys.path.insert(0, MSETUP)
RES = os.path.join(HERE, "results")
NAME_ID = f"{_MODELS}/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
G2G = f"{_DATA}/perturb/gene2go_all.pkl"
TS = f"{_DATA}/raw"
PANEL = "tabula_sapiens_immune_subset_20000.h5ad"
# Axis is parameterised so the causal test can be shown for MORE than one functional direction and layer
# (a reviewer's cherry-pick worry). Defaults reproduce the headline nuclear-vs-surface @ layer 4 run.
AX_DEFS = {
    "nuc_surf":       (["GO:0005634", "GO:0000785", "GO:0003677"], ["GO:0005886", "GO:0005576", "GO:0005615"]),
    "mito_cyto":      (["GO:0005739"], ["GO:0005856"]),
    "trans_transport": (["GO:0006355", "GO:0003700"], ["GO:0006811", "GO:0038023"]),
}
AXIS = os.environ.get("AXIS", "nuc_surf")
NUC, SURF = AX_DEFS[AXIS]
SITE = int(os.environ.get("SITE", 3))   # inject after this layer == the hidden_states[SITE+1] tap
OUTNAME = "ctx_causal.json" if (AXIS == "nuc_surf" and SITE == 3) else f"ctx_causal_{AXIS}_L{SITE+1:02d}.json"
N_CELLS, MAX_LEN, SEED = 30, 512, 0


def tokenise_cells(tok, n):
    from maxtoki_adapter import MaxTokiTokenizer  # noqa
    with h5py.File(os.path.join(TS, PANEL), "r") as f:
        ens = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["_index"][:]]).astype(str)
        ens = np.array([e.split(".")[0] for e in ens])
        var_idx, token_ids, medians = tok.make_var_mapping(list(ens))
        pos = np.full(len(ens), -1, np.int64); pos[var_idx] = np.arange(len(var_idx))
        X = f["X"]; N = int(X.attrs["shape"][0]); indptr = X["indptr"][:]
        sel = np.sort(np.random.default_rng(SEED).choice(N, n, replace=False))
        seqs = []
        for r in sel:
            s, e = int(indptr[r]), int(indptr[r + 1]); idx, val = X["indices"][s:e], X["data"][s:e].astype(np.float32)
            keep = pos[idx] >= 0
            if not keep.any():
                continue
            j = pos[idx[keep]]
            en = np.log1p(val[keep] / (float(val.sum()) or 1.0) * 1e4); nz = en > 0
            if nz.sum() < 40:
                continue
            norm = en[nz] / np.maximum(medians[j[nz]], 1e-9)
            order = np.argsort(-norm)[: MAX_LEN - 2]
            seqs.append(token_ids[j[nz][order]].astype(np.int64))
    return seqs


def main():
    st = SL.Steerer()                                       # MaxToki-217M
    ens2sym = {e: s.upper() for s, e in pickle.load(open(NAME_ID, "rb")).items()}
    g2g = {k.upper(): set(v) for k, v in pickle.load(open(G2G, "rb")).items() if isinstance(v, (set, list, tuple))}
    tokmap = json.load(open(f"{MSETUP}/token_dictionary.json"))
    ens2tid = {k: int(v) for k, v in tokmap.items()}

    # gene-token sets for the two poles (target/control readout + axis construction)
    def pole_tokens(terms):
        out = []
        for ens, tid in ens2tid.items():
            s = ens2sym.get(ens)
            if s and s in g2g and g2g[s] & set(terms):
                out.append(int(tid))
        return np.array(sorted(set(out)))
    nuc_tok, surf_tok = pole_tokens(NUC), pole_tokens(SURF)
    both = set(nuc_tok) & set(surf_tok)
    nuc_tok = np.array([t for t in nuc_tok if t not in both]); surf_tok = np.array([t for t in surf_tok if t not in both])
    print(f"[setup] {len(nuc_tok)} nuclear-pole tokens, {len(surf_tok)} surface-pole tokens", flush=True)

    # functional-context axis in RAW layer-4 hidden space (from the extraction; M is raw mean hidden state)
    z = np.load(os.path.join(RES, f"ctx_maxtoki_L{SITE+1:02d}.npz"), allow_pickle=True)
    M, counts, cap, genes = z["M"].astype(np.float32), z["counts"], int(z["cap"]), z["genes"].astype(str)
    full = (counts == cap).all(0)
    araw = np.full((len(genes), M.shape[-1]), np.nan, np.float32)
    for gi in range(len(genes)):
        cs = np.where(full[:, gi])[0]
        if len(cs):
            araw[gi] = M[:, cs, gi].mean((0, 1))
    gsym = [ens2sym.get(g) for g in genes]
    ia = [i for i, s in enumerate(gsym) if np.isfinite(araw[i, 0]) and s in g2g and g2g[s] & set(NUC)]
    ib = [i for i, s in enumerate(gsym) if np.isfinite(araw[i, 0]) and s in g2g and g2g[s] & set(SURF)]
    u = araw[ia].mean(0) - araw[ib].mean(0)
    d_func = SL.Direction(vec=u, name=f"{AXIS}@L{SITE+1}", basis=f"ctx_L{SITE+1}")
    d_rand = SL.random_direction(st.xt, seed=1, name="random")
    print(f"[setup] axis from {len(ia)} nuclear / {len(ib)} surface genes; ||u_raw||={np.linalg.norm(u):.2f}", flush=True)

    seqs = tokenise_cells(st.tok, N_CELLS)
    print(f"[setup] {len(seqs)} cells", flush=True)

    # calibrate alpha to the residual norm at the injection site
    h = st.hidden(np.concatenate([[st.tok.BOS], seqs[0], [st.tok.EOS]]), layer=SITE + 1)
    resnorm = float(np.linalg.norm(h[1:-1], axis=1).mean())
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
    alpha_units = [a * resnorm for a in alphas]              # in raw hidden-norm units
    print(f"[setup] residual norm at layer {SITE} ~ {resnorm:.1f}; alphas x that = {alpha_units}", flush=True)

    # SIGNED steering: +u (toward nuclear) vs -u (toward surface). A real causal channel FLIPS the readout with
    # the sign; the baseline nuclear/surface asymmetry and generic-perturbation effects do NOT depend on sign,
    # so they cancel in the swing = spec(+a) - spec(-a). Alphas kept in the non-destructive range.
    alphas_s = [0.25, 0.5, 1.0]
    au = [a * resnorm for a in alphas_s]
    rng = np.random.default_rng(SEED)
    swing = {"functional": {a: [] for a in alphas_s}, "random": {a: [] for a in alphas_s}}
    spec = {"functional_+": {a: [] for a in alphas_s}, "functional_-": {a: [] for a in alphas_s}}
    for si, s in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], s, [st.tok.EOS]]).astype(np.int64)
        gene_pos = np.arange(1, 1 + len(s)); rng.shuffle(gene_pos)
        steer_pos = list(gene_pos[: len(gene_pos) // 2]); read_pos = list(gene_pos[len(gene_pos) // 2:])
        d_rand_cell = SL.random_direction(st.xt, seed=1000 + si, name="random")   # FRESH per cell -> swing->0
        for label, d in [("functional", d_func), ("random", d_rand_cell)]:
            pos_rows = SL.dose_response(st, ids, d, steer_pos, read_pos, nuc_tok, surf_tok, au, site=SITE)
            neg_rows = SL.dose_response(st, ids, d, steer_pos, read_pos, nuc_tok, surf_tok, [-a for a in au], site=SITE)
            for k, a in enumerate(alphas_s):
                swing[label][a].append(pos_rows[k]["specificity"] - neg_rows[k]["specificity"])
                if label == "functional":
                    spec["functional_+"][a].append(pos_rows[k]["specificity"])
                    spec["functional_-"][a].append(neg_rows[k]["specificity"])
        if si % 10 == 0:
            print(f"    cell {si}/{len(seqs)}", flush=True)

    from math import comb
    out = {"alphas_xResidNorm": alphas_s, "site": SITE, "n_cells": len(seqs),
           "n_nuclear_tokens": len(nuc_tok), "n_surface_tokens": len(surf_tok), "signed": {}}
    print(f"\n{'alpha':<8} {'spec(+u)':<12} {'spec(-u)':<12} {'FUNC swing':<20} {'RAND swing':<18} {'func>rand'}")
    for a in alphas_s:
        fp = np.array(spec["functional_+"][a]); fm = np.array(spec["functional_-"][a])
        fs = np.array(swing["functional"][a]); rs = np.array(swing["random"][a])
        k = int((fs > rs).sum()); n = len(fs)
        p = float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
        out["signed"][f"alpha_{a}"] = dict(spec_plus=float(fp.mean()), spec_minus=float(fm.mean()),
                                           func_swing=float(fs.mean()), func_swing_sem=float(fs.std()/np.sqrt(n)),
                                           rand_swing=float(rs.mean()), func_gt_rand=k, n=n, sign_p=p)
        print(f"  {a:<6} {fp.mean():+.3f}       {fm.mean():+.3f}       {fs.mean():+.4f}±{fs.std()/np.sqrt(n):.4f}     "
              f"{rs.mean():+.4f}          {k}/{n} p={p:.1e}")

    mid = out["signed"]["alpha_0.5"]
    clean = (mid["spec_plus"] > 0 > mid["spec_minus"] and mid["func_swing"] > 2 * mid["func_swing_sem"]
             and mid["sign_p"] < 0.05 and mid["func_swing"] > 2 * abs(mid["rand_swing"]))
    out["verdict"] = (
        f"at 0.5xResid: spec(+u)={mid['spec_plus']:+.3f}, spec(-u)={mid['spec_minus']:+.3f}, "
        f"signed swing {mid['func_swing']:+.4f} (random {mid['rand_swing']:+.4f}), func>rand {mid['func_gt_rand']}/"
        f"{mid['n']} p={mid['sign_p']:.1e}. " +
        ("CAUSALLY USED — the readout FLIPS with the sign of the functional push (nuclear-gene logits rise toward "
         "+u, fall toward -u) at disjoint positions, dose-dependently, far beyond the sign-independent random "
         "control. Level 1 upgrades from 'represented' to 'represented AND causally used'."
         if clean else
         "NOT cleanly causal — the readout does not flip with the sign of the functional direction beyond the "
         "random control, so the earlier apparent effect was the baseline nuclear/surface asymmetry under "
         "generic perturbation. Level 1 stands as a representation result only.")
    )
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, OUTNAME), "w"), indent=1)
    print("[done] -> results/ctx_causal.json")


if __name__ == "__main__":
    main()
