"""surface_forms — does C2S carry a gene -> cell-cycle-phase association in its WEIGHTS?

THE ONE ARM OF gene_phase_map.py THAT SURVIVED. That script had three arms; two were invalid and are documented
as such in RESULTS:
  A1 binary vs a covariance baseline  -- premise broken. It stratified genes by "informativeness" = the
     resultant length of their expression-weighted phase, assuming low concentration meant the covariance was
     uninformative. It does not: |r| measures SPREAD, not accuracy. In the low-|r| half the baseline's mean
     angle still lands a median of 14 deg from its own class pole and scores AUROC 0.993. Nothing was held out.
  A2 graded vs Whitfield              -- the model's apparent win (paired Wilcoxon p=0.027 on the low half) is
     an alignment artifact. The model's placement angles are tightly clustered (R=0.856, 10-90pct 278-347 deg),
     so a globally fitted rotation makes any sub-arc look accurate. Circular correlation with Whitfield is
     NEGATIVE (-0.784 all, -0.272 low), i.e. there is no graded correspondence to fit. Retracted.

WHAT ACTUALLY WORKS, and why the baseline comparison was a red herring: in this design the model never sees
expression covariance at all. It sees ONE GENE NAME inserted at a fixed rank into a phase-neutral backbone.
So the model is already held out from the covariance by construction, and the right null is not a covariance
baseline -- it is a token with the same surface statistics but no gene identity. That null is the anagram arm,
and it sits at exactly chance (AUROC 0.501, p=0.51) while real symbols reach 0.966.

THIS SCRIPT sharpens the surface-form comparison, which is the days-of-week analogue proper. Ranked by edit
distance from the panel symbol, so that string proximity and token identity can be told apart:

  human      CCNB1   the in-panel symbol                                          (edit distance 0)
  nearmiss   CQNB1   ONE character substituted, verified absent from the gene      (edit distance 1)
                     universe. CLOSER to the panel symbol than the mouse form is.
  lower      ccnb1   pure lowercase; not a real species convention                 (case-only)
  mouse      Ccnb1   the real mouse ortholog symbol. ZERO occurrences in any K562  (case-only)
                     cell sentence, so it carries no covariance in this dataset.
  anagram    NBCC1   letters permuted, verified absent from the gene universe.     (identity destroyed)

THE DECISIVE CONTRAST IS mouse vs nearmiss. nearmiss is a SMALLER edit than mouse, so if placement tracked mere
string proximity nearmiss would win. If instead mouse >> nearmiss, what the model responds to is the token being
a real gene it has seen -- knowledge in the weights, not similarity of spelling, and not this dataset's input.

lower vs mouse then separates "robust to case" from "knows the mouse symbol specifically": if lower ~= mouse the
effect is surface-form generalisation over case; if mouse > lower it is more specific than that. Either way
neither is in this dataset's covariance, which is the claim under test.

LIMIT ON THE NEGATIVE CONTROLS: "absent from the gene universe" is checked against the union of the K562 and
RPE1 panels (~10k symbols), not against a complete HGNC dump, so a constructed string could in principle be a
real symbol outside both panels. Reported per form so it can be audited.

Out: results/surface_forms.json
"""
from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence
from cc_phase import phase_angle_oriented, S_GENES, G2M_GENES
from synth_sweep import PEAK, insert_at


def cmean_deg(a):
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360)


def cR(a):
    return float(np.abs(np.mean(np.exp(1j * np.radians(np.asarray(a, float))))))


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    r = score.argsort().argsort() + 1.0
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)) if n0 and n1 else float("nan")


def perm_auroc(score, y, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    obs = auroc(score, y)
    null = np.array([auroc(rng.permutation(np.asarray(score, float)), y) for _ in range(n)])
    return obs, float((np.sum(null >= obs) + 1) / (n + 1)), float(null.mean()), float(null.std())


def edit1(a, b):
    """Levenshtein, small strings."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--universe-h5ad", default=None, help="second panel, to widen the not-a-real-gene check")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-backbone", type=int, default=24)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/surface_forms.json")
    a = ap.parse_args()
    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    vn = np.char.upper(np.asarray(ad.var_names).astype(str))
    panel = set(vn)
    universe = set(panel)
    if a.universe_h5ad and os.path.exists(a.universe_h5ad):
        u = anndata.read_h5ad(a.universe_h5ad)
        universe |= set(np.char.upper(np.asarray(u.var_names).astype(str)))
    print(f"gene universe for negative controls: {len(universe)} symbols", flush=True)

    theta, _, _, oinfo = phase_angle_oriented(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sc = StandardScaler().fit(H)
    dec = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
    P0 = dec.predict(sc.transform(H)); C0 = P0.mean(0)
    real_rad = float(np.median(np.linalg.norm(P0 - C0, axis=1)))
    mu_S, mu_G = oinfo["s_peak_deg"], oinfo["g2m_peak_deg"]

    tirosh = {}
    for g in S_GENES:   tirosh[g.upper()] = 0
    for g in G2M_GENES: tirosh[g.upper()] = 1
    probes = sorted(g for g in tirosh if g in panel)
    y = [tirosh[g] for g in probes]
    print(f"probes: {len(probes)} ({len(y) - sum(y)} S / {sum(y)} G2M) | real-cell poles S={mu_S} G2M={mu_G}",
          flush=True)

    rng = np.random.default_rng(0)
    AL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def nearmiss(g):
        for _ in range(500):
            i = int(rng.integers(0, len(g)))
            c = AL[int(rng.integers(0, 26))]
            s = g[:i] + c + g[i + 1:]
            if s != g and s not in universe:
                return s
        return g + "X"

    def anagram(g):
        for _ in range(500):
            s = "".join(rng.permutation(list(g)))
            if s != g and s not in universe:
                return s
        return g[::-1]

    forms = {"human":    {g: g for g in probes},
             "nearmiss": {g: nearmiss(g) for g in probes},
             "lower":    {g: g.lower() for g in probes},
             "mouse":    {g: g.capitalize() for g in probes},
             "anagram":  {g: anagram(g) for g in probes}}
    for f, m in forms.items():
        collide = sum(1 for g in probes if m[g] in universe and f in ("nearmiss", "anagram"))
        print(f"  {f:<9} e.g. {m[probes[0]]:<10} mean edit distance {np.mean([edit1(g, m[g]) for g in probes]):.2f}"
              + (f" | collisions with real symbols: {collide}" if f in ("nearmiss", "anagram") else ""), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager").eval()
    device = next(model.parameters()).device
    maxlen = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    cc_strip = (set(S_GENES) | set(G2M_GENES) | set(PEAK)) & panel
    order = rng.permutation(ad.n_obs)
    backbones = []
    for ci in order:
        gs = [x for x in anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes) if x.upper() not in cc_strip]
        if len(gs) >= 200:
            backbones.append(gs[: a.max_genes - 1])
        if len(backbones) >= a.n_backbone:
            break

    def place(bb, sym):
        cs = build_cell_sentence(insert_at(bb, [sym], [a.rank]))
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=maxlen)
        pos = int(enc["input_ids"].shape[1]) - 1
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        h = out.hidden_states[a.layer + 1][0, pos].float().cpu().numpy()[None, :]
        p = dec.predict(sc.transform(h))[0] - C0
        return float(np.degrees(np.arctan2(p[1], p[0])) % 360), float(np.linalg.norm(p))

    res = {"orientation": oinfo, "poles": {"S": mu_S, "G2M": mu_G}, "real_radius": real_rad,
           "n_probes": len(probes), "n_backbone": len(backbones), "forms": {}}
    print(f"\n{'form':<10}{'edit':>6}{'AUROC':>9}{'p':>9}{'null':>8}{'S mean':>9}{'G2M mean':>10}"
          f"{'sep':>7}{'radius':>8}", flush=True)
    for fname, mapping in forms.items():
        ang, rad = {}, {}
        for g in probes:
            per = [place(bb, mapping[g]) for bb in backbones]
            ang[g] = cmean_deg([p[0] for p in per]); rad[g] = float(np.mean([p[1] for p in per]))
        score = [np.cos(np.radians(ang[g] - mu_G)) - np.cos(np.radians(ang[g] - mu_S)) for g in probes]
        obs, p, nmu, nsd = perm_auroc(score, y)
        sm = cmean_deg([ang[g] for g in probes if tirosh[g] == 0])
        gm = cmean_deg([ang[g] for g in probes if tirosh[g] == 1])
        d = abs(sm - gm) % 360; sep = min(d, 360 - d)
        ed = float(np.mean([edit1(g, mapping[g]) for g in probes]))
        rr = float(np.mean(list(rad.values())) / real_rad)
        res["forms"][fname] = dict(auroc=obs, p=p, null_mean=nmu, null_sd=nsd, mean_edit_distance=ed,
                                   S_mean_deg=sm, G2M_mean_deg=gm, separation_deg=sep, radius_vs_real=rr,
                                   angles=ang, example={g: mapping[g] for g in probes[:6]})
        print(f"{fname:<10}{ed:>6.2f}{obs:>9.3f}{p:>9.4f}{nmu:>8.3f}{sm:>9.1f}{gm:>10.1f}{sep:>7.1f}{rr:>8.2f}",
              flush=True)

    h, n, m_, l_, an = (res["forms"][k]["auroc"] for k in ("human", "nearmiss", "mouse", "lower", "anagram"))
    print(f"\nDECISIVE CONTRAST  mouse {m_:.3f} vs nearmiss {n:.3f} "
          f"(nearmiss is the SMALLER edit: {res['forms']['nearmiss']['mean_edit_distance']:.2f} vs "
          f"{res['forms']['mouse']['mean_edit_distance']:.2f})", flush=True)
    if m_ > n + 0.05 and res["forms"]["mouse"]["p"] < 0.05:
        print("  -> a real unseen-in-this-dataset symbol beats a closer non-symbol: the association tracks TOKEN "
              "IDENTITY, not spelling, and is not in this dataset's covariance.", flush=True)
    elif n >= m_:
        print("  -> string proximity explains it; NO claim about gene knowledge is supported.", flush=True)
    print(f"CASE CONTRAST      mouse {m_:.3f} vs lower {l_:.3f} -> "
          + ("case-robust surface generalisation (not mouse-specific)" if abs(m_ - l_) < 0.05
             else "mouse form is more than a case change"), flush=True)
    print(f"FLOOR              anagram {an:.3f} (p={res['forms']['anagram']['p']:.3f})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
