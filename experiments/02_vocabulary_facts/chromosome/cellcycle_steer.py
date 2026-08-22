"""CAN YOU MOVE AROUND THE CELL-CYCLE CIRCLE? — steering a CYCLIC coordinate (Ihor, 2026-07-18).

THE TARGET. §7 demoted the cell-cycle circle to "not a manifold" because persistent homology found no
IRREDUCIBLE loop (H1 z ≈ −0.08, at the covariance-matched floor). But that null asks "is there a hole not
explained by linear structure", and the answer being no just means the loop is a LINEARLY-EMBEDDED ELLIPSE —
which is exactly a manifold you can move around. Meanwhile the supervised probe reads phase at 0.929 circular
correlation. So the circle was shelved on a criterion irrelevant to traversability. This is the right test.

THE OPERATOR — why a fixed direction MUST fail here, and what to use instead. route_cellcycle/cc_common.py
pre-registers the theory: for ANY closed curve and ANY fixed vector w,   ∮ w·dx = w·∮dx = 0.
A constant steering direction does exactly ZERO net work around a cycle — it advances phase while w·t̂ > 0 and
retreats once w·t̂ < 0, stalling about a quarter-turn in ("ON-MANIFOLD BUT PHASE-STALLED"). That is vector
calculus, not an empirical claim, and it is why the earlier phase-advance operator stalled. Chromosome (§12) and
the lineage switches (§13) are a CATEGORY and a LINEAR axis, so a fixed push is fine there. A CIRCLE needs a
ROTATING operator. So we do not push along one vector; we build the 2-D PHASE PLANE and steer to an ANGLE:

    u = Σ_g cos(φ_g)(e_g − ē),  v = Σ_g sin(φ_g)(e_g − ē)      (first Fourier components of embedding vs phase)
    d(Δ) = cos(Δ)·û + sin(Δ)·v̂                                  (unit; "make this context look like phase Δ")

THE TEST — full-circle tracking, not a binary flip. Sweep Δ over the whole circle and ask whether the model's
own predicted phase ROTATES WITH IT. A usable circular coordinate gives ψ(Δ) tracking Δ all the way round
(circular correlation → 1, slope → 1, and it must WRAP). This is a far stronger signature than "category mass
goes up": it is a predicted functional form over 360°, which noise cannot fake.

NON-CIRCULARITY (the §5 trap). The phase plane is built from a DIRECTION-half of the marker genes in INPUT
(embed_tokens) space; the readout is the model's own logits over the HELD-OUT READ-half, at DIFFERENT positions
than the push. Direction genes and readout genes are disjoint; no fitted probe anywhere.

READOUT. Among the READ-half phase genes only, softmax their logits (relative preference among phase markers)
and take the probability-weighted circular mean of their annotated phases:
    ψ = atan2( Σ p_g sin φ_g , Σ p_g cos φ_g )
Reported as the SHIFT from the unsteered baseline, wrapped.

SUBSTRATE. Replogle K562 non-targeting controls — the project's canonical cell-cycle substrate (~71–81%
cycling, near-uniform phase occupancy). Tissue-gating is now known to dominate these causal tests (§13), so the
cell cycle must be run where the cell cycle actually turns.

Run: ../../.venv_state/bin/python -u cellcycle_steer.py [n_cells=40] [seed=0]
Out: results/cellcycle_steer.json
"""
import os, sys, json, pickle, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "route_cellcycle"))
import gm_lib as G
import steer_lib as SL
from cc_common import WAVE, WAVE_ORDER, REPLOGLE, wrap, circ_mean, circ_corr

PHASE_ANGLE = {k: i * (2 * np.pi / len(WAVE_ORDER)) for i, k in enumerate(WAVE_ORDER)}   # G1S,S,G2,G2M
N_ANGLES = 8                      # steering targets around the full circle
# DOSE-RESPONSE over push strength (units of mean gene-embedding norm). A first pass at alpha=4 -- the value
# validated for chromosome/lineage -- produced a large push-dominated shift in EVERY direction (steer AND
# random landed at -50..-160 deg), the signature of SATURATION: slam the residual hard enough and the readout
# goes to a fixed attractor regardless of which way you pushed. A cyclic coordinate needs a gentle enough push
# to rotate rather than override, so the operative regime has to be found, not assumed.
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_RAND = 3
MAX_LEN = 512
SEED = 0


def load_k562(st, n_cells, seed):
    """Cycling cells, rank-value tokenised exactly as the harness does elsewhere."""
    name_id = pickle.load(open(G.ENSMAP, "rb"))
    with h5py.File(REPLOGLE, "r") as f:
        v = f["var"]["gene_name_index"]
        gn = np.array([x.decode() if isinstance(x, bytes) else x for x in v[:]]).astype(str)
        X = f["X"]
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n, n_cells, replace=False))
        E = np.stack([np.asarray(X[int(i), :], dtype=np.float32) for i in sel])
    var_idx, token_ids, medians = st.tok.make_var_mapping([name_id.get(s) for s in gn])
    seqs = []
    for i in range(len(E)):
        tot = E[i].sum() or 1.0
        en = np.log1p(E[i] / tot * 1e4)[var_idx]
        nz = en > 0
        norm = np.zeros_like(en); norm[nz] = en[nz] / medians[nz]
        order = np.argsort(-norm[nz])
        seqs.append(token_ids[np.nonzero(nz)[0][order][: MAX_LEN - 2]])
    return [s for s in seqs if len(s) >= 16]


def main(n_cells=40, seed=SEED):
    st = SL.Steerer()
    rng = np.random.default_rng(seed)

    # ---- symbol -> token, and the phase-annotated marker genes
    tokmap = json.load(open(SL.TOKMAP))
    ens2sym = {e: s.upper() for s, e in pickle.load(open(G.ENSMAP, "rb")).items()}
    s2t = {}
    for ens, tid in tokmap.items():
        s = ens2sym.get(ens)
        if s is not None and isinstance(tid, int):
            s2t.setdefault(s, tid)

    dir_tok, dir_phi, read_tok, read_phi = [], [], [], []
    for k in WAVE_ORDER:
        toks = [s2t[g] for g in WAVE[k] if g in s2t]
        rng.shuffle(toks)
        h = len(toks) // 2
        dir_tok += toks[:h];  dir_phi += [PHASE_ANGLE[k]] * h
        read_tok += toks[h:]; read_phi += [PHASE_ANGLE[k]] * (len(toks) - h)
        print(f"[phase] {k:<4} φ={PHASE_ANGLE[k]:.2f}rad  {h} direction / {len(toks)-h} readout genes")
    read_tok = np.array(read_tok); read_phi = np.array(read_phi)

    # ---- the 2-D PHASE PLANE: first Fourier components of embedding vs phase (direction genes only)
    EMB = st.model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
    mean_norm = float(np.linalg.norm(EMB, axis=1).mean())
    Ed = EMB[np.array(dir_tok)]
    Ec = Ed - Ed.mean(0)
    phi = np.array(dir_phi)
    u = (np.cos(phi)[:, None] * Ec).mean(0)
    v = (np.sin(phi)[:, None] * Ec).mean(0)
    # Gram-Schmidt -> orthonormal basis of the phase plane
    uh = u / (np.linalg.norm(u) + 1e-12)
    v = v - (v @ uh) * uh
    vh = v / (np.linalg.norm(v) + 1e-12)
    print(f"[plane] built from {len(dir_tok)} genes; |u|,|v| orthonormalised; "
          f"cos-sin overlap after GS = {abs(uh @ vh):.2e}")

    def dir_at(delta):
        w = np.cos(delta) * uh + np.sin(delta) * vh
        return SL.Direction(vec=w, name=f"phase:{delta:.2f}", basis="embed_tokens")

    rand_dirs = [SL.random_direction(st.xt, seed=3000 + k) for k in range(N_RAND)]
    deltas = np.arange(N_ANGLES) * (2 * np.pi / N_ANGLES)

    seqs = load_k562(st, n_cells, seed)
    print(f"[cells] {len(seqs)} K562 cells, mean {np.mean([len(s) for s in seqs]):.0f} tokens\n", flush=True)

    def induced_phase(logits, read_pos):
        """prob-weighted circular mean of READ-gene phases, softmaxed WITHIN the read set."""
        out = []
        for p in read_pos:
            lg = logits[p][read_tok]
            w = np.exp(lg - lg.max()); w = w / (w.sum() + 1e-12)
            out.append(np.arctan2((w * np.sin(read_phi)).sum(), (w * np.cos(read_phi)).sum()))
        return circ_mean(np.array(out))

    rec = {a: {"steer": [], "rand": []} for a in ALPHAS}      # alpha -> rows of (delta, shift)
    for ci, s in enumerate(seqs):
        ids = np.concatenate([[st.tok.BOS], s, [st.tok.EOS]]).astype(np.int64)
        gp = np.arange(1, 1 + len(s))
        sh = rng.permutation(len(gp)); half = len(gp) // 2
        push_mask = np.zeros(len(ids), bool); push_mask[gp[sh[:half]]] = True
        read_pos = gp[sh[half:]]
        psi0 = induced_phase(st.logits(ids).numpy(), read_pos)
        for a in ALPHAS:
            push = a * mean_norm
            for d in deltas:
                with st.steering(dir_at(d), alpha=push, positions=push_mask, site="embed"):
                    psi = induced_phase(st.logits(ids).numpy(), read_pos)
                rec[a]["steer"].append((float(d), float(wrap(psi - psi0))))
                # random control shares the push magnitude; rotate which random dir so it is not one fixed vector
                rd = rand_dirs[int(d / (2 * np.pi / N_ANGLES)) % len(rand_dirs)]
                with st.steering(rd, alpha=push, positions=push_mask, site="embed"):
                    psir = induced_phase(st.logits(ids).numpy(), read_pos)
                rec[a]["rand"].append((float(d), float(wrap(psir - psi0))))
        if (ci + 1) % 8 == 0:
            print(f"  {ci + 1}/{len(seqs)} cells", flush=True)
            try:
                import torch as _t
                if _t.backends.mps.is_available():
                    _t.mps.empty_cache()
            except Exception:
                pass

    # ---- does the induced phase TRACK the steering angle around the full circle, at ANY dose?
    out = dict(seed=seed, n_cells=len(seqs), alphas=ALPHAS, n_angles=N_ANGLES, by_alpha={})
    print(f"\n{'alpha':<7} {'steer circ_corr':<17} {'rand circ_corr':<16} per-angle mean induced shift (deg)")
    print("-" * 108)
    for a in ALPHAS:
        row = {}
        for tag in ("steer", "rand"):
            arr = np.array(rec[a][tag])
            d, shift = arr[:, 0], arr[:, 1]
            per = {f"{np.degrees(dd):.0f}": float(np.degrees(circ_mean(shift[np.isclose(d, dd)])))
                   for dd in deltas}
            row[tag] = dict(circ_corr=float(circ_corr(d, shift)), n=int(len(d)), per_angle_deg=per)
        out["by_alpha"][a] = row
        ps = "  ".join(f"{v:+.0f}" for v in row["steer"]["per_angle_deg"].values())
        print(f"{a:<7} {row['steer']['circ_corr']:<+17.3f} {row['rand']['circ_corr']:<+16.3f} {ps}")

    best = max(ALPHAS, key=lambda a: out["by_alpha"][a]["steer"]["circ_corr"])
    bs = out["by_alpha"][best]["steer"]["circ_corr"]; br = out["by_alpha"][best]["rand"]["circ_corr"]
    tracks = bs > 0.3 and bs > 2 * abs(br)
    out["best_alpha"] = best
    out["verdict"] = (f"TRAVERSABLE at alpha={best} — induced phase tracks the steering angle (r={bs:+.3f} vs "
                      f"random {br:+.3f})" if tracks else
                      f"NOT traversable by this operator at any tested dose (best r={bs:+.3f} at alpha={best})")
    print(f"\n  VERDICT: {out['verdict']}")
    print("  (a real cyclic coordinate: steer r rises then falls with alpha — gentle enough to rotate, not")
    print("   so hard it saturates — while the random control stays ~0 at every dose.)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "cellcycle_steer.json"), "w"), indent=1)
    print(f"\n[done] -> results/cellcycle_steer.json")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    main(n, sd)
