# S3 — what makes a model warp a manifold's metric? And does the real claim survive a noise floor?

**Run 2026-08-19.** Scripts `s3_metric_warp.py` (synthetic, 12 models), `s3b_real_c2s_null.py`
(the real C2S data). Results `results/s3_metric_warp.json`, `results/s3b_real_c2s_null.json`.

## Headline

The synthetic run was designed to find *what causes* a metric warp. Its most important output turned
out to be something else: **knot-gap CV has a large noise floor that nobody in this programme had
measured**, and applying that floor to the real C2S data changes the status of the one certified
model-specific geometry claim in the corpus.

- Under **the paper's own binning**, C2S's gap CV of 0.318 sits **within noise of its own shuffle
  floor** (0.320 measured, null 0.284 ± 0.064, excess +0.036, **z +0.57, p = 0.256**).
- Under **equal cells per bin**, the floor drops to 0.146 and a real effect appears for both
  representations: C2S excess **+0.112 (z +2.88, p = 0.012)**, raw expression excess **+0.081
  (z +4.66, p < 0.001)**.

So the claim does not survive as stated, and survives in weakened form once bin sizes are matched.
The reason is simple and mechanical: the paper's bins hold **70 to 533 cells**, a 7.6× spread, and
unequal bin sizes inflate gap CV because centroid noise is larger where bins are small.

## Part 1 — Synthetic: what causes a warp

**Final: four arms × 8 seeds, 20,000 training cells, 3,000 steps, val_corr 0.71–0.79 throughout.**
`uniform` is the control (nothing manipulated); `sharp` puts half the phase genes inside the arc so
the emitted gene distribution turns over fastest there; `occupancy` puts 3× more cells there; `noisy`
makes cells inside the arc harder to predict. `stretch` = mean gap inside the arc / mean gap outside;
only **model − data** is a model property.

| arm | model stretch | data stretch | **model − data** | seeds > 0 | vs `uniform` |
|---|---:|---:|---:|---:|---|
| uniform (control) | 1.007 ± 0.091 | 0.992 ± 0.015 | +0.015 ± 0.089 | 4/8 | — |
| **sharp** (output-change) | 1.202 ± 0.068 | 1.032 ± 0.013 | **+0.170 ± 0.064** | **8/8** | **+0.155, t = +4.00, p = 0.0016** |
| occupancy (density) | 0.842 ± 0.059 | 0.774 ± 0.012 | +0.068 ± 0.053 | 7/8 | +0.054, t = +1.46, p = 0.17 |
| **noisy** (entropy) | 1.040 ± 0.069 | 1.139 ± 0.017 | **−0.099 ± 0.062** | **0/8** | **−0.114, t = −2.96, p = 0.011** |

**The pre-registered prediction is supported.** A model warps the metric of a manifold it inherits
**where its own output distribution turns over fastest**. That is the only arm that separates from
the control, it does so in 8 of 8 seeds, and the effect is ~11× the control's mean.

**The two rival hypotheses fail, in different ways.**
- **Density does not explain it.** `occupancy` is +0.068 and does not clear the control (p = 0.17).
  Note what the data arm does here: the manipulated arc is *compressed* in both representations
  (data stretch 0.774), and the model mostly inherits that. Density changes the geometry, but the
  model is not adding to it.
- **Entropy is refuted with the sign against it.** `noisy` is **−0.099, 0/8 seeds positive**,
  p = 0.011 against the control. Making an arc harder to predict makes the model's metric *less*
  stretched there than the data's. Whatever drives the warp, it is not prediction difficulty.

**One caveat that limits the strength of this.** CV excess over each representation's own shuffle
floor is ≈ 0 or negative in three of four arms — the *global* non-uniformity of the model's gaps is
not above its noise floor. The `sharp` effect is a **localisation** result (the stretch sits where
the manipulation is) rather than a claim that the model's loop is globally more uneven than the
data's. The arc-specific statistic detects it; the global CV statistic does not.

**The design error this caught.** My first run reported model CV 0.137 against data CV 0.023 in the
*uniform control* and I initially read that as "models warp metrics generically". It is not. The
shuffle floors are 0.144 for the 192-d model space and 0.156 for the 1000-d data space, so the
model's raw CV was *at* its floor and the data's was far *below* its floor. Comparing raw gap CV
across two representations with different centroid noise is not a valid comparison. The run was
stopped and restarted with a per-representation null.

## Part 2 — The same test on the real C2S data

Cached C2S-2B L21 activations (3000 × 2304) and the canonical substrate, 500 shuffle draws.

**Reproduction check.** Rebuilding the 12 knots from the activations and wrapped φ gives gap
CV **0.3200** against the stored artifact's **0.3182** — the paper's number is reproduced.

| binning | representation | CV | null | excess | z | p |
|---|---|---:|---:|---:|---:|---:|
| **paper's (uneven, n 70–533)** | C2S-2B L21 | 0.320 | 0.284 ± 0.064 | +0.036 | **+0.57** | **0.256** |
| | raw expression | 0.227 | 0.252 ± 0.026 | −0.025 | −0.93 | 0.844 |
| **equal n per bin (n = 70)** | C2S-2B L21 | 0.258 | 0.146 ± 0.039 | **+0.112** | **+2.88** | **0.012** |
| | raw expression | 0.134 | 0.053 ± 0.017 | **+0.081** | **+4.66** | **<0.001** |

Whitened PC20 versions agree: C2S excess +0.158 (z +4.99), expression +0.077 (z +2.42).

**What this means.**
1. The headline ratio "C2S is ~1.6× less uniform than the data" (0.318 / 0.193) is **inflated by
   unequal bin sizes**. At matched n the raw CVs are 0.258 vs 0.134, and the excesses over each
   representation's own floor are +0.112 vs +0.081 — a ratio of about **1.4**, not 1.6.
2. The model **is** more non-uniform than the data once floors are applied, so the qualitative claim
   survives. But raw expression is *also* significantly non-uniform (z +4.66) — the paper's framing
   treats the data as the uniform reference, and it is not.
3. Under the paper's own binning the effect is **not distinguishable from noise**. Any restatement of
   this claim must use matched bin sizes and quote a floor.

## Part 3 — The location claim is frame-dependent

My rebuilt gaps and the stored gaps are the **same multiset, reflected and rotated**: mine[10] = 70.44
↔ stored[7] = 70.79, mine[11] ↔ stored[6], mine[0] ↔ stored[5]. Gap CV is invariant to that, which is
why the magnitudes match to three decimals.

But the *location* claim — "the stretch is at the G1→S restriction point" — is **not** frame-invariant,
and the programme's own audit already records that the phase angle has **arbitrary handedness per
dataset** and that handedness is not identifiable from two antipodal clusters. The magnitude claim
can be defended; the "restriction point" interpretation rests on a frame convention that has not
been independently pinned down.

Separately, φ is stored in **[−π, π]** in `k562_cc_substrate.npz`. Feeding it to `manifold_fit.knots`
with `kind='cyclic'` unwrapped dumps 1,715 of 3,000 cells into bin 0 and leaves six bins empty. Any
new consumer must wrap first.

## What changes

For [FORMATION.md](../FORMATION.md) Part 2 and [STORY.md](../STORY.md) P12, which both call the
metric stretch "the only certified model-specific geometry claim":

> **Certified, with corrections.** The warp is real at matched bin sizes (C2S excess +0.112, z +2.88)
> but the data is also non-uniform (+0.081, z +4.66), the model-over-data ratio is ~1.4 rather than
> 1.6, the effect is **not significant under the binning the paper used**, and the "G1→S" location
> depends on a phase frame whose handedness the programme's own audit calls non-identifiable.
> Together with [S0](S0_RESULTS.md) — the warp buys no single-cell resolution — what remains is:
> the model's mean path around the loop is somewhat more unevenly spaced than the data's, for
> reasons unknown, with no demonstrated consequence.

## Next

- More seeds on `sharp` vs `uniform` (the only arm where the pre-registered prediction is live).
- Re-run the 27B knots through the same floor; only 2B was tested here.
- The floor should be applied to every gap-CV number in the cell-cycle paper, not just L21.
