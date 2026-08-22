# S0 — Does the metric stretch buy local phase resolution?

**Written before running. 2026-08-19.**

## The question

The one certified model-specific geometry claim in this programme is that C2S stretches the
cell-cycle metric at the G1→S restriction point: consecutive knot gaps CV 0.318 (2B L21) / 0.295
(27B L44) against raw expression's 0.193, with the largest gaps at knot bins 5→8 (150–240°) where
the data's largest gap sits at bin 10→11.

The open reading is whether that warp is *enrichment*. More representational distance per unit of
phase means finer distinctions are possible there — the model would have higher **resolution** at the
commitment step, even though it carries less total phase information than raw expression
(circ-R² 0.875 vs 0.929).

Nobody has tested this. The metric stretch was measured on **knot centroids** (12 points). Resolution
is a statement about **single cells**, and it has never been measured per position on the loop.

## Pre-registered prediction

**H1.** In the G1→S window (knot bins 5–8, 150–240°), C2S-2B L21's local phase resolution relative to
raw expression is **higher** than its relative resolution averaged over the whole loop.

Formally, with `R_arm(w)` = local resolution in window `w`:
`Δ(w) = R_c2s(w) − R_expr(w)`, and H1 predicts `mean Δ(w ∈ G1→S) > mean Δ(w over all windows)`.

**H0.** `Δ(w)` is flat around the loop — the stretch is a property of the centroid path only and buys
no single-cell resolution.

**Direction matters.** A stretched metric with *unchanged* resolution would mean the model spreads
the same cells further apart with proportionally more noise — a rescaling, not enrichment. That
outcome supports H0 and would retire the "enrichment" reading.

## Metrics

Both computed inside each window; they should agree.

1. **Slope-to-noise.** Within a window, regress pairwise representational distance `‖Δx‖` on
   circular `|Δφ|` by OLS. Resolution = `slope / residual_sd`. Units: representational distance per
   degree, in noise units.
2. **Discriminability d′.** `(mean‖Δx‖ for pairs with |Δφ| ∈ [10°,20°] − mean‖Δx‖ for pairs with
   |Δφ| < 3°) / pooled sd`. The direct psychophysics form: can you tell two cells 15° apart apart,
   given the local noise.

## Controls (all mandatory)

1. **Matched dimensionality.** Every arm reduced to the same `k` whitened principal components
   (k = 20 primary, k = 50 secondary). Whitening makes distances comparable across arms in units of
   each representation's own spread.
2. **Matched n per window.** Cell density around the loop is very uneven (bin counts range 70 to
   537). Subsample every window to the same `n` (the minimum across windows), 50 repeats, report the
   mean. Without this, density drives local geometry quality.
3. **Matched cells.** All arms scored on the same cell indices.
4. **Null.** Shuffle φ within each window. Resolution must go to ≈ 0. If it does not, the metric is
   broken and nothing is reported.
5. **Whole profile reported.** All 12 windows are printed, not just the G1→S ones. The primary
   contrast is pre-stated above and cannot be moved after seeing the profile.

## Arms

- **C2S-2B L21** — the representation where the stretch was measured. Primary.
- **Raw expression** — the data. Primary comparator.
- scGPT L11, MaxToki L8, Geneformer L11, STATE-SE L11 — secondary. The stretch has **never been
  measured** in these, so this is exploratory for them, and any result there is a new question, not a
  replication.

## What would change the story

- **H1 supported:** the warp is functional — the model reallocates resolution to the decisive
  transition. "Reorganised, not richer" becomes "reorganised, and the reorganisation buys something".
  This is the first evidence that a model-specific geometric property does work.
- **H0 supported:** the stretch is a centroid-path property with no single-cell consequence. The
  enrichment reading is retired and the claim stays purely descriptive.

Either outcome is reportable. Neither depends on which way it comes out.

## Known limits, stated in advance

- One cell line (K562), fast-cycling, no G0 compartment.
- The phase coordinate φ is derived from marker expression, so raw expression is circularly
  advantaged — as everywhere in this line. That advantage is *against* H1, which makes a positive
  result the harder one to get.
- Resolution is measured in the representation's own units. It is not an information measure and
  does not contradict the decodability numbers.
