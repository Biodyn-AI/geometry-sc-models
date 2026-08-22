# S0 — Does the metric stretch buy local phase resolution? **No.**

**Run 2026-08-19.** Pre-registered in [S0_PREREGISTRATION.md](S0_PREREGISTRATION.md).
Scripts: `s0_local_resolution.py` (failed instrument, kept for the record),
`s0b_plane_resolution.py` (the working instrument + controls), `s0c_contrast_ci.py` (the CI).
Results: `results/s0b_plane_k20.json`, `results/s0c_contrast_ci_k20.json`.

## Verdict

**H0.** The C2S metric stretch at G1→S does not produce measurable extra local phase resolution at
single-cell level. The pre-registered contrast is **+0.0023, 95% CI [−0.0055, +0.0089]**, against a
phase-shuffled calibrated zero of −0.0002, CI [−0.0074, +0.0056]. The two distributions overlap
almost entirely.

This is a **calibrated** negative, not an underpowered one — see the power argument below.

## The numbers

Contrast = (C2S − raw expression) local resolution, G1→S windows (150–240°) minus the mean over all
12 windows. 60 independent outer splits, each with a fresh train/test split for the plane fit and
fresh density-matched subsamples.

| Arm | mean contrast | 95% CI | fraction of splits > 0 |
|---|---:|---|---:|
| **C2S-2B L21 − expression (real)** | **+0.0023** | **[−0.0055, +0.0089]** | 0.70 |
| same, phase shuffled (calibrated zero) | −0.0002 | [−0.0074, +0.0056] | 0.55 |
| planted **2.0×** stretch vs uniform ring | +0.0633 | [+0.0528, +0.0716] | 1.00 |
| planted **1.5×** stretch vs uniform ring | +0.0324 | [+0.0226, +0.0493] | 1.00 |
| planted **1.2×** stretch vs uniform ring | +0.0104 | [+0.0022, +0.0251] | 1.00 |

## Why this is a real negative and not a power failure

The instrument detects a **1.2×** planted stretch with a CI excluding zero. The stretch actually
reported in C2S is far larger than that: knot-gap CV 0.318 against the data's 0.193, and max/min
2.52 against 1.77. If a warp of that size translated into single-cell resolution, this instrument
would have seen it several times over. It saw +0.0023.

## What this does and does not overturn

**Unchanged.** The metric stretch itself stands. It was measured on the 12 knot centroids — the
average path around the loop — and nothing here touches that measurement. It remains the only
certified model-specific geometry claim in the programme.

**Retired.** The reading that the stretch is *enrichment* — that the model reallocates resolution to
the biologically decisive commitment step. The average path spends more distance there; individual
cells are not thereby better separated. Those are consistent, and the second does not follow from the
first.

**Sharpened.** "Reorganised, not richer" now has a second leg. The model is at or below the data on
total phase information (circ-R² 0.875 vs 0.929) *and* at parity on local resolution. The warp is a
property of the mean trajectory with no measured single-cell consequence, and it is still not shown
to be causal.

## The instrument failure, recorded

The pre-registered version of this test (`s0_local_resolution.py`) measured pairwise distance in the
ambient 20-PC whitened space and returned ≈ 0 for **every** arm, including raw expression, which
carries phase at circ-R² 0.929. That was reported as a null in neither direction, because it was an
instrument failure: phase explains only **R² = 0.020–0.026** of ambient pairwise distance globally,
and **0.0006–0.0034** inside a 30° window. Whitening spreads variance evenly over 20 components while
phase occupies about two of them, so roughly 90% of every measured distance was non-phase variance.

Fix: measure in the fitted 2-D phase plane — the space the stretch claim lives in — with the plane
fitted by ridge on a train half and every number computed on the held-out half, so scored cells never
contributed to the plane.

Two controls were added and both had to pass before any real number was reported: a **uniform**
synthetic ring must give a flat profile (got arc-vs-all +0.0014) and a **planted 2× stretch** must be
found (got +0.0612, with the peak in the correct arc).

## One bug worth carrying forward

`data/act_k562/row_cell_ids.npy` for C2S is `arange(3000)` — **positional rows into the substrate**,
not global cell IDs. Every other cached arm (`expr_k562`, `scgptcc_k562`, `maxtoki_k562`,
`geneformer_k562`, `statecc_k562`) stores global IDs in the range 113–642,597. Joining on `cell_idx`
naively yields **6 common cells out of 3000** and silently produces garbage. This is the same class
of defect as the STATE `cell_idx` bug already recorded in the programme. `cc_benchmark_c2s.py`
handles it correctly by asserting cell order; anything new must do the same.

## Limits

One cell line (K562, fast-cycling, no G0). One layer (L21). The 27B was not tested — its activations
are not cached locally. Resolution is measured in each representation's own units, so this is not an
information comparison and does not bear on the decodability numbers.
