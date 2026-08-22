# S6 — does the model complete a manifold it never observed? **No. It bridges no better than the data.**

**Run 2026-08-19/20.** Script `s6_completion.py`, results `results/s6_completion.json`.
2 seeds × 5 gap widths = 10 trainings, 30,000-cell pool, 20,000 used for training, 3,000 steps.

## Design

The sharpest operationalisation of "invents vs reparametrises". A contiguous arc of the ring is
deleted from training entirely — no cell in that arc is ever seen. Then held-out gap cells are
scored with a readout fitted **only on training cells**. The load-bearing arm is raw expression on
the identical cells: a ring is locally near-linear, so bridging a modest hole is free from the data's
geometry, and only an *advantage over that* would be model invention.

## Result

Held-out gap cells, circular decodability (R_diff; chance ≈ 0.03), mean of 2 seeds:

| gap removed | model | raw expression | **model − data** |
|---:|---:|---:|---:|
| 0° (control) | 0.996 | 0.998 | −0.002 |
| 30° | 0.996 | 0.998 | −0.001 |
| 60° | 0.996 | 0.997 | −0.002 |
| 90° | 0.993 | 0.996 | −0.003 |
| **120°** | 0.983 | 0.990 | **−0.007** |

**The model never beats the data at any gap width. Every difference is negative**, and the deficit
grows slightly with the hole. There is no width at which the model supplies something the data's own
linear structure does not.

The secondary metric agrees. Bridge ratio (chord across the gap ÷ mean arc step of an equal span)
falls in both arms as the hole widens and the two converge:

| gap | model | data |
|---:|---:|---:|
| 0° | 0.79 / 0.89 | 0.94 / 0.94 |
| 30° | 0.85 / 0.94 | 0.94 / 0.94 |
| 60° | 0.60 / 0.64 | 0.73 / 0.73 |
| 90° | 0.40 / 0.51 | 0.50 / 0.51 |
| 120° | 0.32 / 0.33 | 0.35 / 0.35 |

*(two values = the two seeds)*

## Reading

**Both arms bridge remarkably well.** Removing a *third of the ring* (120°) costs the model only
0.996 → 0.983 and the data 0.998 → 0.990. So "the representation is continuous across a region never
observed" is true — and it is true of raw expression too, which is the point. A ring is locally
linear; interpolation across a hole is a property of the geometry, not evidence that the model built
anything.

**This is a clean negative for invention, and it is well-powered.** The instrument resolves
differences far smaller than any candidate effect: the control gap (0°) reproduces at −0.002, and
even the largest deficit is −0.007.

**It matches the real-data result rather than contradicting it.** The single real-data test — delete
an intermediate population, steer toward it, get the wrong lineage — concluded "steering interpolates,
it does not generate". The synthetic sweep says the same thing with a curve behind it, and adds the
part real data could not: the model interpolates *exactly as well as the data does*, no better and
slightly worse.

## What this does not show

- Not that models never extrapolate — only that on a 1-D closed manifold with a contiguous hole, this
  architecture matches its input's geometry and adds nothing.
- The gap is contiguous and the manifold is 1-D. A hole in a higher-dimensional or branching
  structure is a different question and is not tested here.
- Cell-level only. The token store is S4's subject, not this one.
