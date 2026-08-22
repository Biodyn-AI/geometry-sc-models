# S4 — does a model learn facts about its vocabulary? **Yes — and whether it does depends on the architecture.**

**Run 2026-08-19 (first version), fully re-run 2026-08-22 on a standard transformer.**
Scripts: `s4_vocab_facts.py` (sweep), `s4b_training_budget.py`, `s4c_where_is_it.py`,
`s4d_abstraction.py`, `s4e_is_the_table_used.py`, `s4g_steering_matched_null.py`.
Results in `results/s4*.json`.

> **This document was rewritten. Its first version concluded the opposite.** The original runs used
> Re-run on a plain transformer — standard softmax multi-head attention, GELU feed-forward, pre-norm,
> identical corpora and identical scGPT input encoding — **three of the four findings reversed**. The
> between them is now the result.

## The question

The synthetic analogue of the chromosome result. Every gene gets an arbitrary group label that
appears in **no single cell** — recoverable only from corpus-wide co-occurrence, exactly like a gene's
chromosome. `consistency` sets how strongly group membership drives co-occurrence. Ground truth is
known, so all three rungs can be scored honestly.

## Finding 1 — the table holds a map, and how good a map depends on the architecture

22-way probe on the learned gene table, split over genes. Chance 0.050. PPMI is a training-free
factorisation of the identical corpus and is **numerically identical across architectures**, which
confirms the corpora and the baseline are unchanged — the model is the only variable.

|---:|---:|---:|---:|
| 0.0 | 0.047 | 0.051 | 0.061 |
| 0.2 | **0.160** | 0.048 | 0.202 |
| 0.4 | **0.245** | 0.050 | 0.290 |
| 0.6 | **0.296** | 0.051 | 0.376 |
| 0.8 | **0.340** | 0.054 | 0.440 |
| 1.0 | **0.426** | 0.051 | 0.497 |

The standard transformer produces a **dose-response curve that tracks the factorisation** (margin
narrowing to −0.071). The bilinear architecture produced a **flat line at chance**. Both null
conditions read chance, so the instrument is calibrated in both.

## Finding 2 — it generalises to genes it never saw together

The decisive design, and the synthetic form of the chromosome paper's 10-Mb neighbourhood holdout:
split every group's genes into halves H1/H2 and let programs draw from one half only, so within-group
**cross-half** pairs never co-occur in any cell while same-half pairs co-occur constantly.

3 seeds, mean ± sd:

| | observed pairs | held-out pairs (never co-occur) | combined |
|---|---:|---:|---|
| **standard transformer** | **0.1367** ± 0.0068 | **+0.0147** ± 0.0107 | **Stouffer z +20.2** |
| PPMI (reference) | 0.1946 ± 0.0077 | +0.0037 ± 0.0011 | — |

Observed-pair structure is **80× the bilinear model's and 70% of the factorisation's**. On held-out
pairs the standard transformer reaches **four times PPMI's own generalisation**. The bilinear model's
"memorisation without abstraction" does not hold for transformers generally.

## Finding 3 — and the map is load-bearing

|---|---:|---:|
| real | 0.8534 | 0.8480 |
| zeroed | 0.0842 | 0.0912 |
| row-shuffled | 0.0047 | 0.0127 |
| **frozen at random init** | **0.6118 (costs 0.2416)** | **0.8473 (costs 0.0007)** |

A permanently random table costs the standard transformer **a quarter of its performance**; it cost
the bilinear model **nothing**. So the three findings are three faces of one fact: the standard
model's table is a genuine map, the bilinear model's was an index.

## Why the architecture matters — stated as a hypothesis, not a result

*(A comparison against a second, non-standard architecture was run and is reported in the paper's supplement, section F.5. It is omitted here because that architecture is not yet published.)*

## What the exclusions establish (unchanged by the re-run)

These were run on the original architecture and are retained because they exclude explanations rather
than making claims about which architecture does what.

1. **Not task degeneracy.** A per-gene-mean predictor scores val_corr 0.47 against the model's 0.84.
2. **Not a baseline mismatch.** PPMI restricted to the model's own top-128 input still scores 0.453 at
   consistency 1.0 (against 0.491 on all genes).
3. **Not undertraining** (`s4b`). 1500 → 12000 steps moves the probe by 0.003 while val_corr rises
   0.801 → 0.857. *(Note the trap: `corr(log steps, acc) = +0.96` — perfectly monotone, entirely
   inside chance. Effect size, not correlation.)*
4. **Not probe linearity** (`s4c`). A nonlinear MLP probe scored 0.059 on the bilinear table.

## The steering arm is EXCLUDED — the instrument does not calibrate

The group direction is built from real embedding rows, `mean(W_E[group == c]) − mean(W_E)`; the
control was an isotropic random vector matched only on **norm**. At consistency 0.0, where there is no
group structure and the true effect is exactly zero, the standard transformer reads
**−0.98 mean, 38% of groups positive**, reaching **−7.1 and 12% positive** at consistency 1.0. A real
effect measured against a valid null cannot be that one-sided.

The cause is anisotropy: any direction built by averaging real embedding rows sits in the dominant
part of the embedding distribution, and moving along it does something systematically different from
moving along an isotropic vector, whether or not it encodes anything. Norm-matching does not fix that.

**The bilinear architecture had the same flaw**; its null-condition reading was −0.04, small enough to
pass as noise. So the previously reported steering numbers (frac+ 0.78–0.82) were never usable on
either architecture and must not be cited.

Two rebuilds failed to fix it.

1. **Construction-matched null** — build the direction the same way from shuffled labels. Calibration
   point still read −0.28.
2. **Alignment-matched null** — give each null draw its own shuffled labelling and use it for BOTH
   the direction and the readout, each deflated by its own no-push baseline, so push-set and read-set
   are aligned in both arms. Calibration point read +0.097 / −0.029 / **−0.269** across doses: fine at
   low push, drifting with dose.

**The null-vs-null floor then settled it.** Scoring one structureless labelling against another, with
identical machinery:

| α | null-vs-null floor | "real" reading (also structureless) |
|---:|---:|---:|
| 0.5 | −0.105 | +0.039 |
| 1.0 | −0.276 | −0.002 |
| 2.0 | **−0.489** | −0.227 |

**The floor is larger than the signal at every dose.** The estimator is biased, not noisy, and the
bias is structural: the real arm is a SINGLE labelling while the null is the MEAN OF FIVE, and with a
saturating dose response those are not exchangeable — which is why the bias grows with α and why the
low-dose reading looked almost acceptable.

**Verdict: no steering claim is reportable from S4, on either architecture.** This is not "no effect
detected"; the instrument cannot distinguish anything. The synthetic work therefore establishes rung 1
(decodable) and rung 2 (beyond a data baseline) for the gene table, and is **silent on rung 3**. The
real-model causal evidence — chromosome steering positive for 132/132 chromosome × strength × seed
combinations, against a norm-matched random push — is a separate measurement with its own controls and
is unaffected.

A correct design would pair single draw against single draw and repeat, rather than comparing one
against an average. That is left for future work; it is not patched here because three attempts at
this null have now failed and the honest report is the exclusion.

## Instrument notes

- **MPS boolean-indexing fault.** `pred[mask]` and `val[mask]`, with the same mask, returned 3243 and
  40 elements. Does not reproduce at smaller scale. Fixed by removing boolean indexing from the hot
  path — masked MSE is now `((pred−val)²·mask).sum()/mask.sum()`, and remaining mask indexing happens
  on CPU. The failure mode is a hard crash, not silent corruption, so completed runs are unaffected.
- **A duplicate job ran for 3.7 hours** because `pkill` killed a scheduler shell but not the Python
  child it had already spawned. No results were double-counted (verified: 0 duplicate keys).
- **A queue script omitted the seed argument**, so the first vanilla S3 run used seeds (0,1,2) rather
  than 0–7; extended afterwards, writing to a separate file to avoid the clobbering race that
  destroyed an earlier backup.
