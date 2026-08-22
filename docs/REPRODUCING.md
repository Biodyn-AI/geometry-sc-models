# Reproducing

There are three levels, and they cost very different amounts.

## Level 1: nothing to download (minutes)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q     # 78 tests
python paper/make_figures.py                  # all 7 figures from committed JSON
```

`make_figures.py` reads only files in this repository. If a figure changes, the committed result
JSON changed, not the environment.

## Level 2: the synthetic programme (hours, laptop)

```bash
cd experiments/05_synthetic
python s4_vocab_facts.py        # §5.2, the token table
python s4d_abstraction.py       # §5.2, generalisation to pairs that never co-occur
python s4e_is_the_table_used.py # §5.2, the four interventions
python s6_completion.py         # §3.1, no completion of deleted structure
python s3_metric_warp.py        # §5.1, one (arm, seed) per invocation
```

`s3_metric_warp.py` runs one arm and seed at a time; the paper uses 8 seeds across 4 arms, so 32
invocations. The merged file is committed as `results/s3_metric_warp_vanilla_all8seeds.json`.

**Expect the last decimal place to move.** The training loop is seeded, but floating-point reduction
order differs between CPU, CUDA and Apple MPS. Signs, orderings and significance should reproduce;
the third decimal may not.

## Level 3: the real-model analyses (days, and downloads)

Read [`DATA.md`](DATA.md) first, set `GEOMSC_DATA` and `GEOMSC_MODELS`, then work from the module
README of whichever result you want. [`CLAIMS.md`](CLAIMS.md) maps each paper claim to its script.

Most scripts read a cached activation file rather than running a model. Each module has
`extract_*.py` scripts that build those caches; budget the first run accordingly.

## If a number does not match

Check in this order.

1. **The committed result file.** Every script writes JSON next to itself under `results/`. If your
   run disagrees with the committed JSON, the difference is in your environment or assets, not in
   the paper.
2. **The two upstream defects** in `DATA.md`. If you re-extracted activations rather than using a
   cache, you may have the *corrected* values, which differ from some cached numbers in a known
   direction.
3. **Which baseline you built.** This is the most common cause. Section 6 of the paper exists
   because the same experiment gives opposite verdicts depending on how the competitor is built: in
   the cross-cell-line transfer, matching the encoding alone gives "the model adds information"
   (Δ = +0.0153, CI excluding zero) while additionally matching gene selection gives parity
   (Δ = −0.0022) on the same cells.

## A note on what "reproduce" means here

Several results in this paper are **negative**, and a few are **retractions of our own earlier
claims**. Reproducing those means reproducing the absence of an effect, which is sensitive to the
null you use. Section 6 of the paper records six cases where a large z against a permutation null went to
approximately zero against a competitor built from the same cells. If you are checking one of those,
build the competitor, not the permutation.
