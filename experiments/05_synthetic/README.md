# Synthetic experiments: planted ground truth

**This is the only part of the repository that runs end to end with no external assets.** No model
weights, no datasets, no cached activations. A laptop is enough. If you want to check one thing from
this paper yourself, check something here.

Everything else in the paper measures models someone else trained, on data someone else collected,
so the ground truth has to be inferred. Here we choose the latent structure, break exactly one
property of it, and ask what the model's copy of that structure does. That is what lets Section 5 of
the paper state a mechanism rather than a correlation.

The model is a plain standard transformer (`vanilla_model.py`): softmax multi-head attention, GELU
feed-forward, pre-norm, d = 192, 4 layers, 4 heads, about 1.4M parameters. Nothing about it is
novel. The input encoding is scGPT's published scheme, vendored at `src/geomsc/tokenizer.py`.

## Quick start

```bash
pip install -r requirements.txt
cd experiments/05_synthetic
python s4_vocab_facts.py            # the token-table result, ~40 min on CPU
```

Results are written to `results/` as JSON. Set `GEOMSC_RESULTS` to write elsewhere.

## What each script does

| script | question | paper |
|---|---|---|
| `s3_metric_warp.py` | When does a model stretch a metric it inherited? Four arms: nothing changed, output turns over fastest here, 3x more cells here, harder to predict here. | §5.1, Table 2, Fig 5 |
| `s4_vocab_facts.py` | Does the gene table hold a fact that appears in no single cell? Sweeps how strongly an arbitrary group label drives co-occurrence. | §5.2, Supp D.3 |
| `s4b_training_budget.py` | Is the token-table result just undertraining? Sweeps 1,500 to 12,000 steps. | Supp D.3 |
| `s4c_where_is_it.py` | Which component holds the group structure? | Supp D.3 |
| `s4d_abstraction.py` | Does the table generalise to gene pairs that never co-occur in any cell? The synthetic form of the neighbourhood holdout. | §5.2, Supp D.3 |
| `s4e_is_the_table_used.py` | Is the table load-bearing? Zeroed, row-shuffled, and frozen at random initialisation. | §5.2, Supp D.3 |
| `s4g_steering_matched_null.py` | **A failed experiment, kept deliberately.** Three successive nulls all failed, and a null-versus-null floor exceeded the signal. No steering claim is reportable from it. | Supp D.4 |
| `s6_completion.py` | Delete an arc of a circular trajectory from training. Does the model bridge the hole better than raw data? It does not. | §3.1, Supp B.3 |
| `s0_local_resolution.py` | **A failed instrument, kept deliberately.** The first attempt at measuring local resolution had no power: phase explains only R² = 0.02 of ambient pairwise distance. | withdrawn arm |
| `s0b_plane_resolution.py` | The rebuilt instrument, in a fitted phase plane with uniform-ring and planted-stretch controls. | withdrawn arm |
| `s0c_contrast_ci.py` | Confidence intervals for the S0b contrasts. | withdrawn arm |
| `s3b_real_c2s_null.py` | The real-data counterpart of S3, on cached C2S activations. **Needs external assets**, see `docs/DATA.md`. | §5.1 |

`synth_lib.py` holds the corpus generators (`ring_corpus`, `group_corpus`), tokenisation, the
training loop, and the two readouts (`cell_embeddings`, `gene_table`).

## Two scripts that failed, and why they are here

`s0_local_resolution.py` and `s4g_steering_matched_null.py` did not produce a usable result. They are
in the repository because the paper's argument depends on knowing when an instrument is not
trustworthy, and because a reader should be able to see the code that produced the failure. `s4g` in particular is the experiment whose *null-versus-null floor* (−0.489) turned out
larger than its signal (−0.227), which is the concrete case behind test 5 of the paper's protocol.

## Reproducing the headline numbers

`s3_metric_warp.py` runs one (arm, seed) pair per invocation and appends to a JSON. The paper uses
8 seeds x 4 arms = 32 runs. `results/s3_metric_warp_vanilla_all8seeds.json` is that merged file and
is what `paper/make_figures.py` reads for Figure 5.

Expect the numbers to move in the last decimal place on different hardware. The training loop is
seeded, but floating-point reduction order differs between CPU, CUDA and Apple MPS backends. The
signs, orderings and significance in `S3_RESULTS.md` should reproduce; the third decimal may not.

## A note about Apple Silicon

`synth_lib.train` computes the masked loss with an explicit multiply-and-sum rather than boolean
indexing. This is not stylistic. On MPS, `pred[mask]` and `val[mask]` with the *same* mask returned
different element counts at scale (3243 against 40), silently corrupting the loss. That fault is why the line is written the way it is. Do not "simplify" that line back.

## Result files

`results/*.json` are the outputs that back the numbers in `S3_RESULTS.md`, `S4_RESULTS.md`,
`S6_RESULTS.md` and `S0_RESULTS.md`, and through those the paper. They are small and are committed
so that a reader can check a figure without rerunning anything.
