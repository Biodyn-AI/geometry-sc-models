# The geometry of single-cell foundation models

Code, results and the paper for **"The geometry of single-cell foundation models: what they inherit,
what they add, and what shapes it"** (Kendiukhov, Smith and Dooms, 2026).

The paper asks three questions of the representational geometry of single-cell foundation models,
across nine representations plus a model-free control: what does it look like, is any of it the
model's own, and what makes it take the shape it does. The short answer is that the answer splits by
what is being represented. A **cell's** state is determined by that cell's own input, and the models
re-describe it. A **gene's** meaning exists only across the corpus, and the models genuinely learn it.

📄 [`paper/main.pdf`](paper/main.pdf) · 🔎 [**From claim to code**](docs/CLAIMS.md) ·
📦 [What you need to download](docs/DATA.md) · ▶️ [Reproducing](docs/REPRODUCING.md)

---

## Start here

If you want to check one thing yourself, check the synthetic experiments. They are the only part of
this work that runs end to end with no external assets: no model weights, no datasets, no cached
activations. A laptop is enough.

```bash
git clone https://github.com/Biodyn-AI/geometry-sc-models
cd geometry-sc-models
pip install -r requirements.txt

PYTHONPATH=src python -m pytest tests/ -q          # 78 tests, ~1 second
cd experiments/05_synthetic && python s4_vocab_facts.py
```

That reproduces the result in Section 5.2: a model's token table holds a fact that appears in no
single training example, generalises to pairs of genes that never co-occur, and costs a quarter of
the model's performance if you freeze it at random initialisation.

To rebuild every figure in the paper from committed data, with nothing else installed:

```bash
python paper/make_figures.py
```

## What is here

| directory | contents |
|---|---|
| [`paper/`](paper) | LaTeX source, bibliography, figure scripts and the compiled PDF. `make_figures.py` regenerates all seven figures from JSON committed in this repository. |
| [`experiments/`](experiments) | The analyses, grouped by the paper section they support. Each module has a README listing every script, what it computes, and what it needs. |
| [`src/geomsc/`](src/geomsc) | The small shared library: the vendored scGPT input encoding, and asset-path resolution. |
| [`tests/`](tests) | Tests for the parts that can be tested without data. |
| [`docs/`](docs) | The claim-to-code index, the data manifest, and reproduction notes. |

### Experiment modules

| module | paper | needs external assets |
|---|---|---|
| [`05_synthetic`](experiments/05_synthetic) | §5.1, §5.2, §3.1 | **no** |
| [`01_cell_state_geometry/c2s_cell_cycle`](experiments/01_cell_state_geometry/c2s_cell_cycle) | §3.1, §3.2, §4.2, §5.1 | yes |
| [`01_cell_state_geometry/five_models`](experiments/01_cell_state_geometry/five_models) | §3.1, §5.3 | yes |
| [`01_cell_state_geometry/developmental`](experiments/01_cell_state_geometry/developmental) | §3.3 | yes |
| [`01_cell_state_geometry/other_representations`](experiments/01_cell_state_geometry/other_representations) | §3, §6 | yes |
| [`01_cell_state_geometry/supporting_arms`](experiments/01_cell_state_geometry/supporting_arms) | Supp F | yes |
| [`02_vocabulary_facts/chromosome`](experiments/02_vocabulary_facts/chromosome) | §4.1 | yes |
| [`03_what_shapes_geometry/steering_operator`](experiments/03_what_shapes_geometry/steering_operator) | §5.4 | yes |
| [`03_what_shapes_geometry/curvature`](experiments/03_what_shapes_geometry/curvature) | §5.5 | yes |
| [`06_gap_tests`](experiments/06_gap_tests) | §6.4, Supp A.3 | yes |

## What this repository is, and is not

**It is a reproduction repository, not a package.** The scripts are the ones that produced the
numbers, collected from the tree the study ran in and organised by the claim they support. Paths to
large assets have been parameterised and nothing else about the analysis logic was changed. We chose
this over rewriting several hundred scripts into a tidy library we could not have re-validated
against the published numbers.

**Large assets are not here.** Model checkpoints, datasets and cached activations come to roughly
24 GB, most of it third-party material we cannot redistribute. [`docs/DATA.md`](docs/DATA.md) lists
every asset and where to get it. Scripts resolve them through `GEOMSC_DATA` and `GEOMSC_MODELS`, and
fail immediately with the path they wanted if an asset is missing.

**Results are committed.** Around 280 small JSON and CSV files sit beside the scripts that wrote
them, so any number in the paper can be checked without rerunning anything.

**Negative and failed results are included on purpose.** Two synthetic scripts produced nothing
usable and are kept, because the paper's argument depends on knowing when an instrument is not
trustworthy. `s0_local_resolution.py` is an instrument with no power. `s4g_steering_matched_null.py`
is the experiment whose null-versus-null floor came out larger than its signal, which is the case
behind test 5 of the paper's protocol.

## Scope of the paper in this repository

The copy of the paper here omits the co-author email addresses that appear in the submitted version.
Nothing else differs: the text, figures, tables and bibliography are identical.

Every synthetic result in this repository was produced with the plain standard transformer in
[`experiments/05_synthetic/vanilla_model.py`](experiments/05_synthetic/vanilla_model.py): softmax
multi-head attention, GELU feed-forward, pre-norm. There is no second architecture in this work.

## Citing

See [`CITATION.cff`](CITATION.cff). Code is MIT licensed. The datasets and model checkpoints this
work analyses carry their own licences; see [`docs/DATA.md`](docs/DATA.md).
