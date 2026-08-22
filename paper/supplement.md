*Companion to "The geometry of single-cell foundation models: what they inherit, what they add, and
what shapes it".*

Contents:
**A** Models, data and extraction ·
**B** Cell-state geometry in full ·
**C** Vocabulary-level facts in full ·
**D** Synthetic experiments in full ·
**E** A worked negative: the extracted-operator battery ·
**F** Retractions, instrument failures and things that did not work

---

# A. Models, data and extraction

## A.1 Representations

| model | architecture | input encoding | what we read | width |
|---|---|---|---|---|
| scGPT `\citep{cui2024scgpt}`{=latex} | encoder, 12 layers, 8 heads | 51 quantile value bins | L11 residual, mean-pooled; native MVC head | 512 |
| Geneformer V1/V2 `\citep{theodoris2023geneformer}`{=latex} | BERT encoder | expression rank | `hidden_states[12]` | 256 / 768 |
| MaxToki-217M | Llama, 11 blocks, 8 heads | expression rank | L0–L11; `embed_tokens`; untied `lm_head` | 1232 |
| MaxToki-1B | Llama, 20 blocks | expression rank | L0–L20; both tables | 2304 |
| STATE-SE / STATE-ST `\citep{adduri2025state}`{=latex} | bidirectional | ESM-2 gene tokens | L11, skipping CLS | 2048 |
| UCE-100M `\citep{rosen2026uce}`{=latex} | transformer | ESM-2 gene tokens | L2 | n/a |
| C2S-Scale-2B / 27B `\citep{levine2024cell2sentence,rizvi2025c2sscale}`{=latex} | Gemma-2 text LLM | rank-ordered gene-name sentence, top-512, magnitudes discarded | last-token residual, L09/15/18/21/25 and L44 | 2304 / 4608 |
| raw expression | n/a | log1p(CP10k) | PCA | varies |
| ESM-2 `\citep{lin2023esm2}`{=latex} | protein LM | amino-acid sequence | frozen gene embeddings | 5120 |

**Note on gene tokens.** UCE's and STATE-SE's gene tokens *are* frozen ESM-2 protein embeddings `\citep{lin2023esm2}`{=latex}. They
therefore cannot be asked questions about learned gene identity, since they are the sequence control, not a
model arm, for any vocabulary-level question.

**Construct caveat, stated because it is load-bearing.** There is no cell vector in any of these
models. Every "cell representation" is a mean-pool over gene-token residuals at one layer, i.e.
approximately a context-corrected bag of genes. Part of the measured redundancy with raw expression is
therefore built into the construction rather than discovered, and this should be borne in mind
throughout §B.

## A.2 Substrates

| dataset | n | used for |
|---|---:|---|
| Replogle K562 CRISPRi, non-targeting controls `\citep{replogle2022genomewide}`{=latex} | 3,000 cells × 6,546 genes | cell cycle (primary) |
| Replogle RPE1, non-targeting controls | 3,000 | cross-cell-line frozen transfer |
| Setty CD34⁺ human bone marrow `\citep{setty2019palantir}`{=latex} | 5,780 × 14,319 | development (workhorse); Palantir pseudotime, 10 clusters, 5 terminal branches |
| CZ CELLxGENE developing human gut, 6–11 PCW | 4,269 (62,849 for co-expression panels) | development; the strongest co-occurrence panel |
| Tabula Sapiens lung airway `\citep{tabulasapiens2022}`{=latex} | 3,600 | development |
| Bastidas-Ponce mouse pancreas `\citep{bastidasponce2019pancreas}`{=latex} | 3,696 | development; cross-species |
| Weinreb LARRY clonal barcoding `\citep{weinreb2020larry}`{=latex} | 865 undifferentiated / 167 clones | external ground-truth fate |
| Tabula Sapiens immune / lung / kidney `\citep{tabulasapiens2022}`{=latex} | ~7,500 | cell type, gene context, co-expression baselines |
| `species_chrom.csv` | 19,844 human genes | chromosome + genomic start |

**Phase coordinate.** Tirosh S and G2M marker z-scores `\citep{tirosh2016melanoma}`{=latex} → PCA(2) → φ = atan2, oriented G1 → S → G2M.
Validated independently: phase-cluster circular means at G2M 20.3°, G1 150.4°, S 249.5° (correct cyclic
order), concentrations 0.79–0.94, S and G2M score peaks in quadrature (109° ≈ 90°). Discrete
composition G1 747 / S 1,208 / G2M 1,045.

**Circularity, stated up front.** φ and the phase call are *derived from marker expression*, so the
expression baseline is circularly advantaged on every cell-cycle comparison. A model that beats it
anyway is meaningfully better; an expression win is partly definitional. The cell cycle is a
**degenerate testbed at the cell level** and an excellent one at the gene level, where the input can be
stripped of the answer (§C.2).

## A.3 Two extraction defects found in this work

**Input convention (§6.4).** One extraction fed raw counts to a checkpoint configured with
`input_style: "binned"`, `n_bins: 51`. Both versions were cached for three tissues, so the size of the
error is directly measurable on the same cells:

| metric | mean change when corrected | max | sign flips |
|---|---:|---:|---:|
| pseudotime Spearman | **+0.109** | +0.214 (gut) | 0 |
| lineage balanced accuracy | **+0.118** | +0.277 (gut) | 0 |
| linear R² | **+0.210** | +0.402 (gut) | 0 |
| curvature gap (kNN − linear) | **−0.075** | −0.118 (lung) | **1** |
| tangent rotation | −4.2° | 17.0° (gut) | 0 |

The direction is consistent: the defect **understates the model and overstates its curvature**. It
compounds with a known confound (curvature significance tracks linear-decoder headroom at partial
correlation −0.858), so raising linear R² by 0.21 mechanically shrinks apparent curvature. Nine routes
rest on the uncorrected extraction.

**Broken pseudotime.** In the cached fetal-gut arm, mean pseudotime by cluster:

| lowest | | highest | |
|---|---:|---|---:|
| intestine goblet cell | 0.18 | progenitor cell | 0.38 |
| enteroendocrine cell | 0.24 | enterocyte | 0.38 |
| **stem cell** | **0.30** | colon epithelial cell | 0.70 |

Differentiated cells sit below stem cells. The other three tissues are correctly rooted (blood HSC_1
0.11 → Ery_2 0.82; lung basal 0.24 → multiciliated 0.73; pancreas Ductal 0.23 → Alpha 0.78). Gut is
excluded from the transfer analysis in §B.4 for this reason.

**Other data hazards documented in the course of this work:** `replogle_concat.h5ad` already stores
log1p(CP10k), so re-normalising double-normalises and corrupts per-gene variance (it drove one gate
from pass to fail); one cached activation file wrote local row numbers as global cell indices,
producing a spurious "genuinely curved" result; one cell-cycle cache had no phase field and only a
third of its cells matched the intended population; and C2S activation caches key rows **positionally**
while every other cache keys by global cell ID, so a naive join yields 6 common cells out of 3,000 and
silently produces garbage.

---

# B. Cell-state geometry in full

## B.1 The information decomposition and its two traps

Within-dataset, C2S-Scale on 3,000 K562 cells:

| representation | phase ordering (circ-R²) | classification |
|---|---:|---:|
| full expression, 6,546 genes with values | 0.929 | 0.907 |
| top-512 genes with values | 0.885 | 0.853 |
| top-512 set, binary, *the model's input* | 0.882 | 0.846 |
| top-512 set + rank, *the model's input* | 0.868 | 0.844 |
| **C2S-2B L21** | **0.875** | **0.866** |
| top-512 set + rank → MLP-256 | 0.802 | 0.830 |

Loss decomposition: 0.929 → 0.882 (tokenisation, −0.047) → 0.875 (the model, −0.007).

**Cross-cell-line frozen transfer (K562 → RPE1).** RPE1 selected on measured criteria over two rejected
candidates: it carries 93/94 cell-cycle markers, is a single cell line so phase cannot confound with
identity, and its phase is well spread (R = 0.267).

| arm | R_diff | median error |
|---|---:|---:|
| all 6,544 shared genes + magnitudes | 0.878 | 16° |
| top-512 + magnitudes | 0.810 | 19° |
| **top-512, rank only, matched gene selection, the model's input** | **0.792** | 20° |
| **C2S-2B L21** | **0.789** | 23° |
| constant predictor / uniform random | 0.043 / 0.017 | 86° / 88° |

| paired contrast (5,000 draws) | Δ R_diff | 95% CI |
|---|---:|---|
| **model − matched input baseline** | **−0.0022** | **[−0.0144, +0.0105] → parity** |
| tokenisation loss | +0.1042 | [+0.0917, +0.1169] |
| …of which truncation to 512 genes | +0.0683 | [+0.0576, +0.0787] |
| …of which discarding magnitudes | +0.0359 | [+0.0261, +0.0459] |

**Two traps, each producing a different wrong verdict.** Comparing the model against all 6,544 genes
with magnitudes gives "expression wins", which charges the model for its tokeniser. Matching the
*encoding* but not the *selection* gives "the model adds information" (Δ = +0.0153, CI excluding zero),
because a cell sentence draws its top-512 from the cell's **full** panel (RPE1: 8,749 genes) while
that baseline drew from the 6,544 shared, so the model saw ~2,205 genes the baseline never did.
**Both the encoding and the gene-selection procedure must be matched.**

## B.2 Downstream benchmarks

**Five prior models, 1 of 45.** Protocol: matched-dimensionality PCA (k ∈ {10, 20, 50}), small probe,
held-out, three tasks. Expression wins ordering 15/15, classification 14/15, S-vs-G2M AUROC 15/15.

| task (k = 50) | best model | expression | winner |
|---|---|---|---|
| phase ordering | MaxToki 0.924 | **0.932** | expression |
| phase classification | **MaxToki 0.896** | 0.879 | model (the only cell of 45) |
| S-vs-G2M AUROC | MaxToki/STATE 0.991 | **0.992** | expression |

UCE loses all 9 of its cells by the widest margins (ordering 0.77–0.80 vs 0.93).

**C2S-Scale, 0 of 36.** Identical protocol and substrate. Best C2S ordering 0.876 (L25, k = 50) against
expression 0.929, and slightly *worse* than the encoder models.

**Developmental pseudotime, 0 of 20.** Model + ridge(k = 400) minus the best nonlinear expression
baseline, all negative:

| | scGPT | Geneformer | STATE-SE | MaxToki | UCE |
|---|---:|---:|---:|---:|---:|
| blood | −0.034 | −0.063 | −0.119 | −0.026 | −0.139 |
| lung | −0.013 | −0.041 | −0.045 | −0.013 | −0.104 |
| gut | −0.035 | −0.134 | −0.091 | −0.021 | −0.374 |
| pancreas | −0.013 | −0.032 | −0.022 | −0.015 | −0.056 |

Expression + nonlinear ceilings: 0.977 / 0.983 / 0.921 / 0.987. A symmetric version (model given the
same nonlinear probe) is also 0/20. Against a *linear* expression probe at k ≥ 100 the models win
66/180, which an earlier k ≤ 50 cap concealed; that correction is recorded in §F.

**Steering, constrained advance at τ = 1.3:**

| tissue | expr-PCA | scGPT | Geneformer | STATE | MaxToki | models beating expr |
|---|---:|---:|---:|---:|---:|---|
| blood | **2.092** | 1.446 | 0.626 | 1.898 | 0.382 | **0/4** |
| mouse pancreas | 1.728 | 2.152 | 1.426 | **2.145** | 2.087 | 3/4 |
| lung | 0.175 | 0.252 | 1.997 | **2.270** | 0.302 | 4/4 |
| gut | 0.769 | 0.877 | 0.996 | **1.884** | 0.642 | 3/4 |

Robustness: 5 seeds × 3 HVG settings, expression beats scGPT 15/15, Geneformer 15/15, MaxToki 15/15,
STATE 11/15 = 56/60 on blood. **Do not read this as "models lose everywhere".** Blood is the tissue
every earlier headline rested on, and it is the only one where expression wins outright.

## B.3 No completion of deleted structure

Delete a contiguous arc of a circular trajectory from training entirely; score held-out gap cells with
a readout fitted only on training cells. Two seeds, standard transformer, chance R_diff ≈ 0.03:

| gap removed | model | raw expression | model − data |
|---:|---:|---:|---:|
| 0° (control) | 0.993 | 0.998 | −0.005 |
| 30° | 0.994 | 0.998 | −0.004 |
| 60° | 0.991 | 0.997 | −0.007 |
| 90° | 0.989 | 0.996 | −0.007 |
| **120°** | 0.963 | 0.990 | **−0.027** |

Both arms bridge remarkably well (removing a *third* of the ring costs the model only 0.030), because
a ring is locally near-linear. But the model never beats the data, and its deficit grows with the hole.
Bridge ratio (chord across the gap ÷ mean arc step of an equal span) falls in both arms and converges
at wide gaps (120°: model 0.28, data 0.35).

This matches the one real-data test, in which deleting an intermediate population and steering toward it
produces the wrong lineage (held-out peak r = +0.545, rank 5/10, path running through the wrong
branch), and adds the curve behind it: **steering interpolates; it does not generate.**

## B.4 Cross-tissue transfer

Fit a pseudotime probe on one tissue, apply it **unchanged** to another. PCA fitted on the train tissue
only. 10,529 genes shared across four tissues; every arm reduced to 50 components.

**Within tissue** (5-fold CV Spearman), the ceiling each arm transfers from:

| arm | blood | gut | lung | pancreas |
|---|---:|---:|---:|---:|
| scGPT | 0.945 | 0.845 | 0.948 | 0.916 |
| **expression** | 0.928 | 0.820 | 0.935 | 0.913 |
| STATE | 0.914 | 0.837 | 0.920 | 0.908 |
| Geneformer | 0.933 | 0.733 | 0.887 | 0.891 |
| UCE | 0.885 | 0.594 | 0.876 | 0.873 |

**Across tissues**, excluding gut (§A.3), 6 ordered pairs:

| pair | expression | scGPT |
|---|---:|---:|
| blood → lung | −0.208 | −0.023 |
| blood → pancreas | −0.205 | **+0.589** |
| lung → blood | −0.337 | −0.108 |
| lung → pancreas | **+0.569** | +0.409 |
| pancreas → blood | −0.305 | −0.083 |
| pancreas → lung | **+0.629** | +0.491 |

| arm | mean | vs expression | wins | Wilcoxon |
|---|---:|---:|---:|---|
| scGPT | +0.212 | +0.188 | 4/6 | p = 0.156 |
| STATE | +0.138 | +0.114 | 5/6 | p = 0.156 |
| Geneformer | +0.103 | +0.079 | 4/6 | p = 0.438 |
| UCE | +0.087 | +0.063 | 4/6 | p = 0.562 |

**Transfer largely fails for every representation.** Within-tissue is 0.87–0.95; the best cross-tissue
number anywhere is 0.63; only 3 of 12 pairs clear 0.5 and two are won by raw expression.

**And the apparent model advantage is sign, not structure.** Over all 12 pairs scGPT's signed advantage
is +0.278 (Wilcoxon p = 0.016), but under **orientation-invariant |ρ|** it falls to +0.094 (p = 0.233)
and every other model goes to ≈ 0. Expression transfers with an *inverted* sign in 8 of 12 pairs,
worse than useless; scGPT is merely closer to zero. The honest statement is that scGPT resists
catastrophic negative transfer, not that it transfers well.

## B.5 Where pretraining does pay

**Endpoint targeting on hard trajectories.** Steer a held-out root cell toward a terminal branch and
ask whether its nearest real training cell is in that branch:

| tissue | expr | scGPT | Geneformer | STATE | MaxToki | best model gain |
|---|---:|---:|---:|---:|---:|---|
| blood | 0.551 | 0.438 | 0.568 | 0.567 | 0.386 | +0.017 (2/4) |
| **mouse pancreas** | **0.207** | 0.539 | 0.365 | 0.458 | **0.569** | **+0.363 (4/4)** |
| **lung** | 0.599 | 0.652 | **0.862** | 0.833 | 0.729 | **+0.263 (4/4)** |
| gut | 0.728 | 0.669 | 0.661 | **0.764** | 0.711 | +0.037 (1/4) |

The pancreas gain is concentrated in the **rare** fates (expression delta/epsilon 0.010/0.010 against
MaxToki 0.330/0.373), which is where a pretrained prior would be expected to help.

**Linear readability.** Pretraining makes trajectories linearly accessible: model + *linear* probe at
k ≥ 100 wins 66/180 (scGPT 22/36, MaxToki 25/36), a genuine win that dissolves entirely against a
nonlinear probe.

**A pretrained readout that names the fate.** Fate-aimed steering read through the model's own head
gives a clean diagonal: 14/15 target → fate hits across four tissues, pooled exact permutation over
103,680 assignments p = 1.9×10⁻⁵; MaxToki's untied `lm_head` reproduces 5/5 independently
(D = +0.821, p = 0.0083).

## B.6 Full anatomy tables

**Geometry across representations** (K562, 3,000 cells):

| representation | linear circ-R² | H_flat (planar?) | θ_far vs flat null |
|---|---:|---|---|
| scGPT L11 | 0.892 | not rejected (p = 0.905) | 37.5° < 54.2° |
| Geneformer L11 | 0.844 | not rejected | 59.6° < 63.3° |
| MaxToki L8 | 0.918 | not rejected | 35.2° < 53.1° |
| STATE-SE L11 | 0.894 | not rejected | 28.6° < 48.6° |
| **raw expression** | **0.929** | not rejected | n/a |
| C2S-2B L21 | 0.875 | n/a | n/a |
| C2S-27B L44 | 0.876 | n/a | n/a |

**Depth construction in C2S** (cross-validated circular correlation; chance = 90° error):

| relative depth | C2S-2B | C2S-27B |
|---:|---|---|
| 0.00 | 0.068 (81°) | 0.025 (83°) |
| ~0.25 | 0.321 (60°) | 0.225 (67°) |
| ~0.35 | **0.737** (26°) | 0.581 (38°) |
| ~0.58 | 0.847 (17°) | **0.880** (14°) |
| ~0.80 | 0.898 (12°, z = +44) | 0.891 (13°) |
| ~0.96 | 0.894 (12°) | **0.902** (13°) |

Phase is near-absent at the embedding and constructed progressively; the 2B builds it faster in
relative depth and both finish equal. *Caveat:* read at the final token, so early-layer values partly
reflect propagation to that position rather than absence of computation.

**Continuity battery** (0 = continuous, 1 = discrete):

| test | statistic | score |
|---|---|---:|
| behavioural interpolation | largest-step share 0.257 (gradual 0.10, snap 1.00); mid-interpolation entropy +0.0056 | **0.17** |
| metric profile | largest step 0.172 of total rise; monotone in 82% of bins | **0.10** |
| occupancy | decoded entropy 0.976 vs true 0.952; 0 empty bins | **0.00** |

**No activation plateaus.** In language models, interpolating between two input sequences leaves the
output nearly flat for most of the path and then changes sharply part way through; those plateaus
sharpen with depth and track the input-output sensitivity of the MLPs `\citep{shinkle2025plateaus}`{=latex}.
Running the same interpolation here: transition width ≈ 0.5 of the λ range, where a plateau would be
≈ 0.05; plateau strength *decreases* with depth (0.179 → 0.115 → 0.056 at L09/L15/L21), the opposite of
the reported depth trend; and crossing a real phase transition is indistinguishable from staying within
a phase (0.068 vs 0.056). Proposed falsifiable account: a plateau needs a **discrete output alphabet**;
a graded distribution over hundreds of marker genes has no cliff to form one.

---

# C. Vocabulary-level facts in full

## C.1 Chromosome

**Decoding across representations** (22-way balanced accuracy, common gene set n = 14,671):

| basis | balanced acc | × chance | beats co-expression **and** sequence? |
|---|---:|---:|---|
| **MaxToki `lm_head`** | **0.483** | 10.6× | yes |
| MaxToki `embed_tokens` | 0.488 | 10.7× | yes |
| Geneformer V2 | 0.186 | 4.1× | yes |
| Geneformer V1 | 0.137 | 3.0× | yes |
| scGPT | 0.089 | 2.0× | **no** |
| ESM-2 (sequence) | 0.105 | 2.3× | reference |
| co-expression | 0.085 | 1.9× | reference |

*(This table uses a weak linear probe on the common gene set; the headline 0.880 in §4.1
is the 1B with its best probe on 15,135 matched genes under the 10-Mb holdout.)*

**The 10-Mb neighbourhood holdout, the decisive control:**

| basis | random split | **group split (10 Mb)** | retained |
|---|---:|---:|---:|
| MaxToki `lm_head` | 0.433 | **0.347** | **78%** |
| MaxToki `embed_tokens` | 0.433 | 0.373 | 85% |
| Geneformer | 0.174 | 0.111 | 51% |
| scGPT | 0.090 | 0.056 | 24% |
| **ESM-2** | 0.190 | **0.074** | **20%, collapses** |
| co-expression (developmental panel) | 0.054 | 0.043 | n/a |

**Why sequence predicts chromosome at all, and why it fails on HOX.** Gene families arise by *local*
duplication, so a gene's nearest ESM-2 neighbour sits on the same chromosome 22.3% of the time against
5.6% chance. HOX is the natural experiment that decorrelates sequence from locus: HOX arose by
whole-genome duplication onto four chromosomes, so a HOX gene's nearest protein relative is its
paralogue *elsewhere* (same-chromosome only 7.7%), and ESM-2 is at floor (0.023). **One biological
fact predicts both the positive and the negative.**

**Tokeniser confound, and how the first control was wrong.** The vocabulary is ordered by Ensembl
accession, and accessions are assigned in chromosome blocks, so adjacent token IDs share a chromosome
48.9% of the time against 5.6% chance. A *linear* probe on token ID returns chance, which is **false
reassurance**, because the block structure is nonlinear: token ID alone via kNN predicts chromosome at
0.49 (k = 1). The finding survives the confound removed properly: holding out whole 1,000-token
accession blocks retains **96%** (0.433 → 0.415). And the clean dissociation: both tables re-encode
token ID about equally (embedding→token-ID ρ = 0.46 MaxToki, 0.44 Geneformer) yet score 0.433 and
0.174 on chromosome. If chromosome were re-encoded token ID they would match.

**Layer profile (217M, 5,686 well-sampled genes):** 0.453, 0.212, 0.185, 0.173, 0.168, 0.158, 0.146,
0.139, 0.122, 0.115, 0.098, **0.088** across L0–L11. Output table 0.516; input table 0.453. **1B at
matched taps and matched width (k = 256):** 0.813 → 0.220 → 0.091 → 0.066 at L0/L2/L4/L8 (null 0.046),
retaining **8%**; the 217M at the same taps gives 0.231 → 0.070. The 1B's advantage is concentrated at
layer 0 (3.5× ahead) and is gone by layer 8.

**Width is a confound.** Reducing the 1B from 2,304 to 256 components costs almost nothing at L0
(0.910 → 0.813) but collapses L08 (0.265 → 0.066): the deep-layer signal is spread thinly rather than
concentrated. Any native-width cross-model comparison measures width as much as content.

**Causal steering, both models:**

| model | mean specific effect | chromosomes positive | random push |
|---|---:|---|---|
| 217M | +0.056 | 116/132 | ≈ 0 |
| **1B** | **+0.170** | **132/132: every chromosome, every strength, both seeds** | ≈ 0 |

Do **not** read the 3× as "the 1B uses chromosome 3× more": α is in units of each model's own mean
gene-embedding norm and those differ (1B 3.52 vs 217M 0.73). What is not scale-confounded is the
*consistency*: 132/132 against 116/132.

**Depth of the causal effect:** +0.0801 at the input embedding (22/22 chromosomes), ~26× weaker one
layer in, −0.00058 with a CI spanning zero by layer 5 (11/22), exactly zero at the last layer.

**Karyotype validation.** The same directions reproduce K562's measured chromosome-level expression
signature at r = +0.434 (z = +2.40) against a shuffled-karyotype control at ≈ 0.

**Attention: measured, and negative.** Same-chromosome versus different-chromosome attention, matched
on rank distance (essential: the model uses RoPE, genes are ordered by expression rank, and
same-chromosome genes co-express and so land at similar ranks):

| sample | strongest head | heads with \|z\| > 3 |
|---|---|---:|
| 6 cells | L2H5, z = **−1.13** | **0 / 88** |
| 40 cells | L2H4, z = **+0.53** | **0 / 88** |

The strongest head changes identity and flips sign between samples. **Attention is not sorted by
chromosome at any layer or head.** The causal effect is real; the stated mechanism is not supported.
*(217M only; the cell cycle × attention cell remains empty.)*

**What does not hold.** Sub-chromosomal position is decodable (ρ +0.396, 22/22 chromosomes, beating
both baselines) but the factorisation beats **both** models on it (ρ +0.762 vs +0.622 and +0.374), so
position-beyond-co-occurrence does not survive; and the causal response is **flat** along a chromosome,
so fine position is readable but not used. Steering destinations do not replicate across models (3/22
modal agreement) and fail against expression enrichment in both (p = 0.92 and 0.115), so the
destinations are model-specific geometry, not biology. The one positive mechanism result, 5-Mb local
domains at +0.049 and +0.037, both p ≤ 0.0001, is a *characteristic scale*: at ~150 Mb the
correspondence is undetectable and at 2 Mb it weakens.

**Chromosome is entangled with cell identity.** Chromosome steering disturbs cell-type, lineage, tissue
and cycling readouts more than a matched meaningless push, and *more* in the 1B (cell-type disturbance
+0.541 vs +0.208). It is not stored in a subspace separate from cell identity.

## C.2 Gene identity → cell-cycle phase

Design: a real cell with all 99 cell-cycle genes stripped (phase-neutral backbone), plus one gene
symbol at a fixed rank. Readout: which pole of the real-cell phase circle the last-token state lands
on. Full surface-form ladder in §4.2.

**Constructive sufficiency (the user-designed variant).** Insert 8 genes whose **external**
Cyclebase/Whitfield `\citep{whitfield2002cellcycle}`{=latex} peak-phase annotation (a synchronised time-course measurement, not this dataset's
covariance) is nearest a target angle:

| condition | angular separation |
|---|---:|
| **annotated (true phase assignment)** | **114.3° ± 6.9** |
| shuffled, *same 43 genes, permuted labels* | 13.2° ± 3.7 |
| random, 8 non-cycle genes | 3.5° ± 0.8 |
| *real cells of those phases* | *175.7°* |

≈33σ against random and ≈10σ against shuffled: the movement is driven by **which phase the genes
belong to**, not by cell-cycle gene content. Radius behaves as the disk reading predicts: annotated
modules stay at real-cell radius (0.752) while incoherent mixtures collapse toward the centre (0.508).
At 27B the effect grows with depth (43.5° → 69.9° → 96.8°) with flat controls.

**Limits.** Only ~65% of the real separation; 80–116° absolute offset (8 genes ≪ a full programme);
poor within-arc resolution. **Loop closure is untestable** from annotation alone, because ~40% of the circle
(≈270–330°) has no phase-specific transcripts, so those targets return the 0° module, and a
`loop_gap = 0°` printed by the harness is an artefact, not closure.

**Compute-versus-relay.** Deleting all 87 markers from the input retains 95% of phase decodability,
but a matched baseline retains 0.94 under the identical ablation, so this is **not model-specific**.
The cell cycle modulates transcription genome-wide; this rules out "copies the markers", not "reads a
distributed signature".

## C.3 Gene context, function and switches

**Genes are re-referenced by context, early, then it decays.** Crowd-removed pairwise context-shift
agreement minus a gene-shuffled control, MaxToki-217M: L0 **+0.000 exactly** (sanity), L1–2 peak
**+0.834**, monotone decay to L11 +0.63. Different-gene control +0.0008. Rank-residualised retains 96%.
No dimension carries more than 6.6% of variance.

**Architecture differs by ~10×.** EXCESS (gene-specific context response) and func-z (organisation
along a nuclear-vs-surface axis against a random-axis null) at L4: scGPT 0.705 / +2.74; STATE-SE
0.835 / **+1.31**; MaxToki-217M 0.740 / **+21.17**; MaxToki-1B 0.881 / +16.43. STATE-SE is the striking
case: it moves genes hardest in the *least coherent* direction, because its gene tokens are frozen
ESM-2 vectors carrying sequence rather than co-expression.

**How much of contextualisation is learned?** An untrained, random-init model of identical architecture
and vocabulary gives EXCESS **+0.274** (against +0.74–0.76 trained) and func-z **+3.8** (against
+21.2). So there is a real **architectural floor** of about a third, since random embeddings plus
deterministic attention mixing already produce a reproducible gene-specific shift, but the
**functional organisation is almost entirely learned**.

**The functional-context axis is causally used, but it is not model-specific.** Inject a
nuclear↔surface axis built at L4 into half a cell's gene tokens after L3, read native next-gene logits
on GO sets at the *other* half: signed swing **+0.8149**, 28/30 cells, **p = 4.3×10⁻⁷**, dose-monotone
(+0.088 / +0.271 / +0.815 at α = 0.25/0.5/1.0), norm-matched random ≈ −0.01. **But the axis fails its
co-expression-coherence null at +1.16σ.** This is the cleanest single example of "causally used but not
the model's own".

**Antipodal regulator pairs.** RORC/FOXP3 scGPT p < 10⁻⁴, separation 0.476; GATA1/SPI1 p < 10⁻⁴,
separation 0.818, subspace cosine −0.394/−0.466/−0.195 across models against **+0.007 co-expression**
and **+0.699 ESM-2**; PAX5/PRDM1 p ≤ 0.0005 in all three models. TBX21/GATA3 and SOX2/CDX2 are null
everywhere. Beats both baselines on 3 of 5 pairs.

**Steerable bistable switches.** GATA1/SPI1 **+4.211** [+3.892, +4.539], reciprocal (ΔR_A +1.830,
ΔR_B −2.175); PAX5/PRDM1 **+5.565** [+5.225, +5.902]. Off-tissue RORC/FOXP3 reproducibly *anti*
(−1.9 / −1.6). Norm-matched random push ≈ 0. **No expression or sequence baseline arm was run for this
result**: it gates on steerability, not novelty.

**Paralogy.** Leave-one-cluster-out held-out ρ: A +0.891, B +0.709, C +0.833, D +0.900; mean **+0.833**,
null −0.000 ± 0.231, z = +3.61, 4/4 signs (p = 0.005). ESM-2 **+0.889**; co-expression (Tabula Sapiens)
+0.086 (n.s.); co-expression (fetal gut) +0.506. **Beats co-expression, loses to sequence.** The
analogy `HOXA9 − HOXA1 + HOXB1 → HOXB9`: top-1 0.333, median rank 2 of 20,271, **3.95× a strict
within-cluster null** (z = +13.5); co-expression 1.3× (p = 0.333, cannot do it); ESM-2 10.7×; the
ranking *inverts* between loose and strict nulls.

**Gene function beyond co-expression (C2S).** Within (co-expression × mean-expression) bins, Δcos
same-function vs different: L9 0.0619 (null −0.0009 ± 0.0095, **z = 6.57**, p = 0.005); L13 0.0439
(z = 3.86); L17 0.0547 (z = 4.52). Caveat: one tissue, 3–6 bins.

**The shape of the chromosome object.** It is a **clustering**, not a manifold: four HOX clusters
separate at held-out 4-class accuracy 0.974 (1B) / 0.923 (217M), but the axis is invariant to random
re-coding of the labels and inter-cluster offsets have mean cosine ≈ 0.2 against ≈ 1.0 for a real grid.
Causally it behaves as **22 independent, signed, saturating lookups consumed at the input**, not one
continuous genomic manifold. The *paralog* axis, by contrast, is a genuine shared 1-D coordinate that
composes across clusters.

**HOX causal use is cluster-specific and tissue-invariant.** β = +0.093 (fetal gut) and +0.145 (bone
marrow), both t > 6, placebo-clean, replicating across two independent tissues. Per cluster: HOXB
+0.26 / +0.25 (used in both tissues), HOXA −0.03 / +0.055 (null in both, *even where it dominates*).
A rotation test, moving to a tissue where HOXA dominates, **falsified** the tissue-gating
interpretation: the used cluster does not follow the dominant one. The live hypothesis, untested, is
that HOXB is the most consistently corpus-wide co-regulated cluster.

---

# D. Synthetic experiments in full

## D.1 Design

Corpora are generated from a latent structure we choose, pushed through scGPT's **published** input
encoding (51 quantile bins, expression-sorted truncation), into a **plain standard transformer**:
softmax multi-head attention, GELU feed-forward, pre-norm, d = 192, 4 layers, 4 heads. Nothing about
the model is novel. 20,000 training cells, 1,000 genes, 3,000 steps, val_corr 0.71–0.86 throughout.

**Why an unremarkable model.** These experiments need many corpora, not one large model;
and a bespoke architecture would let any result be attributed to its unusual components rather than to
transformers in general. 

## D.2 The metric-warp law

Plant a circle with an even metric; break one thing inside a 90° arc. Statistic = (model stretch − data
stretch), so only a model-specific effect registers. Stretch = mean knot gap inside the arc ÷ mean gap
outside; centroids computed on an equal number of cells per bin so density cannot drive centroid noise.

| arm | model stretch | data stretch | **model − data** | seeds > 0 | vs control |
|---|---:|---:|---:|---:|---|
| uniform (control) | 1.007 | 0.992 | +0.027 ± 0.095 | 4/8 | n/a |
| **sharp** (output turns over fastest) | 1.202 | 1.032 | **+0.187 ± 0.091** | **8/8** | **+0.160, t = 3.43, p = 0.0041** |
| occupancy (3× cells) | 0.842 | 0.774 | +0.097 ± 0.039 | 7/8 | +0.070, n.s. |
| **noisy** (harder to predict) | 1.040 | 1.139 | **−0.161 ± 0.080** | **0/8** | **negative** |

**Localisation, seeds 3–7** (manipulated arc = bins 5–7):

| arm | argmax bin per seed |
|---|---|
| **sharp** | **5, 5, 6, 6, 5**, inside the arc every time |
| uniform | 1, 10, 5, 4, 2, scattered |
| occupancy | 10, 8, 11, 1, 11, mostly outside |
| noisy | 4, 7, 4, 4, 7, clustered at the arc edge |

**Caveat.** Global CV excess over each representation's own shuffle floor is ≈ 0 or negative in three
of four arms. The `sharp` effect is a **localisation** result, meaning the stretch sits where the manipulation
is, rather than a claim that the model's loop is globally more uneven than the data's.

## D.3 The token table

Assign every gene an arbitrary group label appearing in **no single cell**; sweep how strongly group
membership drives co-occurrence. 22-way probe on the table, split over genes, chance 0.050. PPMI is a
training-free factorisation of the identical corpus.

| consistency | model table | PPMI | margin |
|---:|---:|---:|---:|
| 0.0 | 0.047 | 0.061 | −0.014 |
| 0.2 | 0.160 | 0.202 | −0.042 |
| 0.4 | 0.245 | 0.290 | −0.044 |
| 0.6 | 0.296 | 0.376 | −0.080 |
| 0.8 | 0.340 | 0.440 | −0.100 |
| 1.0 | **0.426** | 0.497 | −0.071 |

**Generalisation to genes never seen together.** Split every group's genes into halves and let programs
draw from one half only, so within-group *cross-half* pairs never co-occur, the synthetic form of the
neighbourhood holdout. 3 seeds:

| | observed pairs | held-out pairs | combined |
|---|---:|---:|---|
| **model table** | 0.1367 ± 0.0068 | **+0.0147 ± 0.0107** | **Stouffer z +20.2** |
| PPMI | 0.1946 ± 0.0077 | +0.0037 ± 0.0011 | n/a |

The model generalises to unseen pairs at **four times** the factorisation's own rate.

**The table is load-bearing:**

| `gene_emb` | val_corr | vs real |
|---|---:|---:|
| real | 0.8534 | n/a |
| zeroed | 0.0842 | −0.769 |
| row-shuffled | 0.0047 | −0.849 |
| **frozen at random init** | **0.6118** | **−0.242** |

**Exclusions ruled out.** Not task degeneracy (a per-gene-mean predictor scores 0.47 against the
model's 0.85). Not a baseline mismatch (PPMI restricted to the model's own top-128 input still scores
0.453 at consistency 1.0 against 0.491 on all genes). Not undertraining (1500 → 12000 steps moves the
probe by 0.003 while val_corr rises 0.801 → 0.857; note the trap: `corr(log steps, acc) = +0.96`,
perfectly monotone and entirely inside chance). Not probe linearity.

## D.4 The steering arm, and why it is excluded

The group direction is built from real embedding rows, `mean(W_E[group == c]) − mean(W_E)`. Three
successive nulls failed:

1. **Norm-matched isotropic random.** At consistency 0.0, where the true effect is exactly zero, this
   reads −0.98 mean with 38% of groups positive, reaching −7.1 at consistency 1.0. Cause: any direction
   built by averaging real embedding rows sits in the dominant part of an anisotropic embedding
   distribution.
2. **Construction-matched** (same arithmetic, shuffled labels). Calibration point still −0.28.
3. **Alignment-matched** (each null draw carries its own shuffled labelling used for *both* the
   direction and the readout, each deflated by its own no-push baseline). Calibration point +0.097 /
   −0.029 / −0.269 across doses, fine at low push but drifting with dose.

**The null-versus-null floor settled it.** Scoring one structureless labelling against another with
identical machinery:

| α | null-vs-null floor | "real" reading (also structureless) |
|---:|---:|---:|
| 0.5 | −0.105 | +0.039 |
| 1.0 | −0.276 | −0.002 |
| 2.0 | **−0.489** | −0.227 |

**The floor exceeds the signal at every dose.** The bias is structural: the real arm is a *single*
labelling while the null is the *mean of five*, and with a saturating dose response those are not
exchangeable, which is why the bias grows with α. **No steering claim is reportable from this
experiment.** The synthetic work therefore establishes readability and beyond-baseline for the token
table and is **silent on causal use** there. The real-model causal evidence (§C.1) is a separate
measurement with its own controls.

## D.5 The stall theorem, verified

Synthetic circle lying exactly in a linear 2-plane of a 60-D space (flat by construction):

| arm | peak advance | laps | off-manifold | ‖D‖ min | reversible |
|---|---:|---:|---:|---:|---:|
| fixed | +1.50 | 0.24 | 11.08 | 1.00 | −1.67 |
| **fixed + projection** | **+1.49** (predicted π/2 = **1.571**) | 0.24 | 1.08 | **0.09** | −1.70 |
| fixed + projection + retraction | +1.50 | **0.24, no rescue** | 0.75 | 0.64 | −1.57 |
| local + projection | +10.76 | 1.71 | 1.65 | 0.66 | −10.84 |
| **local + projection + retraction** | **+31.09** | **4.95** | 0.68 | 0.92 | −31.06 |
| transport (label-free) | +2.45 | 0.39 | 1.44 | **1.00** | −2.12 |
| oracle (uses labels) | +24.05 | 3.83 | 0.64 | 0.27 | −0.94 |

The **‖D‖ collapse to 0.09** is a judge-free measurement of w ⊥ t̂: the raw pre-normalisation step
norm *is* |w·t̂|.

**In seven real representations:**

| representation | fixed + proj | + retraction | local + proj + retraction |
|---|---:|---:|---:|
| scGPT L11 | 0.34 laps (dies T24) | 0.38 ✗ | **4.53** |
| Geneformer L11 | 0.14 (T13) | 0.18 ✗ | **5.57** |
| MaxToki L8 | 0.01 (T1) | 0.02 ✗ | **6.03** |
| STATE-SE L11 | 0.31 (T22) | 0.29 ✗ | **4.70** |
| **raw expression** | 0.36 (T40) | 0.34 ✗ | **5.08** |
| C2S-2B L21 | 0.216 ± 0.170 | 0.159 ✗ | **3.452 ± 0.129** |

`retraction_rescues_fixed = false` in every substrate. **Honest negative:** the label-free `transport`
arm fails everywhere (0.15 laps, ‖D‖ collapse ratio exactly 1.00, since the carried direction never
rotates), so there is currently **no label-free version of the working operator**.

## D.6 The operator ladder

Mean over 16 model × tissue cells; every rule integrated at identical L2 path length; advance judged by
an independent kNN regressor fitted on training cells only.

| operator | CA@1.3 | tangent alignment | off-manifold @T=8 | verdict |
|---|---:|---:|---:|---|
| linear (constant direction) | 0.461 | 0.420 | 1.847 | baseline |
| quadratic (global bilinear) | 0.426 | 0.400 | ~1.85 | **refuted 16/16** |
| local (no projection) | 0.389 | 0.402 | 1.864 | worse in 10/16 |
| piecewise-5 / piecewise-10 | 0.288 / 0.228 | 0.486 / 0.540 | 1.675 / 1.658 | monotonically worse |
| **linear + projection** | **1.239** | 0.692 | 1.355 | **+0.778, 16/16** |
| **local + projection** | **1.320** | 0.683 | 1.347 | best; beats oracle 13/16 |
| oracle (uses labels at inference) | 0.676 | 0.774 | 0.686 | not a competitor |

Ablations: projection alone +0.778; local direction alone −0.072. **On a closed loop the balance
reverses**: gain from locality +2.09 (scGPT), +4.66 (MaxToki) against gain from projection +1.45 and
+0.06. The general operator is local direction + projection + retraction, and topology decides which
term matters.

**Why the global quadratic failed, measured rather than inferred:** Q's own gradient field rotates only
12–96° early→late against a measured true tangent rotation of 48–113°. One global quadratic
under-rotates by roughly half and cannot express the direction discontinuity at a bifurcation.

**The direction/projection 2×2.** The original `random_dir` control changed two things at once.
Replaced by {developmental, random} × {projected, not}: read through scGPT's own MVC head, dev-axis
cosine +0.0637 (random + projection) vs +0.3485 (real direction + projection); differentiation contrast
+0.522 vs +2.155. **The projection alone steers nothing; the direction carries the biology.**

---

# E. A worked negative: the extracted-operator battery

An earlier submitted result claimed a compact developmental representation "extracted" from a frozen
model: an operator exported from the model, a low-dimensional adaptor fitted to a curated stage
ontology, and task-specific readouts, benchmarked against scVI `\citep{lopez2018scvi}`{=latex}, Palantir `\citep{setty2019palantir}`{=latex}, DPT `\citep{haghverdi2016dpt}`{=latex}, CellTypist `\citep{dominguezconde2022celltypist}`{=latex}, PCA, SVD and raw expression across 88 donor-holdout splits with BH-corrected Wilcoxon tests. Roughly 25 control arms
were then added. It does not survive them.

**What the operator actually is.** Not a weight matrix. It is a sum of post-softmax attention
probabilities accumulated over 600 cells, never divided by its companion counts file, so row sums equal
per-gene detection counts exactly (r = 1.0000), and 580 of 1,200 genes are undetected in that corpus,
so every operator has 580 identically-zero rows. The model in question is d = 512 with 8 heads, so no
parameter tensor is 1,200 × 1,200; that is the sequence length. "Direct export from frozen weights" is
therefore false: the stage requires a forward pass over target-domain cells.

**It ties a random matrix.** At the cell level there is **no bottleneck width ≥ 10 at which the real
operator separates from a matched Gaussian on the same support**. At d = 10 two identically constructed
random nulls differ from each other by *more* than the real operator differs from them (null-vs-null
p = 0.016 against the claim's p = 0.156). At full capacity all three operators tie within 0.003.

**Training budget inverted the ranking.** At the paper's own 220 epochs the real operator is beaten by
a Gaussian (0.711 vs 0.852), by co-expression (0.762) and by its own shuffled values (0.758), and all
96 heads sit below the worst of 20 random draws. At 2,000 epochs the real operator leads, by +0.009
over its own shuffle, +0.011–0.015 over co-expression and +0.027 over a Gaussian, against a
null-vs-null floor of 0.006. **A fixed epoch budget is not a neutral choice.**

**What the selection score measures.** Across all 192 circuits the head-selection score correlates
**r = 0.088** with the biological endpoint it was chosen for, and **−0.90** with cell-level stage
accuracy. Destroying gene-pair identity while preserving the value distribution on the same support
reproduces 49–80% of the real operator's margin; preserving the exact spectrum while permuting the gene
axis reproduces little. So the score measures "were the genes mixed by a reasonably conditioned dense
matrix", not by *which* matrix. A ~0.42 gap opens between no-operator (0.44) and any operator
(0.77–0.81), and everything after sits in a ~0.07 band that the null-vs-null floor (0.042) nearly
fills.

**Weight-derived operators do not rescue it.** The best is the pure learned gene-embedding gram at
0.8634, against its correct matched null (the same embedding with gene rows permuted, preserving
geometry and spectrum exactly) at 0.8477. The gap is 1.0 sd of the null-vs-null floor. Every pooled
QK and OV operator sits at or below that null. And there is an identity behind it: `x·EEᵀ` is a
rank-512 linear reparametrisation of `x`, so a full-capacity linear probe cannot see anything in it it
could not see in raw `x`.

**The model's own representations lose to a random projection.** On CD4/CD8 subtype discrimination
across 14 held-out donors, a dimension-matched **random projection of raw genes** beats the model's
frozen embeddings by **+0.0507 (14/14 donors, p = 0.0001)**; PCA-512 by +0.0472; raw genes by +0.0424.
L2 normalisation is not the cause (the unnormalised vectors have norm CV 1.03%, and re-extraction with
normalisation off changes the result by +0.00003, p = 0.95); nor is the readout choice (`<cls>` vs
avg-pool: +0.0061, p = 0.36, still 0.043 below the random projection).

**One incidental finding.** That model's preprocessing breaks tied quantile bins with an unseeded global
RNG, so its embeddings are not reproducible run to run: the same cells re-extracted agree at cosine
0.9896, while agreement with the cached version is 0.9901, so the pipeline agrees with itself no better
than with a different run.

**What is left standing:** Stage-1 features outperform raw expression by 0.051 AUROC on T-cell subtype
discrimination across 14 held-out donors. But a random matrix on the same support performs
equivalently, so the gain cannot be attributed to learned structure.

We report this in full because every individual check the original performed was passed. The failure
lies in the choice of checks: none of them was a matched random operator or an information-matched
baseline.

---

# F. Retractions, instrument failures, and things that did not work

## F.1 Claims we withdrew

| withdrawn | replaced by |
|---|---|
| "Chromosome *and* within-chromosome position beat co-expression" | The control was a raw per-gene expression profile, which sits at chance and excludes nothing. A proper factorisation of the identical cells reaches 0.720, which beats the 217M outright and takes *position* away from both models (ρ +0.762 vs +0.622 / +0.374). **Two claims retracted from one methodological error.** |
| "A global quadratic gradient field steers better than a constant direction" | Refuted 16/16. Best real gap +0.206, sign unstable, synthetic-linear control ≈ 0. |
| "The GATA1↔PU.1 saddle is a mutual-repression geometry" | Refuted twice: representationally (product R² −0.004 vs s_ery² +0.114) and in the biology (correlation among undecided progenitors −0.050 against a −0.15 threshold). |
| "The cell-cycle manifold is a low-dimensional ring" | A **filled disk**; the earlier statistic was computed on 12 bin centroids and is void as single-cell dimensionality. |
| "C2S sharpens phase ~3× over its own input" | The comparison used a `1/rank` weighting that collapsed the baseline's PCA onto a few genes. A faithful encoding gives 0.868–0.882, level with the model at 0.875. |
| "Compute, not relay: 95% retention after deleting markers" | Not model-specific: a matched baseline retains 0.94 under the identical ablation. |
| "Human curved, mouse flat" | Significance tracks linear-decoder headroom (partial corr −0.858 vs −0.091 for species). |
| "Steering the manifold is retired: there is no curved arc to ride" | *Global* steering fails; local-tangent projection works. |
| "Projection is the whole story" | True on trees, false on loops (§D.6). |
| "16/180: models are useless on pseudotime" | An artefact of an inherited k ≤ 50 cap; 66/180 at k ≥ 100. The correct verdict is 0/20 against a *nonlinear* expression baseline. |
| Aging/HSC headline | Withdrawn by its own held-out test: disease donors sit inside the healthy range (p = 0.23) and the winning lineage flips between runs. What survives is a "real, small, and on current evidence useless" direction-vs-position signal. |

## F.2 Instruments that broke

- **θ_far must not be used on periodic manifolds.** On a circle the first and last bins are adjacent, so
  a perfect circle scores ≈ 0; the real loop scores *below* its own flat-circle null in every model.
- **The decodability gap (kNN − linear) is not a curvature statistic.** It is structurally
  headroom-limited and correlates −0.926 with linear R². Deprecated.
- **Circular correlation is degenerate against uniform variables.** It uses sin(a − circmean(a)), so a
  constant rotation cancels and the null equals the real value exactly; and it returns r = +0.804 at
  p = 0.0005 for an arm that travelled **zero** laps. Replaced by decoded winding and a time-shift null.
- **Full-space kNN breaks at high dimension** (curvature −0.457 at d = 2048).
- **A whitening bug** made the first nonlinear expression baseline return R² = −2.583, and **the broken
  version would have confirmed a model win**.
- **Permutation-floor p-values** (1/13, 1/21, 1/41, 1/201) were quoted as evidence in several places.
- **Small-n leave-one-out is degenerate:** pure noise scores |ρ| = 0.736 at n = 9.
- **A saturated test cannot distinguish present from absent:** perturbation identity decodes at 1.000
  from a *random-init* model because the label is an injected input.
- **MPS boolean indexing** returned mismatched element counts for the same mask (3243 vs 40) at scale;
  removed from the hot path.

## F.3 Nulls that were wrong, and the general lesson

Six documented cases where a large z against a permutation null went to ≈ 0 against a competitor built
from the same cells: attention z +4.18 → +0.17; the functional-context axis +20σ random-partition, +9σ
tightness, **+1.16σ co-expression**; paralogy beats co-expression and loses to sequence; C2S phase
0.898 against a broken input encoding and parity against a faithful one; cross-model consensus survives
every hygiene attack until a matched co-expression neighbourhood scalar recovers ρ 0.458–0.581; and a
label made of **pure sampling noise** scoring 0.447 at z = 35.8, higher than the real biological label
at 0.357, z = 24.9.

## F.4 Things that did not work

- **Perturbation inversion.** Negative in 4/4 models including the perturbation-trained one. AUROC
  0.487–0.707; 0/4 fates significant; residualised on differential expression all q ≥ 0.26. A model
  trained on a CRISPRi vocabulary passes its validity gate yet ranks the canonical erythroid regulator
  111/131 **with the sign backwards**. *Caveat:* the differential-expression-matched stratum test cannot
  run (0–2 known drivers fall in the low-|DE| half), so the negative is "no signal beyond DE", not
  "provably no regulatory knowledge".
- **Hidden clonal fate (LARRY `\citep{weinreb2020larry}`{=latex}).** A well-powered negative: expression position AUROC 0.767, geometry
  0.540 ≈ its clone-permutation null 0.383 ± 0.061, combined gain −0.006.
- **Reconstructing a deleted intermediate.** Held-out peak r = +0.545, rank 5/10, path running through
  the wrong lineage. Steering interpolates; it does not generate.
- **Commitment as a discrete geometric step.** No cliff (sigmoid transition width 0.27–0.83 of the
  pseudotime range), and a counts baseline wins in 3/4 models.
- **Label-free `transport` steering.** Fails in all five substrates, ‖D‖ collapse ratio exactly 1.0000.
- **Irreducible topology.** No loops or holes anywhere. In-plane H1 null in five substrates including
  raw expression (all p ≥ 0.27); out-of-plane excess negative in 5/5; lineage-tree H0 z = −6.81
  (branches *tighter* than a matched Gaussian).
- **Bilinear/interaction machinery generally.** Supervised bilinear CP probes lose to linear (−0.060);
  SAE dictionaries `\citep{kendiukhov2026sae}`{=latex} are 73–99.7% redundant with linear ones and order-4 is worse than order-2; and
  third-order combinatorial ablation finds **0.00% superadditive interactions over ~2,980 targets**.
- **Scale.** 5× parameters changes rotation excess by −11.1° and Δρ by 0.00; at matched SAE capacity a
  13× larger model has 31× more dead features and 19 points lower annotation.

## F.6 Open questions we consider most important

1. **A neighbourhood-co-regulation baseline for chromosome.** Every existing baseline is a per-gene
   profile or a global factorisation; neither models *local* co-regulation. Scoring against Hi-C A/B
   compartments and replication timing decides whether the strongest positive survives.
2. **What makes a token table a map rather than an index**: architecture, corpus scale, vocabulary
   size, or a separately-used output head.
3. **Cell cycle × attention**, the last empty cell in the coverage grid.
4. **A causal test for token-table facts under a calibrated null**, given §D.4.
5. **The output-change law on a non-cyclic manifold.**
6. **Correcting the two extraction defects** across all affected analyses (§A.3).

