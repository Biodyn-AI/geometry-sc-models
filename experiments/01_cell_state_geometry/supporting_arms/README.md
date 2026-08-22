# Manifold fitting, clonal fate, and a withdrawn result

Supporting and withdrawn arms. Includes the LARRY clonal-fate negative and the aging result that its own held-out test withdrew.

12 scripts, 8 committed result files.

Every script here needs assets this repository does not ship. See [`docs/DATA.md`](../../../docs/DATA.md) for what to download and where to point `GEOMSC_DATA` and `GEOMSC_MODELS`.

*A note on section F.* Earlier drafts of the paper carried a supplement section F recording retractions, instruments that failed, and analyses that did not work. That section was removed from the paper. The code and result files for those analyses are still here, because the reasoning in the published sections rests on them; rows that used to point at F are marked *withdrawn arm* below.

| script | what it computes | supports | needs |
|---|---|---|---|
| `common.py` | Shared config for the Tier-B manifold arm: external drive roots, scGPT checkpoint/vocab paths, layer-block boundaries for the drift operator, ruler loading (scGPT cell-type / Geneformer cell-cycle), t | none directly | nothing itself, but every path it defines points at an external drive; the two m |
| `gates.py` | The four-gate stack: sklearn trustworthiness, Isomap-style geodesic-vs-ruler Spearman on a k-NN graph, 200-permutation blocked null (shuffled within tissue or within cell-cycle phase), and a matched-n | none directly | nothing external; sklearn + scipy only |
| `let.py` | Stage-2 LET adaptor: a linear head z = W_enc(x - b), d_latent=10, trained so beta*arccos(cos(z_i,z_j)) matches the biological-ruler distance, with an alpha=0.05 reconstruction regulariser; scaler froz | none directly | nothing external; pure torch/numpy |
| `lineage_graph.py` | Curated hematopoietic lineage graph: maps each Tabula Sapiens ontology cell_type string to a terminal branch (myeloid/T/B/NK/erythroid/stem), a coarse major axis and a nominal tree depth, by ordered s | none directly | its __main__ audit needs only route_geometry/results/scgpt_L11_ruler.npz (696 KB |
| `extract_external.py` | Stage-3 holdout extractor: runs strict non-overlap cells through frozen scGPT whole-human (mirrors bio-sae scgpt_src/01_extract_activations.py, including the FlashMHA->MHA weight rename) and stores on | none directly | needs the 196 MB scGPT whole-human best_model.pt + vocab.json, the three Tabula  |
| `fate_output.py` | The healthy-cohort (n=8) readout: leave-one-donor-out fate directions from stem centroid to committed centroid, then two readouts — POSITION (projection onto the fate-specific direction) and DIRECTION | Supplement F.1 aging row — the withdrawn headline. Produces position rho -0.175 (shuffle p 0.774), direction r | needs data/aging/aging_setty_schema.h5ad (339 MB), data/branchpoint/scgptbin_agi |
| `lineage_manifold.py` | The full branching-manifold test: (A) linear vs degree-2 vs kNN decode of lineage branch at matched 20-D PCA, (B) LEACE erasure ladder with a linearly-embedded synthetic control, (C) supervised biline | Supplement F.4 'Bilinear/interaction machinery generally' — 'Supervised bilinear CP probes lose to linear (-0. | needs route_geometry/results/emb_cache/scgpt_L11.npz (140.6 MB) and scgpt_L11_ru |
| `mds_test.py` | The held-out disease test that withdrew the headline: extends the identical leave-one-donor-out readout to all 12 donors including the 4 never-touched MDS patients, and tests H3 (MDS lymphoid accessib | Supplement F.1: 'Aging/HSC headline ... Withdrawn by its own held-out test: disease donors sit inside the heal | needs data/aging/agingmds_setty_schema.h5ad (513 MB), data/branchpoint/scgptbin_ |
| `operators.py` | Stage-1 operator library: builds six per-cell feature branches (raw_mean, svd50, linear_sae, bilinear, linear_drift, bilinear_drift) from mean-pooled residual streams, caching each to .npy outside the | none directly | needs 82 GB scGPT + 315 GB Geneformer cached token activations, the trained line |
| `preprocess_aging.py` | Builds GSE180298 (Ainciburu CD34+ HSPC 10x matrices + author metadata) into the Setty h5ad schema so route_steering/extract_scgpt_binned.py runs on it unchanged; computes each donor's committed-lineag | none directly — it is the data build behind the Supplement F.1 aging retraction | needs scanpy (a separate conda env in the recorded reproduce commands), the 449  |
| `run_tier_b.py` | Internal sweep driver: for each of six operators fit LET over 5 seeds, score all four gates, freeze the best-seed head for the holdout, write results/internal_gates_{model}.json. | none — produces geodesic-ruler rho 0.689 (linear_sae) vs 0.528 (bilinear) on scGPT and 0.895 (linear_drift) vs | needs operators.py caches to exist; `python run_tier_b.py [scgpt\|geneformer]` |
| `score_external.py` | Scores the frozen LET heads on the external panels with zero retraining; writes results/external_gates_{panel}.json. | none — produces external geodesic-ruler rho 0.632 linear_sae vs 0.547 bilinear, and the Krasnow control collap | needs the frozen let_heads/*.pkl and the external_*.npz produced by extract_exte |

## Provenance

These files were collected from the working tree in which the study was run. Paths to large assets have been parameterised; nothing else about the analysis logic was changed.
