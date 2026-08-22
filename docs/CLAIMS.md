# From claim to code

Every row is a claim in the paper and the script that produced it. Result files are committed beside each script under `results/`, so a number can be checked without rerunning anything.

Section numbers refer to [`paper/main.pdf`](../paper/main.pdf); `Supp X` refers to its supplement. Rows marked *route notes* map to a working log rather than to a numbered claim, and rows marked *unmapped* are supporting analyses the paper does not cite directly; both are kept because the paper's conclusions rest on the whole battery, not only on the numbers that made it into the text.

| paper | script | what it computes |
|---|---|---|
| **§3** | [`celltype_state.py`](../experiments/01_cell_state_geometry/other_representations/celltype_state.py) | none cited; background for the 'cell-type identity is linearly accessible' line of §3 |
| **§3.2** | [`classifier_sweep.py`](../experiments/02_vocabulary_facts/chromosome/classifier_sweep.py) | §3.2 'each given its own best probe' |
| **§3.2** | [`coexpr_fact.py`](../experiments/02_vocabulary_facts/chromosome/coexpr_fact.py) | §3.2 baseline construction for §4.1 |
| **§3.2** | [`nonlinear_sweep.py`](../experiments/02_vocabulary_facts/chromosome/nonlinear_sweep.py) | §3.2 probe family |
| **§3.3** | [`baseline.py`](../experiments/01_cell_state_geometry/developmental/baseline.py) | §3.3 — 'endpoint targeting on hard, low-headroom trajectories (mouse pancreas endpoint purity 0.207 → 0.569, with 4 of 4 models beating expression; lu |
| **§3.3** | [`common.py`](../experiments/01_cell_state_geometry/developmental/common.py) | none directly; carries the fate and marker definitions behind every §3.3 endpoint-purity and fate-diagonal number. |
| **§3.3** | [`dump_shifts_maxtoki.py`](../experiments/01_cell_state_geometry/developmental/dump_shifts_maxtoki.py) | Cross-model replication of the inversion negative (PERTURB_INVERT_CROSSMODEL.md). Not a §3.3 number. |
| **§3.3** | [`dump_shifts_state.py`](../experiments/01_cell_state_geometry/developmental/dump_shifts_state.py) | Cross-model replication of the inversion negative. Not a §3.3 number. |
| **§3.3** | [`fate_matrix.py`](../experiments/01_cell_state_geometry/developmental/fate_matrix.py) | Not quoted numerically in §3.3, but it is the evidence behind the paper's fate-steering framing: 14 of 15 target→fate hits, pooled exact p = 1.93e-05  |
| **§3.3** | [`gene_decode.py`](../experiments/01_cell_state_geometry/developmental/gene_decode.py) | §3.3/§4.4 supporting evidence — stem program +0.20 → −0.53 (scGPT) under the steer vs +0.20 → +0.34 under a random direction; the original 4/4 fate di |
| **§3.3** | [`perturb_invert.py`](../experiments/01_cell_state_geometry/developmental/perturb_invert.py) | Not cited in §3.3. Supports the paper's navigation-vs-intervention dissociation (Discussion): AUROC 0.487–0.707, 0/4 fates survive BH-FDR, and nothing |
| **§3.3** | [`robustness.py`](../experiments/01_cell_state_geometry/developmental/robustness.py) | Supports §3.3's expression-PCA framing: expression beats scGPT 15/15, Geneformer 15/15, MaxToki 15/15, STATE 11/15 = 56/60. |
| **§3.3** | [`score_shifts.py`](../experiments/01_cell_state_geometry/developmental/score_shifts.py) | Makes the three-model inversion comparison apples-to-apples. Not a §3.3 number. |
| **§3.3** | [`st_fate_invert.py`](../experiments/01_cell_state_geometry/developmental/st_fate_invert.py) | Closes the 'they are only embedding models' escape hatch: gate passes, yet GATA1 ranks 111/131 for anti-erythroid with the sign backwards. Discussion- |
| **§3.3** | [`tangent_diagnostic.py`](../experiments/01_cell_state_geometry/developmental/tangent_diagnostic.py) | §3.3 — theta_far 48–113 deg between first and last pseudotime quintile, exceeding a heteroscedastic residual-permutation null (NULL C) in 16 of 16 cel |
| **§3.4** | [`steer_lib.py`](../experiments/02_vocabulary_facts/chromosome/steer_lib.py) | §3.4 protocol for all of §4.1's causal results |
| **§4.1** | [`coocc_bestprobe.py`](../experiments/02_vocabulary_facts/chromosome/coocc_bestprobe.py) | §4.1 feeds the 0.880 / 0.720 / 0.506 numbers |
| **§4.1** | [`coocc_final.py`](../experiments/02_vocabulary_facts/chromosome/coocc_final.py) | §4.1 factorisation baseline 0.692 group-split at LSA-256 |
| **§4.1** | [`coocc_matched.py`](../experiments/02_vocabulary_facts/chromosome/coocc_matched.py) | §4.1 'raw expression profile 0.044' and 'permuted labels 0.045' |
| **§4.1** | [`coocc_position.py`](../experiments/02_vocabulary_facts/chromosome/coocc_position.py) | §4.1 last sentence / Supp S2: factorisation beats both models on position (rho +0.762 vs +0.622 and +0.374) |
| **§4.1** | [`coocc_strongest.py`](../experiments/02_vocabulary_facts/chromosome/coocc_strongest.py) | §4.1 / Fig 3c headline: MaxToki-1B 0.880, strongest factorisation 0.720, MaxToki-217M 0.506, chance 0.045, n=15,135 |
| **§4.1** | [`ctx_anisotropy.py`](../experiments/02_vocabulary_facts/chromosome/ctx_anisotropy.py) | none in §4.1 |
| **§4.1** | [`ctx_causal.py`](../experiments/02_vocabulary_facts/chromosome/ctx_causal.py) | none in §4.1 |
| **§4.1** | [`ctx_celltype_loadings.py`](../experiments/02_vocabulary_facts/chromosome/ctx_celltype_loadings.py) | none in §4.1 |
| **§4.1** | [`ctx_coexpr_null_v2.py`](../experiments/02_vocabulary_facts/chromosome/ctx_coexpr_null_v2.py) | none in §4.1 (relevant to §4.3's function-beyond-co-expression logic) |
| **§4.1** | [`ctx_cross_model.py`](../experiments/02_vocabulary_facts/chromosome/ctx_cross_model.py) | none in §4.1 |
| **§4.1** | [`ctx_curated_targets.py`](../experiments/02_vocabulary_facts/chromosome/ctx_curated_targets.py) | none in §4.1 |
| **§4.1** | [`ctx_devel_trajectory.py`](../experiments/02_vocabulary_facts/chromosome/ctx_devel_trajectory.py) | none in §4.1 |
| **§4.1** | [`ctx_directional_probe.py`](../experiments/02_vocabulary_facts/chromosome/ctx_directional_probe.py) | none in §4.1 |
| **§4.1** | [`ctx_extract_devel.py`](../experiments/02_vocabulary_facts/chromosome/ctx_extract_devel.py) | none in §4.1 |
| **§4.1** | [`ctx_extract_maxtoki.py`](../experiments/02_vocabulary_facts/chromosome/ctx_extract_maxtoki.py) | none in §4.1 - belongs to the separate gene-context paper |
| **§4.1** | [`ctx_extract_random.py`](../experiments/02_vocabulary_facts/chromosome/ctx_extract_random.py) | none in §4.1 |
| **§4.1** | [`ctx_extract_scgpt.py`](../experiments/02_vocabulary_facts/chromosome/ctx_extract_scgpt.py) | none in §4.1 |
| **§4.1** | [`ctx_extract_state.py`](../experiments/02_vocabulary_facts/chromosome/ctx_extract_state.py) | none in §4.1 |
| **§4.1** | [`ctx_feasibility.py`](../experiments/02_vocabulary_facts/chromosome/ctx_feasibility.py) | none in §4.1 |
| **§4.1** | [`ctx_figures.py`](../experiments/02_vocabulary_facts/chromosome/ctx_figures.py) | none in §4.1 |
| **§4.1** | [`ctx_functional_axes.py`](../experiments/02_vocabulary_facts/chromosome/ctx_functional_axes.py) | none in §4.1 |
| **§4.1** | [`ctx_independent.py`](../experiments/02_vocabulary_facts/chromosome/ctx_independent.py) | none in §4.1 |
| **§4.1** | [`ctx_layer_curve.py`](../experiments/02_vocabulary_facts/chromosome/ctx_layer_curve.py) | none in §4.1 |
| **§4.1** | [`ctx_polysemy.py`](../experiments/02_vocabulary_facts/chromosome/ctx_polysemy.py) | none in §4.1 |
| **§4.1** | [`ctx_position_confound.py`](../experiments/02_vocabulary_facts/chromosome/ctx_position_confound.py) | none in §4.1 |
| **§4.1** | [`ctx_prediction_link.py`](../experiments/02_vocabulary_facts/chromosome/ctx_prediction_link.py) | none in §4.1 |
| **§4.1** | [`ctx_switcher_test.py`](../experiments/02_vocabulary_facts/chromosome/ctx_switcher_test.py) | none in §4.1 |
| **§4.1** | [`ctx_tightness_null.py`](../experiments/02_vocabulary_facts/chromosome/ctx_tightness_null.py) | none in §4.1 |
| **§4.1** | [`final_probe_grid.py`](../experiments/02_vocabulary_facts/chromosome/final_probe_grid.py) | §4.1 probe-capacity control |
| **§4.1** | [`gene_sets.py`](../experiments/02_vocabulary_facts/chromosome/gene_sets.py) | none (pre-§4.1 exploration) |
| **§4.1** | [`genome_causal.py`](../experiments/02_vocabulary_facts/chromosome/genome_causal.py) | §4.1 causal use: +0.055, bootstrap CI [+0.013, +0.108], 18/22 chromosomes positive, random push -0.0004 |
| **§4.1** | [`genome_causal_sweep.py`](../experiments/02_vocabulary_facts/chromosome/genome_causal_sweep.py) | §4.1 'genuine dose-response, significant across three seeds even at a sub-natural strength'; Fig 2a of the route paper |
| **§4.1** | [`genome_deleak.py`](../experiments/02_vocabulary_facts/chromosome/genome_deleak.py) | §4.1 'holding out whole token-ID blocks retains 96%' (0.4332 -> 0.4145) |
| **§4.1** | [`genome_groupsplit.py`](../experiments/02_vocabulary_facts/chromosome/genome_groupsplit.py) | §4.1 / Fig 3a: MaxToki lm_head 0.433->0.347 (78% retained), W_E 0.433->0.373 (85%), Geneformer 51%, scGPT 24%, ESM-2 0.190->0.074 (20%, the collapse) |
| **§4.1** | [`genome_intersection.py`](../experiments/02_vocabulary_facts/chromosome/genome_intersection.py) | §4.1 supporting; explains why ESM-2's own-set 0.19 drops to 0.105 on shared genes |
| **§4.1** | [`genome_seq_mechanism.py`](../experiments/02_vocabulary_facts/chromosome/genome_seq_mechanism.py) | §4.1 'its signal was tandem-duplicate family resemblance' |
| **§4.1** | [`genome_wide.py`](../experiments/02_vocabulary_facts/chromosome/genome_wide.py) | §4.1 decoding row: MaxToki 0.43, Geneformer 0.17, scGPT 0.09, ESM-2 0.19, raw co-expression ~0.05, chance 0.045; also 'removing HOX and the protocadhe |
| **§4.1** | [`gm_lib.py`](../experiments/02_vocabulary_facts/chromosome/gm_lib.py) | underpins every number in §4.1 |
| **§4.1** | [`infercnv_confound.py`](../experiments/02_vocabulary_facts/chromosome/infercnv_confound.py) | an applied side result, not cited in main §4.1 |
| **§4.1** | [`karyotype_sim.py`](../experiments/02_vocabulary_facts/chromosome/karyotype_sim.py) | §4.1 external validation: r_chrom = +0.434 vs shuffled -0.028 +/- 0.19 (z = +2.40), gene-level residual r = -0.02 |
| **§4.1** | [`make_note_pdf.sh`](../experiments/02_vocabulary_facts/chromosome/make_note_pdf.sh) | none in §4.1 |
| **§4.1** | [`make_paper_pdf.sh`](../experiments/02_vocabulary_facts/chromosome/make_paper_pdf.sh) | none in §4.1 |
| **§4.1** | [`maxtoki_layers.py`](../experiments/02_vocabulary_facts/chromosome/maxtoki_layers.py) | §4.1 'Where the variable lives': 0.453 W_E, 0.516 lm_head, decaying monotonically to 0.088 at L11 |
| **§4.1** | [`model_scale.py`](../experiments/02_vocabulary_facts/chromosome/model_scale.py) | §4.1 scale: 1B 0.837 random / 0.703 group vs 217M 0.485 / 0.368; and Fig 3b 1B matched-width 0.813 -> 0.066 |
| **§4.1** | [`steer_layers2.py`](../experiments/02_vocabulary_facts/chromosome/steer_layers2.py) | §4.1 'Steering works at the input embedding (+0.0801, 22/22) and is indistinguishable from zero by layer 5'; Supp S6 depth |
| **§4.1** | [`steer_propagation.py`](../experiments/02_vocabulary_facts/chromosome/steer_propagation.py) | §4.1 '132 of 132 chromosome x strength x seed combinations in the 1B (116/132 in the 217M)' and Fig 3c |
| **§4.1** | [`synteny_transfer.py`](../experiments/02_vocabulary_facts/chromosome/synteny_transfer.py) | §4.1 'Two limits': the ranking inverts, factorisation 0.565 vs both models 0.529 |
| **§4.1** | [`table_grid.py`](../experiments/02_vocabulary_facts/chromosome/table_grid.py) | §4.1 'strong in the vocabulary tables (0.516 output, 0.453 input)' |
| **§4.3** | [`antipodal_subspace.py`](../experiments/02_vocabulary_facts/chromosome/antipodal_subspace.py) | §4.3 'GATA1/SPI1 separation 0.818, subspace cosine -0.394 model vs +0.007 co-expression vs +0.699 ESM-2' |
| **§4.3** | [`hox_analogy_null.py`](../experiments/02_vocabulary_facts/chromosome/hox_analogy_null.py) | §4.3 'HOXA9 - HOXA1 + HOXB1 lands on HOXB9 at 3.95x a strict within-cluster null (z=+13.5)' |
| **§4.3** | [`hox_causal_locus.py`](../experiments/02_vocabulary_facts/chromosome/hox_causal_locus.py) | §4.3 of the route paper: beta = +0.093 (t=7.9) fetal gut, +0.146 (t=6.1) aging marrow; HOXB +0.26/+0.25, HOXA null in both |
| **§4.3** | [`hox_within.py`](../experiments/02_vocabulary_facts/chromosome/hox_within.py) | §4.3 'the paralog coordinate transfers to it at held-out rho=+0.833, correct sign in 4 of 4 clusters (p=0.005)' |
| **§4.3** | [`lineage_steer.py`](../experiments/02_vocabulary_facts/chromosome/lineage_steer.py) | §4.3 '+4.211 [+3.892,+4.539] for GATA1/SPI1 and +5.565 for PAX5/PRDM1' |
| **§4.3** | [`hb2_normal_space.py`](../experiments/03_what_shapes_geometry/steering_operator/hb2_normal_space.py) | none — not cited in main.tex or the supplement. Internal negative (labelled AUROC 0.90 local tangent vs 1.00 full head vs 0.92 empirical co-expression |
| **§4.4** | [`local_steering.py`](../experiments/01_cell_state_geometry/developmental/local_steering.py) | §4.4 sec:operator — constrained advance 0.461 → 1.239, positive in 16 of 16 cells, beats the label-using oracle in 12 of 16; ablation projection alone |
| **§5** | [`extract_scgpt_binned.py`](../experiments/01_cell_state_geometry/developmental/extract_scgpt_binned.py) | §5 'A note on extraction' — produces the corrected scgptbin_*.npz that every later number uses. The exact deltas the paper quotes (+0.109 decodability |
| **§5** | [`native_decode.py`](../experiments/01_cell_state_geometry/developmental/native_decode.py) | §5 'A note on extraction' — the gate FAILS on raw-count embeddings (within-cell Spearman +0.085) and PASSES on correctly binned ones (+0.260); this is |
| **§5.4** | [`benchmark_larry.py`](../experiments/03_what_shapes_geometry/steering_operator/benchmark_larry.py) | §5.4 supporting evidence for the released tool (numbers not quoted in main.tex). Produces 6/6 diagonal hits both arms; mean path off-manifold ratio 1. |
| **§5.4** | [`benchmark_setty.py`](../experiments/03_what_shapes_geometry/steering_operator/benchmark_setty.py) | §5.4 model-free claim; supplement §D.6 ladder in counts. Produces ca13 linear 1.916, linear_proj 2.187, retract 1.229 -> projection gain +0.271, proj- |
| **§5.4** | [`example.py`](../experiments/03_what_shapes_geometry/steering_operator/example.py) | none (documentation/demo for §5.4's released tool) |
| **§5.4** | [`gate0_counts_vs_model.py`](../experiments/03_what_shapes_geometry/steering_operator/gate0_counts_vs_model.py) | §5.4 final sentence ('needs only a cell-by-gene matrix ... model-free tool'). Produces projection gain MODEL +0.95/+0.52/+1.18/+0.24 vs COUNTS +1.35/+ |
| **§5.4** | [`gate1_within_branch_rotation.py`](../experiments/03_what_shapes_geometry/steering_operator/gate1_within_branch_rotation.py) | Not §5.4. It is the premise check under §5.5 (sec:curv, 'the geometry arrives bent') and §5.4's 'the tree shape says the direction must be local'. Pro |
| **§5.4** | [`gate2_retraction.py`](../experiments/03_what_shapes_geometry/steering_operator/gate2_retraction.py) | §5.4 'Retraction cures outward drift but never the stall'; supplement §D.5/§D.6 retraction rows. Produces proj-retract +0.29 (scgptbin model), +0.87 ( |
| **§5.4** | [`geom_common.py`](../experiments/03_what_shapes_geometry/steering_operator/geom_common.py) | supports §5.4 ladder rows via gate0/gate2; the retraction fields are the new code here |
| **§5.4** | [`manifold_steer.py`](../experiments/03_what_shapes_geometry/steering_operator/manifold_steer.py) | §5.4 (sec:operator): 'Local direction + tangent projection + retraction ... The operator needs only a cell-by-gene matrix, so we release it as a model |
| **§5.5** | [`qfit.py`](../experiments/03_what_shapes_geometry/curvature/qfit.py) | §5.5 / Table 4 (tab:curv) — imported UNCHANGED by route_curvature_mech/curvature_mech.py, which is what actually produces angular +0.0480/+0.0598/+0.0 |
| **§6** | [`run_nulls.py`](../experiments/03_what_shapes_geometry/curvature/run_nulls.py) | Both nulls pass: shuffle max −0.0024, linear-synthetic max −0.0003 against real +0.0250 (~60 sd). Supports paper §6 test 2 (permutation null vs compet |
| **Supp A.1** | [`ts_extract.py`](../experiments/01_cell_state_geometry/other_representations/ts_extract.py) | Supplement A.1 STATE-SE row, 'L11, skipping CLS' |
| **Supp A.1** | [`uce_loader.py`](../experiments/01_cell_state_geometry/other_representations/uce_loader.py) | Supplement A.1 representations table, row 'UCE-100M ... transformer ... ESM-2 gene tokens ... L2'; A.1 note that UCE's and STATE-SE's gene tokens are  |
| **Supp A.2** | [`common.py`](../experiments/01_cell_state_geometry/other_representations/common.py) | none directly; provides the phase coordinate recipe described in Supplement A.2 ('Tirosh S and G2M marker z-scores → PCA(2) → φ = atan2') |
| **Supp A.3** | [`extract_scgpt_cls.py`](../experiments/01_cell_state_geometry/other_representations/extract_scgpt_cls.py) | its docstring and route_celltoken/RESULTS.md pitfall 1 document the defect quantified in Supplement A.3 'Input convention' and quoted in §6 'A note on |
| **Supp A.3** | [`g2_scgpt_convention.py`](../experiments/06_gap_tests/g2_scgpt_convention.py) | §6 'A note on extraction' lines 760-765: correcting it 'raises pseudotime decodability by 0.109 and linear fit by 0.210, and lowers the measured curva |
| **Supp B.2** | [`extract_branchpoint.py`](../experiments/01_cell_state_geometry/other_representations/extract_branchpoint.py) | Supplies the UCE column of Supplement B.2 'Developmental pseudotime, 0 of 20' (blood -0.139, lung -0.104, gut -0.374, pancreas -0.056) and the UCE row |
| **Supp B.2** | [`extract_percell.py`](../experiments/01_cell_state_geometry/other_representations/extract_percell.py) | Supplies the UCE cells of Supplement B.2 'Five prior models, 1 of 45' — specifically 'UCE loses all 9 of its cells by the widest margins (ordering 0.7 |
| **Supp B.4** | [`universal_manifold.py`](../experiments/01_cell_state_geometry/other_representations/universal_manifold.py) | no cited number; independently reproduces the direction of Supplement B.4, 'Transfer largely fails for every representation' |
| **Supp B.4** | [`g1_cross_dataset.py`](../experiments/06_gap_tests/g1_cross_dataset.py) | §3.1 (sec:inherit) lines 241-242: 'within-tissue Spearman 0.87 to 0.95, best cross-tissue 0.63, with 2 of the 3 pairs above 0.5 won by raw expression' |
| **Supp B.6** | [`cellcycle_state.py`](../experiments/01_cell_state_geometry/other_representations/cellcycle_state.py) | supports the STATE-SE arm of §3's 'the cell cycle is flat' but is NOT the source of the Supplement B.6 number (STATE-SE L11 linear circ-R2 0.894 on 3, |
| **Supp B.6** | [`cellcycle_uce.py`](../experiments/01_cell_state_geometry/other_representations/cellcycle_uce.py) | not a cited number. Result: curvature -0.127 (control -0.045), curved=False. The paper's own cell-cycle anatomy table (Supplement B.6) is produced by  |
| **Supp C.1** | [`steer_classifier.py`](../experiments/02_vocabulary_facts/chromosome/steer_classifier.py) | Supp C.1 'chromosome is entangled with cell identity' (+0.541 disturbance in the 1B vs +0.208) |
| **Supp C.1** | [`steer_coherence.py`](../experiments/02_vocabulary_facts/chromosome/steer_coherence.py) | Supp C.1 caveat on steer_local |
| **Supp C.1** | [`steer_dosage.py`](../experiments/02_vocabulary_facts/chromosome/steer_dosage.py) | Supp C.1 'the causal response is flat along a chromosome, so fine position is readable but not used' |
| **Supp C.1** | [`steer_local.py`](../experiments/02_vocabulary_facts/chromosome/steer_local.py) | Supp C.1 'the one positive mechanism result, 5-Mb local domains at +0.049 and +0.037, both p <= 0.0001' |
| **Supp C.1** | [`steer_locality.py`](../experiments/02_vocabulary_facts/chromosome/steer_locality.py) | Supp C.1 characteristic scale |
| **Supp C.1** | [`steer_mechanism.py`](../experiments/02_vocabulary_facts/chromosome/steer_mechanism.py) | Supp C.1 'fail against expression enrichment in both (p = 0.92 and 0.115)' |
| **Supp C.1** | [`steer_where.py`](../experiments/02_vocabulary_facts/chromosome/steer_where.py) | Supp C.1 'steering destinations do not replicate across models (3/22 modal agreement)' |
| **Supp C.1** | [`g4_attention_chromosome.py`](../experiments/06_gap_tests/g4_attention_chromosome.py) | §4.1 lines 431-435: 'across 11 layers x 8 heads, same-chromosome gene pairs show no attention enrichment once rank-distance is matched, with 0 of 88 h |
| **Supp F.1** | [`fate_output.py`](../experiments/01_cell_state_geometry/supporting_arms/fate_output.py) | Supplement F.1 aging row — the withdrawn headline. Produces position rho -0.175 (shuffle p 0.774), direction rho +0.416 (p 0.0018), marker baseline +0 |
| **Supp F.1** | [`mds_test.py`](../experiments/01_cell_state_geometry/supporting_arms/mds_test.py) | Supplement F.1: 'Aging/HSC headline ... Withdrawn by its own held-out test: disease donors sit inside the healthy range (p = 0.23) and the winning lin |
| **Supp F.1** | [`preprocess_aging.py`](../experiments/01_cell_state_geometry/supporting_arms/preprocess_aging.py) | none directly — it is the data build behind the Supplement F.1 aging retraction |
| **Supp F.1** | [`arms.py`](../experiments/03_what_shapes_geometry/curvature/arms.py) | The 'irreducible saddle' reading — later falsified. Related to supplement F.1 'The GATA1↔PU.1 saddle is a mutual-repression geometry' retraction famil |
| **Supp F.4** | [`lineage_manifold.py`](../experiments/01_cell_state_geometry/supporting_arms/lineage_manifold.py) | Supplement F.4 'Bilinear/interaction machinery generally' — 'Supervised bilinear CP probes lose to linear (-0.060)' is cp_probe.bilinear_gain = -0.060 |
| **Supp F.4** | [`discriminator.py`](../experiments/03_what_shapes_geometry/curvature/discriminator.py) | R²_SAE = 0.79 for the saddle latent, above the 0.50 novelty threshold ⇒ fully redundant with the linear atlas. Belongs to supplement F.4 'SAE dictiona |
| **Supp F.4** | [`geometry_test.py`](../experiments/03_what_shapes_geometry/curvature/geometry_test.py) | SUPERSEDED. Its positive (bilinear 0.643/0.724 S/G2M R², 0.394 cell-type bal-acc vs linear 0.586/0.660/0.371) appears nowhere in the paper; supplement |
| **route notes** | [`compare_celltoken.py`](../experiments/01_cell_state_geometry/other_representations/compare_celltoken.py) | no cited number. Its finding retracts a cross-model reading of route_uce/RESULTS.md: UCE's cell token is 3rd of 4 on compactness (PR 8.5, d90 38, TwoN |
| **route notes** | [`manifold_celltoken.py`](../experiments/01_cell_state_geometry/other_representations/manifold_celltoken.py) | no cited number, but it is the source of the d90=201 censoring sentinel caveat (only 200 PCs are computed, line 54) that route_celltoken/RESULTS.md co |
| **route notes** | [`causal_ablate.py`](../experiments/02_vocabulary_facts/chromosome/causal_ablate.py) | RESULTS.md §6 negative |
| **route notes** | [`cellcycle_geometry.py`](../experiments/02_vocabulary_facts/chromosome/cellcycle_geometry.py) | RESULTS.md §14 |
| **route notes** | [`cellcycle_steer.py`](../experiments/02_vocabulary_facts/chromosome/cellcycle_steer.py) | RESULTS.md §14 negative |
| **route notes** | [`hox_crossmodel.py`](../experiments/02_vocabulary_facts/chromosome/hox_crossmodel.py) | RESULTS.md §11: Geneformer 0.81, scGPT 0.57; UCE/STATE cannot be asked |
| **route notes** | [`run_probe.py`](../experiments/02_vocabulary_facts/chromosome/run_probe.py) | RESULTS.md §1 (methodology origin of §4.1) |
| **route notes** | [`verify_probe.py`](../experiments/02_vocabulary_facts/chromosome/verify_probe.py) | RESULTS.md §1 |
| **route notes** | [`program_stability.py`](../experiments/03_what_shapes_geometry/curvature/program_stability.py) | Only a coarse program-level read survives; the gene list does not (top-15 overlap 1/15). RESULTS.md §2.3. |
| **route notes** | [`run_substrate.py`](../experiments/03_what_shapes_geometry/curvature/run_substrate.py) | Δ_Q = +0.0250, CI [+0.0221,+0.0282], R²_lin 0.8945 → R²_quad 0.9195 on scGPT/Setty (RESULTS.md §1). Not quoted verbatim in the paper; it is the precur |
| **route notes** | [`spectrum.py`](../experiments/03_what_shapes_geometry/curvature/spectrum.py) | Shows eigen-ordered truncation is not a valid decomposition here (methodological catch, RESULTS.md §2.1). |
| **unmapped** | [`build_ctx_basis.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/build_ctx_basis.py) | Thread A infrastructure |
| **unmapped** | [`c2s_gm_lib.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/c2s_gm_lib.py) | Thread A; supports 4.1 chromosome and 4.4 function/paralogy indirectly, not the four sections inventoried here |
| **unmapped** | [`cc_benchmark_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/cc_benchmark_c2s.py) | 3.1 'the models beat expression in 0 of 36 (C2S)'; supplies the top and bottom rows of Table 1 (expression 0.929, C2S-2B L21 0.875) |
| **unmapped** | [`cc_phase.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/cc_phase.py) | underpins every phase number in 3.1, 3.2, 4.2, 5.1; the orientation fix is what makes the 3.1 cross-cell-line transfer interpretable at all |
| **unmapped** | [`cell_sentences.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/cell_sentences.py) | the top-512-rank encoding that Table 1 in 3.1 is built around |
| **unmapped** | [`continuity_test.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/continuity_test.py) | 3.2 continuity scores 0.17 / 0.10 / 0.00; AND 5.1 the metric warp — knot-gap CV 0.318 (2B) and 0.295 (27B), largest gaps at bins 5 to 8, the G1 to S r |
| **unmapped** | [`ctx_causal_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_causal_c2s.py) | Thread B gate 5, passes |
| **unmapped** | [`ctx_coexpr_null_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_coexpr_null_c2s.py) | Thread B gate 4 FAILS at +2.4 sigma — the divergence from MaxToki's +8 sigma. This is the honest negative worth shipping |
| **unmapped** | [`ctx_extract_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_extract_c2s.py) | Thread B infrastructure |
| **unmapped** | [`ctx_functional_axes_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_functional_axes_c2s.py) | Thread B gate 3, z=+21.3 nuclear vs surface |
| **unmapped** | [`ctx_lib_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_lib_c2s.py) | Thread B |
| **unmapped** | [`ctx_polysemy_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_polysemy_c2s.py) | Thread B gate 1, +0.916 at L09 |
| **unmapped** | [`ctx_position_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ctx_position_c2s.py) | Thread B gate 2, survives at +0.911 |
| **unmapped** | [`encoding_matched_transfer.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/encoding_matched_transfer.py) | 3.1 the model 0.789; and the spurious encoding-only-matched contrast delta +0.0153, CI [+0.0022, +0.0283], which the paper reports as the wrong answer |
| **unmapped** | [`gene_sets.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/gene_sets.py) | Thread A screen; the cellcycle_circle entry is carried as the positive control and fails its margin bootstrap |
| **unmapped** | [`lap_walk.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/lap_walk.py) | 5.4 (sec:operator) 'behaviour follows geometry' — 1.579 behavioural laps for local-tangent against 0.212 for fixed, 12 of 12 cells, p=2.4e-4, isometry |
| **unmapped** | [`manifold_fit.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/manifold_fit.py) | 3.2 the loop lies in a fixed linear 2-plane; supplies the knots whose gap CV becomes the 5.1 result |
| **unmapped** | [`manifold_steer.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/manifold_steer.py) | 5.4 the dose-sweep half of the representation-to-behaviour isometry (0.73 to 0.88 over three runs) |
| **unmapped** | [`manifold_steer_poles.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/manifold_steer_poles.py) | the differentiation/erythroid arm; supporting, not a numbered main-text claim |
| **unmapped** | [`orthogonal_props.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/orthogonal_props.py) | 3.2 'Angle is phase; radius is cycling strength' — cycling strength R2 0.414 radial against 0.040 angular; phase programmes load ~15x more on the angl |
| **unmapped** | [`plateau_test.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/plateau_test.py) | 3.2 'no plateau, and a plateau statistic that weakens with depth (0.179 to 0.115 to 0.056 across layers 9, 15 and 21)' |
| **unmapped** | [`ring_analysis.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/ring_analysis.py) | 3.2 'It is a filled disk rather than a ring' — planted ring 0.15/0.000, planted disk 0.35/0.126, C2S-2B 0.38/0.108, C2S-27B 0.38/0.110 |
| **unmapped** | [`run_genome_wide.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/run_genome_wide.py) | the C2S counterpoint to 4.1 — best ctx layer 0.105 but ESM2 sequence decodes it better at 0.190, so not model-specific in a model with no gene table |
| **unmapped** | [`run_geometric.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/run_geometric.py) | Thread A negative |
| **unmapped** | [`run_steering.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/run_steering.py) | Thread A gate 3 |
| **unmapped** | [`steer_c2s.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/steer_c2s.py) | the intervention machinery behind the 4.3 lineage-switch steering and the Thread A gate-3 runs |
| **unmapped** | [`subset_ts.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/subset_ts.py) | panel construction |
| **unmapped** | [`subset_ts_celltype.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/subset_ts_celltype.py) | panel construction |
| **unmapped** | [`surface_forms.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/surface_forms.py) | the whole of 4.2 — CCNB1 0.989 (p=0.0002, 40.7 deg separation), Ccnb1 0.782, ccnb1 0.767, CQNB1 0.685, NBCC1 0.511 (p=0.43); Figure 4 |
| **unmapped** | [`synth_sweep.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/synth_sweep.py) | 114.3 deg +/- 6.9 against shuffled 13.2 and random 3.5. Not in the main-text sections listed; likely supplement. Its insert_at helper is load-bearing  |
| **unmapped** | [`transfer_test.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/transfer_test.py) | superseded by encoding_matched_transfer.py + selection_matched_arm.py; its results/transfer_test.json and transfer_rpe1.json are marked _RETRACTED in  |
| **unmapped** | [`validate_local.py`](../experiments/01_cell_state_geometry/c2s_cell_cycle/validate_local.py) | none — a smoke test, and a good one to keep as the repo's CI check |
| **unmapped** | [`lymphoid_ablation.py`](../experiments/01_cell_state_geometry/developmental/lymphoid_ablation.py) | Honest negative that the paper's caveat discipline rests on: D moves by +0.003, the caveat stands. |
| **unmapped** | [`maxtoki_native.py`](../experiments/01_cell_state_geometry/developmental/maxtoki_native.py) | Retires the 'scGPT-only readout' caveat: MaxToki passes the gate better than scGPT (+0.302 vs +0.260) and gives 5/5, D = +0.821, p = 0.0083. |
| **unmapped** | [`adapt_cached.py`](../experiments/01_cell_state_geometry/five_models/adapt_cached.py) | Sec 5.3: supplies the Geneformer / MaxToki / STATE-SE / UCE rows of the stall table; Sec 3.1: supplies the five model panels of the 1-of-45 benchmark |
| **unmapped** | [`cc_benchmark.py`](../experiments/01_cell_state_geometry/five_models/cc_benchmark.py) | Sec 3.1 exactly: 'On matched-dimensionality benchmarks the models beat expression in 1 of 45 cells'. The stored value is literally "model_wins_over_ex |
| **unmapped** | [`cc_common.py`](../experiments/01_cell_state_geometry/five_models/cc_common.py) | Sec 3.1 / Sec 3.2 substrate: the 3,000 K562 cells and the phase angle every circ-R^2 number is scored against |
| **unmapped** | [`cc_decode.py`](../experiments/01_cell_state_geometry/five_models/cc_decode.py) | Supports Sec 5.4 'Behaviour follows geometry' in spirit but is a negative on this substrate; the number the paper uses for that claim (1.579 vs 0.212  |
| **unmapped** | [`cc_geometry.py`](../experiments/01_cell_state_geometry/five_models/cc_geometry.py) | Sec 3.2 'It is flat': out-of-plane rotation 22.7 deg vs planar null 25.6 +/- 2.3, p = 0.905; and 'Persistent homology inside the fitted phase plane ag |
| **unmapped** | [`cc_mechanism.py`](../experiments/01_cell_state_geometry/five_models/cc_mechanism.py) | Sec 5.3: 'Measured cell by cell, the phase rate peaks exactly at the perpendicular crossing (90.7 deg), banking 55.3% of total advance before it'. Par |
| **unmapped** | [`cc_steering.py`](../experiments/01_cell_state_geometry/five_models/cc_steering.py) | Sec 5.3: 'in seven representations including raw expression with no model at all, fixed direction 0.01 to 0.36 laps against local-tangent 4.5 to 6.0'. |
| **unmapped** | [`cc_summary.py`](../experiments/01_cell_state_geometry/five_models/cc_summary.py) | Sec 5.3 Fig 6b: the bar values in the paper's figure script (scGPT 0.34/4.53, Geneformer 0.14/5.57, MaxToki 0.01/6.03, STATE-SE 0.31/4.70, raw expr 0. |
| **unmapped** | [`cc_synthetic_check.py`](../experiments/01_cell_state_geometry/five_models/cc_synthetic_check.py) | Sec 5.3: 'On a synthetic zero-curvature circle, stall at 1.49 rad against the predicted pi/2 = 1.571, with the pre-normalisation step norm collapsing  |
| **unmapped** | [`extract_scgpt_cc.py`](../experiments/01_cell_state_geometry/five_models/extract_scgpt_cc.py) | Sec 3.2 and Sec 5.3: produces the scGPT row (H_flat p=0.905, out-of-plane 22.7 deg; fixed_proj 0.34 laps vs local+retract 4.53 laps) |
| **unmapped** | [`make_expr.py`](../experiments/01_cell_state_geometry/five_models/make_expr.py) | Sec 3.2 and Sec 5.3: the 'raw expression, no model at all' row (circ-R^2 0.929, fixed 0.36 laps vs local 5.08 laps) — the claim that the loop belongs  |
| **unmapped** | [`celltype_uce.py`](../experiments/01_cell_state_geometry/other_representations/celltype_uce.py) | not a cited number. Result: bal-acc 0.433 vs null 0.031, curvature +0.011 against a synthetic control at -0.310, curved=False |
| **unmapped** | [`extract_geneformer_cls.py`](../experiments/01_cell_state_geometry/other_representations/extract_geneformer_cls.py) | none cited |
| **unmapped** | [`extract_maxtoki_cls.py`](../experiments/01_cell_state_geometry/other_representations/extract_maxtoki_cls.py) | none — this arm produced no data |
| **unmapped** | [`extract_regulatory.py`](../experiments/01_cell_state_geometry/other_representations/extract_regulatory.py) | none in the current paper — this is the Route-B regulatory-logic substrate, background rather than a cited number |
| **unmapped** | [`extract_state_cls.py`](../experiments/01_cell_state_geometry/other_representations/extract_state_cls.py) | none — this arm produced no data |
| **unmapped** | [`train_compare.py`](../experiments/01_cell_state_geometry/other_representations/train_compare.py) | none in the current paper. Result: fidelity gate PASS (0.826 / 0.864), redundancy 81.1%, genuinely nonlinear 24.0%, novel-and-nonlinear 11.3% (170 lat |
| **unmapped** | [`common.py`](../experiments/01_cell_state_geometry/supporting_arms/common.py) | none directly |
| **unmapped** | [`extract_external.py`](../experiments/01_cell_state_geometry/supporting_arms/extract_external.py) | none directly |
| **unmapped** | [`gates.py`](../experiments/01_cell_state_geometry/supporting_arms/gates.py) | none directly |
| **unmapped** | [`let.py`](../experiments/01_cell_state_geometry/supporting_arms/let.py) | none directly |
| **unmapped** | [`lineage_graph.py`](../experiments/01_cell_state_geometry/supporting_arms/lineage_graph.py) | none directly |
| **unmapped** | [`operators.py`](../experiments/01_cell_state_geometry/supporting_arms/operators.py) | none directly |
| **unmapped** | [`run_tier_b.py`](../experiments/01_cell_state_geometry/supporting_arms/run_tier_b.py) | none — produces geodesic-ruler rho 0.689 (linear_sae) vs 0.528 (bilinear) on scGPT and 0.895 (linear_drift) vs 0.796 (bilinear) vs 0.886/matched-null  |
| **unmapped** | [`score_external.py`](../experiments/01_cell_state_geometry/supporting_arms/score_external.py) | none — produces external geodesic-ruler rho 0.632 linear_sae vs 0.547 bilinear, and the Krasnow control collapse to 0.109-0.321; not cited in the pape |
| **unmapped** | [`chrom_cnv_origin.py`](../experiments/02_vocabulary_facts/chromosome/chrom_cnv_origin.py) | Supp S5 DNA-dosage alternative |
| **unmapped** | [`cnv_gene_level_control.py`](../experiments/02_vocabulary_facts/chromosome/cnv_gene_level_control.py) | Supp S5 control |
| **unmapped** | [`cnv_gene_level_test.py`](../experiments/02_vocabulary_facts/chromosome/cnv_gene_level_test.py) | Supp S5 'its per-gene success pattern tracks the normal panel over the aneuploid one' |
| **unmapped** | [`coocc_diagnose.py`](../experiments/02_vocabulary_facts/chromosome/coocc_diagnose.py) | Supp S4 panel-dependence of the co-occurrence baseline |
| **unmapped** | [`coocc_fair.py`](../experiments/02_vocabulary_facts/chromosome/coocc_fair.py) | supporting; its width-matched block is explicitly retracted inside coocc_final.py |
| **unmapped** | [`corpus_s1_audit.py`](../experiments/02_vocabulary_facts/chromosome/corpus_s1_audit.py) | Supp S5 'only ~5-6% of its cells come from studies that sampled cancer' |
| **unmapped** | [`genome_position2.py`](../experiments/02_vocabulary_facts/chromosome/genome_position2.py) | Supp S2: MaxToki excess +0.396 (22/22 chromosomes) vs ESM-2 +0.253 vs co-expression +0.063 |
| **unmapped** | [`genome_position_direction.py`](../experiments/02_vocabulary_facts/chromosome/genome_position_direction.py) | Supp S2 geometry |
| **unmapped** | [`genome_position_geometry.py`](../experiments/02_vocabulary_facts/chromosome/genome_position_geometry.py) | Supp S2 'a chromosome-specific linear gradient' |
| **unmapped** | [`h_tokenizer_rank.py`](../experiments/02_vocabulary_facts/chromosome/h_tokenizer_rank.py) | none (screen loader) |
| **unmapped** | [`hox_axes.py`](../experiments/02_vocabulary_facts/chromosome/hox_axes.py) | route paper Supp S7 'approximately a bilinear grid, not a rigid one' (R^2 = 0.53) |
| **unmapped** | [`hypotheses.py`](../experiments/02_vocabulary_facts/chromosome/hypotheses.py) | none (screening program) |
| **unmapped** | [`make_fig5.py`](../experiments/02_vocabulary_facts/chromosome/make_fig5.py) | route paper Supp S2 figure |
| **unmapped** | [`make_fig6.py`](../experiments/02_vocabulary_facts/chromosome/make_fig6.py) | route paper Supp S3 figure; main-paper Fig 3b data |
| **unmapped** | [`make_figures.py`](../experiments/02_vocabulary_facts/chromosome/make_figures.py) | produces the route paper's Figures 1-4 (Fig 1c is the 0.880/0.720/0.506 panel) |
| **unmapped** | [`make_graphical_abstract.py`](../experiments/02_vocabulary_facts/chromosome/make_graphical_abstract.py) | route paper front matter |
| **unmapped** | [`make_pdf.sh`](../experiments/02_vocabulary_facts/chromosome/make_pdf.sh) | build script |
| **unmapped** | [`panel_sweep.py`](../experiments/02_vocabulary_facts/chromosome/panel_sweep.py) | Supp S4 panel dependence |
| **unmapped** | [`position_improve.py`](../experiments/02_vocabulary_facts/chromosome/position_improve.py) | Supp S2 |
| **unmapped** | [`purity_decomposition.py`](../experiments/02_vocabulary_facts/chromosome/purity_decomposition.py) | Supp S4 / mechanism discussion |
| **unmapped** | [`specificity_matched.py`](../experiments/02_vocabulary_facts/chromosome/specificity_matched.py) | Supp S4; supersedes specificity_mechanism.py |
| **unmapped** | [`specificity_mechanism.py`](../experiments/02_vocabulary_facts/chromosome/specificity_mechanism.py) | Supp S4 |
| **unmapped** | [`steer_algebra.py`](../experiments/02_vocabulary_facts/chromosome/steer_algebra.py) | Supp S6 'algebra': 22 independent, signed, saturating lookups, not one continuous manifold |
| **unmapped** | [`steer_relative.py`](../experiments/02_vocabulary_facts/chromosome/steer_relative.py) | Supp S6 operator check (did not beat global) |
| **unmapped** | [`steer_signed.py`](../experiments/02_vocabulary_facts/chromosome/steer_signed.py) | Supp S6 'bidirectional but asymmetric' |
| **unmapped** | [`steer_swap.py`](../experiments/02_vocabulary_facts/chromosome/steer_swap.py) | Supp S6 substitution ceiling |
| **unmapped** | [`steer_units.py`](../experiments/02_vocabulary_facts/chromosome/steer_units.py) | Supp S6 magnitude: substitution ceiling +0.0082 (+0.37x baseline) vs the vector's +0.055 |
| **unmapped** | [`abundance_control.py`](../experiments/03_what_shapes_geometry/curvature/abundance_control.py) | SUPERSEDED. Reports the margin tripling (S: +0.058 → +0.186; G2M: +0.067 → +0.139). Caveat recorded in GEOMETRY_SYNTHESIS.md #15: the covariate is LIN |
| **unmapped** | [`arms_identifiability.py`](../experiments/03_what_shapes_geometry/curvature/arms_identifiability.py) | At (r=8, λ=1e-3) both arms are harmful (−0.0568/−0.0681); at (r=32, λ=1e-2) both are useful (+0.0162/+0.0123) with Δ_Q +0.0274 vs +0.0238. Instance of |
| **unmapped** | [`build_rulers.py`](../experiments/03_what_shapes_geometry/curvature/build_rulers.py) | none. CONTAINS A KNOWN BUG at lines 91-92: replogle_concat.h5ad X is already log1p(CP10k), and this applies log1p(X/rowsum·1e4) a second time. This co |
| **unmapped** | [`extract_atlas_feats.py`](../experiments/03_what_shapes_geometry/curvature/extract_atlas_feats.py) | The domain-shift guard that stops a crippled atlas from inflating the novelty claim. Method note only. |
| **unmapped** | [`feat_identifiability.py`](../experiments/03_what_shapes_geometry/curvature/feat_identifiability.py) | Pair Jaccard 0.43–0.50 vs the reference config; eigen-feature Jaccard 0.94–1.00. Named products are only partly identified. |
| **unmapped** | [`make_figure.py`](../experiments/03_what_shapes_geometry/curvature/make_figure.py) | figures/route_q_summary.png — an internal figure, NOT paper Figure 7. Paper Figure 7 is produced by manifolds/paper/make_figures.py fig7() with the an |
| **unmapped** | [`run_features.py`](../experiments/03_what_shapes_geometry/curvature/run_features.py) | Feature-space Δ_Q = +0.0293, CI [+0.0262,+0.0327], R²_lin(atlas) 0.8731 (feat_q.log). Undocumented in RESULTS.md. |
| **unmapped** | [`run_generality.sh`](../experiments/03_what_shapes_geometry/curvature/run_generality.sh) | The 16-cell generality table in GENERALITY.md; the pre-registered mouse-pancreas negative control FAILS (4/4 positive). |
| **unmapped** | [`structure_geometry.py`](../experiments/03_what_shapes_geometry/curvature/structure_geometry.py) | SUPERSEDED, same as geometry_test.py. Produces margins +0.058±0.002 (S), +0.067±0.004 (G2M), +0.024±0.006 (cell-type) and the sweep ranking atomic 0.3 |
| **unmapped** | [`summarize.py`](../experiments/03_what_shapes_geometry/curvature/summarize.py) | Produces the Δ_kernel column — the ONLY place either assigned path touches the deprecated kNN-minus-linear family, and only by reading route_branchpoi |
| **unmapped** | [`_data.py`](../experiments/03_what_shapes_geometry/steering_operator/_data.py) | none (data plumbing) |
| **unmapped** | [`ha2_onmanifold_perturb.py`](../experiments/03_what_shapes_geometry/steering_operator/ha2_onmanifold_perturb.py) | supplement §F.4 'Perturbation inversion' — this hardens that negative. Produces maxtoki raw 0.475 / tangent 0.474 / DE 0.931; state raw 0.553 / tangen |
| **unmapped** | [`ha3_heldout_population.py`](../experiments/03_what_shapes_geometry/steering_operator/ha3_heldout_population.py) | supplement §F.4 'Reconstructing a deleted intermediate. Held-out peak r = +0.545, rank 5/10, path running through the wrong lineage.' Exact match (0.5 |
| **unmapped** | [`hb1_larry_curvature.py`](../experiments/03_what_shapes_geometry/steering_operator/hb1_larry_curvature.py) | supplement §F.4 'Hidden clonal fate (LARRY)': expression position AUROC 0.767, geometry 0.540 vs clone-permutation null 0.383 +/- 0.061, combined gain |
| **unmapped** | [`hb3_tangent_dimension.py`](../experiments/03_what_shapes_geometry/steering_operator/hb3_tangent_dimension.py) | supplement §F.4 'Commitment as a discrete geometric step. No cliff (sigmoid transition width 0.27-0.83 of the pseudotime range), and a counts baseline |

221 entries, of which 109 map to a numbered section or supplement.

Two groups are absent from this table because they need no external assets and are documented in place: [`experiments/05_synthetic`](../experiments/05_synthetic/README.md), which reproduces Sections 5.1 and 5.2 end to end on a laptop, and the figure scripts in [`paper/`](../paper), which rebuild every figure from committed JSON.
