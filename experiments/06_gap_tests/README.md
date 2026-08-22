# Gap tests run for this paper

Four checks that closed open questions: cross-dataset replication, the scGPT input-convention defect, the 1B layer sweep, and chromosome attention.

3 scripts, 4 committed result files.

Every script here needs assets this repository does not ship. See [`docs/DATA.md`](../../../docs/DATA.md) for what to download and where to point `GEOMSC_DATA` and `GEOMSC_MODELS`.

| script | what it computes | supports | needs |
|---|---|---|---|
| `g1_cross_dataset.py` | Cross-tissue pseudotime transfer: fit a PCA+ridge probe on one of four tissues (blood/Setty, fetal gut, lung airway, mouse pancreas; 10,529 shared genes, 50 components) and apply it unchanged to anoth | §3.1 (sec:inherit) lines 241-242: 'within-tissue Spearman 0.87 to 0.95, best cross-tissue 0.63, with 2 of the  | Needs data/branchpoint/{scgptbin,geneformer,state,uce}_{setty,gut,lung,pancreas} |
| `g2_scgpt_convention.py` | Measures how much the scGPT input-convention bug (raw counts fed to a binned checkpoint) changes the answers, by comparing the already-cached scgpt_* vs scgptbin_* embeddings for gut/lung/pancreas on  | §6 'A note on extraction' lines 760-765: correcting it 'raises pseudotime decodability by 0.109 and linear fit | Needs data/branchpoint/{scgpt,scgptbin}_{gut,lung,pancreas}.npz only (~48 MB). N |
| `g4_attention_chromosome.py` | Direct test of whether attention is sorted by chromosome: on real Setty cells through MaxToki-217M with eager attention (11 layers x 8 heads), compares same-chromosome vs different-chromosome gene-pai | §4.1 lines 431-435: 'across 11 layers x 8 heads, same-chromosome gene pairs show no attention enrichment once  | Needs the MaxToki-217M-HF weights (828 MB), torch + transformers, and four modul |

## Provenance

These files were collected from the working tree in which the study was run. Paths to large assets have been parameterised; nothing else about the analysis logic was changed.
