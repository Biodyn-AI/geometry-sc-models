# What you need to download

Everything outside [`experiments/05_synthetic`](../experiments/05_synthetic) reads model weights,
public datasets, or activations cached from them. Together these come to roughly **24 GB of working
assets**, drawn from about **133 GB of source atlases**. None of it is in this repository: most is
third-party material we cannot redistribute, and the rest is derived caches that are cheaper to
regenerate than to host.

Scripts resolve assets through two environment variables:

```bash
export GEOMSC_DATA=/somewhere/with/space      # datasets and cached activations
export GEOMSC_MODELS=/somewhere/with/space    # model checkpoints
```

Both default to `./data` and `./models` inside the repository, which are gitignored. A script that
cannot find an asset raises immediately with the path it wanted, rather than failing part way
through a long run.

---

## Models

| model | where | size | used by |
|---|---|---|---|
| **scGPT** whole-human | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) checkpoint release | 205 MB | §3, §5.4, §5.5 |
| **Geneformer** V2-104M and V1-10M | [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) on Hugging Face, with its token dictionaries and `gene_name_id_dict` | ~1 GB | §3, §4.1 |
| **MaxToki-217M** and **MaxToki-1B** | the authors' Hugging Face release (`LlamaForCausalLM` safetensors). Use the **HF safetensors** variants, not the BioNeMo distcp ones: the analyses need `output_hidden_states` and `output_attentions` through standard `transformers` hooks | 0.9 GB and 3.9 GB | §4.1, §5.5 |
| **STATE-SE (600M)** and **STATE-ST** | [Arc Institute](https://arcinstitute.org) release | 3.0 GB and 542 MB | §3, §6 |
| **UCE-100M** | [minwoosun/uce-100m](https://huggingface.co/minwoosun/uce-100m) | 3.4 GB | §3, §6 |
| **C2S-Scale-Gemma-2 2B / 27B** | [vandijklab](https://huggingface.co/vandijklab) on Hugging Face, **gated**, request access first. The 27B needs an 80 GB H100. Load with **eager attention**; the default kernel changes the activations | 5 GB and 55 GB | §3.1, §3.2, §4.2, §5.1 |
| **ESM-2** gene-symbol embeddings | `minwoosun/uce-misc` on Hugging Face (`.pt`). This is the **sequence control**, not a model arm | ~1 GB | §4.1, §4.3 |

## Datasets

| dataset | where | size | used by |
|---|---|---|---|
| **Replogle K562** and **RPE1** non-targeting controls | [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013), GEO/figshare release. 3,000 cells each are used | 30 GB source | §3.1, §4.2, §5.1 |
| **Setty CD34⁺ bone marrow** | [Setty et al. 2019](https://doi.org/10.1038/s41587-019-0068-4) | 162 MB | §3.3 |
| **Fetal gut atlas** (62,849 cells) | CZ CELLxGENE. The panel the strongest co-occurrence factorisation is built on | 898 MB | §4.1 |
| **Lung airway**, **mouse pancreas** | Tabula Sapiens; [Bastidas-Ponce et al. 2019](https://doi.org/10.1242/dev.173849) | ~250 MB | §3.3 |
| **Tabula Sapiens** kidney, lung, immune | [Tabula Sapiens Consortium 2022](https://doi.org/10.1126/science.abl4896) | ~600 MB | §4.3, §6 |
| **LARRY** clonal barcoding | [Weinreb et al. 2020](https://doi.org/10.1126/science.aaw3381), cospar-preprocessed subsample | 84 MB | Supp F.4 |
| **`species_chrom.csv`** | the `minwoosun/uce-misc` snapshot. Chromosome and genomic start for 19,844 human genes; this is the label table for every §4.1 result | 3.9 MB | §4.1 |
| **GO / gene2go** annotations | NCBI | small | §4.3 |
| Cell-cycle marker sets | **no download**: the Tirosh S and G2M lists are in the code | — | §3.1, §3.2 |

## Cached activations

Most scripts read a cache rather than running a model. Regenerate them with the `extract_*.py`
scripts in each module, or expect the first run to be slow. The largest caches, for scale:

- per-gene contextual bases, MaxToki 12-layer scan: 2.4 GB
- STATE activations tree: 9.4 GB
- fitted curvature matrices `q_*.npz`: 1.6 GB, dominated by STATE at d = 2048
- UCE layer-2 activations: 2.4 GB
- gene-by-gene co-expression over Tabula Sapiens: 885 MB

## One thing that cannot be regenerated

The **C2S-27B activations no longer exist**. Every `*_27b` number in the paper was produced on a
rented H100 that was terminated with no network volume attached. The committed result JSONs are the
only surviving record. Re-deriving them means renting an 80 GB GPU and re-extracting. The 2B results
are fully reproducible.

## Two defects to know about before you start

Both are documented in Supplement A.3, and both are in the *upstream assets*, not in this code.

1. **Input convention.** One cached extraction fed raw counts to a checkpoint configured with
   `input_style: "binned"`, `n_bins: 51`. Correcting it raises pseudotime decodability by 0.109 and
   linear fit by 0.210, and *lowers* measured curvature by 0.075. If you re-extract, you will get
   the corrected numbers, which differ from nine routes' cached values in a known direction.
2. **Broken pseudotime in one cached gut arm.** Differentiated cells sit *below* stem cells. The gut
   arm is excluded from the transfer analysis for this reason. The other three tissues are correctly
   rooted.

## Licences

Each dataset and checkpoint carries its own licence and terms; several of the Hugging Face models
are gated and require you to accept terms before download. The MIT licence in this repository covers
our code only.
