Here is a comprehensive README for the `PEACE.py` script:

# PEACE: Per-position Embedding Alignment-free Constraint Estimation

**PEACE** is a dual-mode computational tool for analyzing evolutionary constraints and directional divergence in protein families directly from sequence embeddings. Unlike traditional alignment-based methods, PEACE operates in an alignment-free manner by leveraging per-position embedding templates and advanced correspondence matching.

It supports two operational modes:

- **DTW Consensus Mode:** Uses Dynamic Time Warping (DTW) barycenter averaging to build a consensus trajectory without requiring a reference sequence.
- **Median-Reference Mode:** Uses a median-length reference sequence with soft cosine similarity correspondence for faster analysis.

## Key Features

- **Alignment-Free Analysis:** Estimates per-position dispersion and Shannon entropy using ESM-2 protein language model embeddings.
- **Directionality Detection:** Identifies structured divergence vs. isotropic relaxation using effective rank, GMM clustering, and silhouette scores.
- **Phylogenetic Structure Testing:** Optional integration with Newick trees to test if embedding clusters map to specific clades (candidate episodic positive selection).
- **FDR Correction:** Built-in Benjamini-Hochberg correction for phylogenetic permutation tests (requires `statsmodels`).
- **Flexible Visualization:** Supports PCA, t-SNE, and UMAP projections for directionality cluster visualization.
- **Comprehensive Reporting:** Auto-generates publication-ready figures and detailed text reports with top constrained/variable/divergent positions.

## Installation

### Dependencies

```bash
pip install torch transformers biopython scikit-learn tslearn matplotlib pandas numpy tqdm statsmodels umap-learn
```

> **Note:** `tslearn` is required only for DTW mode. `umap-learn` is optional and only needed if using `--dim-reduction umap`. `statsmodels` is recommended for FDR-corrected phylogenetic p-values.

## Usage

### Basic Syntax

```bash
python PEACE.py --mode <dtw|reference> --fasta <input.fasta> [options]
```

### Arguments

| Argument               | Required | Default                        | Description                              |
| ---------------------- | -------- | ------------------------------ | ---------------------------------------- |
| `--mode`               | ✅        | -                              | Analysis mode: `dtw` or `reference`      |
| `--fasta`              | ✅        | -                              | Input FASTA file path                    |
| `-o, --output-dir`     | ❌        | `peace_results`                | Output directory                         |
| `--model-name`         | ❌        | `facebook/esm2_t33_650M_UR50D` | HuggingFace PLM checkpoint               |
| `--device`             | ❌        | auto                           | `cuda` or `cpu`                          |
| `--max-length`         | ❌        | 1022                           | Max token length for PLM                 |
| `--min-effective-seqs` | ❌        | 30                             | Min matched seqs per position            |
| `--dim-reduction`      | ❌        | `pca`                          | Projection method: `pca`, `tsne`, `umap` |
| `--tree`               | ❌        | None                           | Newick tree for phylo-structure test     |
| `--seed`               | ❌        | 42                             | Random seed                              |

### DTW-Specific Options

| Argument             | Default | Description                              |
| -------------------- | ------- | ---------------------------------------- |
| `--window-size`      | 11      | Sliding window size for trajectories     |
| `--pca-components`   | 50      | PCA components before DBA                |
| `--dtw-max-iter`     | 10      | Max iterations for DBA convergence       |
| `--dtw-subset`       | 0       | Subset size for DBA (0 = use all)        |
| `--consensus-length` | 0       | Force consensus length (0 = auto median) |
| `--skip-dba`         | False   | Skip DBA computation                     |

### Reference-Specific Options

| Argument                     | Default | Description                            |
| ---------------------------- | ------- | -------------------------------------- |
| `--sim-threshold-percentile` | 5.0     | Percentile threshold for soft matching |

### Phylogeny Options

| Argument               | Default | Description                               |
| ---------------------- | ------- | ----------------------------------------- |
| `--phylo-permutations` | 500     | Number of permutations for structure test |

## Examples

### DTW Mode (Recommended for diverse families)

```bash
python PEACE.py --mode dtw --fasta kinase_family.fasta \
    --tree kinase_tree.nwk \
    --dim-reduction tsne \
    -o kinase_peace_dtw
```

### Reference Mode (Faster, good for closely related sequences)

```bash
python PEACE.py --mode reference --fasta gpcr_subfamily.fasta \
    --sim-threshold-percentile 3.0 \
    --min-effective-seqs 20 \
    -o gpcr_peace_ref
```

### With UMAP and FDR-Corrected Phylogeny

```bash
python PEACE.py --mode dtw --fasta enzyme_family.fasta \
    --tree enzyme_tree.nwk \
    --dim-reduction umap \
    --phylo-permutations 1000 \
    -o enzyme_peace_umap
```

## Output Files

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `dispersion_data.csv`        | Per-position metrics: dispersion, Shannon entropy, effective rank, directional score, phylo z/p/q values |
| `dispersion_profile.png/pdf` | Dispersion profile with N-term/core coloring and dominant AA track |
| `diagnostics.png/pdf`        | Validation plots: match sim vs dispersion, position offset std, Shannon correlation |
| `directionality.png/pdf`     | Effective rank vs dispersion scatter, top directional position cluster projections |
| `phylo_structure.png/pdf`    | Phylo-directional score profile and z-score scatter (only if `--tree` provided) |
| `report.txt`                 | Full text report with top constrained, variable, directional, and phylo-structure positions |
| `embedding_cache/`           | Cached `.npy` embeddings for re-use across runs              |
| `dba_consensus.pkl`          | Cached DTW consensus (DTW mode only)                         |

## Interpretation Guide

| Metric                | Low Value                                   | High Value                                       |
| --------------------- | ------------------------------------------- | ------------------------------------------------ |
| **Dispersion**        | Purifying selection / structural constraint | Relaxed constraint / diversified                 |
| **Effective Rank**    | Directional divergence (structured)         | Isotropic relaxation                             |
| **Directional Score** | Uniform conservation                        | Family-wide structured divergence                |
| **Phylo Z-score**     | Pervasive family-wide signal                | Clade-specific episodic selection                |
| **Phylo Q-value**     | Not significant after FDR                   | Significant branch-specific candidate (Q < 0.05) |

## Recent Updates (2026-07-20)

- Added **FDR correction** (Benjamini-Hochberg) for phylogenetic permutation p-values
- Added **t-SNE and UMAP** projection support via `--dim-reduction` flag
- Improved fallback handling when optional dependencies are missing

## Citation

If you use PEACE in your research, please cite the associated publication *()*.