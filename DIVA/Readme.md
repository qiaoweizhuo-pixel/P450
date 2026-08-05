# DIVA v3.0: Differential Vector Attribution

**DIVA (Differential Vector Attribution)** is an unsupervised, alignment-free per-residue importance metric that identifies which residues in a protein sequence carry features aligned with the protein’s own functional specificity. Unlike supervised methods, DIVA requires **no functional labels, no predefined clusters, no classifier training, and no phylogenetic tree**. It operates entirely on ESM-2 embeddings by measuring how strongly each residue’s local embedding aligns with the protein’s intrinsic specificity vector.

DIVA was developed as part of a three-pronged analytical framework alongside **PEACE** (constraint estimation) and **PIVOT** (supervised ablation), and has been validated to capture a biologically meaningful **membrane-topological template** in plant cytochrome P450 N-termini.

## How DIVA Works

For each protein sequence independently:

1. **Extract embeddings** from the final hidden layer of ESM-2 (`esm2_t33_650M_UR50D`)
2. **Compute the specificity vector**: Δv = CLS token embedding − mean residue embedding
3. **Unit-normalize** the specificity vector
4. **Score each residue**: absolute cosine similarity between its embedding and the normalized specificity direction
5. **Normalize** scores to [0, 1] per protein (divide by within-protein maximum)

A high DIVA score indicates that a residue’s embedding points in the same high-dimensional direction as the protein’s functional signature. A low score indicates orthogonality to that direction.

> ⚠️ **Key Distinction:** DIVA does **not** measure conservation, hydrophobicity, or catalytic importance. It measures whether a residue’s embedding (including its sequence context) aligns with what makes *this particular protein* functionally distinct in ESM-2 space.

## Installation

```bash
pip install torch transformers numpy pandas scikit-learn tqdm matplotlib
```

> **Note:** DIVA does **not** require `tslearn`, `statsmodels`, `umap-learn`, or `biopython`. Those are dependencies of PEACE or PIVOT, not DIVA.

## Usage

### Basic Syntax

```bash
python DIVA_V3.0.py --fasta <input.fasta> [options]
```

### Arguments

| Argument           | Required | Default                        | Type  | Description                                                  |
| ------------------ | -------- | ------------------------------ | ----- | ------------------------------------------------------------ |
| `--fasta`          | ✅        | —                              | str   | Input FASTA file                                             |
| `--output-dir`     | ❌        | `residue_importance`           | str   | Output directory                                             |
| `--model-name`     | ❌        | `facebook/esm2_t33_650M_UR50D` | str   | HuggingFace ESM-2 checkpoint                                 |
| `--device`         | ❌        | auto (`cuda` if available)     | str   | `cuda` or `cpu`                                              |
| `--batch-size`     | ❌        | 1                              | int   | Batch size for embedding extraction (recommended: 1)         |
| `--dpi`            | ❌        | 300                            | int   | Figure DPI for output plots                                  |
| `--top-n`          | ❌        | 20                             | int   | Number of top important residues to highlight per sequence   |
| `--max-length`     | ❌        | 1022                           | int   | Max token length for ESM-2                                   |
| `--n-bins`         | ❌        | 200                            | int   | Number of bins for population-level importance profile       |
| `--top-percentage` | ❌        | 0.2                            | float | Top fraction threshold for hotspot detection (e.g., 0.2 = top 20%) |
| `--cv-threshold`   | ❌        | 0.5                            | float | CV threshold for hotspot stability (kept for compatibility; currently unused) |

### Examples

#### Basic run

```bash
python DIVA_V3.0.py --fasta p450_family.fasta
# Outputs to ./residue_importance/ with default parameters
```

#### Custom hotspot threshold and highlighting

```bash
python DIVA_V3.0.py --fasta kinase_family.fasta \
    --output-dir kinase_diva \
    --top-percentage 0.15 \
    --top-n 30 \
    --n-bins 300 \
    --dpi 600
```

#### GPU with larger batch (use cautiously)

```bash
python DIVA_V3.0.py --fasta large_family.fasta \
    --device cuda \
    --batch-size 4 \
    --output-dir large_diva
```

#### Alternative model with shorter max length

```bash
python DIVA_V3.0.py --fasta membrane_proteins.fasta \
    --model-name facebook/esm2_t36_3B_UR50D \
    --max-length 512 \
    --output-dir mem_diva_3b
```

## Output Files

| File                           | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `per_residue_scores.csv`       | Per-protein, per-position normalized DIVA scores             |
| `population_profile.csv`       | Binned population-level importance density (mean ± SD per bin) |
| `hotspots.csv`                 | Identified hotspot regions (contiguous bins above percentile threshold) |
| `population_profile.png/pdf`   | Population importance density plot with hotspot shading      |
| `per_protein_profiles.png/pdf` | Heatmap of per-protein normalized importance across positions |

## Interpretation Guide

| Concept                | Meaning                                                      |
| ---------------------- | ------------------------------------------------------------ |
| **High DIVA score**    | Residue embedding aligns with protein’s functional specificity direction |
| **Low DIVA score**     | Residue embedding is orthogonal to specificity direction     |
| **Population hotspot** | Region where high DIVA scores are consistent across the protein family |
| **N-terminal hotspot** | In membrane proteins, often corresponds to topological template (validated) |

## Citation

If you use DIVA in your research, please cite:

> Qiao Z, Wang J, Qin B, Wei F, Liang Y. The N-Terminus of *Sophora tonkinensis* Cytochrome P450s Evolves Neutrally yet Encodes Rich Functional Information: A Protein Language Model Analysis. *(Manuscript under review, 2026)*