# PIVOT:  Perturbation-based Importance Vector Orientation via Transformers

**PIVOT** (internally `AA_Position_Importance_V3.py`) is a supervised, perturbation-based per-residue importance framework that identifies which amino acid positions are most discriminative for protein family classification in ESM-2 embedding space. Unlike unsupervised methods, PIVOT explicitly measures how much each position *contributes to* or *detracts from* correct family assignment by quantifying classifier decision-function changes upon systematic embedding perturbation.

PIVOT operates as a **two-phase pipeline**:

- **Phase 1 (`population`)**: Sliding-window scan across all sequences to identify population-level importance hotspots
- **Phase 2 (`individual` / `batch-individual`)**: Single-residue perturbation on target sequence(s), guided by Phase 1 results, to pinpoint exact discriminatory residues

------

## How PIVOT Works

### Phase 1: Population-Level Importance Profiling

1. Extract CLS embeddings from ESM-2 for all sequences
2. Train a **PCA + LinearSVC** classifier (`class_weight='balanced'`) to predict cluster membership
3. Slide a window (default size=5, stride=2) across aligned positions
4. For each window: replace window embeddings with mean embedding, recompute SVM decision function
5. Compute importance = L1 norm of decision function difference (original vs. perturbed)
6. Aggregate across seeds and normalize to [0, 1] per sequence length
7. Detect hotspots using top-percentage threshold and CV stability filtering

### Phase 2: Individual Residue Pinpointing

1. Load Phase 1 hotspot regions from JSON output
2. For target sequence(s): compute single-residue perturbation at every position
3. Perturbation formula: `new_emb = (sum_all − emb[p]) / (L − 1)` (mean removal, not masking)
4. Compute importance = L1 norm of SVM decision function change
5. Normalize and rank residues; highlight top-N per sequence

> ⚠️ **Critical Interpretation Notes:**
>
> - Importance scores are **always non-negative** (L1 norm); there are no negative scores in this implementation
> - Higher score = more discriminative for family classification
> - Score ≈ 0 means the position is neutral/redundant for the SVM's decision boundary
> - The metric reflects **SVM decision function sensitivity**, not probability or biological conservation directly

------

## Installation

```bash
pip install torch transformers numpy pandas scikit-learn tqdm matplotlib seaborn
```

------

## Cluster File Format

The `--cluster` argument requires a **CSV with header** containing exactly two columns: `Gene_List` and `Cluster`.

```csv
Gene_List,Cluster
StCYP71A1;StCYP71A2;StCYP71A3,CYP71A
StCYP81Q1;StCYP81Q3,CYP81Q
StCYP93A5,CYP93A
```

| Column      | Format                        | Description                                   |
| ----------- | ----------------------------- | --------------------------------------------- |
| `Gene_List` | Semicolon-separated (`;`) IDs | Sequence IDs belonging to this cluster        |
| `Cluster`   | String                        | Family/cluster label for all genes in the row |

- Sequence IDs must match FASTA headers (text before first `|` or first whitespace)
- Multiple rows can share the same `Cluster` value
- FASTA sequences **not listed** in any `Gene_List` entry are silently excluded

------

## Arguments

| Argument            | Required | Default                        | Type      | Description                                                  |
| ------------------- | -------- | ------------------------------ | --------- | ------------------------------------------------------------ |
| `--mode`            | ✅        | —                              | str       | `population`, `individual`, or `batch-individual`            |
| `--fasta`           | ✅        | —                              | str       | Input FASTA file                                             |
| `--cluster`         | ✅        | —                              | str       | Cluster CSV (see format above)                               |
| `--output-dir`      | ❌        | `output_v3`                    | str       | Output directory                                             |
| `--model-name`      | ❌        | `facebook/esm2_t33_650M_UR50D` | str       | HuggingFace ESM-2 checkpoint                                 |
| `--device`          | ❌        | `cpu`                          | str       | `cuda` or `cpu`                                              |
| `--batch-size`      | ❌        | 8                              | int       | Batch size for embedding extraction                          |
| `--pca-components`  | ❌        | 50                             | int       | PCA components before SVM                                    |
| `--svm-c`           | ❌        | 1.0                            | float     | LinearSVC regularization parameter                           |
| `--random-seeds`    | ❌        | `[42]`                         | list[int] | Random seed(s) for SVM initialization                        |
| `--window-size`     | ❌        | 5                              | int       | Sliding window size (Phase 1 only)                           |
| `--stride`          | ❌        | 2                              | int       | Window stride (Phase 1 only)                                 |
| `--n-bins`          | ❌        | 200                            | int       | Bins for normalized position profile                         |
| `--top-percentage`  | ❌        | 0.2                            | float     | Top fraction for hotspot detection                           |
| `--cv-threshold`    | ❌        | 0.5                            | float     | CV threshold for hotspot stability filtering                 |
| `--population-json` | ⚠️        | —                              | str       | Phase 1 JSON path (**required** for `individual`/`batch-individual`) |
| `--target-seq`      | ⚠️        | —                              | str       | Target sequence ID (**required** for `individual` mode only) |
| `--output-csv`      | ❌        | `all_residues_summary.csv`     | str       | Summary CSV path (`batch-individual` mode)                   |
| `--dpi`             | ❌        | 300                            | int       | Figure DPI                                                   |
| `--top-n`           | ❌        | 20                             | int       | Top residues to highlight per sequence                       |

------

## Usage Examples

### Phase 1: Population-Level Scan

```bash
python PIVOT.py --mode population \
    --fasta P450_full.fasta \
    --cluster clusters.csv \
    --output-dir full_length_pop \
    --device cuda \
    --random-seeds 42 123 456
```

### Phase 2: Single-Sequence Analysis

```bash
python PIVOT.py --mode individual \
    --fasta P450_full.fasta \
    --cluster clusters.csv \
    --population-json full_length_pop/full_results_v3.json \
    --target-seq CYP82E4_v1 \
    --output-dir individual_CYP82E4
```

### Phase 2 Batch: All-Sequence Scan + Summary

```bash
python PIVOT.py --mode batch-individual \
    --fasta P450_full.fasta \
    --cluster clusters.csv \
    --population-json full_length_pop/full_results_v3.json \
    --output-csv all_residues_summary.csv \
    --output-dir batch_results \
    --top-n 30
```

------

## Output Files

### Phase 1 (`population`)

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `full_results_v3.json`       | Complete results: per-sequence scores, hotspot regions, metadata |
| `population_profile.png/pdf` | Normalized importance density across binned positions        |
| `importance_heatmap.png/pdf` | Per-sequence importance heatmap                              |
| `hotspot_regions.csv`        | Detected hotspot start/end positions per cluster             |

### Phase 2 (`individual` / `batch-individual`)

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `per_residue_scores.csv`     | Per-position importance scores for target sequence(s)        |
| `all_residues_summary.csv`   | Aggregated summary across all sequences (`batch-individual` only) |
| `residue_importance.png/pdf` | Bar plot of top-N important residues                         |

If you use PIVOT in your research, please cite:

> Qiao Z, Wang J, Qin B, Wei F, Liang Y. The N-Terminus of *Sophora tonkinensis* Cytochrome P450s Evolves Neutrally yet Encodes Rich Functional Information: A Protein Language Model Analysis. *(Manuscript under review, 2026)*
