# ESM_EVOL_ALS: Protein Sequence Clustering and Evolutionary Analysis Tool

## 📖 Introduction
`ESM_EVOL_ALS` is a comprehensive bioinformatics pipeline that integrates deep learning feature extraction with phylogenetic analysis. It utilizes the **ESM2 (Evolutionary Scale Modeling)** protein language model to extract high-dimensional embeddings, performs hierarchical clustering, and systematically compares the results against monophyletic clustering derived from a user-provided phylogenetic tree.

**Key Features:**
- 🧬 **ESM2 Embedding**: Supports both ESM2-650M and ESM2-15B models for deep protein representation.
- 🌳 **Phylogenetic Integration**: Automatically partitions a phylogenetic tree into strictly **monophyletic groups**, aligning the number of clusters with the ESM-based clusters.
- 📊 **Multi-dimensional Comparison**: Computes clustering consistency metrics (V-measure, AMI), identifies "divergent sequences" (sequences that cluster differently across methods), and generates side-by-side UMAP visualizations.
- 🗂️ **Rich Data Export**: Outputs detailed cluster assignments, distance matrices, interactive Sankey diagrams, PCoA comparisons, and more.

---

## 📦 Environment and Dependencies
Python 3.12 is recommended. Due to the large size of ESM models, it is highly recommended to create a fresh virtual environment using `conda` or `venv`.

### Install Dependencies
Run the following command in your terminal:
```bash
pip install torch pandas numpy biopython transformers scikit-learn umap-learn matplotlib seaborn scipy ete3 plotly
```
> **Note:** If you have an NVIDIA GPU, ensure you install the correct CUDA-compatible version of `torch`. The ESM2-650M model requires approximately ≥12GB of VRAM, while ESM2-15B requires ≥40GB.

---

## 🚀 Usage

### Command Line Arguments

| Argument               | Short | Type    | Required | Default     | Description                                                  |
| :--------------------- | :---- | :------ | :------- | :---------- | :----------------------------------------------------------- |
| `--input`              | `-i`  | String  | **Yes**  | None        | Path to the input protein sequences in FASTA format          |
| `--tree`               | `-t`  | String  | **Yes**  | None        | Path to the input phylogenetic tree file (Newick format)     |
| `--model`              | `-m`  | String  | No       | `esm2-650M` | Choose the ESM2 model (`esm2-650M` or `esm2-15B`)            |
| `--linkage`            | `-l`  | String  | No       | `average`   | Linkage method for hierarchical clustering (`average`, `ward`, `complete`, `single`) |
| `--metric`             | None  | String  | No       | `cosine`    | Distance metric for ESM embeddings (`euclidean`, `cosine`, `correlation`) |
| `--distance-threshold` | `-d`  | Float   | No       | None        | Distance threshold for cutting the hierarchical tree. If omitted, clusters are determined automatically based on the tree structure |
| `--phylo-clusters`     | `-p`  | Integer | No       | None        | Specific number of clusters to force on the phylogenetic tree. If omitted, the script will use the exact same number of clusters as the ESM hierarchical clustering |

### Basic Run Example
Assume your FASTA file and Newick tree are in the current directory:
```bash
python ESM_EVOL_ALS.py -i P450_sequences.fasta -t phylogenetic_tree.nwk -m esm2-650M
```

**Advanced Example (Custom linkage and fixed clusters):**
```bash
python ESM_EVOL_ALS.py -i P450_sequences.fasta -t phylogenetic_tree.nwk -l ward --metric correlation -p 12
```
*(The command above uses Ward's linkage and correlation distance for ESM clustering, and forces the phylogenetic tree to be partitioned into exactly 12 monophyletic clusters)*

---

## 📂 Output Files
Upon execution, the script generates a series of files prefixed by the input FASTA filename (`$PREFIX`).

| Output File                                | Description                                                  |
| :----------------------------------------- | :----------------------------------------------------------- |
| `$PREFIX_esm2_embeddings.csv`              | The high-dimensional ESM2 feature matrix (rows = proteins, columns = embedding dimensions) |
| **`$PREFIX_cluster_assignments.csv`**      | **🌟 Core Output**: Contains UMAP coordinates, ESM cluster ID, phylogenetic cluster ID, and a binary flag (`is_divergent`) for each sequence |
| `$PREFIX_hclust_tree.nwk`                  | A Newick-formatted hierarchical clustering tree derived from the ESM embeddings |
| `$PREFIX_phylogenetic_distance_matrix.csv` | The patristic distance matrix calculated from the input phylogenetic tree |
| `$PREFIX_cluster_comparison.png`           | Side-by-side UMAP projections of ESM clusters (left) and Phylogenetic clusters (right), with divergent sequences highlighted as **hollow black circles** |
| `$PREFIX_sankey_diagram.html`              | **🌟 Interactive Sankey Diagram**: Visualizes the flow of sequences from ESM clusters (left) to Phylogenetic clusters (right) |
| `$PREFIX_pcoa_comparison.png`              | PCoA (Principal Coordinate Analysis) comparing the ESM embedding space versus the phylogenetic tree space |
| `$PREFIX_hclust_dendrogram.png`            | A static dendrogram of the hierarchical clustering           |
| `$PREFIX_divergent_sequences.csv`          | A simple list of sequence IDs identified as divergent (misclassified across the two clustering methods) |
| `$PREFIX_sankey_flow_genes.csv`            | A detailed table showing sequence flows between clusters, useful for identifying major evolutionary transitions |
| `$PREFIX_evolution_analysis.csv`           | Statistical metrics (V-measure and Adjusted Mutual Information) evaluating the consistency between the two clustering strategies |

---

## ⚙️ Core Algorithm Details
1. **ESM Clustering**: The script uses `scipy.cluster.hierarchy` to perform hierarchical clustering on the extracted ESM embeddings.
2. **Phylogenetic Clustering (Monophyletic Verification)**: The script implements a **custom dynamic programming algorithm**. It ensures that every cluster extracted from the input phylogenetic tree is a strict **monophyletic group**. If a cluster is found to be paraphyletic or polyphyletic, the algorithm recursively splits it into smaller, strictly monophyletic subgroups to maintain biological accuracy.
3. **Divergent Sequence Definition**: A protein is flagged as "divergent" (`is_divergent = 1`) if its assigned ESM cluster ID does not match its phylogenetic cluster ID (`hcluster != phylogeny_cluster`). This flag is an indicator of potential convergent evolution, lateral gene transfer, or differing evolutionary constraints detected by the two models.

---

## 💡 Notes & FAQ

**1. Why are my UMAP plots changing even when I run the script repeatedly?**
Although the script uses a fixed `random_state=42` in `umap.UMAP()` for reproducibility, updating the `umap-learn` or `numpy` packages to different versions might cause slight variations in the 2D coordinate layout. **This is normal**. It does **not** affect the actual cluster assignments (the `hcluster` and `phylogeny_cluster` columns in the CSV file), which are entirely deterministic.

**2. What if some sequence IDs are missing from my phylogenetic tree?**
If a sequence ID in your FASTA file is not present as a leaf node in the input Newick tree, the script will issue a warning. These missing sequences will be ignored during phylogenetic distance calculations, and they will be assigned to the nearest existing cluster based on patristic distance.

**3. Running Time Estimates**
- For the `esm2-650M` model, processing ~500 sequences with an average length of 400 amino acids takes approximately 5–10 minutes on an NVIDIA A100 GPU.
- The `esm2-15B` model is extremely memory-intensive. If you encounter an Out-of-Memory (OOM) error, consider reducing the number or length of your input sequences.

---

## 📚 Citation
If you use this tool in your research, please cite Zhu Q. et al.,**The N-Terminus of \*Sophora tonkinensis\* Cytochrome P450s Evolves Neutrally yet Encodes Rich Functional Information: A Protein Language Model Analysis** (2026) & the ESM2 model paper and the relevant Python libraries (`PyTorch`, `scikit-learn`, `ETE3`, etc.) as appropriate.