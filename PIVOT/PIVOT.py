#!/usr/bin/env python3
"""
AA_Position_Importance_V3.py (Optimized for CPU, Multi-class Compatible, Batch Mode, Full-length Phase2)

Two-stage analysis for full-length variable-length P450 sequences:
- Phase 1 (population): normalized sliding window perturbation across all sequences,
  outputs importance density profile and hotspot regions.
- Phase 2 (individual): full-length single-residue knockout, outputs importance for all positions,
  with enhanced visualization (profile + amino acid bar).
- Phase 2 Batch (batch-individual): run Phase 2 on all sequences, save per-sequence full importance CSV,
  and a summary CSV of top 5 residues per hotspot.

Usage examples:
  # Phase 1: population analysis on full-length sequences
  python AA_Position_Importance_V3.py --mode population --fasta P450_full.fasta \
      --cluster clusters.csv --output-dir full_length_pop

  # Phase 2: individual pinpointing for a specific sequence (full-length scan)
  python AA_Position_Importance_V3.py --mode individual --fasta P450_full.fasta \
      --cluster clusters.csv --output-dir individual_results \
      --population-json full_length_pop/full_results_v3.json \
      --target-seq CYP82E4_v1

  # Phase 2 Batch: run individual on all sequences and generate summary CSV
  python AA_Position_Importance_V3.py --mode batch-individual \
      --fasta P450_full.fasta --cluster clusters.csv \
      --population-json full_length_pop/full_results_v3.json \
      --output-csv all_residues_summary.csv
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import argparse
import json
import pickle

from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==================== 性能优化设置 ====================
torch.set_num_threads(16)

# ==================== JSON 序列化辅助 ====================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

# ==================== 数据加载 ====================
def load_sequences(fasta_file: str, cluster_file: str, n_aa: Optional[int] = None) -> Tuple[List[str], List[str], Dict[str, str]]:
    cluster_df = pd.read_csv(cluster_file)
    gene_cluster = {}
    for _, row in cluster_df.iterrows():
        genes = str(row['Gene_List']).split(';')
        for g in genes:
            g_clean = g.strip()
            if g_clean:
                gene_cluster[g_clean] = str(row['Cluster'])

    sequences = {}
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                header = line[1:]
                if '|' in header:
                    current_id = header.split('|')[0]
                else:
                    current_id = header.split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)

    common = set(sequences.keys()) & set(gene_cluster.keys())
    seq_list = []
    label_list = []
    for g in common:
        seq = sequences[g]
        if n_aa is not None:
            if len(seq) >= n_aa:
                seq_list.append(seq[:n_aa])
                label_list.append(gene_cluster[g])
        else:
            seq_list.append(seq)
            label_list.append(gene_cluster[g])
    return seq_list, label_list, sequences

# ==================== ESM2 嵌入提取 ====================
class ESM2Embedder:
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cpu", batch_size: int = 4):
        self.device = device
        self.batch_size = batch_size
        print(f"Loading ESM2 model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.embed_dim = self.model.config.hidden_size

    def get_all_embeddings(self, sequences: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        mean_embs = []
        pos_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), self.batch_size), desc="Extracting embeddings"):
                batch = sequences[i:i+self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state.cpu().numpy()
                for j, seq in enumerate(batch):
                    seq_len = len(seq)
                    pos_emb = last_hidden[j, 1:seq_len+1, :]
                    pos_embs.append(pos_emb)
                    mean_embs.append(pos_emb.mean(axis=0))
        return np.array(mean_embs), pos_embs

# ==================== SVM 模型封装 ====================
class SVMAttributor:
    def __init__(self, pca_components: int = 50, svm_c: float = 1.0, random_state: int = 42):
        self.pca_components = pca_components
        self.svm_c = svm_c
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        self.clf = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: List[str]):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        n_comp = min(self.pca_components, X_scaled.shape[0]-1, X_scaled.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        self.clf = LinearSVC(C=self.svm_c, class_weight='balanced', max_iter=10000,
                             random_state=self.random_state, dual=False)
        self.clf.fit(X_pca, y_enc)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.clf.decision_function(X_pca)

    def compute_perturbation_impact(self, orig_vec: np.ndarray, new_vec: np.ndarray) -> float:
        return float(np.linalg.norm(orig_vec - new_vec, ord=1))

# ==================== Phase 1: 群体归一化扫描 ====================
def population_normalized_scan(sequences: List[str], labels: List[str],
                               embedder: ESM2Embedder, args) -> Dict:
    print("\n=== Phase 1: Population Normalized Scan ===")
    print(f"Analyzing {len(sequences)} full-length sequences (variable lengths).")

    X_mean, pos_embs = embedder.get_all_embeddings(sequences)

    attributor = SVMAttributor(pca_components=args.pca_components, svm_c=args.svm_c,
                               random_state=args.random_seeds[0])
    attributor.fit(X_mean, labels)

    window_size = args.window_size
    stride = args.stride
    all_window_data = []

    for i, seq in enumerate(tqdm(sequences, desc="Scanning sequences")):
        L = len(seq)
        if L < window_size:
            continue
        emb = pos_embs[i]
        mean_emb = X_mean[i]
        orig_score = attributor.decision_function(mean_emb.reshape(1, -1)).flatten()

        starts = list(range(0, L - window_size + 1, stride))
        for start in starts:
            end = start + window_size
            sum_all = emb.sum(axis=0)
            sum_window = emb[start:end].sum(axis=0)
            perturbed_mean = (sum_all - sum_window) / (L - window_size)
            new_score = attributor.decision_function(perturbed_mean.reshape(1, -1)).flatten()
            delta = attributor.compute_perturbation_impact(orig_score, new_score)

            rel_mid = (start + window_size/2) / L
            all_window_data.append({'rel_mid': rel_mid, 'importance': delta})

    n_bins = args.n_bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for w in all_window_data:
        bin_idx = np.digitize(w['rel_mid'], bins) - 1
        if 0 <= bin_idx < n_bins:
            bin_means[bin_idx] += w['importance']
            bin_stds[bin_idx] += w['importance']**2
            bin_counts[bin_idx] += 1

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_means[i] /= bin_counts[i]
            bin_stds[i] = np.sqrt(bin_stds[i]/bin_counts[i] - bin_means[i]**2)
        else:
            bin_means[i] = 0.0
            bin_stds[i] = 0.0

    cv = bin_stds / (bin_means + 1e-10)
    threshold = np.percentile(bin_means[bin_counts > 0], 100 * (1 - args.top_percentage))
    significant = (bin_means >= threshold) & (cv < args.cv_threshold)

    hotspots = []
    i = 0
    while i < n_bins:
        if significant[i]:
            j = i
            while j < n_bins and significant[j]:
                j += 1
            rel_start = bins[i]
            rel_end = bins[j] if j < n_bins else 1.0
            mean_imp = bin_means[i:j].mean()
            hotspots.append({
                'rel_start': float(rel_start),
                'rel_end': float(rel_end),
                'mean_importance': float(mean_imp),
                'bins': [int(i), int(j-1)]
            })
            i = j
        else:
            i += 1

    return {
        'n_bins': n_bins,
        'bin_centers': bin_centers.tolist(),
        'bin_means': bin_means.tolist(),
        'bin_stds': bin_stds.tolist(),
        'hotspots': hotspots,
        'attributor': attributor,
        'X_mean': X_mean,
        'pos_embs': pos_embs,
        'sequences': sequences,
        'labels': labels,
    }

# ==================== Phase 2: 全序列单残基扫描 ====================
def pinpoint_individual(seq_id: str, seq: str, attributor: SVMAttributor,
                        hotspots: List[Dict], embedder: ESM2Embedder) -> Dict:
    print(f"\n=== Phase 2: Full-length residue scanning for {seq_id} ===")

    X_mean, pos_emb_list = embedder.get_all_embeddings([seq])
    mean_emb = X_mean[0]
    pos_emb = pos_emb_list[0]
    L = len(seq)

    orig_score = attributor.decision_function(mean_emb.reshape(1, -1)).flatten()

    importance = np.zeros(L)
    for p in tqdm(range(L), desc="Scanning residues", leave=False):
        sum_all = pos_emb.sum(axis=0)
        perturbed = (sum_all - pos_emb[p]) / (L - 1)
        new_score = attributor.decision_function(perturbed.reshape(1, -1)).flatten()
        delta = attributor.compute_perturbation_impact(orig_score, new_score)
        importance[p] = delta

    hotspot_results = {}
    for hs_idx, hs in enumerate(hotspots):
        rel_start = hs['rel_start']
        rel_end = hs['rel_end']
        abs_start = int(rel_start * L)
        abs_end = min(int(rel_end * L) + 1, L)

        if abs_start >= L:
            continue

        scores = [(p+1, importance[p]) for p in range(abs_start, abs_end)]
        scores.sort(key=lambda x: x[1], reverse=True)
        hotspot_results[f"hotspot_{hs_idx+1}"] = {
            'rel_range': [rel_start, rel_end],
            'abs_range': [abs_start+1, abs_end],
            'top_residues': [{'position': pos, 'importance': imp} for pos, imp in scores[:10]]
        }

    return {
        'seq_id': seq_id,
        'length': L,
        'importance': importance.tolist(),
        'hotspot_results': hotspot_results
    }

# ==================== Phase 2 批量模式 ====================
def batch_individual_all(sequences_dict: Dict[str, str], attributor: SVMAttributor,
                         hotspots: List[Dict], embedder: ESM2Embedder,
                         output_csv: str, out_dir: Path):
    all_results = []
    for seq_id, seq in tqdm(sequences_dict.items(), desc="Batch individual"):
        try:
            res = pinpoint_individual(seq_id, seq, attributor, hotspots, embedder)
            # 保存全长重要性 CSV
            df_full = pd.DataFrame({
                'position': np.arange(1, res['length']+1),
                'amino_acid': list(seq),
                'importance': res['importance']
            })
            df_full.to_csv(out_dir / f"{seq_id}_full_importance.csv", index=False)

            # 生成可视化
            plot_individual_pinpoint(res, seq, out_dir)

            # 汇总热点区域 Top5
            for hs_name, hs_data in res['hotspot_results'].items():
                for rank, r in enumerate(hs_data['top_residues'][:5], 1):
                    all_results.append({
                        'seq_id': seq_id,
                        'hotspot': hs_name,
                        'rel_start': hs_data['rel_range'][0],
                        'rel_end': hs_data['rel_range'][1],
                        'abs_start': hs_data['abs_range'][0],
                        'abs_end': hs_data['abs_range'][1],
                        'rank': rank,
                        'position': r['position'],
                        'importance': r['importance']
                    })
        except Exception as e:
            print(f"Error processing {seq_id}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Batch individual complete. Results saved to {output_csv}")
    return df

# ==================== 可视化 ====================
def plot_population_profile(pop_data: Dict, out_dir: Path):
    bin_centers = np.array(pop_data['bin_centers'])
    bin_means = np.array(pop_data['bin_means'])
    bin_stds = np.array(pop_data['bin_stds'])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bin_centers, bin_means, 'b-', linewidth=2, label='Importance density')
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                    color='blue', alpha=0.2, label='±1 SD')

    for hs in pop_data['hotspots']:
        ax.axvspan(hs['rel_start'], hs['rel_end'], alpha=0.2, color='red', label='Hotspot' if hs==pop_data['hotspots'][0] else '')

    ax.set_xlabel('Normalized sequence position (0 = N-term, 1 = C-term)')
    ax.set_ylabel('Importance score')
    ax.set_title('Population-level Importance Density Profile')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'population_profile.png', dpi=300)
    plt.close()

def plot_individual_pinpoint(pinpoint_data: Dict, seq: str, out_dir: Path):
    seq_id = pinpoint_data['seq_id']
    L = pinpoint_data['length']
    importance = np.array(pinpoint_data['importance'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    positions = np.arange(1, L+1)
    ax1.plot(positions, importance, 'b-', linewidth=1.5, label='Residue importance')

    colors_hotspot = ['#FFDDC1', '#C1E1C1', '#C1D4FF']
    for i, (hs_name, hs_data) in enumerate(pinpoint_data['hotspot_results'].items()):
        start, end = hs_data['abs_range']
        ax1.axvspan(start, end, alpha=0.15, color=colors_hotspot[i % len(colors_hotspot)],
                    label=f"{hs_name} ({start}-{end})")

    top_indices = np.argsort(importance)[::-1][:5]
    for idx in top_indices:
        pos = idx + 1
        imp = importance[idx]
        aa = seq[idx]
        ax1.scatter(pos, imp, color='red', s=80, zorder=5)
        ax1.annotate(f'{aa}{pos}', xy=(pos, imp), xytext=(5, 5),
                     textcoords='offset points', fontsize=9, fontweight='bold', color='darkred')

    ax1.set_xlabel('Amino acid position')
    ax1.set_ylabel('Importance score (Δ decision)')
    ax1.set_title(f'{seq_id} - Full-length residue importance profile')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 氨基酸条带图
    aa_colors = {
        'A': '#C0C0C0', 'C': '#FFFF00', 'D': '#FF0000', 'E': '#FF0000',
        'F': '#00FF00', 'G': '#FFA500', 'H': '#0000FF', 'I': '#808080',
        'K': '#0000FF', 'L': '#808080', 'M': '#FFFF00', 'N': '#FF00FF',
        'P': '#00FFFF', 'Q': '#FF00FF', 'R': '#0000FF', 'S': '#FFA500',
        'T': '#FFA500', 'V': '#808080', 'W': '#00FF00', 'Y': '#00FF00'
    }
    colors = [aa_colors.get(aa, '#000000') for aa in seq]
    ax2.bar(positions, [1]*L, color=colors, width=1.0, edgecolor='none')
    ax2.set_xlim(1, L)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Amino acid position')
    ax2.set_yticks([])
    ax2.set_title('Amino acid sequence (color-coded by residue type)')

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=aa_colors[aa], edgecolor='none', label=aa)
                       for aa in sorted(set(seq))[:10]]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=5)

    plt.tight_layout()
    plt.savefig(out_dir / f"{seq_id}_full_profile.png", dpi=300)
    plt.savefig(out_dir / f"{seq_id}_full_profile.pdf")
    plt.close()

# ==================== 命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="AA_Position_Importance_V3: Optimized two-stage analysis with batch mode")
    parser.add_argument("--mode", required=True, choices=["population", "individual", "batch-individual"],
                        help="Analysis mode: population (Phase1), individual (Phase2), batch-individual (Phase2 on all)")
    parser.add_argument("--fasta", required=True, help="FASTA file path")
    parser.add_argument("--cluster", required=True, help="CSV file with Gene_List and Cluster columns")
    parser.add_argument("--output-dir", default="output_v3", help="Output directory")

    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[42])

    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-bins", type=int, default=200)
    parser.add_argument("--top-percentage", type=float, default=0.2)
    parser.add_argument("--cv-threshold", type=float, default=0.5)

    parser.add_argument("--population-json", help="Path to population results JSON (required for individual/batch)")
    parser.add_argument("--target-seq", help="Target sequence ID (required for individual)")
    parser.add_argument("--output-csv", default="all_residues_summary.csv", help="Output CSV for batch-individual mode")

    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = ESM2Embedder(model_name=args.model_name, device=args.device, batch_size=args.batch_size)

    if args.mode == "population":
        seqs, labels, _ = load_sequences(args.fasta, args.cluster, n_aa=None)
        print(f"Loaded {len(seqs)} full-length sequences.")
        pop_results = population_normalized_scan(seqs, labels, embedder, args)

        save_data = {
            'n_bins': pop_results['n_bins'],
            'bin_centers': pop_results['bin_centers'],
            'bin_means': pop_results['bin_means'],
            'bin_stds': pop_results['bin_stds'],
            'hotspots': pop_results['hotspots'],
            'args': vars(args),
        }
        with open(out_dir / 'full_results_v3.json', 'w') as f:
            json.dump(convert_to_serializable(save_data), f, indent=2, cls=NumpyEncoder)

        with open(out_dir / 'attributor.pkl', 'wb') as f:
            pickle.dump(pop_results['attributor'], f)

        plot_population_profile(pop_results, out_dir)

        print(f"\n✅ Population analysis complete. Hotspots:")
        for i, hs in enumerate(pop_results['hotspots']):
            print(f"  Hotspot {i+1}: {hs['rel_start']:.3f} - {hs['rel_end']:.3f} (mean imp={hs['mean_importance']:.4f})")

    elif args.mode == "individual":
        if not args.population_json or not args.target_seq:
            raise ValueError("--population-json and --target-seq are required for individual mode.")

        with open(args.population_json, 'r') as f:
            pop_data = json.load(f)
        hotspots = pop_data['hotspots']

        pop_dir = Path(args.population_json).parent
        with open(pop_dir / 'attributor.pkl', 'rb') as f:
            attributor = pickle.load(f)

        _, _, seq_dict = load_sequences(args.fasta, args.cluster, n_aa=None)
        if args.target_seq not in seq_dict:
            raise ValueError(f"Target sequence {args.target_seq} not found in FASTA.")
        target_seq = seq_dict[args.target_seq]

        pinpoint_data = pinpoint_individual(args.target_seq, target_seq, attributor, hotspots, embedder)

        with open(out_dir / f"{args.target_seq}_pinpoint.json", 'w') as f:
            json.dump(convert_to_serializable(pinpoint_data), f, indent=2, cls=NumpyEncoder)

        plot_individual_pinpoint(pinpoint_data, target_seq, out_dir)

        print(f"\n✅ Pinpoint analysis complete for {args.target_seq}.")

    elif args.mode == "batch-individual":
        if not args.population_json:
            raise ValueError("--population-json required for batch-individual mode.")

        with open(args.population_json, 'r') as f:
            pop_data = json.load(f)
        hotspots = pop_data['hotspots']

        pop_dir = Path(args.population_json).parent
        with open(pop_dir / 'attributor.pkl', 'rb') as f:
            attributor = pickle.load(f)

        _, _, seq_dict = load_sequences(args.fasta, args.cluster, n_aa=None)

        batch_individual_all(seq_dict, attributor, hotspots, embedder, args.output_csv, out_dir)

if __name__ == "__main__":
    main()