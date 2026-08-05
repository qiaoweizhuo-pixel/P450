#!/usr/bin/env python3
"""
DIVA_V3.0.py

Memory-efficient single-sequence processing for DIVA (only cosine projection).
Collects per-residue importance scores, generates per-sequence CSV/plots,
and produces population-level importance density profile with hotspot identification.

Usage:
    python DIVA_V3.0.py --fasta seqs.fasta --output-dir diva_results
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== 命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="DIVA: unsupervised residue importance (cosine-only)")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", default="residue_importance", help="Output directory")
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D", help="ESM2 model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (recommended 1)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top important residues to highlight per sequence")
    parser.add_argument("--max-length", type=int, default=1022, help="Max sequence length for ESM2")
    # 群体统计参数
    parser.add_argument("--n-bins", type=int, default=200, help="Number of bins for population profile")
    parser.add_argument("--top-percentage", type=float, default=0.2, help="Top percentage threshold for hotspot detection")
    parser.add_argument("--cv-threshold", type=float, default=0.5, help="CV threshold for hotspot stability (dummy, kept for compatibility)")
    return parser.parse_args()

# ==================== 单序列处理 ====================
def process_single_sequence(seq, tokenizer, model, device, top_n, out_dir, seq_id, max_length):
    """Compute DIVA scores (cosine-only) and save individual CSV and plot."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length+2)
    attention_mask = inputs['attention_mask'][0]
    valid_len = attention_mask.sum().item()
    L = valid_len - 2
    if L <= 0:
        raise ValueError(f"Sequence {seq_id} has no valid residues.")
    if L < len(seq):
        print(f"Warning: {seq_id} truncated from {len(seq)} to {L} residues.")
    seq_trimmed = seq[:L]

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        last_hidden = outputs.last_hidden_state.cpu().numpy()

    residue_hidden = last_hidden[0, 1:1+L, :]   # (L, D)
    cls_vec = last_hidden[0, 0, :]
    mean_vec = residue_hidden.mean(axis=0)

    # Delta and cosine projection (no attention)
    delta = cls_vec - mean_vec
    delta_norm = delta / (np.linalg.norm(delta) + 1e-10)
    cos_sim = np.dot(residue_hidden, delta_norm) / (np.linalg.norm(residue_hidden, axis=1) + 1e-10)
    abs_cos = np.abs(cos_sim)
    final_score = abs_cos
    if final_score.max() > 0:
        final_score_norm = final_score / final_score.max()
    else:
        final_score_norm = final_score

    # Save per-sequence CSV
    df = pd.DataFrame({
        'position': np.arange(1, L+1),
        'amino_acid': list(seq_trimmed),
        'cosine_similarity': cos_sim,
        'abs_cosine': abs_cos,
        'final_score': final_score,
        'final_score_norm': final_score_norm
    })
    df.to_csv(out_dir / f"{seq_id}_residue_importance.csv", index=False)

    # Individual plot
    plot_single_importance(seq_id, seq_trimmed, final_score_norm, top_n, out_dir)

    # Return normalized scores with their actual positions (for population stats)
    # We store (relative_position, score) for binning later
    rel_pos = np.arange(L) / L  # 0 to 1
    return rel_pos, final_score_norm

# ==================== 单序列绘图 ====================
def plot_single_importance(seq_id, seq, scores, top_n, out_dir):
    L = len(seq)
    positions = np.arange(1, L+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(positions, scores, 'b-', linewidth=1.5, label='DIVA score')
    ax1.fill_between(positions, 0, scores, alpha=0.2, color='blue')
    ax1.set_xlabel('Amino acid position')
    ax1.set_ylabel('Importance score (normalized)')
    ax1.set_title(f'{seq_id} - DIVA importance (cosine projection only)')
    ax1.grid(True, alpha=0.3)

    top_indices = np.argsort(scores)[::-1][:top_n]
    ax1.scatter(top_indices+1, scores[top_indices], color='red', s=60, zorder=5, label=f'Top {top_n} residues')
    annotate_n = min(10, top_n)
    for idx in top_indices[:annotate_n]:
        pos = idx + 1
        imp = scores[idx]
        aa = seq[idx]
        ax1.annotate(f'{aa}{pos}', xy=(pos, imp), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, fontweight='bold', color='darkred')
    if top_n > 10:
        extra = top_indices[10:]
        ax1.scatter(extra+1, scores[extra], color='orange', s=40, zorder=4, alpha=0.7, label=f'Top {top_n} (others)')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color='blue', alpha=0.3, label='DIVA profile'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=f'Top {min(10,top_n)} annotated'),
    ]
    if top_n > 10:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, label='Top 11-20'))
    ax1.legend(handles=legend_elements, loc='upper right')

    # Amino acid color bar
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
    ax2.set_title('Amino acid sequence (color-coded)')
    unique_aas = sorted(set(seq))[:10]
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=aa_colors[aa], edgecolor='none', label=aa)
                       for aa in unique_aas]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=5)

    plt.tight_layout()
    plt.savefig(out_dir / f"{seq_id}_importance.png", dpi=300)
    plt.savefig(out_dir / f"{seq_id}_importance.pdf")
    plt.close()

# ==================== 群体统计与绘图 ====================
def population_profile(all_rel_positions, all_scores, args, out_dir):
    """
    Bin normalized positions and compute mean/std, then identify hotspots.
    all_rel_positions: list of arrays (each length L_i) with values in [0,1)
    all_scores: list of arrays (each length L_i) with normalized importance scores
    """
    n_bins = args.n_bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for rel_pos, scores in zip(all_rel_positions, all_scores):
        # digitize: find bin index for each position
        idxs = np.digitize(rel_pos, bins) - 1
        # clamp to valid range
        idxs = np.clip(idxs, 0, n_bins-1)
        for i, idx in enumerate(idxs):
            val = scores[i]
            bin_means[idx] += val
            bin_stds[idx] += val**2
            bin_counts[idx] += 1

    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_means[i] /= bin_counts[i]
            bin_stds[i] = np.sqrt(bin_stds[i]/bin_counts[i] - bin_means[i]**2)
        else:
            bin_means[i] = 0.0
            bin_stds[i] = 0.0

    # Compute coefficient of variation (CV) but not used for hotspot detection here
    # We'll use threshold based on percentile of bin_means (ignoring empty bins)
    valid_mask = bin_counts > 0
    if np.any(valid_mask):
        threshold = np.percentile(bin_means[valid_mask], 100 * (1 - args.top_percentage))
    else:
        threshold = 0.0

    # Identify hotspots: bins with mean >= threshold and CV < threshold (CV ignored)
    significant = (bin_means >= threshold) & (bin_counts > 0)
    # Merge adjacent significant bins
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
                'bins': [i, j-1]
            })
            i = j
        else:
            i += 1

    # Save hotspot list to JSON
    hotspot_data = {
        'n_bins': n_bins,
        'bin_centers': bin_centers.tolist(),
        'bin_means': bin_means.tolist(),
        'bin_stds': bin_stds.tolist(),
        'hotspots': hotspots,
        'threshold': threshold
    }
    import json
    with open(out_dir / 'population_hotspots.json', 'w') as f:
        json.dump(hotspot_data, f, indent=2)

    # Plot population profile
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(bin_centers, bin_means, 'b-', linewidth=2, label='DIVA population mean')
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                    color='blue', alpha=0.2, label='±1 SD')

    for hs in hotspots:
        ax.axvspan(hs['rel_start'], hs['rel_end'], alpha=0.2, color='red',
                   label='Hotspot' if hs == hotspots[0] else '')

    ax.set_xlabel('Normalized sequence position (0 = N-term, 1 = C-term)')
    ax.set_ylabel('Importance score (normalized)')
    ax.set_title('Population-level DIVA importance density profile')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'population_profile.png', dpi=300)
    plt.savefig(out_dir / 'population_profile.pdf')
    plt.close()

    return hotspots

# ==================== 主函数 ====================
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        print("Error: No sequences found.")
        return
    print(f"Loaded {len(records)} sequences.")
    print("Attention mode: cosine projection only (no attention).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_attentions=True).to(args.device).eval()

    all_rel_positions = []
    all_scores = []
    all_summary = []

    for rec in tqdm(records, desc="Processing"):
        seq_id = rec.id
        seq = str(rec.seq)
        try:
            rel_pos, scores = process_single_sequence(seq, tokenizer, model, args.device,
                                                      args.top_n, out_dir, seq_id, args.max_length)
            all_rel_positions.append(rel_pos)
            all_scores.append(scores)

            # Collect top N for summary CSV (optional)
            top_indices = np.argsort(scores)[::-1][:args.top_n]
            for rank, pos in enumerate(top_indices, 1):
                all_summary.append({
                    'seq_id': seq_id,
                    'rank': rank,
                    'position': pos+1,
                    'amino_acid': seq[pos] if len(seq) > pos else 'X',
                    'score': scores[pos]
                })
        except Exception as e:
            print(f"Error processing {seq_id}: {e}")
            continue
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Save summary of top residues across all sequences
    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        df_summary.to_csv(out_dir / "top_residues_summary.csv", index=False)
        print(f"✅ Per-sequence top residues summary saved to {out_dir / 'top_residues_summary.csv'}")

    # Generate population profile
    if all_rel_positions:
        hotspots = population_profile(all_rel_positions, all_scores, args, out_dir)
        print("\n=== Population hotspots detected ===")
        for i, hs in enumerate(hotspots):
            print(f"Hotspot {i+1}: {hs['rel_start']:.3f} - {hs['rel_end']:.3f} (mean importance {hs['mean_importance']:.4f})")
    else:
        print("No data for population profile.")

    print(f"\n✅ Analysis complete. Results saved to {out_dir}/")
    print("   - Per-sequence CSV and plots")
    print("   - population_profile.png/pdf")
    print("   - population_hotspots.json")

if __name__ == "__main__":
    main()