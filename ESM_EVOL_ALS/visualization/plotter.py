import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ete3 import TreeStyle
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

# 添加桑基图相关的导入
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def plot_sankey_diagram(esm_clusters, phylo_clusters, seq_ids, output_path):
    """
    绘制桑基图，显示ESM2聚类和系统发育树聚类之间的关系
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly 不可用，跳过桑基图生成")
        return
    
    # 准备数据
    esm_labels = [f"ESM_Cluster_{c}" for c in esm_clusters]
    phylo_labels = [f"Phylo_Cluster_{phylo_clusters.get(seq_id, -1)}" for seq_id in seq_ids]
    
    all_nodes = sorted(list(set(esm_labels))) + sorted(list(set(phylo_labels)))
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    link_counts = {}
    for esm_label, phylo_label in zip(esm_labels, phylo_labels):
        if esm_label in node_dict and phylo_label in node_dict:
            link_counts.setdefault((node_dict[esm_label], node_dict[phylo_label]), 0)
            link_counts[(node_dict[esm_label], node_dict[phylo_label])] += 1
    
    source = [src for (src, tgt), count in link_counts.items()]
    target = [tgt for (src, tgt), count in link_counts.items()]
    value = [count for (src, tgt), count in link_counts.items()]
    
    n_esm_nodes = len(set(esm_labels))
    node_colors = ["#0076B4"] * n_esm_nodes + ["#D6641E"] * len(set(phylo_labels))
    link_colors = ["rgba(150, 150, 150, 0.6)"] * len(source)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color=node_colors
        ),
        link=dict(
            source=source, target=target, value=value, color=link_colors
        ))])
    
    fig.update_layout(title_text="ESM2聚类 vs 系统发育树聚类 - 桑基图", font_size=12, width=1200, height=800)
    
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"桑基图已保存至: {output_path}")

def plot_dendrogram(linkage_matrix, labels, output_path, title_suffix=""):
    """
    绘制层次聚类树状图
    """
    plt.figure(figsize=(15, 10))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(f"Hierarchical Clustering Dendrogram {title_suffix}")
    plt.xlabel("Protein Sequences")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"层次聚类树状图已保存至: {output_path}")

def plot_phylogenetic_tree(tree, output_path):
    """
    绘制系统发育树
    """
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.mode = "c"
    ts.arc_start = -180
    ts.arc_span = 180
    
    tree.render(output_path, tree_style=ts)
    print(f"系统发育树图已保存至: {output_path}")

def plot_cluster_comparison(umap_coords, esm_clusters, phylo_clusters, seq_ids, divergent_sequences, output_path):
    """
    绘制ESM聚类与系统发育聚类的比较图，高亮显示分歧序列
    """
    esm_labels = [esm_clusters.get(seq_id, -1) for seq_id in seq_ids]
    phylo_labels = [phylo_clusters.get(seq_id, -1) for seq_id in seq_ids]
    is_divergent = [seq_id in divergent_sequences for seq_id in seq_ids]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ESM聚类结果
    scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], c=esm_labels, cmap="Spectral", s=30, alpha=0.8)
    ax1.scatter(umap_coords[is_divergent, 0], umap_coords[is_divergent, 1], s=100, facecolors='none', edgecolors='black', linewidths=2, marker='o', label='Divergent Sequences')
    ax1.set_title("ESM-based Clustering (Divergent Sequences Highlighted)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1)
    
    # 系统发育聚类结果
    scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c=phylo_labels, cmap="Spectral", s=30, alpha=0.8)
    ax2.scatter(umap_coords[is_divergent, 0], umap_coords[is_divergent, 1], s=100, facecolors='none', edgecolors='black', linewidths=2, marker='o', label='Divergent Sequences')
    ax2.set_title("Phylogenetic Clustering (Divergent Sequences Highlighted)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"聚类比较图已保存至: {output_path}")

def plot_convergence_heatmap(embeddings, clusters, seq_ids, output_path):
    """
    绘制趋同进化热图
    """
    sorted_indices = np.argsort(clusters)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_clusters = clusters[sorted_indices]
    correlation_matrix = np.corrcoef(sorted_embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    plt.title("Protein Sequence Similarity Heatmap")
    plt.xlabel("Protein Sequences")
    plt.ylabel("Protein Sequences")
    
    current_cluster = sorted_clusters[0]
    cluster_boundaries = []
    for i, cluster in enumerate(sorted_clusters):
        if cluster != current_cluster:
            cluster_boundaries.append(i)
            current_cluster = cluster
    
    for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color='black', linestyle='-', linewidth=1)
        plt.axvline(x=boundary, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"趋同进化热图已保存至: {output_path}")

def plot_pcoa_comparison(esm_coords, phylo_coords, seq_ids, output_path):
    """
    绘制ESM和系统发育PCoA结果的比较图
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.scatter(esm_coords[:, 0], esm_coords[:, 1], c='blue', alpha=0.7, s=50, label='ESM PCoA')
    ax.scatter(phylo_coords[:, 0], phylo_coords[:, 1], c='red', alpha=0.7, s=50, label='Phylogeny PCoA')
    
    for i in range(len(esm_coords)):
        ax.plot([esm_coords[i, 0], phylo_coords[i, 0]], [esm_coords[i, 1], phylo_coords[i, 1]], 'gray', alpha=0.3, linewidth=0.5)
    
    # 添加凸包
    try:
        esm_hull = ConvexHull(esm_coords)
        phylo_hull = ConvexHull(phylo_coords)
        for simplex in esm_hull.simplices:
            ax.plot(esm_coords[simplex, 0], esm_coords[simplex, 1], 'b--', alpha=0.5)
        for simplex in phylo_hull.simplices:
            ax.plot(phylo_coords[simplex, 0], phylo_coords[simplex, 1], 'r--', alpha=0.5)
    except:
        print("警告: 无法计算凸包，可能点太少了。")
    
    ax.set_xlabel("PCoA 1")
    ax.set_ylabel("PCoA 2")
    ax.set_title("PCoA Comparison: ESM vs Phylogenetic Tree")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"PCoA比较图已保存至: {output_path}")