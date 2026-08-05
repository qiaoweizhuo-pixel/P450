import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import umap
# 导入绘图依赖（用于距离矩阵热图）
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# 导入核心功能模块
from core.data_processor import parse_fasta, ESM2FeatureExtractor
from core.clustering import hierarchical_clustering, parse_phylogenetic_tree, extract_phylogenetic_clusters
from core.metrics import calculate_divergence
from core.utils import calculate_patroistic_distance_matrix, perform_pcoa
from visualization.exporter import save_linkage_matrix, linkage_to_newick, save_newick_tree, export_sankey_data
from visualization.plotter import (
    plot_dendrogram, 
    plot_phylogenetic_tree, 
    plot_cluster_comparison, 
    plot_convergence_heatmap, 
    plot_pcoa_comparison, 
    plot_sankey_diagram
)
# 注意：PLOTLY_AVAILABLE 变量已在 plotter.py 内部处理

def main():
    parser = argparse.ArgumentParser(description="基于ESM2的P450蛋白聚类与进化分析")
    parser.add_argument("-i", "--input", required=True, help="输入FASTA文件路径")
    parser.add_argument("-t", "--tree", required=True, help="系统发育树文件路径(Newick格式)")
    parser.add_argument("-m", "--model", default="esm2-650M", 
                        choices=["esm2-15B", "esm2-650M"], help="ESM模型选择")
    parser.add_argument("-l", "--linkage", default="average", 
                        choices=["average", "ward", "complete", "single"], 
                        help="层次聚类连接方法")
    parser.add_argument("--metric", default="cosine", 
                        choices=["euclidean", "cosine", "correlation"], 
                        help="距离度量方法")
    parser.add_argument("-d", "--distance-threshold", type=float, 
                        help="层次聚类距离阈值，用于确定聚类数量")
    parser.add_argument("-p", "--phylo-clusters", type=int, 
                        help="系统发育树聚类数量，如果不指定则使用与ESM聚类相同的数量")
    
    args = parser.parse_args()

    # 模型名称映射
    model_mapping = {
        "esm2-650M": "facebook/esm2_t33_650M_UR50D",
        "esm2-15B": "facebook/esm2_t48_15B_UR50D"
    }
    
    print(f"使用模型: {args.model}")
    print(f"层次聚类连接方法: {args.linkage}")
    print(f"距离度量方法: {args.metric}")
    if args.distance_threshold:
        print(f"层次聚类距离阈值: {args.distance_threshold}")
    if args.phylo_clusters:
        print(f"系统发育树聚类数量: {args.phylo_clusters}")
    
    # 读取FASTA文件
    print(f"读取FASTA文件: {args.input}")
    seq_ids, sequences = parse_fasta(args.input)
    print(f"载入{len(sequences)}条蛋白质序列")
    
    # 初始化特征提取器
    extractor = ESM2FeatureExtractor(model_mapping[args.model])
    
    # 提取ESM2嵌入特征
    print("正在提取ESM嵌入特征...")
    start_time = time.time()
    embeddings = extractor.embed_sequences(sequences)
    print(f"特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 保存嵌入向量
    output_prefix = Path(args.input).stem
    embeddings_df = pd.DataFrame(embeddings, index=seq_ids)
    embeddings_df.to_csv(f"{output_prefix}_esm2_embeddings.csv")
    print(f"嵌入向量已保存至: {output_prefix}_esm2_embeddings.csv")
    
    # UMAP降维
    print("使用UMAP降维...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    
    # 层次聚类 - 现在返回聚类数量
    print("执行层次聚类...")
    hclust_clusters, linkage_matrix, dist_matrix, n_esm_clusters = hierarchical_clustering(
        embeddings, 
        method=args.linkage, 
        metric=args.metric,
        threshold=args.distance_threshold
    )
    
    # 保存连接矩阵
    save_linkage_matrix(linkage_matrix, seq_ids, f"{output_prefix}_linkage_matrix.csv")
    
    # 将层次聚类结果转换为Newick格式并保存
    print("生成Newick格式的层次聚类树...")
    newick_tree = linkage_to_newick(linkage_matrix, seq_ids)
    save_newick_tree(newick_tree, f"{output_prefix}_hclust_tree.nwk")
    
    # 解析系统发育树
    print("解析系统发育树...")
    tree = parse_phylogenetic_tree(args.tree)
    
    # 提取系统发育聚类 - 使用用户指定的数量或ESM聚类的数量
    if args.phylo_clusters:
        n_phylo_clusters = args.phylo_clusters
        print(f"使用用户指定的系统发育树聚类数量: {n_phylo_clusters}")
    else:
        n_phylo_clusters = n_esm_clusters
        print(f"使用与ESM层次聚类相同的聚类数量: {n_phylo_clusters}")
    
    phylo_clusters = extract_phylogenetic_clusters(tree, seq_ids, n_clusters=n_phylo_clusters)
    
    # 准备ESM聚类结果字典
    esm_clusters_dict = {seq_id: cluster for seq_id, cluster in zip(seq_ids, hclust_clusters)}
    
    # 计算差异指标和分歧序列
    print("计算聚类差异指标...")
    v_measure, ami, divergent_sequences, common_sequences = calculate_divergence(
        esm_clusters_dict, phylo_clusters
    )
    
    # 保存聚类分配结果
    results_df = pd.DataFrame({
        "sequence_id": seq_ids,
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
        "hcluster": hclust_clusters,
        "phylogeny_cluster": [phylo_clusters.get(seq_id, -1) for seq_id in seq_ids],
        "is_divergent": [1 if seq_id in divergent_sequences else 0 for seq_id in seq_ids]
    })
    results_df.to_csv(f"{output_prefix}_cluster_assignments.csv", index=False)
    print(f"聚类分配已保存至: {output_prefix}_cluster_assignments.csv")
    
    # 保存进化分析结果
    evolution_df = pd.DataFrame({
        "sequence_id": common_sequences,
        "v_measure": [v_measure] * len(common_sequences),
        "ami": [ami] * len(common_sequences)
    })
    evolution_df.to_csv(f"{output_prefix}_evolution_analysis.csv", index=False)
    print(f"进化分析结果已保存至: {output_prefix}_evolution_analysis.csv")
    
    # 保存分歧序列
    divergent_df = pd.DataFrame({"sequence_id": divergent_sequences})
    divergent_df.to_csv(f"{output_prefix}_divergent_sequences.csv", index=False)
    print(f"分歧序列列表已保存至: {output_prefix}_divergent_sequences.csv")
    
    # 保存距离矩阵（使用 squareform 转换为方阵）
    dist_df = pd.DataFrame(squareform(dist_matrix), index=seq_ids, columns=seq_ids)
    dist_df.to_csv(f"{output_prefix}_distance_matrix.csv")
    print(f"距离矩阵已保存至: {output_prefix}_distance_matrix.csv")
    
    # 计算系统发育树的patristic距离矩阵
    print("计算系统发育树的patristic距离矩阵...")
    phylo_dist_matrix = calculate_patroistic_distance_matrix(tree, seq_ids)
    phylo_dist_df = pd.DataFrame(phylo_dist_matrix, index=seq_ids, columns=seq_ids)
    phylo_dist_df.to_csv(f"{output_prefix}_phylogenetic_distance_matrix.csv")
    print(f"系统发育距离矩阵已保存至: {output_prefix}_phylogenetic_distance_matrix.csv")
    
    # 执行PCoA分析
    print("执行PCoA分析...")
    esm_dist_matrix_square = squareform(dist_matrix)
    esm_pcoa = perform_pcoa(esm_dist_matrix_square)
    phylo_pcoa = perform_pcoa(phylo_dist_matrix)
    
    # 保存PCoA结果
    pcoa_df = pd.DataFrame({
        "sequence_id": seq_ids,
        "esm_pcoa1": esm_pcoa[:, 0],
        "esm_pcoa2": esm_pcoa[:, 1],
        "phylo_pcoa1": phylo_pcoa[:, 0],
        "phylo_pcoa2": phylo_pcoa[:, 1]
    })
    pcoa_df.to_csv(f"{output_prefix}_pcoa_results.csv", index=False)
    print(f"PCoA结果已保存至: {output_prefix}_pcoa_results.csv")
    
    # 可视化
    print("生成可视化图表...")
    
    # 层次聚类树状图
    plot_dendrogram(linkage_matrix, seq_ids, f"{output_prefix}_hclust_dendrogram.png", 
                   f"({args.linkage} linkage, {args.metric} distance)")
    
    # 系统发育树可视化
    plot_phylogenetic_tree(tree, f"{output_prefix}_phylogeny_tree.png")
    
    # 聚类比较图（高亮分歧序列）
    plot_cluster_comparison(umap_coords, esm_clusters_dict, phylo_clusters, seq_ids, 
                           divergent_sequences, f"{output_prefix}_cluster_comparison.png")
    
    # 趋同进化热图
    plot_convergence_heatmap(embeddings, hclust_clusters, seq_ids, 
                            f"{output_prefix}_convergence_heatmap.png")
    
    # PCoA比较图
    plot_pcoa_comparison(esm_pcoa, phylo_pcoa, seq_ids, f"{output_prefix}_pcoa_comparison.png")
    
    # 绘制距离矩阵热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_df.iloc[:50, :50], cmap="viridis")  # 只显示前50个序列以免过于拥挤
    plt.title("Distance Matrix Heatmap (First 50 Sequences)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance_heatmap.png", dpi=300)
    plt.close()
    print(f"距离矩阵热图已保存至: {output_prefix}_distance_heatmap.png")
    
    # 导出桑基图数据表
    print("导出桑基图数据表...")
    export_sankey_data(hclust_clusters, phylo_clusters, seq_ids, output_prefix)
    
    # 生成桑基图
    print("生成桑基图...")
    plot_sankey_diagram(hclust_clusters, phylo_clusters, seq_ids, f"{output_prefix}_sankey_diagram.html")
    
    print("分析完成!")

if __name__ == "__main__":
    main()