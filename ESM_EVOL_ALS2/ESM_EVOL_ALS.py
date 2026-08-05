import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import time
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
from ete3 import Tree, TreeStyle, NodeStyle
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# 添加桑基图相关的导入
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: Plotly 未安装，无法生成桑基图。请使用 'pip install plotly' 安装。")


class ESM2FeatureExtractor:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """
        初始化ESM2模型和分词器
        
        参数:
            model_name: Hugging Face模型名称或路径
        """
        print(f"加载模型: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 设置为评估模式
        print(f"模型已加载到设备: {self.device}")
    
    def embed_sequences(self, sequences, batch_size=8, layer_index=33):
        """
        提取蛋白质序列的ESM2嵌入特征
        
        参数:
            sequences: 蛋白质序列列表
            batch_size: 批处理大小
            layer_index: 要提取的层索引（ESM2-t33使用33）
        
        返回:
            嵌入向量矩阵 (n_sequences, embedding_dim)
        """
        all_embeddings = []
        
        # 分批处理序列
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # 分词和编码
            inputs = self.tokenizer(
                batch_sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024  # ESM2最大长度
            ).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 获取指定层的隐藏状态
                hidden_states = outputs.hidden_states[layer_index]
                
                # 提取每个序列的[CLS]标记对应的特征（位于位置0）
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
            
            # 打印进度
            if (i // batch_size) % 10 == 0:
                print(f"已处理 {min(i+batch_size, len(sequences))}/{len(sequences)} 条序列")
        
        # 合并所有批次的嵌入
        embeddings = np.vstack(all_embeddings)
        print(f"特征矩阵形状: {embeddings.shape}")
        return embeddings

def parse_fasta(file_path):
    """解析FASTA文件，返回序列ID和序列列表"""
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))
    
    return seq_ids, sequences

def hierarchical_clustering(embeddings, method='average', metric='cosine', threshold=None):
    """
    执行层次聚类分析
    
    参数:
        embeddings: 嵌入向量矩阵
        method: 连接方法 ('average', 'ward', 'complete', 'single')
        metric: 距离度量 ('cosine', 'correlation', 'euclidean')
        threshold: 距离阈值，用于确定聚类数量
    
    返回:
        聚类标签数组和连接矩阵
    """
    print(f"执行层次聚类，方法: {method}, 度量: {metric}")
    
    # 数据标准化 - 对某些距离度量很重要
    if metric in ['correlation', 'cosine']:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    # 计算距离矩阵
    dist_matrix = pdist(embeddings, metric=metric)
    
    # 执行层次聚类
    Z = linkage(dist_matrix, method=method)
    
    # 确定聚类
    if threshold is not None:
        clusters = fcluster(Z, threshold, criterion='distance')
    else:
        # 使用默认方法确定聚类数量
        clusters = fcluster(Z, t=0.7 * max(Z[:, 2]), criterion='distance')
    
    unique_clusters = len(np.unique(clusters))
    print(f"层次聚类完成，共得到 {unique_clusters} 个簇")
    return clusters, Z, dist_matrix, unique_clusters

def parse_phylogenetic_tree(tree_file):
    """
    解析系统发育树文件
    
    参数:
        tree_file: 系统发育树文件路径(Newick格式)
    
    返回:
        ETE Tree 对象
    """
    try:
        tree = Tree(tree_file)
        print(f"成功解析系统发育树，包含 {len(tree.get_leaf_names())} 个叶节点")
        return tree
    except Exception as e:
        print(f"解析系统发育树时出错: {e}")
        return None

def validate_monophyly(tree, clusters, sequences):
    """
    验证每个簇是否都是单系群
    
    参数:
        tree: ETE Tree 对象
        clusters: 聚类字典
        sequences: 序列ID列表
    """
    cluster_groups = {}
    for seq_id, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(seq_id)
    
    monophyletic_count = 0
    total_monophyletic_sequences = 0
    for cluster_id, seq_list in cluster_groups.items():
        if len(seq_list) > 1:
            try:
                # 找到这些序列的最近共同祖先
                mrca = tree.get_common_ancestor(seq_list)
                
                # 检查MRCA的所有后代是否都属于同一簇
                mrca_leaves = mrca.get_leaf_names()
                if set(mrca_leaves).issubset(set(seq_list)):
                    monophyletic_count += 1
                    total_monophyletic_sequences += len(seq_list)
                    print(f"簇 {cluster_id} 是单系群 (包含 {len(seq_list)} 个序列)")
                else:
                    extra_seqs = set(mrca_leaves) - set(seq_list)
                    print(f"警告: 簇 {cluster_id} 不是单系群")
                    print(f"  - MRCA包含 {len(mrca_leaves)} 个序列，但簇只有 {len(seq_list)} 个序列")
                    print(f"  - 额外序列: {len(extra_seqs)} 个")
            except Exception as e:
                print(f"验证簇 {cluster_id} 单系性时出错: {e}")
        else:
            # 单序列簇自动视为单系
            monophyletic_count += 1
            total_monophyletic_sequences += 1
            print(f"簇 {cluster_id} 是单序列簇")
    
    total_sequences = len(sequences)
    monophyly_percentage = (total_monophyletic_sequences / total_sequences) * 100 if total_sequences > 0 else 0
    
    print(f"单系群比例: {monophyletic_count}/{len(cluster_groups)} 个簇 ({monophyly_percentage:.1f}% 的序列在单系群中)")

def extract_phylogenetic_clusters(tree, sequences, n_clusters=None):
    """
    改进的系统发育树聚类算法：先按ESM簇数目生成簇，再分割非单系群
    
    参数:
        tree: ETE Tree 对象
        sequences: 序列ID列表
        n_clusters: 期望的簇数量
    
    返回:
        基于系统发育树的聚类标签字典
    """
    print("使用改进的两步聚类算法")
    
    # 获取树中的所有叶节点
    leaves = tree.get_leaf_names()
    
    # 确保所有序列都在树中
    missing_seqs = set(sequences) - set(leaves)
    if missing_seqs:
        print(f"警告: {len(missing_seqs)} 个序列在系统发育树中找不到")
        sequences = [seq for seq in sequences if seq in leaves]
    
    # 第一步：使用动态规划算法生成与ESM相同数量的簇
    print(f"第一步：使用动态规划算法生成 {n_clusters} 个簇")
    clusters = dynamic_programming_clustering(tree, sequences, n_clusters)
    
    # 第二步：验证并修复非单系群
    print("第二步：验证并修复非单系群")
    clusters = fix_non_monophyletic_clusters(tree, clusters, sequences)
    
    # 重新编号簇ID，使其从1开始连续
    clusters = renumber_clusters(clusters)
    
    final_cluster_count = len(set(clusters.values()))
    print(f"最终得到 {final_cluster_count} 个单系群簇")
    
    # 最终验证单系性
    validate_monophyly(tree, clusters, sequences)
    
    return clusters

def dynamic_programming_clustering(tree, sequences, n_clusters):
    """
    使用动态规划算法将树划分为指定数量的单系群
    
    参数:
        tree: ETE Tree 对象
        sequences: 序列ID列表
        n_clusters: 目标簇数量
    
    返回:
        聚类字典
    """
    print(f"使用动态规划算法生成 {n_clusters} 个簇")
    
    # 获取所有可能的单系群
    all_monophyletic_groups = find_all_monophyletic_groups(tree, sequences)
    
    # 使用贪心算法选择最优的单系群组合
    selected_groups = select_optimal_groups(all_monophyletic_groups, n_clusters, len(sequences),tree)
    
    # 为每个选定的单系群分配簇ID
    clusters = {}
    used_sequences = set()
    
    for cluster_id, group in enumerate(selected_groups):
        actual_cluster_id = cluster_id + 1
        for seq in group:
            if seq not in used_sequences:
                clusters[seq] = actual_cluster_id
                used_sequences.add(seq)
    
    # 处理未分配的序列（如果有）
    unassigned = set(sequences) - used_sequences
    if unassigned:
        print(f"有 {len(unassigned)} 个序列未分配，将它们分配到最近的簇")
        clusters = assign_remaining_sequences(tree, clusters, unassigned)
    
    return clusters

def find_all_monophyletic_groups(tree, sequences, min_size=1):
    """
    找到树中所有可能的单系群
    
    参数:
        tree: ETE Tree 对象
        sequences: 序列ID列表
        min_size: 最小群大小
    
    返回:
        单系群列表
    """
    monophyletic_groups = []
    
    # 遍历所有内部节点
    for node in tree.traverse("preorder"):
        if not node.is_leaf():
            # 获取该节点下的所有叶节点
            group_leaves = node.get_leaf_names()
            
            # 只考虑包含在目标序列中的叶节点
            group_leaves = [leaf for leaf in group_leaves if leaf in sequences]
            
            # 检查大小限制
            if len(group_leaves) >= min_size:
                monophyletic_groups.append(group_leaves)
    
    # 按大小排序
    monophyletic_groups.sort(key=len, reverse=True)
    
    print(f"找到 {len(monophyletic_groups)} 个可能的单系群")
    return monophyletic_groups

def select_optimal_groups(all_groups, target_clusters, total_sequences, tree):
    """
    使用贪心算法选择最优的单系群组合
    
    参数:
        all_groups: 所有单系群列表
        target_clusters: 目标簇数量
        total_sequences: 总序列数
    
    返回:
        选定的单系群列表
    """
    # 按大小排序
    sorted_groups = sorted(all_groups, key=len, reverse=True)
    
    selected_groups = []
    covered_sequences = set()
    
    # 选择前k-1个最大的不重叠单系群
    for group in sorted_groups:
        if len(selected_groups) >= target_clusters - 1:
            break
            
        group_set = set(group)
        
        # 检查是否与已选群重叠
        if not any(group_set & set(selected) for selected in selected_groups):
            selected_groups.append(group)
            covered_sequences |= group_set
    
    # 最后一个簇包含所有未覆盖的序列
    all_sequences = set()
    for group in all_groups:
        all_sequences |= set(group)
    
    remaining_sequences = all_sequences - covered_sequences
    if remaining_sequences:
        selected_groups.append(list(remaining_sequences))
    
    # 如果簇数量不够，分割最大的簇
    while len(selected_groups) < target_clusters and selected_groups:
        # 找到最大的簇
        largest_idx = max(range(len(selected_groups)), key=lambda i: len(selected_groups[i]))
        largest_group = selected_groups[largest_idx]
        
        if len(largest_group) <= 1:
            break
            
        # 尝试分割最大的簇
        subgroup1, subgroup2 = split_monophyletic_group(tree, largest_group)
        
        if subgroup1 and subgroup2:
            # 替换原来的簇为两个子簇
            del selected_groups[largest_idx]
            selected_groups.append(subgroup1)
            selected_groups.append(subgroup2)
        else:
            break
    
    print(f"选择了 {len(selected_groups)} 个单系群，覆盖 {len(covered_sequences)} 个序列")
    return selected_groups

def split_monophyletic_group(tree, group):
    """
    将一个单系群分割为两个单系群
    
    参数:
        tree: ETE Tree 对象
        group: 单系群序列列表
    
    返回:
        两个子群（如果可能分割）
    """
    if len(group) <= 1:
        return None, None
    
    # 找到这些序列的MRCA
    mrca = tree.get_common_ancestor(group)
    
    # 获取MRCA的直接子节点
    children = mrca.get_children()
    
    if len(children) < 2:
        return None, None
    
    # 计算每个子节点下的群序列
    subgroup1 = []
    subgroup2 = []
    
    for child in children:
        child_leaves = set(child.get_leaf_names())
        group_leaves_in_child = child_leaves & set(group)
        
        if group_leaves_in_child:
            if not subgroup1:
                subgroup1 = list(group_leaves_in_child)
            elif not subgroup2:
                subgroup2 = list(group_leaves_in_child)
            else:
                # 如果有超过两个子节点，将额外的序列添加到较小的子群
                if len(subgroup1) <= len(subgroup2):
                    subgroup1.extend(list(group_leaves_in_child))
                else:
                    subgroup2.extend(list(group_leaves_in_child))
    
    return subgroup1, subgroup2

def assign_remaining_sequences(tree, clusters, unassigned_sequences):
    """
    将未分配的序列分配到最近的簇
    
    参数:
        tree: ETE Tree 对象
        clusters: 当前聚类字典
        unassigned_sequences: 未分配的序列集合
    
    返回:
        更新后的聚类字典
    """
    # 创建簇到序列的映射
    cluster_sequences = {}
    for seq, cluster_id in clusters.items():
        if cluster_id not in cluster_sequences:
            cluster_sequences[cluster_id] = []
        cluster_sequences[cluster_id].append(seq)
    
    # 为每个未分配序列找到最近的簇
    for seq in unassigned_sequences:
        try:
            seq_node = tree & seq
            min_distance = float('inf')
            nearest_cluster = 0
            
            for cluster_id, cluster_seqs in cluster_sequences.items():
                # 计算与簇中所有序列的平均距离
                total_distance = 0
                count = 0
                
                for cluster_seq in cluster_seqs:
                    try:
                        cluster_node = tree & cluster_seq
                        distance = tree.get_distance(seq_node, cluster_node)
                        total_distance += distance
                        count += 1
                    except:
                        continue
                
                if count > 0:
                    avg_distance = total_distance / count
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        nearest_cluster = cluster_id
            
            clusters[seq] = nearest_cluster
        except Exception as e:
            print(f"无法为序列 {seq} 找到最近簇: {e}")
            # 分配到第一个簇
            clusters[seq] = 1
    
    return clusters

def fix_non_monophyletic_clusters(tree, clusters, sequences):
    """
    修复非单系群，将其分割为单系群
    
    参数:
        tree: ETE Tree 对象
        clusters: 当前聚类字典
        sequences: 序列ID列表
    
    返回:
        修复后的聚类字典
    """
    print("修复非单系群...")
    
    # 获取簇分组
    cluster_groups = {}
    for seq_id, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(seq_id)
    
    fixed_clusters = clusters.copy()
    new_cluster_id = max(clusters.values()) + 1
    
    # 检查每个簇的单系性
    for cluster_id, seq_list in cluster_groups.items():
        if len(seq_list) <= 1:
            continue  # 单序列簇自动是单系的
            
        try:
            # 找到这些序列的最近共同祖先
            mrca = tree.get_common_ancestor(seq_list)
            
            # 检查MRCA的所有后代是否都属于同一簇
            mrca_leaves = mrca.get_leaf_names()
            cluster_set = set(seq_list)
            
            if not set(mrca_leaves).issubset(cluster_set):
                # 非单系群，需要分割
                print(f"簇 {cluster_id} 是非单系群，包含 {len(seq_list)} 个序列")
                
                # 简单分割：将非单系群分割为两个单系子群
                subgroups = simple_split_non_monophyletic(tree, seq_list, mrca)
                
                if len(subgroups) > 1:
                    # 为每个子群分配新的簇ID
                    for subgroup in subgroups:
                        for seq in subgroup:
                            fixed_clusters[seq] = new_cluster_id
                        new_cluster_id += 1
                    print(f"  - 成功分割为 {len(subgroups)} 个单系子群")
                else:
                    print(f"  - 无法进一步分割，保持原状")
                    
        except Exception as e:
            print(f"验证簇 {cluster_id} 单系性时出错: {e}")
    
    return fixed_clusters

def simple_split_non_monophyletic(tree, sequences, mrca):
    """
    简单分割非单系群为单系子群
    
    参数:
        tree: ETE Tree 对象
        sequences: 非单系群序列列表
        mrca: 最近共同祖先节点
    
    返回:
        单系子群列表
    """
    # 特殊情况：如果只有两个序列，直接分成两个单序列簇
    if len(sequences) == 2:
        return [[sequences[0]], [sequences[1]]]
    
    subgroups = []
    processed_sequences = set()
    
    # 获取MRCA的直接子节点
    children = mrca.get_children()
    
    # 对每个子节点，找到属于目标序列的单系群
    for child in children:
        child_leaves = set(child.get_leaf_names())
        sequences_in_child = child_leaves & set(sequences)
        
        if sequences_in_child and not sequences_in_child.issubset(processed_sequences):
            # 检查这个子群是否单系
            try:
                child_mrca = tree.get_common_ancestor(list(sequences_in_child))
                child_mrca_leaves = set(child_mrca.get_leaf_names())
                
                if child_mrca_leaves.issubset(set(sequences)):
                    # 这个子群是单系的
                    subgroups.append(list(sequences_in_child))
                    processed_sequences.update(sequences_in_child)
                else:
                    # 递归分割非单系子群
                    recursive_subgroups = simple_split_non_monophyletic(tree, list(sequences_in_child), child_mrca)
                    subgroups.extend(recursive_subgroups)
                    processed_sequences.update(sequences_in_child)
            except Exception as e:
                print(f"分割子群时出错: {e}")
                # 如果出错，直接作为一个子群
                subgroups.append(list(sequences_in_child))
                processed_sequences.update(sequences_in_child)
    
    # 处理未被任何子节点覆盖的序列
    remaining_sequences = set(sequences) - processed_sequences
    if remaining_sequences:
        # 为剩余序列创建单独的子群
        for seq in remaining_sequences:
            subgroups.append([seq])
    
    # 如果无法分割，返回原始序列作为单个簇
    if not subgroups:
        subgroups = [sequences]
    
    return subgroups

def renumber_clusters(clusters):
    """
    重新编号簇ID，使其从1开始连续
    
    参数:
        clusters: 聚类字典
    
    返回:
        重新编号后的聚类字典
    """
    unique_clusters = sorted(set(clusters.values()))
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters, 1)}
    
    return {seq_id: cluster_mapping[cluster_id] for seq_id, cluster_id in clusters.items()}

def calculate_divergence(esm_clusters, phylo_clusters):
    """
    计算ESM聚类与系统发育聚类之间的差异
    
    参数:
        esm_clusters: ESM聚类结果字典 {序列ID: 簇标签}
        phylo_clusters: 系统发育聚类结果字典 {序列ID: 簇标签}
    
    返回:
        V-measure, AMI, 分歧序列列表
    """
    # 确保两个聚类结果的序列顺序一致
    common_sequences = list(set(esm_clusters.keys()) & set(phylo_clusters.keys()))
    
    # 确保顺序一致地构建标签列表
    labels_esm = []
    labels_phylo = []
    for seq_id in common_sequences:
        labels_esm.append(esm_clusters[seq_id])
        labels_phylo.append(phylo_clusters[seq_id])

    # 计算V-measure和AMI
    v_measure = v_measure_score(labels_phylo, labels_esm)
    ami = adjusted_mutual_info_score(labels_phylo, labels_esm)
    
    print(f"V-measure: {v_measure:.3f}")
    print(f"Adjusted Mutual Information (AMI): {ami:.3f}")
    
    # 识别分歧序列 (在两个树中分配到不同簇的序列)
    divergent_sequences = []
    for seq_id in common_sequences:
        if esm_clusters[seq_id] != phylo_clusters[seq_id]:
            divergent_sequences.append(seq_id)
    
    print(f"发现 {len(divergent_sequences)} 个分歧序列")
    
    return v_measure, ami, divergent_sequences, common_sequences

def export_sankey_data(esm_clusters, phylo_clusters, seq_ids, output_prefix):
    """
    导出桑基图数据表，包括每个簇的基因列表和流量基因列表
    
    参数:
        esm_clusters: ESM聚类标签数组
        phylo_clusters: 系统发育聚类字典
        seq_ids: 序列ID列表
        output_prefix: 输出文件前缀
    """
    print("导出桑基图数据表...")
    
    # 准备数据
    esm_labels = [f"ESM_Cluster_{c}" for c in esm_clusters]
    phylo_labels = [f"Phylo_Cluster_{phylo_clusters[seq_id]}" for seq_id in seq_ids]
    
    # 创建源节点和目标节点的映射
    all_nodes = list(set(esm_labels)) + list(set(phylo_labels))
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # 1. 导出ESM2聚类基因列表
    esm_cluster_genes = {}
    for i, (seq_id, cluster_label) in enumerate(zip(seq_ids, esm_labels)):
        if cluster_label not in esm_cluster_genes:
            esm_cluster_genes[cluster_label] = []
        esm_cluster_genes[cluster_label].append(seq_id)
    
    esm_data = []
    for cluster, genes in esm_cluster_genes.items():
        esm_data.append({
            "Cluster": cluster,
            "Gene_Count": len(genes),
            "Gene_List": "; ".join(genes)
        })
    
    esm_df = pd.DataFrame(esm_data)
    esm_df.to_csv(f"{output_prefix}_esm_clusters_genes.csv", index=False)
    print(f"ESM2聚类基因列表已保存至: {output_prefix}_esm_clusters_genes.csv")
    
    # 2. 导出系统发育树聚类基因列表
    phylo_cluster_genes = {}
    for seq_id in seq_ids:
        cluster_label = f"Phylo_Cluster_{phylo_clusters[seq_id]}"
        if cluster_label not in phylo_cluster_genes:
            phylo_cluster_genes[cluster_label] = []
        phylo_cluster_genes[cluster_label].append(seq_id)
    
    phylo_data = []
    for cluster, genes in phylo_cluster_genes.items():
        phylo_data.append({
            "Cluster": cluster,
            "Gene_Count": len(genes),
            "Gene_List": "; ".join(genes)
        })
    
    phylo_df = pd.DataFrame(phylo_data)
    phylo_df.to_csv(f"{output_prefix}_phylogenetic_clusters_genes.csv", index=False)
    print(f"系统发育树聚类基因列表已保存至: {output_prefix}_phylogenetic_clusters_genes.csv")
    
    # 3. 导出流量基因列表
    flow_data = {}
    for i, (seq_id, esm_label, phylo_label) in enumerate(zip(seq_ids, esm_labels, phylo_labels)):
        flow_key = (esm_label, phylo_label)
        if flow_key not in flow_data:
            flow_data[flow_key] = []
        flow_data[flow_key].append(seq_id)
    
    flow_list = []
    for (esm_cluster, phylo_cluster), genes in flow_data.items():
        flow_list.append({
            "ESM_Cluster": esm_cluster,
            "Phylogenetic_Cluster": phylo_cluster,
            "Flow_Count": len(genes),
            "Gene_List": "; ".join(genes)
        })
    
    flow_df = pd.DataFrame(flow_list)
    flow_df = flow_df.sort_values(["ESM_Cluster", "Phylogenetic_Cluster"])
    flow_df.to_csv(f"{output_prefix}_sankey_flow_genes.csv", index=False)
    print(f"桑基图流量基因列表已保存至: {output_prefix}_sankey_flow_genes.csv")
    
    # 4. 导出详细的连接数据（用于高级分析）
    detailed_flow_data = []
    for i, (seq_id, esm_label, phylo_label) in enumerate(zip(seq_ids, esm_labels, phylo_labels)):
        detailed_flow_data.append({
            "Gene_ID": seq_id,
            "ESM_Cluster": esm_label,
            "Phylogenetic_Cluster": phylo_label,
            "ESM_Cluster_ID": esm_clusters[i],
            "Phylogenetic_Cluster_ID": phylo_clusters[seq_id],
            "Is_Divergent": 1 if esm_clusters[i] != phylo_clusters[seq_id] else 0
        })
    
    detailed_flow_df = pd.DataFrame(detailed_flow_data)
    detailed_flow_df.to_csv(f"{output_prefix}_sankey_detailed_assignments.csv", index=False)
    print(f"详细桑基图分配数据已保存至: {output_prefix}_sankey_detailed_assignments.csv")
    
    return esm_df, phylo_df, flow_df

def plot_sankey_diagram(esm_clusters, phylo_clusters, seq_ids, output_path):
    """
    绘制桑基图，显示ESM2聚类和系统发育树聚类之间的关系
    
    参数:
        esm_clusters: ESM聚类标签数组
        phylo_clusters: 系统发育聚类字典
        seq_ids: 序列ID列表
        output_path: 输出文件路径
    """
    if not PLOTLY_AVAILABLE:
        print("警告: Plotly 不可用，跳过桑基图生成")
        return
    
    # 准备数据
    esm_labels = [f"ESM_Cluster_{c}" for c in esm_clusters]
    phylo_labels = [f"Phylo_Cluster_{phylo_clusters[seq_id]}" for seq_id in seq_ids]
    
    # 创建源节点和目标节点的映射
    all_nodes = list(set(esm_labels)) + list(set(phylo_labels))
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # 计算连接权重
    link_counts = {}
    for esm_label, phylo_label in zip(esm_labels, phylo_labels):
        key = (node_dict[esm_label], node_dict[phylo_label])
        link_counts[key] = link_counts.get(key, 0) + 1
    
    # 准备桑基图数据
    source = []
    target = []
    value = []
    
    for (src, tgt), count in link_counts.items():
        source.append(src)
        target.append(tgt)
        value.append(count)
    
    # 节点颜色 - 使用指定的颜色
    n_esm_nodes = len(set(esm_labels))
    n_phylo_nodes = len(set(phylo_labels))
    
    node_colors = []
    # ESM2聚类节点颜色 (#0076B4)
    for i in range(n_esm_nodes):
        node_colors.append("#0076B4")
    # 系统发育树聚类节点颜色 (#D6641E)
    for i in range(n_phylo_nodes):
        node_colors.append("#D6641E")
    
    # 链接颜色 - 使用渐变
    link_colors = []
    for src in source:
        # 计算从ESM蓝色到系统发育橙色的渐变
        if src < n_esm_nodes:  # 确保源节点是ESM聚类
            # 简单的线性插值
            ratio = 0.5  # 使用中间值
            color = f"rgba({int(0 * (1-ratio) + 214 * ratio)}, {int(118 * (1-ratio) + 100 * ratio)}, {int(180 * (1-ratio) + 30 * ratio)}, 0.6)"
            link_colors.append(color)
        else:
            link_colors.append("rgba(200, 200, 200, 0.6)")
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        ))])
    
    fig.update_layout(
        title_text="ESM2聚类 vs 系统发育树聚类 - 桑基图",
        font_size=12,
        width=1200,
        height=800
    )
    
    # 保存为HTML文件
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"桑基图已保存至: {output_path}")

def plot_dendrogram(linkage_matrix, labels, output_path, title_suffix=""):
    """
    绘制层次聚类树状图
    
    参数:
        linkage_matrix: 连接矩阵
        labels: 叶节点标签
        output_path: 输出文件路径
        title_suffix: 标题后缀
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
    
    参数:
        tree: ETE Tree 对象
        output_path: 输出文件路径
    """
    # 设置树样式
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
    
    参数:
        umap_coords: UMAP坐标
        esm_clusters: ESM聚类标签字典
        phylo_clusters: 系统发育聚类标签字典
        seq_ids: 序列ID列表
        divergent_sequences: 分歧序列列表
        output_path: 输出文件路径
    """
    # 准备数据
    esm_labels = [esm_clusters.get(seq_id, -1) for seq_id in seq_ids]
    phylo_labels = [phylo_clusters.get(seq_id, -1) for seq_id in seq_ids]
    
    # 创建分歧序列的布尔掩码
    is_divergent = [seq_id in divergent_sequences for seq_id in seq_ids]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ESM聚类结果
    scatter1 = ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                          c=esm_labels, cmap="Spectral", s=30, alpha=0.8)
    
    # 高亮显示分歧序列
    ax1.scatter(umap_coords[is_divergent, 0], umap_coords[is_divergent, 1],
               s=100, facecolors='none', edgecolors='black', linewidths=2, 
               marker='o', label='Divergent Sequences')
    
    ax1.set_title("ESM-based Clustering (Divergent Sequences Highlighted)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1)
    
    # 系统发育聚类结果
    scatter2 = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                          c=phylo_labels, cmap="Spectral", s=30, alpha=0.8)
    
    # 高亮显示分歧序列
    ax2.scatter(umap_coords[is_divergent, 0], umap_coords[is_divergent, 1],
               s=100, facecolors='none', edgecolors='black', linewidths=2, 
               marker='o', label='Divergent Sequences')
    
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
    
    参数:
        embeddings: 嵌入向量矩阵
        clusters: 聚类标签数组
        seq_ids: 序列ID列表
        output_path: 输出文件路径
    """
    # 按聚类标签排序
    sorted_indices = np.argsort(clusters)
    sorted_embeddings = embeddings[sorted_indices]
    sorted_clusters = clusters[sorted_indices]
    sorted_seq_ids = [seq_ids[i] for i in sorted_indices]
    
    # 计算相关性矩阵
    correlation_matrix = np.corrcoef(sorted_embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    plt.title("Protein Sequence Similarity Heatmap")
    plt.xlabel("Protein Sequences")
    plt.ylabel("Protein Sequences")
    
    # 添加聚类边界
    unique_clusters = np.unique(sorted_clusters)
    cluster_boundaries = []
    current_cluster = sorted_clusters[0]
    
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

def tree_to_linkage(tree, seq_ids):
    """
    将系统发育树转换为连接矩阵格式
    
    参数:
        tree: ETE Tree 对象
        seq_ids: 序列ID列表，用于确定顺序
    
    返回:
        连接矩阵
    """
    # 获取所有叶节点
    leaves = tree.get_leaf_names()
    
    # 创建一个映射：序列名 -> 索引
    idx_map = {name: i for i, name in enumerate(seq_ids)}
    
    # 初始化连接矩阵
    Z = []
    
    # 递归遍历树，构建连接矩阵
    def traverse(node, node_id):
        nonlocal Z
        if node.is_leaf():
            return idx_map[node.name], 0.0  # 返回叶节点的索引和高度0
        
        children = node.get_children()
        if len(children) != 2:
            # 对于多分枝的树，我们需要二分化
            # 这里简化处理，只取前两个子节点
            children = children[:2]
        
        left_child, left_height = traverse(children[0], node_id)
        right_child, right_height = traverse(children[1], node_id)
        
        # 当前节点的高度是子节点高度的最大值加上分支长度
        current_height = max(left_height, right_height) + (node.dist or 1.0)
        
        # 添加到连接矩阵
        Z.append([left_child, right_child, current_height, len(node.get_leaves())])
        
        return node_id, current_height
    
    # 开始遍历
    traverse(tree, len(seq_ids))
    
    # 转换为numpy数组
    Z = np.array(Z)
    return Z

def calculate_patroistic_distance_matrix(tree, sequences):
    """
    计算系统发育树的patristic距离矩阵
    
    参数:
        tree: ETE Tree 对象
        sequences: 序列ID列表
    
    返回:
        patristic距离矩阵
    """
    n_seqs = len(sequences)
    dist_matrix = np.zeros((n_seqs, n_seqs))
    
    # 创建序列ID到索引的映射
    id_to_idx = {seq_id: i for i, seq_id in enumerate(sequences)}
    
    # 计算每对序列之间的patristic距离
    for i, seq_id1 in enumerate(sequences):
        for j, seq_id2 in enumerate(sequences):
            if i < j:  # 只计算上三角部分
                try:
                    node1 = tree & seq_id1
                    node2 = tree & seq_id2
                    dist = tree.get_distance(node1, node2)
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                except Exception as e:
                    print(f"无法计算 {seq_id1} 和 {seq_id2} 之间的距离: {e}")
                    # 如果无法计算距离，使用一个较大的默认值
                    dist_matrix[i, j] = 10.0
                    dist_matrix[j, i] = 10.0
    
    return dist_matrix

def perform_pcoa(distance_matrix, n_components=2):
    """
    执行PCoA分析
    
    参数:
        distance_matrix: 距离矩阵
        n_components: 主坐标数量
    
    返回:
        PCoA坐标
    """
    # 确保距离矩阵是二维的
    if distance_matrix.ndim == 1:
        distance_matrix = squareform(distance_matrix)
    
    # 中心化距离矩阵
    n = distance_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ distance_matrix**2 @ H
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # 按特征值降序排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 选择前n_components个主坐标
    coordinates = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
    
    return coordinates

def plot_pcoa_comparison(esm_coords, phylo_coords, seq_ids, output_path):
    """
    绘制ESM和系统发育PCoA结果的比较图
    
    参数:
        esm_coords: ESM PCoA坐标
        phylo_coords: 系统发育PCoA坐标
        seq_ids: 序列ID列表
        output_path: 输出文件路径
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制ESM PCoA结果
    scatter1 = ax.scatter(esm_coords[:, 0], esm_coords[:, 1], 
                         c='blue', alpha=0.7, s=50, label='ESM PCoA')
    
    # 绘制系统发育PCoA结果
    scatter2 = ax.scatter(phylo_coords[:, 0], phylo_coords[:, 1], 
                         c='red', alpha=0.7, s=50, label='Phylogeny PCoA')
    
    # 连接相同的样本点
    for i in range(len(esm_coords)):
        ax.plot([esm_coords[i, 0], phylo_coords[i, 0]], 
                [esm_coords[i, 1], phylo_coords[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    # 添加地毯图
    for i in range(len(esm_coords)):
        ax.plot([esm_coords[i, 0], esm_coords[i, 0]], 
                [esm_coords[i, 1]-0.02, esm_coords[i, 1]+0.02], 
                'blue', alpha=0.5, linewidth=1)
        ax.plot([phylo_coords[i, 0]-0.02, phylo_coords[i, 0]+0.02], 
                [phylo_coords[i, 1], phylo_coords[i, 1]], 
                'red', alpha=0.5, linewidth=1)
    
    # 添加凸包
    try:
        esm_hull = ConvexHull(esm_coords)
        phylo_hull = ConvexHull(phylo_coords)
        
        for simplex in esm_hull.simplices:
            ax.plot(esm_coords[simplex, 0], esm_coords[simplex, 1], 'b--', alpha=0.5)
        
        for simplex in phylo_hull.simplices:
            ax.plot(phylo_coords[simplex, 0], phylo_coords[simplex, 1], 'r--', alpha=0.5)
    except:
        print("无法计算凸包，可能点太少了")
    
    ax.set_xlabel("PCoA 1")
    ax.set_ylabel("PCoA 2")
    ax.set_title("PCoA Comparison: ESM vs Phylogenetic Tree")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"PCoA比较图已保存至: {output_path}")

def save_linkage_matrix(linkage_matrix, seq_ids, output_path):
    """
    保存连接矩阵到CSV文件，并添加序列ID信息
    
    参数:
        linkage_matrix: 连接矩阵
        seq_ids: 序列ID列表
        output_path: 输出文件路径
    """
    # 创建连接矩阵的DataFrame
    linkage_df = pd.DataFrame(linkage_matrix, 
                             columns=['cluster1', 'cluster2', 'distance', 'sample_count'])
    
    # 添加序列ID信息
    n_seqs = len(seq_ids)
    
    # 创建一个映射，将索引映射到序列ID（对于原始序列）
    id_map = {i: seq_ids[i] for i in range(n_seqs)}
    
    # 对于连接矩阵中的簇，创建一个映射
    cluster_map = {}
    for i, row in enumerate(linkage_df.iterrows()):
        idx = i + n_seqs  # 簇的索引从n_seqs开始
        cluster_map[idx] = f"cluster_{i}"
    
    # 合并映射
    full_map = {**id_map, **cluster_map}
    
    # 添加序列ID列
    linkage_df['cluster1_id'] = linkage_df['cluster1'].map(full_map)
    linkage_df['cluster2_id'] = linkage_df['cluster2'].map(full_map)
    
    # 保存到CSV
    linkage_df.to_csv(output_path, index=False)
    print(f"连接矩阵已保存至: {output_path}")

def linkage_to_newick(Z, labels):
    """
    将层次聚类的连接矩阵转换为Newick格式的树字符串
    
    参数:
        Z: 连接矩阵
        labels: 叶节点标签列表
    
    返回:
        Newick格式的树字符串
    """
    # 使用scipy的to_tree函数将连接矩阵转换为树结构
    tree = to_tree(Z, rd=False)
    
    # 递归函数构建Newick字符串
    def build_newick(node, parent_dist):
        if node.is_leaf():
            # 叶节点：返回序列ID和距离
            return f"{labels[node.id]}:{parent_dist - node.dist:.6f}"
        else:
            # 内部节点：递归构建左右子树的Newick字符串
            left = build_newick(node.left, node.dist)
            right = build_newick(node.right, node.dist)
            return f"({left},{right}):{parent_dist - node.dist:.6f}"
    
    # 从根节点开始构建Newick字符串
    newick_str = build_newick(tree, tree.dist) + ";"
    return newick_str

def save_newick_tree(newick_str, output_path):
    """
    将Newick字符串保存到文件
    
    参数:
        newick_str: Newick格式的树字符串
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        f.write(newick_str)
    print(f"Newick树已保存至: {output_path}")

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
    # 新增参数：系统发育树聚类数量
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
    
    # 保存距离矩阵
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
    # 确保距离矩阵是方阵形式
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