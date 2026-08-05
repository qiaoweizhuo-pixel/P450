import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from ete3 import Tree
import warnings
warnings.filterwarnings('ignore')

def hierarchical_clustering(embeddings, method='average', metric='cosine', threshold=None):
    """
    执行层次聚类分析
    """
    print(f"执行层次聚类，方法: {method}, 度量: {metric}")
    
    if metric in ['correlation', 'cosine']:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    dist_matrix = pdist(embeddings, metric=metric)
    Z = linkage(dist_matrix, method=method)
    
    if threshold is not None:
        clusters = fcluster(Z, threshold, criterion='distance')
    else:
        # 默认使用 70% 的最大高度作为阈值
        clusters = fcluster(Z, t=0.7 * max(Z[:, 2]), criterion='distance')
    
    unique_clusters = len(np.unique(clusters))
    print(f"层次聚类完成，共得到 {unique_clusters} 个簇")
    return clusters, Z, dist_matrix, unique_clusters

def parse_phylogenetic_tree(tree_file):
    """
    解析系统发育树文件
    """
    try:
        tree = Tree(tree_file)
        print(f"成功解析系统发育树，包含 {len(tree.get_leaf_names())} 个叶节点")
        return tree
    except Exception as e:
        print(f"解析系统发育树时出错: {e}")
        return None

def validate_monophyly(tree, clusters, sequences):
    """验证每个簇是否都是单系群 (仅打印摘要信息)"""
    cluster_groups = {}
    for seq_id, cluster_id in clusters.items():
        cluster_groups.setdefault(cluster_id, []).append(seq_id)
    
    monophyletic_count = 0
    for cluster_id, seq_list in cluster_groups.items():
        if len(seq_list) > 1:
            try:
                mrca = tree.get_common_ancestor(seq_list)
                mrca_leaves = mrca.get_leaf_names()
                # 检查 MRCA 的所有后代是否都在当前簇中
                if set(mrca_leaves).issubset(set(seq_list)):
                    monophyletic_count += 1
            except:
                pass 
        else:
            monophyletic_count += 1
    
    monophyly_percentage = (monophyletic_count / len(cluster_groups)) * 100 if len(cluster_groups) > 0 else 0
    
    print(f"单系群比例: {monophyletic_count}/{len(cluster_groups)} 个簇 ({monophyly_percentage:.1f}% 的序列在单系群中)")

def renumber_clusters(clusters):
    """重新编号簇ID，使其从1开始连续"""
    unique_clusters = sorted(set(clusters.values()))
    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters, 1)}
    return {seq_id: cluster_mapping[cluster_id] for seq_id, cluster_id in clusters.items()}

# --- 辅助函数（动态规划和修复） ---

def split_monophyletic_group(tree, group):
    if len(group) <= 1:
        return None, None
    try:
        mrca = tree.get_common_ancestor(group)
    except:
        return None, None
        
    children = mrca.get_children()
    if len(children) < 2:
        return None, None
    
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
                if len(subgroup1) <= len(subgroup2):
                    subgroup1.extend(list(group_leaves_in_child))
                else:
                    subgroup2.extend(list(group_leaves_in_child))
    
    if not subgroup1 or not subgroup2:
        return None, None
        
    return subgroup1, subgroup2

def find_all_monophyletic_groups(tree, sequences, min_size=1):
    monophyletic_groups = []
    for node in tree.traverse("preorder"):
        if not node.is_leaf():
            group_leaves = [leaf for leaf in node.get_leaf_names() if leaf in sequences]
            if len(group_leaves) >= min_size:
                monophyletic_groups.append(group_leaves)
    
    monophyletic_groups.sort(key=len, reverse=True)
    # *** 修复点: 拼写错误修正 ***
    print(f"找到 {len(monophyletic_groups)} 个可能的单系群")
    return monophyletic_groups

def select_optimal_groups(all_groups, target_clusters, sequences, tree):
    sorted_groups = sorted(all_groups, key=len, reverse=True)
    selected_groups = []
    covered_sequences = set()
    
    for group in sorted_groups:
        if len(selected_groups) >= target_clusters - 1:
            break
        group_set = set(group)
        if not any(group_set & set(selected) for selected in selected_groups):
            selected_groups.append(group)
            covered_sequences |= group_set
    
    remaining_sequences = set(sequences) - covered_sequences
    if remaining_sequences:
        selected_groups.append(list(remaining_sequences))
    
    while len(selected_groups) < target_clusters and selected_groups:
        largest_idx = -1
        max_size = -1
        
        for i, group in enumerate(selected_groups):
            if len(group) > 1 and len(group) > max_size:
                max_size = len(group)
                largest_idx = i
        
        if largest_idx == -1: 
            break
            
        largest_group = selected_groups[largest_idx]
        subgroup1, subgroup2 = split_monophyletic_group(tree, largest_group)
        
        if subgroup1 and subgroup2:
            del selected_groups[largest_idx]
            selected_groups.append(subgroup1)
            selected_groups.append(subgroup2)
        else:
            if len(selected_groups) < target_clusters:
                mid = len(largest_group) // 2
                subgroup1 = largest_group[:mid]
                subgroup2 = largest_group[mid:]
                if subgroup1 and subgroup2:
                    del selected_groups[largest_idx]
                    selected_groups.append(subgroup1)
                    selected_groups.append(subgroup2)
                else:
                    break
            else:
                break
    
    return selected_groups

def assign_remaining_sequences(tree, clusters, unassigned_sequences):
    cluster_sequences = {}
    for seq, cluster_id in clusters.items():
        cluster_sequences.setdefault(cluster_id, []).append(seq)
    
    for seq in unassigned_sequences:
        try:
            seq_node = tree & seq
            min_distance = float('inf')
            nearest_cluster = 0
            
            for cluster_id, cluster_seqs in cluster_sequences.items():
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
            clusters[seq] = 1 
    return clusters

def dynamic_programming_clustering(tree, sequences, n_clusters):
    print(f"使用动态规划算法生成 {n_clusters} 个簇")
    
    all_monophyletic_groups = find_all_monophyletic_groups(tree, sequences) 
    selected_groups = select_optimal_groups(all_monophyletic_groups, n_clusters, sequences, tree) 
    
    clusters = {}
    used_sequences = set()
    
    for cluster_id, group in enumerate(selected_groups):
        actual_cluster_id = cluster_id + 1
        for seq in group:
            if seq not in used_sequences:
                clusters[seq] = actual_cluster_id
                used_sequences.add(seq)
    
    unassigned = set(sequences) - used_sequences
    if unassigned:
        print(f"有 {len(unassigned)} 个序列未分配，将它们分配到最近的簇")
        clusters = assign_remaining_sequences(tree, clusters, unassigned)
    
    return clusters

def simple_split_non_monophyletic(tree, sequences, mrca):
    if len(sequences) == 2:
        return [[sequences[0]], [sequences[1]]]
    
    subgroups = []
    processed_sequences = set()
    children = mrca.get_children()
    
    for child in children:
        child_leaves = set(child.get_leaf_names())
        sequences_in_child = child_leaves & set(sequences)
        
        if sequences_in_child and not sequences_in_child.issubset(processed_sequences):
            try:
                child_mrca = tree.get_common_ancestor(list(sequences_in_child))
                child_mrca_leaves = set(child_mrca.get_leaf_names())
                
                if child_mrca_leaves.issubset(set(sequences)):
                    subgroups.append(list(sequences_in_child))
                    processed_sequences.update(sequences_in_child)
                else:
                    recursive_subgroups = simple_split_non_monophyletic(tree, list(sequences_in_child), child_mrca)
                    subgroups.extend(recursive_subgroups)
                    processed_sequences.update(sequences_in_child)
            except:
                subgroups.append(list(sequences_in_child))
                processed_sequences.update(sequences_in_child)
    
    remaining_sequences = set(sequences) - processed_sequences
    if remaining_sequences:
        for seq in remaining_sequences:
            subgroups.append([seq])
    
    if not subgroups:
        subgroups = [sequences]
    
    return subgroups

def fix_non_monophyletic_clusters(tree, clusters, sequences):
    print("修复非单系群...")
    
    cluster_groups = {}
    for seq_id, cluster_id in clusters.items():
        cluster_groups.setdefault(cluster_id, []).append(seq_id)
    
    fixed_clusters = clusters.copy()
    new_cluster_id = max(clusters.values()) + 1
    
    for cluster_id, seq_list in cluster_groups.items():
        if len(seq_list) <= 1:
            continue
            
        try:
            mrca = tree.get_common_ancestor(seq_list)
            mrca_leaves = mrca.get_leaf_names()
            cluster_set = set(seq_list)
            
            if not cluster_set.issuperset(set(mrca_leaves)):
                # print(f"簇 {cluster_id} 是非单系群，包含 {len(seq_list)} 个序列")
                subgroups = simple_split_non_monophyletic(tree, seq_list, mrca)
                
                for seq in seq_list:
                    del fixed_clusters[seq]
                
                if len(subgroups) > 1:
                    for subgroup in subgroups:
                        for seq in subgroup:
                            fixed_clusters[seq] = new_cluster_id
                        new_cluster_id += 1
                else:
                    for seq in seq_list:
                        fixed_clusters[seq] = cluster_id
                    
        except:
            pass 
    
    return fixed_clusters

def extract_phylogenetic_clusters(tree, sequences, n_clusters=None):
    """
    改进的系统发育树聚类算法：先按ESM簇数目生成簇，再分割非单系群
    """
    print("使用改进的两步聚类算法")
    
    leaves = tree.get_leaf_names()
    missing_seqs = set(sequences) - set(leaves)
    if missing_seqs:
        sequences = [seq for seq in sequences if seq in leaves]
    
    print(f"第一步：使用动态规划算法生成 {n_clusters} 个簇")
    clusters = dynamic_programming_clustering(tree, sequences, n_clusters)
    
    print("第二步：验证并修复非单系群")
    clusters = fix_non_monophyletic_clusters(tree, clusters, sequences)
    
    clusters = renumber_clusters(clusters)
    
    final_cluster_count = len(set(clusters.values()))
    print(f"最终得到 {final_cluster_count} 个单系群簇")
    
    validate_monophyly(tree, clusters, sequences)
    
    return clusters