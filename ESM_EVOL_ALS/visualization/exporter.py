import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import to_tree
import warnings
warnings.filterwarnings('ignore')

def save_linkage_matrix(linkage_matrix, seq_ids, output_path):
    """
    保存连接矩阵到CSV文件，并添加序列ID信息
    """
    linkage_df = pd.DataFrame(linkage_matrix, 
                             columns=['cluster1', 'cluster2', 'distance', 'sample_count'])
    
    n_seqs = len(seq_ids)
    id_map = {i: seq_ids[i] for i in range(n_seqs)}
    
    cluster_map = {i + n_seqs: f"cluster_{i}" for i in range(len(linkage_df))}
    full_map = {**id_map, **cluster_map}
    
    linkage_df['cluster1_id'] = linkage_df['cluster1'].map(full_map)
    linkage_df['cluster2_id'] = linkage_df['cluster2'].map(full_map)
    
    linkage_df.to_csv(output_path, index=False)
    print(f"连接矩阵已保存至: {output_path}")

def linkage_to_newick(Z, labels):
    """
    将层次聚类的连接矩阵转换为Newick格式的树字符串
    """
    tree = to_tree(Z, rd=False)
    
    def build_newick(node, parent_dist):
        if node.is_leaf():
            return f"{labels[node.id]}:{parent_dist - node.dist:.6f}"
        else:
            left = build_newick(node.left, node.dist)
            right = build_newick(node.right, node.dist)
            branch_length = parent_dist - node.dist if parent_dist else 0.0
            return f"({left},{right}):{branch_length:.6f}"
    
    newick_str = build_newick(tree, tree.dist) + ";"
    return newick_str

def save_newick_tree(newick_str, output_path):
    """
    将Newick字符串保存到文件
    """
    with open(output_path, 'w') as f:
        f.write(newick_str)
    print(f"Newick树已保存至: {output_path}")


def export_sankey_data(esm_clusters, phylo_clusters, seq_ids, output_prefix):
    """
    导出桑基图数据表
    """
    print("导出桑基图数据表...")
    
    esm_labels = [f"ESM_Cluster_{c}" for c in esm_clusters]
    phylo_labels = [f"Phylo_Cluster_{phylo_clusters.get(seq_id, -1)}" for seq_id in seq_ids]
    
    # 1. ESM2聚类基因列表
    esm_cluster_genes = {}
    for seq_id, cluster_label in zip(seq_ids, esm_labels):
        esm_cluster_genes.setdefault(cluster_label, []).append(seq_id)
    esm_data = [{"Cluster": c, "Gene_Count": len(g), "Gene_List": "; ".join(g)} for c, g in esm_cluster_genes.items()]
    pd.DataFrame(esm_data).to_csv(f"{output_prefix}_esm_clusters_genes.csv", index=False)
    print(f"ESM2聚类基因列表已保存至: {output_prefix}_esm_clusters_genes.csv")
    
    # 2. 系统发育树聚类基因列表
    phylo_cluster_genes = {}
    for seq_id in seq_ids:
        if seq_id in phylo_clusters:
            cluster_label = f"Phylo_Cluster_{phylo_clusters[seq_id]}"
            phylo_cluster_genes.setdefault(cluster_label, []).append(seq_id)
    phylo_data = [{"Cluster": c, "Gene_Count": len(g), "Gene_List": "; ".join(g)} for c, g in phylo_cluster_genes.items()]
    pd.DataFrame(phylo_data).to_csv(f"{output_prefix}_phylogenetic_clusters_genes.csv", index=False)
    print(f"系统发育树聚类基因列表已保存至: {output_prefix}_phylogenetic_clusters_genes.csv")
    
    # 3. 流量基因列表
    flow_data = {}
    for seq_id, esm_label, phylo_label in zip(seq_ids, esm_labels, phylo_labels):
        flow_data.setdefault((esm_label, phylo_label), []).append(seq_id)
    
    flow_list = [{"ESM_Cluster": e, "Phylogenetic_Cluster": p, "Flow_Count": len(g), "Gene_List": "; ".join(g)} 
                 for (e, p), g in flow_data.items()]
    flow_df = pd.DataFrame(flow_list).sort_values(["ESM_Cluster", "Phylogenetic_Cluster"])
    flow_df.to_csv(f"{output_prefix}_sankey_flow_genes.csv", index=False)
    print(f"桑基图流量基因列表已保存至: {output_prefix}_sankey_flow_genes.csv")
    
    # 4. 详细的连接数据
    detailed_flow_data = [{"Gene_ID": seq_id, "ESM_Cluster": esm_label, "Phylogenetic_Cluster": phylo_label, 
                           "ESM_Cluster_ID": esm_clusters[i], "Phylogenetic_Cluster_ID": phylo_clusters.get(seq_id, -1),
                           "Is_Divergent": 1 if esm_clusters[i] != phylo_clusters.get(seq_id, -1) else 0} 
                          for i, (seq_id, esm_label, phylo_label) in enumerate(zip(seq_ids, esm_labels, phylo_labels))]
    pd.DataFrame(detailed_flow_data).to_csv(f"{output_prefix}_sankey_detailed_assignments.csv", index=False)
    print(f"详细桑基图分配数据已保存至: {output_prefix}_sankey_detailed_assignments.csv")
    
    return pd.DataFrame(esm_data), pd.DataFrame(phylo_data), flow_df