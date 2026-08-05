from sklearn.metrics import v_measure_score, adjusted_mutual_info_score

def calculate_divergence(esm_clusters, phylo_clusters):
    """
    计算ESM聚类与系统发育聚类之间的差异
    """
    common_sequences = list(set(esm_clusters.keys()) & set(phylo_clusters.keys()))
    
    labels_esm = []
    labels_phylo = []
    for seq_id in common_sequences:
        labels_esm.append(esm_clusters[seq_id])
        labels_phylo.append(phylo_clusters[seq_id])

    v_measure = v_measure_score(labels_phylo, labels_esm)
    ami = adjusted_mutual_info_score(labels_phylo, labels_esm)
    
    print(f"V-measure: {v_measure:.3f}")
    print(f"Adjusted Mutual Information (AMI): {ami:.3f}")
    
    divergent_sequences = []
    for seq_id in common_sequences:
        if esm_clusters[seq_id] != phylo_clusters[seq_id]:
            divergent_sequences.append(seq_id)
    
    print(f"发现 {len(divergent_sequences)} 个分歧序列")
    
    return v_measure, ami, divergent_sequences, common_sequences