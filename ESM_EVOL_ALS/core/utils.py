import numpy as np
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

def calculate_patroistic_distance_matrix(tree, sequences):
    """
    计算系统发育树的patristic距离矩阵
    """
    n_seqs = len(sequences)
    dist_matrix = np.zeros((n_seqs, n_seqs))
    
    for i, seq_id1 in enumerate(sequences):
        for j, seq_id2 in enumerate(sequences):
            if i < j:
                try:
                    node1 = tree & seq_id1
                    node2 = tree & seq_id2
                    dist = tree.get_distance(node1, node2)
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                except Exception as e:
                    dist_matrix[i, j] = 10.0
                    dist_matrix[j, i] = 10.0
    
    return dist_matrix

def perform_pcoa(distance_matrix, n_components=2):
    """
    执行PCoA分析 (主坐标分析)
    """
    if distance_matrix.ndim == 1:
        distance_matrix = squareform(distance_matrix)
    
    n = distance_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ distance_matrix**2 @ H
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    positive_eigenvalues = np.maximum(eigenvalues[:n_components], 0)
    coordinates = eigenvectors[:, :n_components] * np.sqrt(positive_eigenvalues)
    
    return coordinates