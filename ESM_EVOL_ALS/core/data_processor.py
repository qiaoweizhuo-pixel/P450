import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

class ESM2FeatureExtractor:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
        """
        初始化ESM2模型和分词器
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
                max_length=1024
            ).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_index]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
            
            # 打印进度
            if (i // batch_size) % 10 == 0:
                print(f"已处理 {min(i+batch_size, len(sequences))}/{len(sequences)} 条序列")
        
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