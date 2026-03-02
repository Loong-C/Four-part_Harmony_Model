import torch
import torch.nn as nn
import math
from model_blocks import HarmonyEmbedding

class BachHarmonyTransformer(nn.Module):
    def __init__(self, vocab_sizes, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        巴赫四部和声生成模型 (Autoregressive Transformer)
        """
        super(BachHarmonyTransformer, self).__init__()
        self.d_model = d_model
        
        # 1. 接入我们之前写好的特征嵌入大门
        self.embedding = HarmonyEmbedding(vocab_sizes, d_model)
        
        # 2. Transformer 主干网络
        # batch_first=True 让我们输入和输出的张量形状都是 (batch, seq, feature)，更符合直觉
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 多头输出层 (预测 6 个维度的概率分布)
        # 将 d_model 维度的隐藏状态，分别映射到各自的词表大小
        self.head_chord = nn.Linear(d_model, vocab_sizes['chord'])
        self.head_dur   = nn.Linear(d_model, vocab_sizes['duration'])
        self.head_s     = nn.Linear(d_model, vocab_sizes['pitch'])
        self.head_a     = nn.Linear(d_model, vocab_sizes['pitch'])
        self.head_t     = nn.Linear(d_model, vocab_sizes['pitch'])
        self.head_b     = nn.Linear(d_model, vocab_sizes['pitch'])

    def generate_square_subsequent_mask(self, sz, device):
        """
        生成因果掩码，防止模型在训练时“偷看”未来的时间步。
        返回一个对角线及左下为 0.0，右上为 -inf 的矩阵。
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        x 形状: (batch_size, seq_len, 6)
        """
        seq_len = x.size(1)
        device = x.device
        
        # 1. 嵌入特征 (batch_size, seq_len, d_model)
        emb = self.embedding(x)
        
        # 2. 生成因果掩码 (seq_len, seq_len)
        causal_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # 3. 通过 Transformer 主干
        # 注意: PyTorch 的 TransformerEncoder 需要 is_causal=True 来开启高效的注意力机制 (PyTorch 2.0+ 特性)
        hidden_states = self.transformer(emb, mask=causal_mask, is_causal=True)
        
        # 4. 通过六个输出头预测未来
        # 每个输出的形状都将是: (batch_size, seq_len, 对应的词表大小)
        logits_chord = self.head_chord(hidden_states)
        logits_dur   = self.head_dur(hidden_states)
        logits_s     = self.head_s(hidden_states)
        logits_a     = self.head_a(hidden_states)
        logits_t     = self.head_t(hidden_states)
        logits_b     = self.head_b(hidden_states)
        
        # 将六个输出打包成字典返回
        return {
            'chord': logits_chord,
            'duration': logits_dur,
            'S': logits_s,
            'A': logits_a,
            'T': logits_t,
            'B': logits_b
        }