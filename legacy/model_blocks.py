import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class HarmonyEmbedding(nn.Module):
    def __init__(self, vocab_sizes, d_model):
        super(HarmonyEmbedding, self).__init__()
        self.d_model = d_model
        
        # 六个独立词表
        self.chord_emb = nn.Embedding(vocab_sizes['chord'], d_model)
        self.dur_emb   = nn.Embedding(vocab_sizes['duration'], d_model)
        self.s_emb = nn.Embedding(vocab_sizes['pitch'], d_model)
        self.a_emb = nn.Embedding(vocab_sizes['pitch'], d_model)
        self.t_emb = nn.Embedding(vocab_sizes['pitch'], d_model)
        self.b_emb = nn.Embedding(vocab_sizes['pitch'], d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        c_idx, d_idx = x[:, :, 0], x[:, :, 1]
        s_idx, a_idx, t_idx, b_idx = x[:, :, 2], x[:, :, 3], x[:, :, 4], x[:, :, 5]
        
        c_vec = self.chord_emb(c_idx)
        d_vec = self.dur_emb(d_idx)
        s_vec = self.s_emb(s_idx)
        a_vec = self.a_emb(a_idx)
        t_vec = self.t_emb(t_idx)
        b_vec = self.b_emb(b_idx)
        
        # 特征相加融合
        combined_vec = c_vec + d_vec + s_vec + a_vec + t_vec + b_vec
        combined_vec = combined_vec * math.sqrt(self.d_model)
        out = self.pos_encoder(combined_vec)
        out = self.norm(out)
        return out