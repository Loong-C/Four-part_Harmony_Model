import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # 用于显示极其优雅的进度条

# 导入我们之前写的模块
from dataset import HarmonyDataset
from harmony_model import BachHarmonyTransformer
from vocab_manager import load_vocab

def encode_dataset(raw_data, chord2id, dur2id, pitch2id):
    """
    将人类可读的字符串 JSON 数据，转化为纯数字 ID 张量
    """
    tensor_data = []
    for step in raw_data:
        chord, dur, s, a, t, b = step
        
        # 查字典，如果遇到极其罕见的未登录词（通常不会发生），默认给 0 (<PAD>)
        c_id = chord2id.get(str(chord), 0)
        d_id = dur2id.get(str(dur), 0)
        s_id = pitch2id.get(str(s), 0)
        a_id = pitch2id.get(str(a), 0)
        t_id = pitch2id.get(str(t), 0)
        b_id = pitch2id.get(str(b), 0)
        
        tensor_data.append([c_id, d_id, s_id, a_id, t_id, b_id])
        
    return torch.tensor(tensor_data, dtype=torch.long)

def train_model():
    # 1. 自动检测计算设备 (支持 Nvidia GPU, Apple M芯片 MPS, 或是普通 CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🚀 正在使用的计算设备: {device}")

    # 2. 加载本地数据与词表
    print("正在加载数据集与词表...")
    with open("data/bach_chorales_raw.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    chord2id = load_vocab("data/chord_vocab.json")
    dur2id = load_vocab("data/duration_vocab.json")
    pitch2id = load_vocab("data/pitch_vocab.json")
    
    vocab_sizes = {
        'chord': len(chord2id), 'duration': len(dur2id), 'pitch': len(pitch2id)
    }

    # 3. 数据编码与组装 DataLoader
    tensor_data = encode_dataset(raw_data, chord2id, dur2id, pitch2id)
    context_length = 32 # 模型每次观察 32 个时间步的历史
    batch_size = 64     # 每次并行训练 64 个片段
    
    dataset = HarmonyDataset(tensor_data, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"数据准备完毕！总训练样本数: {len(dataset)}，共分为 {len(dataloader)} 个 Batch。")

    # 4. 初始化模型与优化器
    # 这里我们定义一个中等规模的模型（参数量约几百万），适合在本地快速训练
    model = BachHarmonyTransformer(
        vocab_sizes=vocab_sizes, 
        d_model=256, 
        nhead=8, 
        num_layers=4, 
        dim_feedforward=1024, 
        dropout=0.1
    ).to(device)
    
    # 使用目前最主流的大模型优化器 AdamW，学习率设为 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # 定义损失函数 (忽略 <PAD> 的 index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 5. 开始史诗级的训练循环
    epochs = 20 # 暂定训练 20 轮
    
    print("\n" + "="*50)
    print("🎵 巴赫和声大模型 - 训练正式开始 🎵")
    print("="*50)

    for epoch in range(epochs):
        model.train() # 设置为训练模式 (启用 Dropout 等)
        total_loss = 0
        
        # 使用 tqdm 包装 dataloader 以显示极其优雅的进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in pbar:
            # 将数据推送到 GPU/Apple Silicon 上
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 清空上一轮的梯度
            optimizer.zero_grad()
            
            # 前向传播 (Forward)
            outputs = model(batch_x)
            
            # 准备计算 Loss
            # outputs 里的每个 tensor 形状为 (batch_size, seq_len, vocab_size)
            # PyTorch 的 CrossEntropyLoss 要求输入 shape 为 (N, C)，目标 shape 为 (N)
            # 所以我们需要把前两个维度展平 (batch_size * seq_len, vocab_size)
            
            # 提取目标值 Y (对应 6 个特征) 并展平为一维
            target_chord = batch_y[:, :, 0].reshape(-1)
            target_dur   = batch_y[:, :, 1].reshape(-1)
            target_s     = batch_y[:, :, 2].reshape(-1)
            target_a     = batch_y[:, :, 3].reshape(-1)
            target_t     = batch_y[:, :, 4].reshape(-1)
            target_b     = batch_y[:, :, 5].reshape(-1)
            
            # 计算 6 个独立特征的交叉熵损失
            loss_c = criterion(outputs['chord'].reshape(-1, vocab_sizes['chord']), target_chord)
            loss_d = criterion(outputs['duration'].reshape(-1, vocab_sizes['duration']), target_dur)
            loss_s = criterion(outputs['S'].reshape(-1, vocab_sizes['pitch']), target_s)
            loss_a = criterion(outputs['A'].reshape(-1, vocab_sizes['pitch']), target_a)
            loss_t = criterion(outputs['T'].reshape(-1, vocab_sizes['pitch']), target_t)
            loss_b = criterion(outputs['B'].reshape(-1, vocab_sizes['pitch']), target_b)
            
            # 融合为总损失！
            loss = loss_c + loss_d + loss_s + loss_a + loss_t + loss_b
            
            # 反向传播 (Backward) 与 权重更新 (Step)
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping)，防止梯度爆炸导致 Loss 变成 NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录并更新进度条上的 Loss 显示
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] 完成! 平均 Loss: {avg_loss:.4f}")
        
        # 定期保存模型权重 (Checkpoint)
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/bach_model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存至 {save_path}")

    print("🎉 训练大功告成！")

if __name__ == "__main__":
    # 如果你没有安装 tqdm，可以在终端运行: pip install tqdm
    train_model()