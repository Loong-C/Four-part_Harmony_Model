import torch
import json
from harmony_model import BachHarmonyTransformer

def load_vocab_sizes(data_dir="data"):
    """读取我们之前用数据挖掘机存下来的词表，获取真实的大小"""
    with open(f"{data_dir}/chord_vocab.json", 'r', encoding='utf-8') as f:
        chord_vocab = json.load(f)
    with open(f"{data_dir}/duration_vocab.json", 'r', encoding='utf-8') as f:
        dur_vocab = json.load(f)
    with open(f"{data_dir}/pitch_vocab.json", 'r', encoding='utf-8') as f:
        pitch_vocab = json.load(f)
        
    return {
        'chord': len(chord_vocab),
        'duration': len(dur_vocab),
        'pitch': len(pitch_vocab)
    }

if __name__ == "__main__":
    print("--- 1. 加载本地词表大小 ---")
    try:
        vocab_sizes = load_vocab_sizes("data")
        print(f"真实词表大小: {vocab_sizes}")
    except FileNotFoundError:
        print("未找到 data 目录下的 JSON 词表，请先运行 build_dataset.py！")
        exit()

    print("\n--- 2. 初始化完整模型 ---")
    # 我们先建一个小型模型用来测试，d_model=128, 层数=2
    model = BachHarmonyTransformer(vocab_sizes, d_model=128, nhead=4, num_layers=2)
    print("模型主体组装完毕！")

    print("\n--- 3. 模拟一次完整的前向传播 ---")
    batch_size = 2
    seq_len = 16 # 一次看 16 个和声步长
    
    # 模拟输入张量 (Batch, Seq_Len, 6个特征)
    dummy_input = torch.zeros((batch_size, seq_len, 6), dtype=torch.long)
    dummy_input[:, :, 0] = torch.randint(0, vocab_sizes['chord'], (batch_size, seq_len))
    dummy_input[:, :, 1] = torch.randint(0, vocab_sizes['duration'], (batch_size, seq_len))
    for i in range(2, 6):
        dummy_input[:, :, i] = torch.randint(0, vocab_sizes['pitch'], (batch_size, seq_len))
        
    print(f"输入张量形状: {dummy_input.shape}")
    
    # 前向传播！
    outputs = model(dummy_input)
    
    print("\n--- 4. 查看多头输出层的结果 ---")
    for key, logits in outputs.items():
        print(f"[{key}] 输出形状: {logits.shape} -> (batch_size, seq_len, vocab_size)")
    
    print("\n✅ 测试成功！模型成功吐出了 6 个维度的概率分布预测值 (Logits)。")