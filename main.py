import torch
# 从我们写好的模块中导入函数和类
from data_extractor import extract_structured_harmony_data
from model_blocks import HarmonyEmbedding

def build_vocabularies(raw_data):
    """
    遍历原始数据，为 和弦(Chord), 时长(Duration), 音高(Pitch) 建立独立的映射字典。
    这三个字典的结构都是: { "特征值": 整数ID }
    """
    # 基础特殊 Token 初始化 (从 0 开始编号)
    # 我们预留了 0 作为 <PAD> (用于以后不同长度序列的对齐填充)
    chord2id = {"<PAD>": 0, "NC": 1}
    dur2id = {"<PAD>": 0}
    # 音高词表共享给 SATB 四个声部
    pitch2id = {"<PAD>": 0, "<REST>": 1, "<HOLD>": 2}
    
    # 开始遍历提取出的二维数据进行收录
    for step in raw_data:
        chord, dur, s, a, t, b = step
        
        # 1. 录入和弦
        if chord not in chord2id: chord2id[chord] = len(chord2id)
        # 2. 录入时长
        if dur not in dur2id: dur2id[dur] = len(dur2id)
        # 3. 录入四个声部的音高
        for pitch_state in [s, a, t, b]:
            if pitch_state not in pitch2id: pitch2id[pitch_state] = len(pitch2id)
            
    return chord2id, dur2id, pitch2id

def convert_data_to_tensor(raw_data, chord2id, dur2id, pitch2id):
    """
    利用构建好的词表，将混杂着字符串和数字的列表转换为全数字的 PyTorch 张量
    """
    tensor_data = []
    for step in raw_data:
        chord, dur, s, a, t, b = step
        
        # 查字典，转换！
        c_id = chord2id[chord]
        d_id = dur2id[dur]
        s_id = pitch2id[s]
        a_id = pitch2id[a]
        t_id = pitch2id[t]
        b_id = pitch2id[b]
        
        tensor_data.append([c_id, d_id, s_id, a_id, t_id, b_id])
        
    # 转换为 PyTorch 可以理解的 LongTensor (整数张量)
    # 形状: (sequence_length, 6)
    return torch.tensor(tensor_data, dtype=torch.long)

if __name__ == "__main__":
    from dataset import HarmonyDataset
    from torch.utils.data import DataLoader
    
    print("--- 1. 开始提取原始数据 ---")
    raw_dataset = extract_structured_harmony_data('bach/bwv253.mxl')
    
    print("\n--- 2. 开始构建模型词表 (Tokenizer) ---")
    chord2id, dur2id, pitch2id = build_vocabularies(raw_dataset)
    vocab_sizes = {
        'chord': len(chord2id),
        'duration': len(dur2id),
        'pitch': len(pitch2id)
    }
    
    print("\n--- 3. 将原始数据转换为 PyTorch 张量 ---")
    tensor_sequence = convert_data_to_tensor(raw_dataset, chord2id, dur2id, pitch2id)
    print(f"生成的整曲张量形状: {tensor_sequence.shape}")
    
    # ================= 新增的代码在这里 =================
    
    print("\n--- 4. 构建数据传送带 (DataLoader) ---")
    # 设定模型的“视野大小”（上下文长度），比如让它一次看 16 个和声切片
    context_length = 16 
    
    # 实例化我们刚刚写的 Dataset
    dataset = HarmonyDataset(tensor_sequence, context_length)
    
    # 实例化 DataLoader，它会帮我们自动打乱顺序 (shuffle) 并打包成 Batch
    # batch_size=4 意味着模型一次并行处理 4 个不同的乐句切片
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"数据集切分完成！共产生 {len(dataset)} 个训练样本。")
    print(f"以 batch_size=4 打包，共有 {len(dataloader)} 个 Batch。")
    
    print("\n--- 5. 模拟训练时的前向传播流动 ---")
    d_model = 256 # Transformer 的隐藏层维度
    embed_layer = HarmonyEmbedding(vocab_sizes, d_model)
    
    # 从 DataLoader 中抽取第一个 Batch 看看
    for batch_x, batch_y in dataloader:
        print(f"获取到的输入批次 X 的形状: {batch_x.shape} -> (batch_size, context_length, features)")
        print(f"获取到的目标批次 Y 的形状: {batch_y.shape} -> Y 是 X 向未来偏移了一步")
        
        # 把 X 送进我们的特征嵌入大门！
        embedded_x = embed_layer(batch_x)
        print(f"流经 Embedding 层后，张量形状变为: {embedded_x.shape} -> 完美契合 Transformer 的输入要求！")
        
        # 为了演示，我们只看第一个 Batch 就退出循环
        break