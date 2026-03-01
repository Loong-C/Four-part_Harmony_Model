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
    print("--- 1. 开始提取原始数据 ---")
    raw_dataset = extract_structured_harmony_data('bach/bwv253.mxl')
    
    print("\n--- 2. 开始构建模型词表 (Tokenizer) ---")
    chord2id, dur2id, pitch2id = build_vocabularies(raw_dataset)
    # 统计各个特征的总词汇量大小，供 Embedding 层初始化使用
    vocab_sizes = {
        'chord': len(chord2id),
        'duration': len(dur2id),
        'pitch': len(pitch2id)
    }
    print(f"提取完成: 发现 {vocab_sizes['chord']} 种和弦, {vocab_sizes['duration']} 种时长, {vocab_sizes['pitch']} 种音高状态。")
    
    print("\n--- 3. 将原始数据转换为 PyTorch 张量 ---")
    tensor_sequence = convert_data_to_tensor(raw_dataset, chord2id, dur2id, pitch2id)
    print(f"生成的张量形状: {tensor_sequence.shape} (长度为 {tensor_sequence.shape[0]} 个时间步，每个时间步包含 6 个特征)")
    
    print("\n--- 4. 初始化 Embedding 层并流经数据 ---")
    d_model = 256 # 隐藏层维度
    embed_layer = HarmonyEmbedding(vocab_sizes, d_model)
    
    # 深度学习模型通常处理一个 Batch (批量) 的数据，即输入通常是 3 维的：
    # (batch_size, sequence_length, features)
    # 我们现在只有一首曲子，所以手动在最前面加一个 batch_size=1 的维度
    input_batch = tensor_sequence.unsqueeze(0) 
    print(f"送入模型的 Batch 形状: {input_batch.shape}")
    
    # 见证奇迹的时刻：前向传播！
    output_vectors = embed_layer(input_batch)
    print(f"成功获取稠密向量! 最终融合张量形状: {output_vectors.shape}")