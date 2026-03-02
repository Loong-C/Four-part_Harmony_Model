import json
import os

def build_vocabularies(raw_dataset):
    """
    遍历整个庞大的数据集，建立全局词表
    """
    chord2id = {"<PAD>": 0, "NC": 1}
    dur2id = {"<PAD>": 0}
    pitch2id = {"<PAD>": 0, "<REST>": 1, "<HOLD>": 2}
    
    for step in raw_dataset:
        chord, dur, s, a, t, b = step
        if chord not in chord2id: chord2id[chord] = len(chord2id)
        if dur not in dur2id: dur2id[dur] = len(dur2id)
        for pitch_state in [s, a, t, b]:
            if pitch_state not in pitch2id: pitch2id[pitch_state] = len(pitch2id)
            
    return chord2id, dur2id, pitch2id

def save_vocab(vocab_dict, filepath):
    """将字典保存为 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

def load_vocab(filepath):
    """从 JSON 文件加载字典，并确保 Key 的类型正确"""
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # 对于时长，JSON 会把浮点数 0.5 变成字符串 "0.5"，需要转回 float
    # 但为了稳妥，我们在词表里统一把 Key 当作字符串处理更简单
    return vocab