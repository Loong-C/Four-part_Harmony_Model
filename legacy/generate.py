import torch
import json
import os
from harmony_model import BachHarmonyTransformer


def load_vocab_and_reverse_mapping(data_dir="data"):
    """加载词表，并建立从 ID 反推回字符串/数字的反向映射字典"""
    with open(f"{data_dir}/chord_vocab.json", 'r', encoding='utf-8') as f:
        chord2id = json.load(f)
    with open(f"{data_dir}/duration_vocab.json", 'r', encoding='utf-8') as f:
        dur2id = json.load(f)
    with open(f"{data_dir}/pitch_vocab.json", 'r', encoding='utf-8') as f:
        pitch2id = json.load(f)
        
    # 构建反向映射 {ID: "特征值"}
    id2chord = {v: k for k, v in chord2id.items()}
    id2dur   = {v: k for k, v in dur2id.items()}
    id2pitch = {v: k for k, v in pitch2id.items()}
    
    return chord2id, dur2id, pitch2id, id2chord, id2dur, id2pitch

def midi_to_note_name(midi_str):
    """将 MIDI 数字 (如 '60') 转换为音符名称 (如 'C4')，方便人类阅读"""
    if not midi_str.isdigit():
        return midi_str # 如果是 <REST> 或 <HOLD>，直接返回
    
    midi_val = int(midi_str)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_val // 12) - 1
    note = notes[midi_val % 12]
    return f"{note}{octave}"

import torch.nn.functional as F
import music21 as m21

def get_key_scale_mask(key_name, pitch2id, device):
    """
    创建一个掩码：在当前调式音阶外的音符，权重设为 -inf
    """
    mask = torch.zeros(len(pitch2id)).to(device)
    try:
        k = m21.key.Key(key_name)
        scale_pitches = [p.name for p in k.getScale().getPitches()]
    except:
        return mask # 如果调式非法，不进行过滤
    
    for pitch_str, p_id in pitch2id.items():
        if pitch_str in ["<PAD>", "<REST>", "<HOLD>"]: continue
        
        # 将 MIDI 数字转为音名进行比对
        p_obj = m21.pitch.Pitch(int(pitch_str))
        if p_obj.name not in scale_pitches:
            mask[p_id] = -1e10 # 极小的数，相当于屏蔽
    return mask

def generate_harmony_v2(model, device, target_chords, target_durs, key_name, 
                        chord2id, dur2id, pitch2id, id2pitch, temperature=0.8):
    model.eval()
    seq_len = len(target_chords)
    current_seq = torch.zeros((1, 1, 6), dtype=torch.long).to(device)
    
    # 预计算调式掩码
    scale_mask = get_key_scale_mask(key_name, pitch2id, device)
    generated_steps = []

    with torch.no_grad():
        for i in range(seq_len):
            c_id = chord2id.get(target_chords[i], 1)
            d_id = dur2id.get(str(target_durs[i]), 0)
            
            outputs = model(current_seq)
            
            step_notes = []
            for voice in ['S', 'A', 'T', 'B']:
                logits = outputs[voice][:, -1, :] / temperature
                # 应用调式掩码 (仅针对音高层)
                logits = logits + scale_mask
                
                # 采样：增加多样性
                probs = F.softmax(logits, dim=-1)
                n_id = torch.multinomial(probs, num_samples=1).item()
                step_notes.append(n_id)
            
            s_id, a_id, t_id, b_id = step_notes
            new_step = torch.tensor([[[c_id, d_id, s_id, a_id, t_id, b_id]]], dtype=torch.long).to(device)
            current_seq = torch.cat([current_seq, new_step], dim=1)
            
            generated_steps.append({
                'Chord': target_chords[i], 'Dur': target_durs[i],
                'Soprano': id2pitch[s_id], 'Alto': id2pitch[a_id],
                'Tenor': id2pitch[t_id], 'Bass': id2pitch[b_id]
            })
    return generated_steps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载词表
    chord2id, dur2id, pitch2id, id2chord, id2dur, id2pitch = load_vocab_and_reverse_mapping()
    vocab_sizes = {'chord': len(chord2id), 'duration': len(dur2id), 'pitch': len(pitch2id)}
    
    # 2. 加载模型 (注意：这里的参数必须和你训练时设定的一模一样！)
    model = BachHarmonyTransformer(vocab_sizes, d_model=256, nhead=8, num_layers=4).to(device)
    
    # 找到你最后一次保存的模型权重 (请根据你 checkpoints 文件夹里的实际文件名修改)
    model_path = "checkpoints/bach_model_epoch_20.pt" 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print("未找到模型权重文件！请检查路径。")
        exit()

    # 3. 终极命题：给模型出题！
    # 这是一个经典的 C大调 / a小调 常见进行： 主 -> 下属 -> 属七 -> 主
    my_chords = ["I", "IV", "V", "I", "IV6", "I64", "V", "I6"]
    my_durs   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0] # 最后一个音符拉长两拍
    my_key = "D"
    print("\n" + "="*50)
    print(f"你的调式：{my_key}")
    print(f"🎵 你的命题进行: {' -> '.join(my_chords)}")
    print("="*50)
    
    # 4. 生成！
    result = generate_harmony_v2(model, device, my_chords, my_durs, my_key, chord2id, dur2id, pitch2id, id2pitch)
    
    # 5. 打印优美的表格
    print(f"{'Step':<6} | {'Chord':<6} | {'Dur':<5} | {'Soprano':<7} | {'Alto':<7} | {'Tenor':<7} | {'Bass':<7}")
    print("-" * 65)
    for i, step in enumerate(result):
        print(f"{i+1:<6} | {step['Chord']:<6} | {step['Dur']:<5} | {midi_to_note_name(step['Soprano']):<7} | {midi_to_note_name(step['Alto']):<7} | {midi_to_note_name(step['Tenor']):<7} | {midi_to_note_name(step['Bass']):<7}")
        
    # 将结果保存下来供 evaluation 使用
    with open("generated_score.json", "w") as f:
        json.dump(result, f, indent=2)