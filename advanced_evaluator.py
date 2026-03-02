import json
import numpy as np

# 定义巴赫时代的标准音域 (MIDI)
VOICE_RANGES = {
    'Soprano': (60, 81), # C4 - A5
    'Alto':    (53, 74), # F3 - D5
    'Tenor':   (48, 69), # C3 - A4
    'Bass':    (41, 64)  # F2 - E4
}

def evaluate_theory_logic(samples):
    stats = {
        "range_violation": 0,
        "voice_crossing": 0,
        "parallel_errors": 0,
        "total_notes": 0
    }
    
    for sample in samples:
        notes = sample['notes']
        for i in range(len(notes)):
            step = notes[i]
            stats["total_notes"] += 4
            
            # 1. 检查音域
            for v in ['Soprano', 'Alto', 'Tenor', 'Bass']:
                m_val = note_name_to_midi(step[v])
                if m_val:
                    low, high = VOICE_RANGES[v]
                    if not (low <= m_val <= high):
                        stats["range_violation"] += 1
            
            # 2. 检查声部交叉 (S < A < T < B 为错误，因为 MIDI 越大音越高)
            s = note_name_to_midi(step['Soprano'])
            a = note_name_to_midi(step['Alto'])
            t = note_name_to_midi(step['Tenor'])
            b = note_name_to_midi(step['Bass'])
            
            if s and a and s < a: stats["voice_crossing"] += 1
            if a and t and a < t: stats["voice_crossing"] += 1
            if t and b and t < b: stats["voice_crossing"] += 1
            
        # 3. 检查平行 (调用之前的 check_parallels)
        # ... 迭代调用并累加到 stats["parallel_errors"]
        
    print("--- 批量乐理评估报告 ---")
    print(f"测试样本数: {len(samples)}")
    print(f"音域违规率: {stats['range_violation']/stats['total_notes']:.2%}")
    print(f"声部交叉率: {stats['voice_crossing']/stats['total_notes']:.2%}")
    return stats