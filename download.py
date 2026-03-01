import music21 as m21
import pandas as pd

def extract_structured_harmony_data(bwv_number='bach/bwv253.mxl'):
    print(f"Parsing {bwv_number}...")
    score = m21.corpus.parse(bwv_number)
    key = score.analyze('key')
    chordified = score.chordify()
    
    parts = score.parts
    voice_parts = {'S': parts[0], 'A': parts[1], 'T': parts[2], 'B': parts[3]}
    
    # 【改动 1】提前提取纯净的 notesAndRests 序列，避免后续混入 Instrument/Clef 等无用对象
    flat_voices = {
        name: part.flatten().notesAndRests.stream()
        for name, part in voice_parts.items()
    }
    flat_chords = chordified.flatten().getElementsByClass('Chord').stream()
    
    # 【改动 2】使用 round 保留 4 位小数，彻底解决浮点数精度导致的切片错位
    all_offsets = set()
    for p in flat_voices.values():
        for el in p:
            all_offsets.add(round(float(el.offset), 4))
            all_offsets.add(round(float(el.offset + el.quarterLength), 4))
            
    sorted_offsets = sorted(list(all_offsets))
    dataset = []
    
    for i in range(len(sorted_offsets) - 1):
        start_time = sorted_offsets[i]
        end_time = sorted_offsets[i+1]
        duration = round(end_time - start_time, 4)
        
        if duration <= 0:
            continue
            
        # --- A. 获取和弦级数 ---
        roman_numeral = "NC"
        for c in flat_chords:
            c_start = round(float(c.offset), 4)
            c_end = round(float(c.offset + c.quarterLength), 4)
            # 找到覆盖当前 start_time 的和弦
            if c_start <= start_time < c_end:
                roman_numeral = m21.roman.romanNumeralFromChord(c, key).figure
                break
                
        step_data = {'Chord': roman_numeral, 'Duration': duration}
        
        # --- B. 获取四个声部状态 ---
        for v_name, flat_stream in flat_voices.items():
            state = "<REST>"
            for el in flat_stream:
                el_start = round(float(el.offset), 4)
                el_end = round(float(el.offset + el.quarterLength), 4)
                
                # 【改动 3】使用严谨的区间包含判断：如果当前切片被该音符覆盖
                if el_start <= start_time < el_end:
                    if el.isRest:
                        state = "<REST>"
                    elif el.isNote:
                        # 只有当切片起点刚好等于音符起点时，才是新按下的音 (Attack)
                        if el_start == start_time:
                            state = el.pitch.midi
                        else:
                            state = "<HOLD>"
                    elif el.isChord:
                        if el_start == start_time:
                            state = el.sortAscending().notes[-1].pitch.midi
                        else:
                            state = "<HOLD>"
                    break # 找到了就立即跳出内层循环
            
            step_data[v_name] = state
            
        dataset.append([
            step_data['Chord'], 
            step_data['Duration'], 
            step_data['S'], step_data['A'], step_data['T'], step_data['B']
        ])
        
    return dataset

if __name__ == "__main__":
    raw_tensor_data = extract_structured_harmony_data('bach/bwv253.mxl')
    df = pd.DataFrame(raw_tensor_data, columns=['Chord', 'Duration', 'Soprano', 'Alto', 'Tenor', 'Bass'])
    print("\n--- 提取的多维张量数据 (前 10 个时间步) ---")
    print(df.head(10).to_string(index=True))