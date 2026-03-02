import music21 as m21
import pandas as pd
def extract_structured_harmony_data(bwv_number='bach/bwv253.mxl'):
    """
    提取巴赫众赞歌的多维结构化数据，返回一个二维列表。
    每一行对应 [Chord, Duration, Soprano, Alto, Tenor, Bass]。
    """
    print(f"Parsing {bwv_number}...")
    score = m21.converter.parse(bwv_number)
    key = score.analyze('key')
    chordified = score.chordify()
    
    parts = score.parts
    voice_parts = {'S': parts[0], 'A': parts[1], 'T': parts[2], 'B': parts[3]}
    
    flat_voices = {name: part.flatten().notesAndRests.stream() for name, part in voice_parts.items()}
    flat_chords = chordified.flatten().getElementsByClass('Chord').stream()
    
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
        
        if duration <= 0: continue
            
        roman_numeral = "NC"
        for c in flat_chords:
            c_start = round(float(c.offset), 4)
            c_end = round(float(c.offset + c.quarterLength), 4)
            if c_start <= start_time < c_end:
                roman_numeral = m21.roman.romanNumeralFromChord(c, key).figure
                break
                
        step_data = {'Chord': roman_numeral, 'Duration': duration}
        
        for v_name, flat_stream in flat_voices.items():
            state = "<REST>"
            for el in flat_stream:
                el_start = round(float(el.offset), 4)
                el_end = round(float(el.offset + el.quarterLength), 4)
                
                if el_start <= start_time < el_end:
                    if el.isRest: state = "<REST>"
                    elif el.isNote: state = el.pitch.midi if el_start == start_time else "<HOLD>"
                    elif el.isChord: state = el.sortAscending().notes[-1].pitch.midi if el_start == start_time else "<HOLD>"
                    break
            step_data[v_name] = state
            
        dataset.append([
            step_data['Chord'], step_data['Duration'], 
            step_data['S'], step_data['A'], step_data['T'], step_data['B']
        ])
        
    return dataset
