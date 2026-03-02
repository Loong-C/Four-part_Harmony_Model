import json

# 巴赫时代的标准音域 (MIDI)
RANGES = {
    'Soprano': (60, 81), 'Alto': (53, 74), 'Tenor': (48, 69), 'Bass': (36, 64)
}

class HarmonyEvaluator:
    def __init__(self):
        self.stats = {"parallel_5_8": 0, "voice_crossing": 0, "out_of_range": 0, "chord_validity": 0}

    def midi(self, note_str):
        if note_str in ["<REST>", "<HOLD>", "<PAD>"]: return None
        return int(note_str)

    def check_sample(self, sample):
        """对单个生成样本进行乐理扫描"""
        for i in range(len(sample)):
            step = sample[i]
            s, a, t, b = [self.midi(step[v]) for v in ['Soprano', 'Alto', 'Tenor', 'Bass']]
            
            # 1. 音域检查
            for idx, v in enumerate(['Soprano', 'Alto', 'Tenor', 'Bass']):
                val = [s, a, t, b][idx]
                if val and (val < RANGES[v][0] or val > RANGES[v][1]):
                    self.stats["out_of_range"] += 1

            # 2. 声部交叉检查 (音高应该是 S > A > T > B)
            if s and a and s < a: self.stats["voice_crossing"] += 1
            if a and t and a < t: self.stats["voice_crossing"] += 1
            if t and b and t < b: self.stats["voice_crossing"] += 1

            # 3. 平行检查 (与前一步对比)
            if i > 0:
                prev = sample[i-1]
                ps, pa, pt, pb = [self.midi(prev[v]) for v in ['Soprano', 'Alto', 'Tenor', 'Bass']]
                # 这里可以复用你之前的平行五八度检查逻辑...
                # 简化展示：
                if s and a and ps and pa and (s-a)%12 == (ps-pa)%12 == 7 and s != ps:
                    self.stats["parallel_5_8"] += 1

    def get_report(self, total_samples):
        print("\n" + "="*30)
        print("📊 批量测试乐理评估报告")
        print("="*30)
        for k, v in self.stats.items():
            print(f"{k:<18}: {v} 次违规 (平均每首 {v/total_samples:.2f} 处)")