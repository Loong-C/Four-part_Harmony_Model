import torch
import json
from generate import load_vocab_and_reverse_mapping, generate_harmony_v2
from harmony_model import BachHarmonyTransformer
from theory_evaluator import HarmonyEvaluator
from tqdm import tqdm

def run_massive_test(num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c2id, d2id, p2id, id2c, id2d, id2p = load_vocab_and_reverse_mapping()
    v_sizes = {'chord': len(c2id), 'duration': len(d2id), 'pitch': len(p2id)}
    
    model = BachHarmonyTransformer(v_sizes).to(device)
    model.load_state_dict(torch.load("checkpoints/bach_model_epoch_20.pt", map_location=device))
    
    evaluator = HarmonyEvaluator()
    
    # 定义测试集：不同的调式和进行
    test_cases = [
        {"key": "C", "prog": ["I", "IV", "V", "I"]},
        {"key": "G", "prog": ["I", "ii", "V7", "I"]},
        {"key": "a", "prog": ["i", "iv", "V", "i"]}, # 小调测试
        {"key": "D", "prog": ["I", "vi", "ii", "V", "I"]}
    ]
    
    print(f"开始大规模自动化测试，预计生成 {num_samples} 个样本...")
    
    for _ in tqdm(range(num_samples)):
        # 随机选一个测试案例
        case = test_cases[torch.randint(0, len(test_cases), (1,)).item()]
        
        # 生成
        sample = generate_harmony_v2(
            model, device, case["prog"], [1.0]*len(case["prog"]), 
            case["key"], c2id, d2id, p2id, id2p
        )
        
        # 评估
        evaluator.check_sample(sample)
    
    evaluator.get_report(num_samples)

if __name__ == "__main__":
    run_massive_test(200) # 运行 200 次生成测试