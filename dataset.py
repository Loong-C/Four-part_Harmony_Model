import torch
from torch.utils.data import Dataset

class HarmonyDataset(Dataset):
    """
    自回归和声数据集：
    将整首乐曲的张量切分为固定长度的上下文窗口。
    """
    def __init__(self, tensor_data, context_length):
        """
        参数:
            tensor_data: 我们之前提取出的完整乐曲张量，形状为 (total_seq_len, 6)
            context_length: 模型一次能看到的历史步数 (比如 16 步或 32 步)
        """
        self.data = tensor_data
        self.context_length = context_length

    def __len__(self):
        # 能够切出的样本总数 = 总长度 - 窗口长度
        # 比如一首曲子 100 步，窗口是 10 步，那么可以有 90 个不同的起点
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # 取出一段作为输入 X (从 idx 到 idx + context_length - 1)
        x = self.data[idx : idx + self.context_length]
        
        # 取出目标预测 Y (向右平移 1 步：从 idx + 1 到 idx + context_length)
        y = self.data[idx + 1 : idx + self.context_length + 1]
        
        return x, y