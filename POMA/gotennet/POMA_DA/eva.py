# eva.py
import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from gotennet.model import GotenNetWrapper
from gotennet.data import DADataModule
from config import TrainingConfig

@torch.no_grad()
def evaluate(model, loader, device):
    if loader is None:
        return 0.0
    model.eval()
    mae = 0.0
    cnt = 0
    
    pbar = tqdm(loader, desc=">>> 正在评估测试集", leave=True)
    for b in pbar:
        if b is None: continue
        b = b.to(device)
        res = model(b.z, b.pos, b.batch, b.sub_batch)
        pred = res[0] 
        
        mae += (pred - b.y.view(pred.shape)).abs().sum().item()
        cnt += b.y.size(0)
        
    return mae / cnt if cnt > 0 else 0.0

def main():
    config = TrainingConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # 指向你的那个旧权重
    checkpoint_path = "s200_best_model.pth" # 换成你报错测试加载的那个确切的 pth/pt 文件名
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到权重文件 {checkpoint_path}")
        return

    print(">>> 正在加载数据配置...")
    dm = DADataModule(vars(config))
    dm.prepare_dataset()
    test_loader = dm.target_test_loader()
    
    mean = dm.mean.to(device)
    std = dm.std.to(device)

    print(">>> 正在初始化 GotenNet 模型...")
    model = GotenNetWrapper(
        config=config, 
        mean=mean, 
        std=std, 
        num_tasks=len(config.label_names)
    ).to(device)

    print(f">>> 正在直接加载专家权重 (无映射): {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # ✨ 见证奇迹的时刻：直接加载，严格模式！如果没报错，说明彻底搞定了！
    model.load_state_dict(state_dict, strict=True)
    print(">>> 权重加载成功！(Strict=True 通关 ✅)")

    print(">>> 开始执行最终评估...")
    mae = evaluate(model, test_loader, device)
    print("=" * 50)
    print(f"🎯 最终 Target 域测试集 MAE: {mae:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()