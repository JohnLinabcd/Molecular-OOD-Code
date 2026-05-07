import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# 导入你项目中的组件
from visnet.model import ViSNet
from visnet.data import DADataModule
from config import TrainingConfig

@torch.no_grad()
def evaluate(model, loader, device):
    """
    核心评估函数：计算 Mean Absolute Error
    """
    if loader is None:
        return 0.0
    model.eval()
    mae = 0.0
    cnt = 0
    
    pbar = tqdm(loader, desc=">>> 正在评估测试集", leave=True)
    for b in pbar:
        if b is None: continue
        b = b.to(device)
        # ViSNet 返回 (out, mol_feat, sub_feat)，我们只取 out
        res = model(b.z, b.pos, b.batch, b.sub_batch)
        pred = res[0] 
        
        # 计算绝对误差
        mae += (pred - b.y.view(pred.shape)).abs().sum().item()
        cnt += b.y.size(0)
        
    final_mae = mae / cnt if cnt > 0 else 0.0
    return final_mae

def main():
    # 1. 配置初始化
    config = TrainingConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = "s200_best_model.pth" # 你的模型路径
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到权重文件 {checkpoint_path}")
        return

    # 2. 数据准备 (获取测试集和归一化参数)
    print(">>> 正在加载数据配置...")
    dm = DADataModule(vars(config))
    dm.prepare_dataset()
    test_loader = dm.target_test_loader()
    
    # 必须使用训练时的 mean 和 std，否则结果会偏离
    mean = dm.mean.to(device)
    std = dm.std.to(device)

    # 3. 模型初始化
    print(">>> 正在初始化 ViSNet 模型...")
    model = ViSNet(
        hidden_channels=config.hidden_channels, 
        num_layers=config.num_layers, 
        num_heads=config.num_heads, 
        num_rbf=config.num_rbf, 
        mean=mean, 
        std=std, 
        num_tasks=len(config.label_names)
    ).to(device)

    # 4. 加载权重
    print(f">>> 正在加载专家权重: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 处理两种常见的保存格式：仅 state_dict 或 包含 epoch 的字典
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        saved_mae = ckpt.get('mae', 'N/A')
        print(f">>> 权重包含的历史 MAE 记录为: {saved_mae}")
    else:
        model.load_state_dict(ckpt)

    # 5. 执行评估
    print(">>> 开始计算测试集 MAE...")
    target_mae = evaluate(model, test_loader, device)
    
    print("-" * 30)
    print(f"评估完成！")
    print(f"目标文件: {config.target_paths[0]}")
    print(f"专家模型测试集 MAE: {target_mae:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()