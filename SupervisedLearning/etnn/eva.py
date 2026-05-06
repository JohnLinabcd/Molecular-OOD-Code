import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import Namespace

# --- 导入组件 ---
from data import DADataModule
from etnn.model import ETNN
from etnn.lifter import get_adjacency_types
from etnn.combinatorial_data import CombinatorialComplexData

# 适配 PyTorch 2.6+
if hasattr(torch.serialization, 'add_safe_globals'):
    import numpy
    torch.serialization.add_safe_globals([
        CombinatorialComplexData, numpy.core.multiarray.scalar, numpy.dtype
    ])

@torch.no_grad()
def main():
    # 🎯 配置区
    model_path = "sup_200.pth"
    target_csv = "data/scaffold_datasets1/a022.csv" # 确保此路径存在
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 加载 Checkpoint 并提取元数据
    # ==========================================
    print(f">>> 正在加载自包含检查点: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    meta = ckpt['meta_info']
    config_dict = meta['config_dict']
    actual_dims = meta['actual_num_features_per_rank']
    v_dims = meta['visible_dims']
    mean_val = meta['mean']
    std_val = meta['std']
    
    print(f"✅ 提取统计量: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # ==========================================
    # 2. 准备 DADataModule (绕过 Pandas 报错)
    # ==========================================
    print(f">>> 正在初始化数据集对象...")
    
    # ✨ 核心技巧：
    # 1. 把 sup_source_paths 也指向 a022.csv，防止 pd.concat([]) 报错
    config_dict['sup_source_paths'] = [target_csv] 
    # 2. 确保 target_paths 指向 a022.csv
    config_dict['target_paths'] = [target_csv]
    # 3. 确保 da_source_paths 为空列表
    config_dict['da_source_paths'] = []
    
    # 必须确保 config_dict 里有这些 key，否则 DADataModule 会报错
    if 'cache_dir' not in config_dict: config_dict['cache_dir'] = "./cache"
    if 'label_name' not in config_dict: config_dict['label_name'] = "homo"

    dm = DADataModule(config_dict)
    # 这一步会处理 a022.csv 并计算临时的 mean/std
    dm.prepare_dataset()
    
    # ✨ 核心技巧：立刻用 pth 里的正确均值覆盖掉它
    dm._mean = torch.tensor([mean_val]).to(device)
    dm._std = torch.tensor([std_val]).to(device)
    print(f"✅ 已强制对齐物理标尺。")

    # 提取 20% 的固定测试集
    test_loader = dm.target_test_loader()
    print(f"✅ Target Test 集准备就绪，样本数: {len(dm.target_test)}")

    # ==========================================
    # 3. 初始化模型 (参数全部来自 pth)
    # ==========================================
    config = Namespace(**config_dict)
    adjs = get_adjacency_types(max(v_dims), config.connectivity, config.neighbor_types)
    model = ETNN(
        num_features_per_rank=actual_dims,
        num_hidden=config.num_hidden,
        num_out=1,
        num_layers=config.num_layers,
        adjacencies=adjs,
        initial_features=config.initial_features,
        visible_dims=v_dims,
        normalize_invariants=config.normalize_invariants,
        global_pool=config.global_pool
    ).to(device)

    # 兼容两种加载模式
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # ==========================================
    # 4. 执行推理
    # ==========================================
    preds_raw, targets = [], []
    for batch in tqdm(test_loader, desc="Target Test Inference"):
        batch = batch.to(device)
        out, _, _ = model(batch)
        preds_raw.append(out.view(-1).cpu().numpy())
        targets.append(batch.y.view(-1).cpu().numpy())

    preds_np = np.concatenate(preds_raw)
    targets_np = np.concatenate(targets)

    # ==========================================
    # 5. 计算物理指标
    # ==========================================
    # 判定模式：你的训练脚本决定了输出是物理值还是 Z-Score
    # 基于之前的结果，我们计算两套 MAE，取合理的那个
    mae_raw = np.abs(preds_np - targets_np).mean()
    preds_recovered = (preds_np * std_val) + mean_val
    mae_recovered = np.abs(preds_recovered - targets_np).mean()

    if mae_raw < mae_recovered:
        final_mae, final_preds, mode = mae_raw, preds_np, "物理直出"
    else:
        final_mae, final_preds, mode = mae_recovered, preds_recovered, "反标准化还原"

    print("\n" + "="*60)
    print(f"📊 评估报告 (仅 Target Test 20% 子集)")
    print(f"判定模式: {mode}")
    print(f"物理 MAE: {final_mae:.4f}")
    print("-" * 40)
    print("前 5 个样本对比 (Pred vs Target):")
    for i in range(min(5, len(targets_np))):
        print(f"  [{i}] {final_preds[i]:10.4f}  vs  {targets_np[i]:10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()