import os
import torch
import numpy as np
import random
from tqdm import tqdm
from argparse import Namespace

# 💥 核心：强制单线程计算，配合底层 sorted()，保证结果小数点后 8 位都不变
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# 导入你的底层组件
from data import DADataModule
from etnn.model import ETNN
from etnn.lifter import get_adjacency_types
from etnn.combinatorial_data import CombinatorialComplexData

# 适配 PyTorch 2.6+ 安全加载
if hasattr(torch.serialization, 'add_safe_globals'):
    import numpy
    torch.serialization.add_safe_globals([
        CombinatorialComplexData, numpy.core.multiarray.scalar, numpy.dtype
    ])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def main():
    # ==========================================
    # 🎯 1. 核心配置 (根据你的实际路径修改)
    # ==========================================
    MODEL_PATH = "sup_200.pth"                             # 监督学习生成的权重
    TARGET_CSV = "data/scaffold_datasets1/a013.csv"        # 目标数据集 CSV
    # ✨ 指向你之前确认的那个 pkl 所在的根目录
    TARGET_CACHE_ROOT = "/data/lzh/etnnDA/etnnDA1/etnnDA_cache"
    DEVICE = torch.device("cpu") # 建议用 CPU 运行以确保 100% 结果可复现

    set_seed(42)
    print(f"\n>>> [1/4] 正在从 {MODEL_PATH} 提取模型标尺与权重...")
    
    # 加载 Checkpoint (weights_only=False 以加载 meta_info 字典)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    meta = ckpt['meta_info']
    
    # 提取物理标尺和架构参数
    train_mean = meta['mean']
    train_std = meta['std']
    actual_dims = meta['actual_num_features_per_rank']
    v_dims = meta['visible_dims']
    train_config_dict = meta['config_dict']

    # ==========================================
    # 🎯 2. 数据对齐：通过 DA 逻辑精准提取 20% 测试集
    # ==========================================
    print(f">>> [2/4] 正在加载本地缓存数据: {TARGET_CACHE_ROOT}")
    
    # 构造最简 DA 配置，锁定本地路径
    eval_config = train_config_dict.copy()
    eval_config.update({
        'target_paths': [TARGET_CSV],
        'sup_source_paths': [TARGET_CSV], # 占位防报错
        'da_source_paths': [],
        'cache_dir': TARGET_CACHE_ROOT
    })

    dm = DADataModule(eval_config)
    dm.prepare_dataset()
    
    # ✨ 核心操作：不再重新计算，直接强行覆盖为 pth 里的物理标尺
    dm._mean = torch.tensor([train_mean]).to(DEVICE)
    dm._std = torch.tensor([train_std]).to(DEVICE)
    
    test_loader = dm.target_test_loader()
    print(f"✅ 缓存路径确认: {dm.target_test.dataset.cache_file}")
    print(f"✅ 样本对齐完成: 目标测试集共 {len(dm.target_test)} 个样本")

    # ==========================================
    # 3. 模型初始化 (参数完全来自 pth)
    # ==========================================
    print(f">>> [3/4] 正在初始化 ETNN 模型架构...")
    cfg = Namespace(**eval_config)
    
    # 这里会调用你已经加上 sorted() 的 get_adjacency_types
    adjs = get_adjacency_types(max(v_dims), cfg.connectivity, cfg.neighbor_types)
    
    model = ETNN(
        num_features_per_rank=actual_dims,
        num_hidden=cfg.num_hidden,
        num_out=1,
        num_layers=cfg.num_layers,
        adjacencies=adjs,
        initial_features=cfg.initial_features,
        visible_dims=v_dims,
        normalize_invariants=cfg.normalize_invariants,
        global_pool=cfg.global_pool
    ).to(DEVICE)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ==========================================
    # 4. 执行全量推理 (物理尺度)
    # ==========================================
    print(f">>> [4/4] 正在进行最终推理...")
    preds, targets = [], []
    
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(DEVICE)
        out, _, _ = model(batch)
        
        preds.append(out.view(-1).cpu().numpy())
        targets.append(batch.y.view(-1).cpu().numpy())

    preds_np = np.concatenate(preds)
    targets_np = np.concatenate(targets)
    
    # 计算物理 MAE
    final_mae = np.abs(preds_np - targets_np).mean()

    # ==========================================
    # 输出最终评估报告
    # ==========================================
    print("\n" + "█"*60)
    print(f"📊 最终评估报告 (确定性 & 路径解耦模式)")
    print(f"🎯 最终物理 MAE: {final_mae:.6f}")
    print(f"统计标尺: Mean={train_mean:.4f}, Std={train_std:.4f}")
    print("█"*60)
    
    print("\n样本预测细节 (前 5 个):")
    for i in range(min(5, len(targets_np))):
        print(f"  [{i:03d}] 预测: {preds_np[i]:10.4f} | 真实: {targets_np[i]:10.4f} | 误差: {abs(preds_np[i]-targets_np[i]):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()