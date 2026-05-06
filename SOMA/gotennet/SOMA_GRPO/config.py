# config.py
import torch
from dataclasses import dataclass
import os
import re
import glob

@dataclass
class TrainingConfig:
    # ==========================================================
    # 1. 核心架构参数 (遵照官方标准与你的200轮设定)
    # ==========================================================
    hidden_channels: int = 128
    num_layers: int = 6
    num_heads: int = 8      # GotenNet底层不强依赖head，但保留以防扩展
    num_rbf: int = 32
    cutoff: float = 5.0
    reduce_op: str = "sum"
    epochs: int = 200       # 按你的要求：200轮
    
    # 修改：GotenNet最佳学习率
    lr: float = 5e-4
    weight_decay: float = 1e-8
    warmup_steps: int = 2000
    
    patience: int = 500
    lr_patience: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # ==========================================================
    # 2. 训练与显存控制 (针对 2080Ti 11G 优化)
    # ==========================================================
    batch_size: int = 28           
    inference_batch_size: int = 28 
    num_workers: int = 0           
    standardize: bool = True       
    cache_dir: str = "./cache_substructure"
    seed: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ========= 稳定性 =========
    grad_clip_norm: float = 0.5
    grad_clip_value: float = 0.5
    loss_upper_bound: float = 20.0
    label_name: str = "homo"

    # ==========================================================
    # 3. 领域自适应 (DA) 设置 (保留你所有的 SOMA 逻辑)
    # ==========================================================
    da_type: str = "coral"      
    sub_sample_size: int = 128  
    init_w_mol: float = 3e-4
    init_w_sub: float = 4e-5

    # ========= 数据路径 =========
    train_paths: list = None
    val_paths: list = None
    test_paths: list = None
    da_source_paths: list = None
    target_paths: list = None
    da_source_weights: list = None
    sup_source_paths: list = None

    def __post_init__(self):
        # 你的路径自适应逻辑
        if self.train_paths is None:
            self.train_paths = [f"data/a{i}.csv" for i in [0, 1, 6, 8, 9, 11]]
        if self.val_paths is None:
            self.val_paths = ["data/a2.csv", "data/a3.csv", "data/a4.csv", "data/a5.csv", "data/a7.csv"]
        if self.test_paths is None:
            self.test_paths = ["data/scaffold_datasets1/a033.csv"]
        if self.target_paths is None:
            self.target_paths = ["data/scaffold_datasets1/a013.csv"]
        if self.sup_source_paths is None:
            self.sup_source_paths = [f"data/a{i}.csv" for i in [0, 1, 6, 8, 9, 11]]
            
        if self.da_source_paths is None:
            auto_dir = "final_auto_da_sources"
            if os.path.exists(auto_dir) and len(os.listdir(auto_dir)) > 0:
                print(f">>> [Config] 识别到 GRPO 智能优化结果，加载精选骨架...")
                self.da_source_paths = sorted(glob.glob(os.path.join(auto_dir, "*.csv")))
            else:
                self.da_source_paths = []

        if self.da_source_paths and self.da_source_weights is None:
            raw_weights = []
            for path in self.da_source_paths:
                fname = os.path.basename(path)
                try:
                    sim_m = re.search(r'sim_([\d\.]+)', fname)
                    cnt_m = re.search(r'(?:cnt|count)_(\d+)', fname)
                    wl_sim = float(sim_m.group(1)) if sim_m else 0.5
                    count = float(cnt_m.group(1)) if cnt_m else 100
                    raw_weights.append(wl_sim * np.log1p(count))
                except:
                    raw_weights.append(1.0)
            sw = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
            self.da_source_weights = [w / sw for w in raw_weights]