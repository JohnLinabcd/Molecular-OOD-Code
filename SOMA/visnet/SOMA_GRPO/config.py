import torch
from dataclasses import dataclass
import os
import re
import numpy as np
import glob

@dataclass
class TrainingConfig:
    # ==========================================================
    # 1. 核心架构参数 (遵照官方标准，严禁修改)
    # ==========================================================
    hidden_channels: int = 128
    num_layers: int = 6
    num_heads: int = 8
    num_rbf: int = 32
    cutoff: float = 5.0

    # ==========================================================
    # 2. 训练与显存控制 (针对 2080Ti 11G 优化)
    # ==========================================================
    batch_size: int = 28           # 128通道下，32是11G显存的稳健上限
    inference_batch_size: int = 28 # 推理阶段同步控制，防止 OOM
    num_workers: int = 0           # 0最稳，防止多进程死锁
    standardize: bool = True       # 标签标准化
    cache_dir: str = "./cache_substructure"
    seed: int = 42
    device: str = "cuda:0"         # 默认主卡

    # ==========================================================
    # 3. 领域自适应 (DA) 设置
    # ==========================================================
    da_type: str = "coral"      
    sub_sample_size: int = 128  
    init_w_mol: float = 3e-4
    init_w_sub: float = 4e-5

    # ==========================================================
    # 4. 优化器与周期设置 (实战参数)
    # ==========================================================
    epochs: int = 200           
    lr: float = 1e-4            
    weight_decay: float = 5e-2  
    grad_clip_norm: float = 1.0 
    resume_checkpoint: str = "" 
    checkpoint_dir: str = "./checkpoints"

    # ==========================================================
    # 5. 路径容器 (动态初始化)
    # ==========================================================
    label_names: list = None 
    sup_source_paths: list = None 
    da_source_paths: list = None  
    target_paths: list = None
    da_source_weights: list = None 

    def __post_init__(self):
        # 默认预测任务
        if self.label_names is None: self.label_names = ["homo"] 
        
        # 默认全量监督源路径
        if self.sup_source_paths is None:
            self.sup_source_paths = [f"data/a{i}.csv" for i in [0, 1, 6, 8, 9, 11]]

        # 默认目标域路径 (实战目标)
        if self.target_paths is None:
            self.target_paths = ["data/scaffold_datasets1/a013.csv"]

        # 自动识别 DA 数据源：如果 GRPO 跑出了结果，优先用结果
        if self.da_source_paths is None:
            auto_dir = "final_auto_da_sources"
            if os.path.exists(auto_dir) and len(os.listdir(auto_dir)) > 0:
                print(f">>> [Config] 识别到 GRPO 智能优化结果，加载精选骨架...")
                self.da_source_paths = sorted(glob.glob(os.path.join(auto_dir, "*.csv")))
            else:
                self.da_source_paths = []

        # 自动计算初始权重 (Sim * log(Count))
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
            
            sum_w = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
            self.da_source_weights = [w / sum_w for w in raw_weights]

if __name__ == "__main__":
    config = TrainingConfig()
    print(">>> TrainingConfig 初始化成功。")