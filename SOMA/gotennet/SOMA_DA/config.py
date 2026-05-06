import torch
from dataclasses import dataclass
import os
import re
import glob
import numpy as np

@dataclass
class TrainingConfig:
    # ==========================================================
    # 1. 模型架构参数 (GotenNet 纯净版)
    # ==========================================================
    hidden_channels: int = 256
    num_layers: int = 4
    num_rbf: int = 50
    cutoff: float = 5.0
    reduce_op: str = "sum"
    
    # ==========================================================
    # 2. 训练与显存控制
    # ==========================================================
    epochs: int = 200       
    lr: float = 5e-4
    weight_decay: float = 1e-8
    warmup_steps: int = 2000
    
    batch_size: int = 28           
    inference_batch_size: int = 28 
    num_workers: int = 0           
    standardize: bool = True       
    cache_dir: str = "./cache_substructure"
    seed: int = 42
    device: str = "cuda:2" if torch.cuda.is_available() else "cpu"

    grad_clip_norm: float = 1.0
    label_names: list = None
    
    # ==========================================================
    # 3. 领域自适应 (DA) 参数
    # ==========================================================     
    da_type: str = "coral"      
    init_w_mol: float = 6e-7
    init_w_sub: float = 8e-8

    # ==========================================================
    # 4. 路径容器
    # ==========================================================
    train_paths: list = None
    val_paths: list = None
    test_paths: list = None
    target_paths: list = None
    sup_source_paths: list = None
    da_source_paths: list = None
    da_source_weights: list = None

    def __post_init__(self):
        if self.label_names is None:
            self.label_names = ["homo"]
        
        # 定义你的基础隔离数据路径
        if self.sup_source_paths is None:
            self.sup_source_paths = [f"data/a{i}.csv" for i in [0, 1, 6, 8, 9, 11]]
        if self.target_paths is None:
            self.target_paths = ["data/scaffold_datasets1/a013.csv"]
            
        # 智能骨架加载逻辑
        if self.da_source_paths is None:
            auto_dir = "smart_selected_csvs"
            if os.path.exists(auto_dir) and len(os.listdir(auto_dir)) > 0:
                print(f">>> [Config] 识别到 Selector 智能优化结果，加载精选骨架...")
                self.da_source_paths = sorted(glob.glob(os.path.join(auto_dir, "*.csv")))
            else:
                self.da_source_paths = []

        # 动态权重解析
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