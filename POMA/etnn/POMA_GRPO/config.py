import torch
from dataclasses import dataclass, field
from typing import Optional, List
import os
import glob
import re
import numpy as np

@dataclass
class TrainingConfig:
    use_official_features: bool = True
    use_brics: bool = True
    # ==========================================================
    # 1. 核心架构参数 (ETNN 拓扑配置)
    # ==========================================================
    lifters: list = field(default_factory=lambda: ["atom:0", "bond:1", "clique:c", "ring:2"])
    connectivity: str = "self_and_next"
    neighbor_types: list = field(default_factory=lambda: ["+1", "-1"])
    num_hidden: int = 128
    num_layers: int = 4
    initial_features: list = field(default_factory=lambda: ["hetero", "node"])
    normalize_invariants: bool = True
    batch_norm: bool = False
    lean: bool = True
    global_pool: bool = True
    sparse_invariant_computation: bool = False
    pos_update: bool = False

    # ==========================================================
    # 2. 训练与显存控制 (针对 2080Ti 优化)
    # ==========================================================
    batch_size: int = 40           
    inference_batch_size: int = 40
    num_workers: int = 0           
    standardize: bool = False      # GRPO 中通常手动传入 mean/std
    # cache_dir: str = "./cache_substructure"
    # 强制将缓存目录设在 /data 分区，解决 /home 空间不足
    cache_dir: str = "/data/etnngrpo_cache"
    seed: int = 42
    device: str = "cuda:0"         

    # ==========================================================
    # 3. 领域自适应 (DA) 与 优化器设置
    # ==========================================================
    da_type: str = "coral"      
    epochs: int = 200           
    lr: float = 1e-4            
    weight_decay: float = 5e-2  
    grad_clip_norm: float = 0.5 
    loss_upper_bound: float = 20.0

    # ==========================================================
    # 4. 路径容器
    # ==========================================================
    label_name: str = "homo" 
    sup_source_paths: Optional[List[str]] = None 
    da_source_paths: Optional[List[str]] = None  
    target_paths: Optional[List[str]] = None

    def __post_init__(self):
        if self.sup_source_paths is None:
            # self.sup_source_paths = ["data/a0.csv", "data/a1.csv", "data/a6.csv", "data/a8.csv", "data/a9.csv", "data/a11.csv"]
            self.sup_source_paths = ["data/train_giant.csv"]
        if self.target_paths is None:
            self.target_paths = ["data/scaffold_datasets1/a022.csv"]
        if self.da_source_paths is None:
            self.da_source_paths = []