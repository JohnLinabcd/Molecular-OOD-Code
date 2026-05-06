# config.py
import torch
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    use_official_features: bool = True
    use_brics: bool = True
    # ========= 1. 数据路径 (严格还原要求) =========
    label_name: str = "lumo"
    train_paths: Optional[List[str]] = None
    val_paths: Optional[List[str]] = None
    test_paths: Optional[List[str]] = None

    # ========= 2. 数据处理配置 (关键修改点) =========
    batch_size: int = 32  # 调小以减轻计算瞬间压力
    inference_batch_size: int = 64
    # 调试 SegFault 期间必须为 0，否则看不到报错行
    num_workers: int = 0 
    standardize: bool = True
    reload: bool = False
    # 强制将缓存目录设在 /data 分区，解决 /home 空间不足
    cache_dir: str = "/data/etnn_cache"

    # ========= 3. ETNN 拓扑配置 (Lifting) =========
    # 理由：根据 ring.py 定义，Rank 设为 2，以匹配化学特征
    lifters: list = ("atom:0", "bond:1", "clique:c", "ring:2")
    connectivity: str = "self_and_next"
    neighbor_types: list = ("+1", "-1")
    
    # ========= 4. ETNN 模型参数 =========
    num_hidden: int = 128
    num_layers: int = 4
    initial_features: list = ("hetero", "node")
    normalize_invariants: bool = True
    batch_norm: bool = False
    lean: bool = True
    global_pool: bool = True
    sparse_invariant_computation: bool = False
    pos_update: bool = False

    # ========= 5. 训练与优化配置 =========
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 2000
    patience: int = 50
    lr_patience: int = 15
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip_norm: float = 0.5
    loss_upper_bound: float = 30.0

    # ========= 6. 运行环境 =========
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def __post_init__(self):
        if self.train_paths is None:
            self.train_paths = ["data/train_giant.csv"]
        if self.val_paths is None:
            self.val_paths = ["data/val_giant.csv"]
        if self.test_paths is None:
            self.test_paths = ["data/scaffold_datasets1/a022.csv"]
        
        # 自动创建缓存目录
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except:
                print(f"Warning: Cannot create {self.cache_dir}, falling back to ./cache")
                self.cache_dir = "./cache"
                os.makedirs(self.cache_dir, exist_ok=True)