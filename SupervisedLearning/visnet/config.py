# config.py
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # ========= DataModule 必需 =========
    label_name: str = "gap"
    batch_size: int = 32
    inference_batch_size: int = 64
    num_workers: int = 4
    standardize: bool = True
    reload: bool = False
    cache_dir: str = "./cache"

    # ========= 训练 =========
    epochs: int = 200
    
    # ##### 修改: 降低学习率 (3e-4 -> 1e-4) 防止梯度爆炸 #####
    lr: float = 1e-4
    
    weight_decay: float = 1e-2
    
    # ##### 修改: 增加热身步数 (1000 -> 2000) 平滑启动 #####
    warmup_steps: int = 2000
    
    patience: int = 500

    lr_patience: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # ========= 稳定性 =========
    # ##### 修改: 加严梯度裁剪 (1.0 -> 0.5) #####
    grad_clip_norm: float = 0.5
    grad_clip_value: float = 0.5
    loss_upper_bound: float = 20.0

    # ========= 模型 =========
    hidden_channels: int = 128
    num_heads: int = 8
    num_layers: int = 6
    num_rbf: int = 32
    cutoff: float = 5.0
    max_num_neighbors: int = 32
    lmax: int = 1
    vertex: bool = False
    reduce_op: str = "sum"

    # ========= 其他 =========
    device: str = "cuda:3" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ========= 数据路径 =========
    train_paths: list = None
    val_paths: list = None
    test_paths: list = None

    def __post_init__(self):
        if self.train_paths is None:
            self.train_paths = [
                "data/a0.csv", "data/a1.csv", "data/a6.csv",
                "data/a8.csv", "data/a9.csv", "data/a11.csv"
            ]
        if self.val_paths is None:
            self.val_paths = [
                "data/a10.csv"
            ]
        if self.test_paths is None:
            self.test_paths = ["data/scaffold_datasets1/a033.csv"]