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

    # ========= 训练参数 (融合你的 200 轮与官方设定) =========
    epochs: int = 200            
    lr: float = 5e-4             # 3D GNN 最佳启动学习率
    weight_decay: float = 1e-8   # 防止等变特征坍缩
    warmup_steps: int = 1000     
    patience: int = 500
    
    lr_patience: int = 15        
    lr_factor: float = 0.8       
    min_lr: float = 1e-6

    # ========= 稳定性 =========
    grad_clip_norm: float = 10.0 
    loss_upper_bound: float = 50.0

    # ========= 模型参数 (GotenNet 官方架构) =========
    hidden_channels: int = 256   
    num_layers: int = 4          
    cutoff: float = 5.0          
    reduce_op: str = "sum"       

    # ========= 其他 =========
    device: str = "cuda:2" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ========= 数据路径 =========
    train_paths: list = None
    val_paths: list = None
    test_paths: list = None

    def __post_init__(self):
        if self.train_paths is None:
            self.train_paths = ["data/a0.csv", "data/a1.csv", "data/a6.csv", "data/a8.csv", "data/a9.csv", "data/a11.csv"]
        if self.val_paths is None:
            self.val_paths = ["data/a10.csv"]
        if self.test_paths is None:
            self.test_paths = ["data/scaffold_datasets1/a022.csv"]