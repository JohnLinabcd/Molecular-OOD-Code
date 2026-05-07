# config.py
import torch
from dataclasses import dataclass, field
import os, re, glob

@dataclass
class TrainingConfig:
    # ==========================================================
    # 1. ETNN 模型架构参数 (严格对齐之前的 GRPO 与监督学习)
    # ==========================================================
    num_hidden: int = 128
    num_layers: int = 4
    initial_features: list = field(default_factory=lambda: ["hetero", "node"])
    lifters: list = field(default_factory=lambda: ["atom:0", "bond:1", "clique:c", "ring:2"])
    connectivity: str = "self_and_next"
    neighbor_types: list = field(default_factory=lambda: ["+1", "-1"])
    
    # 特征消融开关 (默认全开：11维+BRICS)
    use_official_features: bool = True
    use_brics: bool = True
    global_pool: bool = True
    
    # ==========================================================
    # 2. 训练与显存控制
    # ==========================================================
    batch_size: int = 32
    inference_batch_size: int = 128
    seed: int = 42
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    # ✨ 核心修改：域适应实验专用的缓存目录
    cache_dir: str = "/etnnDA/etnnDA1/etnnDA_cache" 
    
    # ==========================================================
    # 3. 领域自适应 (DA) 设置
    # ==========================================================
    init_w_mol: float = 3e-4  
    init_w_sub: float = 4e-5  
    da_type: str = "coral"      
    sub_sample_size: int = 128 
    
    # ==========================================================
    # 4. 优化器设置
    # ==========================================================
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 5e-2
    grad_clip_norm: float = 1.0

    # ==========================================================
    # 5. 路径容器
    # ==========================================================
    label_name: str = "homo"
    # 全量源域
    sup_source_paths: list = field(default_factory=lambda: ["data/train_giant.csv"])
    # 目标域测试集
    target_paths: list = field(default_factory=lambda: ["data/scaffold_datasets1/a013.csv"])
    
    # 由 Selector 动态填充的路径和权重
    da_source_paths: list = field(default_factory=list)
    da_source_weights: list = field(default_factory=list)

    def __post_init__(self):
        """
        [ETNN 域适应专用] 自动加载智能选择器选中的 Top 5 骨架。
        没有复杂的审计，直接选用。
        """
        auto_dir = "./smart_selected_csvs"
        candidates = sorted(glob.glob(os.path.join(auto_dir, "score_*.csv")))
        
        if not candidates:
            print(">>> [Warning] smart_selected_csvs 目录为空！将仅执行纯监督训练。")
            self.da_source_paths = []
            self.da_source_weights = []
            return

        valid_paths = []
        valid_scores = []

        print(f"\n" + "="*60)
        print(f">>> [DA Config] 正在加载智能选取的 Top 5 辅助源...")
        print("="*60)

        for path in candidates[:5]:
            fname = os.path.basename(path)
            try:
                # 从文件名解析 GRPO 跑出来的评分
                score_match = re.search(r'score_([\d\.]+)', fname)
                score = float(score_match.group(1)) if score_match else 1.0
                
                valid_paths.append(path)
                valid_scores.append(score)
                print(f"    [Selected] {fname[:30]}... Score: {score:.4f}")
            except Exception:
                continue

        if valid_paths:
            self.da_source_paths = valid_paths
            sum_s = sum(valid_scores)
            self.da_source_weights = [s / sum_s for s in valid_scores]
            print(f">>> 最终选用辅助源数量: {len(self.da_source_paths)}")