import torch
from dataclasses import dataclass, field
import torch
from dataclasses import dataclass, field
import os, re, glob, sys
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

@dataclass
class TrainingConfig:
    # ==========================================================
    # 1. 模型架构参数
    # ==========================================================
    hidden_channels: int = 128
    num_layers: int = 6
    num_heads: int = 8
    num_rbf: int = 32
    
    # ==========================================================
    # 2. 训练与显存控制
    # ==========================================================
    batch_size: int = 32
    inference_batch_size: int = 128
    seed: int = 42
    device: str = "cuda:0"
    cache_dir: str = "./cache_substructure" 
    
    # ==========================================================
    # 3. 领域自适应 (DA) 设置
    # ==========================================================
    init_w_mol: float = 3e-4
    init_w_sub: float = 4e-5
    da_type: str = "coral"      
    sub_sample_size: int = 128
    
    # ==========================================================
    # 4. 优化器与周期设置
    # ==========================================================
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 5e-2
    grad_clip_norm: float = 1.0
    resume_checkpoint: str = ""

    # ==========================================================
    # 5. 路径容器
    # ==========================================================
    label_names: list = field(default_factory=lambda: ["homo"])
    sup_source_paths: list = field(default_factory=lambda: [f"data/a{i}.csv" for i in [0, 1, 6, 8, 9, 11]])
    target_paths: list = field(default_factory=lambda: ["data/scaffold_datasets1/a033.csv"])
    da_source_paths: list = field(default_factory=list)
    da_source_weights: list = field(default_factory=list)
    
    # 审计状态位
    audit_passed: bool = False

    @torch.no_grad()
    def _estimate_target_mean(self, device):
        """
        [严谨审计] 利用预训练模型预测目标域伪标签均值
        """
        from visnet.model import ViSNet
        from visnet.data import MultiTaskDataset
        import shutil

        ckpt_path = "s200_best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f">>> [Warning] 找不到预训练模型 {ckpt_path}，无法执行能级审计。")
            return None

        temp_path = os.path.join(self.cache_dir, "temp_config_audit")
        if os.path.exists(temp_path): shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
        
        target_ds = MultiTaskDataset(self.target_paths, temp_path, self.label_names)
        loader = DataLoader(target_ds, batch_size=self.inference_batch_size, shuffle=False)

        checkpoint = torch.load(ckpt_path, map_location=device)
        m_val = checkpoint['model_state_dict'].get('mean', torch.zeros(1,1))
        s_val = checkpoint['model_state_dict'].get('std', torch.ones(1,1))
        
        model = ViSNet(
            hidden_channels=self.hidden_channels, num_layers=self.num_layers,
            num_heads=self.num_heads, num_rbf=self.num_rbf,
            mean=m_val, std=s_val, num_tasks=1
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        preds = []
        for batch in loader:
            batch = batch.to(device)
            out, _, _ = model(batch.z, batch.pos, batch.batch, batch.sub_batch)
            preds.append(out.cpu().numpy())
        
        shutil.rmtree(temp_path)
        return np.mean(np.concatenate(preds))

    def __post_init__(self):
        # 1. 识别 Selector 选出的 Top 5
        auto_dir = "./smart_selected_csvs"
        candidates = sorted(glob.glob(os.path.join(auto_dir, "*.csv"))) if os.path.exists(auto_dir) else []
        
        if not candidates:
            self.audit_passed = False
            return

        # 辅助函数：解析文件名中的 score 用于排序
        def get_score_from_name(path):
            s_match = re.search(r'score_([\d\.]+)', os.path.basename(path))
            return float(s_match.group(1)) if s_match else 0.0

        # 2. 估计目标域能级基准
        main_device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        est_target_mean = self._estimate_target_mean(main_device)
        
        valid_paths = []
        valid_raw_weights = []

        print(f"\n" + "="*60)
        print(f">>> [V4 Audit] 正在执行物理一致性审计...")
        if est_target_mean is not None:
            print(f">>> 预测目标能级基准: {est_target_mean:.3f} eV")
        print("="*60)

        for path in candidates:
            fname = os.path.basename(path)
            try:
                # 解析 Sim
                sim_match = re.search(r'sim_([\d\.]+)', fname)
                sim = float(sim_match.group(1)) if sim_match else 0.5
                
                # --- A. 结构熔断 (Sim < 0.45) ---
                if sim < 0.45:
                    print(f"    [Block] {fname[:15]}... 结构不匹配 (Sim:{sim:.3f})")
                    continue

                df = pd.read_csv(path)
                y_vals = df[self.label_names[0]].values
                y_min, y_max, y_mean = y_vals.min(), y_vals.max(), y_vals.mean()
                y_median = np.median(y_vals)
                y_width = y_max - y_min
                sample_smi = df['smiles'].iloc[0]

                # --- B. 宽度审计 (2.5eV) ---
                if y_width < 2.5:
                    print(f"    [Block] {fname[:15]}... 范围过窄 ({y_width:.2f}eV)")
                    continue

                # --- C. 离子拦截 ---
                if '+' in sample_smi or '-' in sample_smi:
                    print(f"    [Block] {fname[:15]}... 离子态冲突")
                    continue

                # --- D. 能量核心区包含度 (15% 边际) ---
                if est_target_mean is not None:
                    margin = 0.15 * y_width
                    if not (y_min + margin <= est_target_mean <= y_max - margin):
                        print(f"    [Block] {fname[:15]}... 能量区间错位 (Inclusion OOR)")
                        continue

                # --- E. 分布偏度审计 (10% 阈值) ---
                if abs(y_mean - y_median) > 0.10 * y_width:
                    print(f"    [Block] {fname[:15]}... 分布严重偏斜")
                    continue

                # 通过审计，解析权重 (Score 线性比例)
                score_match = re.search(r'score_([\d\.]+)', fname)
                raw_w = float(score_match.group(1)) if score_match else sim
                
                valid_paths.append(path)
                valid_raw_weights.append(raw_w)

            except Exception as e:
                continue

        # ==========================================================
        # 新增功能：多样性补足 (Rescue Logic)
        # ==========================================================
        min_required = 2
        if 0 < len(valid_paths) < min_required:
            print(f">>> [Diversity Rescue] 审计通过数量({len(valid_paths)})不足，正在按 Score 补足至 {min_required} 个...")
            
            # 获取所有未入选的候选者，并按 Score 降序排序
            others = [c for c in candidates if c not in valid_paths]
            others.sort(key=lambda x: get_score_from_name(x), reverse=True)
            
            for extra_path in others:
                if len(valid_paths) >= min_required:
                    break
                valid_paths.append(extra_path)
                valid_raw_weights.append(get_score_from_name(extra_path))
                print(f"    [Rescued] 强制补回骨架: {os.path.basename(extra_path)}")

        # 3. 最终决策
        if valid_paths:
            self.da_source_paths = valid_paths
            self.audit_passed = True
            sum_s = sum(valid_raw_weights)
            self.da_source_weights = [s / sum_s for s in valid_raw_weights]
            print(f">>> [Audit Pass] 最终选用源数量: {len(self.da_source_paths)}")
            print(f">>> [Final Weights] {[round(w, 3) for w in self.da_source_weights]}\n")
        else:
            self.da_source_paths = []
            self.da_source_weights = []
            self.audit_passed = False
            print(">>> [Audit Failed] 所有候选骨架均未通过物理审计且无可用候选。\n")