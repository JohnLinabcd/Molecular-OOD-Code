# train.py
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import TrainingConfig
from data import DADataModule
from controller import AdaptiveDAWeightController
from etnn.model import ETNN
from etnn.lifter import get_adjacency_types
from utils import set_seed, ensure_dir, get_logger

# 解决 PyTorch 2.6+ 兼容性问题
from etnn.combinatorial_data import CombinatorialComplexData
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([CombinatorialComplexData])

# --- 核心：相关性对齐 (CORAL) 损失 ---
def coral_loss(source, target):
    """计算源域特征与目标域特征协方差矩阵之间的差异"""
    if source.size(0) < 2 or target.size(0) < 2: 
        return torch.tensor(0.0, device=source.device)
    
    d = source.size(1)
    
    # 均值中心化
    xm = source - torch.mean(source, 0, keepdim=True)
    xc = torch.matmul(xm.t(), xm) / (source.size(0) - 1)
    
    xmt = target - torch.mean(target, 0, keepdim=True)
    xct = torch.matmul(xmt.t(), xmt) / (target.size(0) - 1)
    
    # 弗罗贝尼乌斯范数 (Frobenius norm)
    loss = torch.norm(xc - xct, p='fro').pow(2) / (4 * d * d)
    return loss

def sample_substructures(feat, sample_size):
    """如果子结构数量过多，随机采样以稳定梯度的计算和内存占用"""
    if feat.size(0) <= sample_size: 
        return feat
    # 随机打乱并截取前 sample_size 个
    indices = torch.randperm(feat.size(0), device=feat.device)[:sample_size]
    return feat[indices]

@torch.no_grad()
def evaluate(model, loader, device, std, mean):
    model.eval()
    mae_sum = 0.0
    total_cnt = 0
    
    for batch in loader:
        if batch is None: continue
        batch = batch.to(device)
        
        out, _, _ = model(batch)
        if out.ndim == 1: 
            out = out.unsqueeze(-1)
            
        # --- 修改这里：直接使用 out，不要再乘 std 加 mean ---
        pred_real = out 
        target_real = batch.y
        
        mae_sum += (pred_real - target_real).abs().sum().item()
        total_cnt += batch.y.size(0)
        
    return mae_sum / total_cnt if total_cnt > 0 else 0.0

def get_actual_dims(dm, config):
    """
    自动推断输入的特征维度，防止写死配置导致模型初始化失败。
    """
    sample = dm.source_sup_dataset[0]
    v_dims = list(sample.num_features_per_rank.keys())
    
    base_hetero_dims = {}
    for r in v_dims:
        attr_name = f"x_{r}"
        base_hetero_dims[r] = getattr(sample, attr_name).shape[1] if hasattr(sample, attr_name) else 0

    node_feat_dim = sample.x.shape[1] if hasattr(sample, 'x') else 0
    num_lifters = len(config.lifters)

    actual_dims = {}
    for r in v_dims:
        total_d = 0
        for f_type in config.initial_features:
            if f_type == "hetero":
                total_d += base_hetero_dims.get(r, 0)
            elif f_type == "node":
                total_d += node_feat_dim
            elif f_type == "mem":
                total_d += num_lifters
        actual_dims[r] = total_d
        
    return actual_dims, v_dims

def train_curriculum(config: TrainingConfig):
    set_seed(config.seed)
    
    # 1. 触发 Config 初始化 (读取强化学习输出的 Top 5 骨架)
    config.__post_init__() 
    
    target_name = os.path.basename(config.target_paths[0])
    out_dir = f"./outputs/DA_ETNN_{target_name}"
    ensure_dir(out_dir)
    logger = get_logger(os.path.join(out_dir, "train.log"))

    if len(config.da_source_paths) == 0:
        logger.warning("未找到任何智能选取的辅助源！将退化为纯监督基准训练。")

    logger.info(f"🚀 开始 ETNN 域适应训练 | 目标: {target_name} | 设备: {config.device}")
    if config.da_source_weights:
        logger.info(f"🧬 辅助源融合权重: {[round(w, 4) for w in config.da_source_weights]}")

    # 2. 准备数据
    cfg_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    dm = DADataModule(cfg_dict)
    dm.prepare_dataset()
    
    source_sup_loader = dm.source_sup_loader()
    source_da_loaders = dm.source_da_loaders()
    da_weights = torch.tensor(config.da_source_weights, device=config.device) if config.da_source_weights else None
    
    target_train_loader = dm.target_train_loader()
    target_test_loader = dm.target_test_loader()
    
    mean, std = dm.mean.to(config.device), dm.std.to(config.device)
    
    # 3. 初始化 ETNN 模型
    actual_dims, v_dims = get_actual_dims(dm, config)
    adjs = get_adjacency_types(max(v_dims), config.connectivity, config.neighbor_types)
    
    model = ETNN(
        num_features_per_rank=actual_dims, 
        num_hidden=config.num_hidden, 
        num_out=1, 
        num_layers=config.num_layers,
        adjacencies=adjs, 
        initial_features=config.initial_features,
        visible_dims=v_dims, 
        normalize_invariants=True, 
        global_pool=config.global_pool
    ).to(config.device)

    # 4. 优化器与动态权重控制器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 110, 140, 170], gamma=0.5)
    controller = AdaptiveDAWeightController(config.init_w_mol, config.init_w_sub)
    
    w_mol, w_sub, w_reg = config.init_w_mol, config.init_w_sub, 1.0
    best_mae = float('inf')

    # ================= 训练主循环 =================
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        
        epoch_reg_sum, epoch_mol_sum, epoch_sub_sum = 0.0, 0.0, 0.0
        
        # 前 10 个 Epoch 只做回归 (Warmup)，之后开启 DA
        active_w_mol = w_mol if epoch > 10 else 0.0
        active_w_sub = w_sub if epoch > 10 else 0.0
        active_w_reg = w_reg

        da_iters = [iter(l) for l in source_da_loaders]
        target_iter = iter(target_train_loader)
        
        num_batches = len(source_sup_loader)
        pbar = tqdm(source_sup_loader, desc=f"Ep{epoch:03d}", leave=False)
        
        for batch_s_sup in pbar:
            optimizer.zero_grad()
            batch_s_sup = batch_s_sup.to(config.device)
            
            # ✨ A. 前向传播：全量源域 (回归任务)
            out_s, _, _ = model(batch_s_sup)
            if out_s.ndim == 1: out_s = out_s.unsqueeze(-1)
            
            loss_reg = F.l1_loss((out_s - mean)/std, (batch_s_sup.y - mean)/std)
            loss_da = torch.tensor(0.0, device=config.device)
            
            current_mol_raw = 0.0
            current_sub_raw = 0.0

            # ✨ B. 域适应对齐 (DA 任务)
            if (active_w_mol > 0 or active_w_sub > 0) and da_weights is not None:
                # 获取 Target Batch
                try: 
                    batch_t = next(target_iter).to(config.device)
                except StopIteration: 
                    target_iter = iter(target_train_loader)
                    batch_t = next(target_iter).to(config.device)
                
                # 提取 Target 的特征
                _, mol_t, sub_t = model(batch_t)
                
                l_m_total = torch.tensor(0.0, device=config.device)
                l_s_total = torch.tensor(0.0, device=config.device)
                
                # 遍历所有被选中的辅助源 (DA Sources)
                for i, s_iter in enumerate(da_iters):
                    try: 
                        batch_s = next(s_iter).to(config.device)
                    except StopIteration: 
                        # # 如果某一个小源用完了，重置它
                        # da_iters[i] = iter(source_da_loaders[i])
                        # batch_s = next(da_iters[i]).to(config.device)
                        continue
                    
                    # 提取 Auxiliary Source 的特征
                    _, mol_s, sub_s = model(batch_s)
                    
                    alpha_i = da_weights[i]
                    
                    # 分子级 (Global) 对齐
                    if active_w_mol > 0: 
                        l_m_total += alpha_i * coral_loss(mol_s, mol_t)
                    
                    # 子结构级 (BRICS) 对齐
                    if active_w_sub > 0: 
                        sub_s_sampled = sample_substructures(sub_s, config.sub_sample_size)
                        sub_t_sampled = sample_substructures(sub_t, config.sub_sample_size)
                        l_s_total += alpha_i * coral_loss(sub_s_sampled, sub_t_sampled)
                
                current_mol_raw = l_m_total.item()
                current_sub_raw = l_s_total.item()
                
                # 加权并截断异常梯度
                loss_da = (active_w_mol * torch.clamp(l_m_total, max=10.0)) + \
                          (active_w_sub * torch.clamp(l_s_total, max=5.0))
                          
                epoch_mol_sum += current_mol_raw
                epoch_sub_sum += current_sub_raw

            # ✨ C. 最终梯度反向传播
            total_loss = (active_w_reg * loss_reg) + loss_da
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            
            epoch_reg_sum += loss_reg.item()

        # --- 结算与评估 ---
        avg_reg = epoch_reg_sum / num_batches
        avg_mol = epoch_mol_sum / num_batches
        avg_sub = epoch_sub_sum / num_batches
        
        # 动态更新控制器权重
        if epoch >= 10: 
            w_mol, w_sub, w_reg = controller.update(epoch, avg_reg, avg_mol, avg_sub)
        
        # 测试目标域表现 (真实物理尺度 MAE)
        tgt_mae = evaluate(model, target_test_loader, config.device, std, mean)
        
        epoch_dur = time.time() - epoch_start_time
        curr_lr = optimizer.param_groups[0]['lr']

        # --- 日志输出 ---
        log_line = (
            f"Ep {epoch:03d}/{config.epochs} | Time: {epoch_dur:.1f}s | LR: {curr_lr:.1e} | "
            f"MAE: {tgt_mae:.4f} "
        )
        
        if tgt_mae < best_mae:
            best_mae = tgt_mae
            log_line += "⭐ | "
            torch.save({'model_state_dict': model.state_dict(), 'mae': tgt_mae, 'epoch': epoch}, 
                       os.path.join(out_dir, "best_model.pt"))
        else:
            log_line += "   | "
        
        log_line += (
            f"Reg: {avg_reg:.4f} | molDA: {avg_mol*1000:.3f} | subDA: {avg_sub*1000000:.3f} | "
            f"W(r,m,s): ({w_reg:.2f}, {active_w_mol:.1e}, {active_w_sub:.1e})"
        )
        logger.info(log_line)

        if epoch == config.epochs:
            torch.save({'model_state_dict': model.state_dict(), 'mae': tgt_mae, 'epoch': epoch}, 
                       os.path.join(out_dir, "last_model.pth"))

        scheduler.step()

if __name__ == "__main__":
    cfg = TrainingConfig()
    
    # 1. 强制运行 Selector
    from selector_utils import run_smart_selection
    run_smart_selection(cfg)
    
    # 2. 运行完后，此时 smart_selected_csvs 文件夹里有文件了
    # 我们需要手动让 cfg 重新扫描一次该文件夹
    cfg.__post_init__() 
    
    # 3. 开始训练
    train_curriculum(cfg)