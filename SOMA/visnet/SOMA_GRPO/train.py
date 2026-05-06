import os
import time
import torch  
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 导入你的核心模块
from visnet.model import ViSNet
from visnet.data import DADataModule  
from config import TrainingConfig
from utils import set_seed, ensure_dir, get_logger
from controller import AdaptiveDAWeightController 

# ================= 1. 损失函数 (与模拟战保持一致) =================

def coral_loss(source, target):
    """Correlation Alignment loss"""
    if source.size(0) < 2 or target.size(0) < 2: 
        return torch.tensor(0.0, device=source.device)
    d = source.data.shape[1]
    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(xm.t(), xm) / (source.size(0) - 1)
    # Target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(xmt.t(), xmt) / (target.size(0) - 1)
    # Frobenius norm
    return torch.norm(xc - xct, p='fro').pow(2) / (4 * d * d)

def mmd_loss(source, target):
    """Maximum Mean Discrepancy with multi-kernel"""
    if source.size(0) < 2 or target.size(0) < 2: 
        return torch.tensor(0.0, device=source.device)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_dist = ((total0 - total1)**2).sum(2)
    bandwidth = torch.sum(L2_dist.data) / (source.size(0) + target.size(0) - 1)
    kernels = [torch.exp(-L2_dist / (bandwidth * (2.0**i))) for i in range(5)]
    res = sum(kernels)
    n = source.size(0)
    XX, YY, XY = res[:n, :n], res[n:, n:], res[:n, n:]
    return torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)

def sample_substructures(feat, sample_size):
    """子结构特征随机采样以控制计算量"""
    if feat.size(0) <= sample_size: 
        return feat
    return feat[torch.randperm(feat.size(0), device=feat.device)[:sample_size]]

@torch.no_grad()
def evaluate(model, loader, device):
    """评估函数，返回真实尺度下的 MAE"""
    if loader is None: return 0.0
    model.eval()
    mae, cnt = 0, 0
    for b in loader:
        if b is None: continue
        b = b.to(device)
        # res[0] 为预测值
        res = model(b.z, b.pos, b.batch, b.sub_batch)
        mae += (res[0] - b.y.view(res[0].shape)).abs().sum().item()
        cnt += b.y.size(0)
    return mae / cnt if cnt > 0 else 0.0

# ================= 2. 核心训练流程 =================

def train_full_process(config: TrainingConfig):
    set_seed(config.seed)
    
    # 输出目录设置
    target_name = os.path.basename(config.target_paths[0]).split('.')[0]
    out_dir = f"./outputs/AutoDA_{target_name}"
    ensure_dir(out_dir)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    logger.info(f"🚀 开始实战训练！目标域: {target_name}")
    logger.info(f"📂 选用的DA源域: {[os.path.basename(p) for p in config.da_source_paths]}")
    
    # 准备数据加载器
    dm = DADataModule(vars(config))
    dm.prepare_dataset()
    
    source_sup_loader = dm.source_sup_loader() # 全量监督源
    source_da_loaders = dm.source_da_loaders() # GRPO选出的精英DA源
    da_source_weights = torch.tensor(config.da_source_weights, device=config.device)
    
    target_train_loader = dm.target_train_loader()
    target_test_loader = dm.target_test_loader()
    
    # 初始化 ViSNet
    mean, std = dm.mean.to(config.device), dm.std.to(config.device)
    model = ViSNet(
        hidden_channels=config.hidden_channels, 
        num_layers=config.num_layers, 
        num_heads=config.num_heads,
        num_rbf=config.num_rbf, 
        mean=mean, std=std, 
        num_tasks=len(config.label_names)
    ).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # 初始化 Controller
    controller = AdaptiveDAWeightController(config.init_w_mol, config.init_w_sub)
    
    # 权重初始化
    w_mol, w_sub, w_reg = 0.0, 0.0, 1.0 # 初始状态 (预热期)
    best_mae = float('inf')
    
    # --- 开始 Epoch 循环 ---
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_reg_loss = 0.0
        epoch_mol_raw = 0.0
        epoch_sub_raw = 0.0
        
        # 11 轮开始正式开启 DA 和权重控制
        if epoch > 10:
            active_w_mol = w_mol
            active_w_sub = w_sub
            active_w_reg = w_reg
        else:
            active_w_mol, active_w_sub, active_w_reg = 0.0, 0.0, 1.0

        da_iters = [iter(l) for l in source_da_loaders]
        target_iter = iter(target_train_loader)
        pbar = tqdm(source_sup_loader, desc=f"Epoch {epoch:03d}", leave=False) 
        
        for batch_s_sup in pbar:
            optimizer.zero_grad()
            
            # A. 监督回归损失 (全量库)
            batch_s_sup = batch_s_sup.to(config.device)
            out_s, _, _ = model(batch_s_sup.z, batch_s_sup.pos, batch_s_sup.batch, batch_s_sup.sub_batch)
            loss_reg = F.l1_loss(out_s, batch_s_sup.y.view(out_s.shape))
            
            # B. 领域自适应损失 (选中骨架组合)
            loss_da = torch.tensor(0.0, device=config.device)
            current_mol_raw = 0.0
            current_sub_raw = 0.0
            
            if active_w_mol > 0 or active_w_sub > 0:
                try: batch_t = next(target_iter).to(config.device)
                except StopIteration:
                    target_iter = iter(target_train_loader)
                    batch_t = next(target_iter).to(config.device)
                
                _, mol_t, sub_t = model(batch_t.z, batch_t.pos, batch_t.batch, batch_t.sub_batch)
                
                weighted_l_mol = torch.tensor(0.0, device=config.device)
                weighted_l_sub = torch.tensor(0.0, device=config.device)
                
                for i, s_iter in enumerate(da_iters):
                    try: batch_s_da = next(s_iter).to(config.device)
                    except StopIteration: continue
                    
                    _, mol_s, sub_s = model(batch_s_da.z, batch_s_da.pos, batch_s_da.batch, batch_s_da.sub_batch)
                    
                    alpha_i = da_source_weights[i]
                    
                    # 分子级对齐
                    l_m = coral_loss(mol_s, mol_t) if config.da_type == "coral" else mmd_loss(mol_s, mol_t)
                    weighted_l_mol += alpha_i * l_m
                    
                    # 子结构级对齐
                    s_s = sample_substructures(sub_s, config.sub_sample_size)
                    s_t = sample_substructures(sub_t, config.sub_sample_size)
                    l_s = coral_loss(s_s, s_t) if config.da_type == "coral" else mmd_loss(s_s, s_t)
                    weighted_l_sub += alpha_i * l_s
                
                current_mol_raw = weighted_l_mol.item()
                current_sub_raw = weighted_l_sub.item()
                loss_da = (active_w_mol * weighted_l_mol) + (active_w_sub * weighted_l_sub)

            # 总损失
            total_loss = (active_w_reg * loss_reg) + loss_da
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            
            epoch_reg_loss += loss_reg.item()
            epoch_mol_raw += current_mol_raw
            epoch_sub_raw += current_sub_raw

        # 计算本轮均值
        num_batches = len(source_sup_loader)
        avg_reg = epoch_reg_loss / num_batches
        avg_mol = epoch_mol_raw / num_batches
        avg_sub = epoch_sub_raw / num_batches
        
        # 更新自适应权重
        if epoch >= 10:
            w_mol, w_sub, w_reg = controller.update(epoch, avg_reg, avg_mol, avg_sub)

        # 验证集评估
        tgt_mae = evaluate(model, target_test_loader, config.device)
        scheduler.step(tgt_mae)
        
        # 记录日志
        logger.info(f"Ep {epoch:03d} | Reg:{avg_reg:.4f} molDA:{avg_mol*1000:.2f} subDA:{avg_sub*1000:.2f} | "
                    f"W_reg:{active_w_reg:.2f} W_mol:{active_w_mol:.1e} W_sub:{active_w_sub:.1e} | "
                    f"TgtMAE:{tgt_mae:.4f}")

        # 保存模型
        if tgt_mae < best_mae:
            best_mae = tgt_mae
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'mae': tgt_mae}, 
                       os.path.join(out_dir, "best_model.pt"))
            logger.info(f" >>> ✨ 发现更优模型，已保存。")

        if epoch == config.epochs:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, 
                       os.path.join(out_dir, "last_model.pt"))

if __name__ == "__main__":
    train_full_process(TrainingConfig())