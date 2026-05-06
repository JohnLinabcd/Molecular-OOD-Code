# train.py
import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import TrainingConfig
from data import DataModule
from utils import (
    get_logger, set_seed, ensure_dir, save_config,
    EarlyStopping, compute_metrics, plot_training_curves
)
from etnn.model import ETNN
from etnn.lifter import get_adjacency_types

# ✨ 新增：一个全能的检查点保存函数，代替 utils.save_model
def save_full_checkpoint(model, optimizer, epoch, val_metrics, config, mean, std, actual_dims, v_dims, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        # 把关键的复现信息全部固化到 pth 里
        'meta_info': {
            'mean': mean,
            'std': std,
            'actual_num_features_per_rank': actual_dims,
            'visible_dims': v_dims,
            'config_dict': config.__dict__ if hasattr(config, '__dict__') else config
        }
    }
    torch.save(checkpoint, path)

def train_one_epoch(model, dataloader, optimizer, criterion, device, mean, std, config, warmup_scheduler, global_step):
    model.train()
    total_loss, total_mae, n_batch = 0.0, 0.0, 0
    safe_std = max(std.item() if hasattr(std, 'item') else std, 1e-6)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred, _, _ = model(batch)
        if pred.ndim == 1: pred = pred.unsqueeze(-1)
        
        if torch.isnan(pred).any(): continue

        # 物理尺度的 TrainMAE 用于日志监测
        with torch.no_grad():
            batch_mae = (pred - batch.y).abs().mean().item()
            total_mae += batch_mae

        # 仅在计算 Loss 时使用标准化
        loss = criterion((pred - mean)/safe_std, (batch.y - mean)/safe_std)

        if loss.item() > config.loss_upper_bound: continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if global_step < config.warmup_steps:
            warmup_scheduler.step()
        
        global_step += 1
        total_loss += loss.item()
        n_batch += 1

    return total_loss / max(1, n_batch), total_mae / max(1, n_batch), global_step

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = batch.to(device)
        pred, _, _ = model(batch)
        if pred.ndim == 1: pred = pred.unsqueeze(-1)
        preds.append(pred.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    if not preds: return {'mae': float('inf')}
    
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    return compute_metrics(preds, targets)


def main(config: TrainingConfig):
    set_seed(config.seed)
    device = torch.device(config.device)
    
    out_dir = f"./outputs/{config.label_name}_ETNN_Project"
    ensure_dir(out_dir)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    save_config(config, os.path.join(out_dir, "config.json"))

    # 1. 准备数据
    dm = DataModule(config)
    dm.prepare_dataset()
    
    # 提取并保存纯数值的 mean 和 std，方便序列化
    mean_val = dm._mean.item() if config.standardize and dm._mean is not None else 0.0
    std_val = dm._std.item() if config.standardize and dm._std is not None else 1.0
    
    # --- 维度推断逻辑 ---
    sample = dm.train_ds[0]
    
    # ✨ 修改：强制将维度按 0, 1, 2 升序排列，杜绝乱序 Bug
    v_dims = sorted(list(sample.num_features_per_rank.keys()))
    
    base_hetero_dims = {}
    for r in v_dims:
        attr_name = f"x_{r}"
        if hasattr(sample, attr_name):
            base_hetero_dims[r] = getattr(sample, attr_name).shape[1]
        else:
            base_hetero_dims[r] = 0

    node_feat_dim = sample.x.shape[1] if hasattr(sample, 'x') else 0
    num_lifters = len(config.lifters)

    actual_num_features_per_rank = {}
    for r in v_dims:
        total_d = 0
        for f_type in config.initial_features:
            if f_type == "hetero": total_d += base_hetero_dims.get(r, 0)
            elif f_type == "node": total_d += node_feat_dim
            elif f_type == "mem": total_d += num_lifters
        actual_num_features_per_rank[r] = total_d

    logger.info(f">>> 初始特征类型: {config.initial_features}")
    logger.info(f">>> 各阶输入维度 (已固定对齐): {actual_num_features_per_rank}")
    logger.info(f">>> 训练统计量: Mean={mean_val:.4f}, Std={std_val:.4f}")

    adjacencies = get_adjacency_types(max(v_dims), config.connectivity, config.neighbor_types)

    # 2. 模型初始化
    model = ETNN(
        num_features_per_rank=actual_num_features_per_rank,
        num_hidden=config.num_hidden,
        num_out=1,
        num_layers=config.num_layers,
        adjacencies=adjacencies,
        initial_features=config.initial_features,
        visible_dims=v_dims,
        normalize_invariants=config.normalize_invariants,
        global_pool=config.global_pool
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: s/config.warmup_steps if s < config.warmup_steps else 1.0
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    
    early_stopping = EarlyStopping(patience=config.patience, mode='min')
    criterion = nn.L1Loss()

    best_val_mae = float('inf')
    global_step = 0
    last_epoch = 0 

    # 3. 训练主循环
    logger.info(f">>> 开始训练...")
    for epoch in range(1, config.epochs + 1):
        # 传入 device 上的 mean 和 std 用于 Loss 计算
        train_loss, train_mae, global_step = train_one_epoch(
            model, dm.train_dataloader(), optimizer, criterion, 
            device, torch.tensor(mean_val).to(device), torch.tensor(std_val).to(device), 
            config, warmup_scheduler, global_step
        )

        val_metrics = evaluate(model, dm.val_dataloader(), device)
        current_val_mae = val_metrics['mae']
        test_metrics = evaluate(model, dm.test_dataloader(), device)
        current_test_mae = test_metrics['mae']

        scheduler.step(current_val_mae)
        
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | TrainMAE: {train_mae:.4f} | "
            f"ValMAE: {current_val_mae:.4f} | TestMAE(Monitor): {current_test_mae:.4f} | LR: {curr_lr:.2e}"
        )

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            # ✨ 修改：使用新的全能保存函数
            save_full_checkpoint(
                model, optimizer, epoch, val_metrics, config, 
                mean_val, std_val, actual_num_features_per_rank, v_dims, 
                os.path.join(out_dir, "best_model.pth")
            )
            logger.info(f"--- 保存当前最佳模型 (MAE: {best_val_mae:.4f})")

        if early_stopping(current_val_mae, model):
            logger.info(">>> 早停触发！")
            last_epoch = epoch
            break
        
        last_epoch = epoch 

    # ✨ 修改：保存最后一轮模型
    final_model_path = os.path.join(out_dir, "last_model.pth")
    save_full_checkpoint(
        model, optimizer, last_epoch, {'val_mae': current_val_mae if 'current_val_mae' in locals() else float('inf')}, 
        config, mean_val, std_val, actual_num_features_per_rank, v_dims, final_model_path
    )
    logger.info(f">>> 保存最后一轮模型 (Epoch {last_epoch}) 到: {final_model_path}")
    logger.info(">>> 训练流程执行完毕。")

if __name__ == "__main__":
    main(TrainingConfig())