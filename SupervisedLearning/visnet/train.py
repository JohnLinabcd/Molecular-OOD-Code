import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch 

from utils import (
    get_logger, set_seed, ensure_dir,
    plot_training_curves, plot_predictions_vs_targets,
    plot_error_distribution, compute_metrics,
    save_model, save_config, EarlyStopping
)

from config import TrainingConfig
from visnet.data import DataModule
from visnet.model import ViSNet


# =====================================================
# 初始完整性检查 (Sanity Check)
# =====================================================
@torch.no_grad()
def check_data_integrity(model, dataloader, device, logger):
    model.eval()
    logger.info(">>> 正在进行数据完整性检查 (Sanity Check)...")
    
    has_error = False
    
    for i, batch in enumerate(tqdm(dataloader, desc="Checking Data", leave=False)):
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        batch_idx = batch.batch.to(device)
        
        try:
            pred = model(z, pos, batch_idx)
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                has_error = True
                logger.error(f"!!! 发现坏数据 Batch {i} !!!")
                
                data_list = batch.to_data_list()
                for mol_idx, data in enumerate(data_list):
                    single_batch = Batch.from_data_list([data]).to(device)
                    pred_single = model(single_batch.z, single_batch.pos, single_batch.batch)
                    
                    if torch.isnan(pred_single).any() or torch.isinf(pred_single).any():
                        smiles = getattr(data, 'smiles', 'Unknown')
                        logger.error(f"  -> 问题分子索引: {mol_idx}")
                        logger.error(f"  -> SMILES: {smiles}")
                        
                        pos_chk = data.pos
                        dist = torch.cdist(pos_chk, pos_chk)
                        dist.fill_diagonal_(float('inf'))
                        min_dist = dist.min().item()
                        logger.error(f"  -> 最小原子间距: {min_dist:.6f}")
                
                return False 

        except Exception as e:
            logger.error(f"Batch {i} 发生运行错误: {e}")
            return False

    if not has_error:
        logger.info(">>> 数据完整性检查通过！")
        return True
    return False


# =====================================================
# 训练一个 epoch (修复了 LR 冲突问题)
# =====================================================
def train_one_epoch(
    model, dataloader, optimizer, criterion, device,
    mean, std, config, warmup_scheduler=None,
    global_step=0  # <--- 新增: 接收全局步数
):
    model.train()
    total_loss, n_batch = 0.0, 0
    safe_std = max(std, 1e-6)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device)

        if torch.isnan(z).any() or torch.isnan(pos).any() or torch.isnan(y).any():
            continue

        optimizer.zero_grad(set_to_none=True)

        pred = model(z, pos, batch_idx)

        # ---------- 输出检查 ----------
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            # print(f"Warning: NaN detected in prediction at batch {n_batch}")
            continue

        # ---------- 标准化 ----------
        pred_n = (pred - mean) / safe_std
        y_n = (y - mean) / safe_std

        loss = criterion(pred_n, y_n)

        # ---------- loss 爆炸保护 ----------
        if (
            torch.isnan(loss)
            or torch.isinf(loss)
            or loss.item() > config.loss_upper_bound
        ):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # ---------- 梯度清洗 ----------
        for p in model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_norm
        )

        optimizer.step()

        # ============================================================
        # 关键修复: 只有在 step < warmup_steps 时才调用 warmup_scheduler
        # 这样当 ReduceLROnPlateau 降低 LR 时，就不会被这里的 step 覆盖回去
        # ============================================================
        if global_step < config.warmup_steps:
            if warmup_scheduler is not None:
                warmup_scheduler.step()
        
        global_step += 1  # 更新步数
        # ============================================================

        total_loss += loss.item()
        n_batch += 1

    # 返回 loss 和 更新后的 global_step
    return total_loss / max(1, n_batch), global_step


# =====================================================
# 验证 / 测试
# =====================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, mean, std, return_pred=False):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    safe_std = max(std, 1e-6)

    # 这里的 desc 会根据调用处不同显示 Val 或 Test
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        z = batch.z.to(device)
        pos = batch.pos.to(device)
        batch_idx = batch.batch.to(device)
        y = batch.y.to(device)

        pred = model(z, pos, batch_idx)
        
        if torch.isnan(pred).any():
             continue

        pred_n = (pred - mean) / safe_std
        y_n = (y - mean) / safe_std

        loss = criterion(pred_n, y_n)
        total_loss += loss.item()

        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    if len(preds) == 0:
        return {"mae": float('inf'), "loss": float('inf')}

    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    # 计算指标 (compute_metrics 可能返回 numpy 类型)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / max(1, len(dataloader))

    # JSON 兼容处理
    safe_metrics = {}
    for k, v in metrics.items():
        if hasattr(v, 'item'):
            safe_metrics[k] = v.item()
        else:
            safe_metrics[k] = float(v)
    
    if return_pred:
        return safe_metrics, preds, targets
    return safe_metrics


# =====================================================
# 主函数
# =====================================================
def main(config: TrainingConfig):
    set_seed(config.seed)
    device = config.device

    out_dir = f"./outputs/{config.label_name}"
    ensure_dir(out_dir)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    save_config(config, os.path.join(out_dir, "config.json"))

    # ---------- Data ----------
    dm = DataModule(vars(config)) 
    dm.prepare_dataset()

    mean = dm.mean.item() if config.standardize else 0.0
    std = dm.std.item() if config.standardize else 1.0

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # ---------- Model ----------
    model = ViSNet(
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_rbf=config.num_rbf,
        cutoff=config.cutoff,
        max_num_neighbors=config.max_num_neighbors,
        lmax=config.lmax,
        vertex=config.vertex,
        reduce_op=config.reduce_op,
        mean=mean,
        std=std,
        derivative=False,
    ).to(device)

    # ---------- Optimizer ----------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ---------- Warmup ----------
    def warmup_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )

    # 调度器基于 Validation MAE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=config.patience)

    criterion = nn.L1Loss()

    # ==================== 修改1：注释掉 best_mae ====================
    # best_mae = float("inf")
    best_path = os.path.join(out_dir, "best_model.pth")
    
    # =====================================================
    # Sanity Check
    # =====================================================
    logger.info("检查训练数据...")
    if not check_data_integrity(model, train_loader, device, logger):
        logger.error("数据检查失败！请先清理坏数据。")
        return 

    # =====================================================
    # Train Loop
    # =====================================================
    global_step = 0  # <--- 初始化全局 step

    for epoch in range(1, config.epochs + 1):
        # 1. 训练 (接收更新后的 global_step)
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, mean, std, config, warmup_scheduler,
            global_step=global_step
        )

        # 2. 验证 (用于 Model Selection 和 Scheduler)
        val_metrics = evaluate(
            model, val_loader, criterion, device, mean, std
        )
        current_val_mae = val_metrics.get("mae", float('inf'))

        # 3. 测试集监控 (Monitor Only)
        test_metrics_monitor = evaluate(
            model, test_loader, criterion, device, mean, std
        )
        current_test_mae = test_metrics_monitor.get("mae", float('inf'))

        # 4. 调整 LR (基于 Val)
        scheduler.step(current_val_mae)
        
        # 获取当前 LR
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch} | "
            f"Loss={train_loss:.4f} | "
            f"ValMAE={current_val_mae:.4f} | "
            f"TestMAE(Monitor)={current_test_mae:.4f} | "
            f"LR={current_lr:.2e}"
        )

        # ==================== 修改2：直接保存当前模型 ====================
        save_model(
            model, optimizer, epoch, val_metrics,
            config, best_path
        )
        logger.info(f"已保存当前轮次模型到 {best_path}")

        # 6. Early Stopping (基于 Val)
        early_stopping(current_val_mae, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # ---------- Final Test ----------
    # ==================== 修改3：直接使用当前模型测试 ====================
    logger.info("使用最后一轮模型进行最终测试...")
    # 模型已经是最后一轮的状态，直接测试，不需要加载checkpoint
    
    test_metrics, preds, targets = evaluate(
        model, test_loader, criterion, device, mean, std, True
    )

    plot_predictions_vs_targets(
        preds, targets, os.path.join(out_dir, "pred_vs_true.png")
    )
    plot_error_distribution(
        preds, targets, os.path.join(out_dir, "error_dist.png")
    )

    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"最终测试 MAE = {test_metrics['mae']:.4f}")


if __name__ == "__main__":
    main(TrainingConfig())