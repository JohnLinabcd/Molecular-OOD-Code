# train.py
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
    plot_predictions_vs_targets, plot_error_distribution, 
    compute_metrics, save_model, save_config, EarlyStopping
)

from config import TrainingConfig
from gotennet.data import DataModule
from gotennet.model import GotenNetWrapper

@torch.no_grad()
def check_data_integrity(model, dataloader, device, logger):
    model.eval()
    logger.info(">>> 正在进行数据完整性检查...")
    for i, batch in enumerate(tqdm(dataloader, desc="Checking", leave=False)):
        z, pos, batch_idx = batch.z.to(device), batch.pos.to(device), batch.batch.to(device)
        try:
            pred = model(z, pos, batch_idx)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                logger.error(f"!!! Batch {i} 存在 NaN !!!")
                return False 
        except Exception as e:
            logger.error(f"Batch {i} 发生运行错误: {e}")
            return False
    return True

def train_one_epoch(model, dataloader, optimizer, criterion, device, mean, std, config, warmup_scheduler=None, global_step=0):
    model.train()
    # 🌟 修改点 1：增加 total_mae 的累加器
    total_loss, total_mae, n_batch = 0.0, 0.0, 0
    safe_std = max(std, 1e-6)

    for batch in tqdm(dataloader, desc="Train", leave=False):
        z, pos, batch_idx, y = batch.z.to(device), batch.pos.to(device), batch.batch.to(device), batch.y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        pred = model(z, pos, batch_idx)
        if torch.isnan(pred).any() or torch.isinf(pred).any(): continue

        # 🌟 修改点 2：在计算标准化 Loss 前，截获原始尺度下的 MAE
        batch_mae = (pred.detach() - y).abs().mean().item()

        loss = criterion((pred - mean) / safe_std, (y - mean) / safe_std)
        if torch.isnan(loss) or loss.item() > config.loss_upper_bound: continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        if global_step < config.warmup_steps and warmup_scheduler is not None:
            warmup_scheduler.step()
        
        global_step += 1  
        total_loss += loss.item()
        total_mae += batch_mae   # 🌟 累加当前批次的 MAE
        n_batch += 1

    # 🌟 修改点 3：返回结果时把计算好的平均 Train MAE 也带上
    return total_loss / max(1, n_batch), total_mae / max(1, n_batch), global_step

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, mean, std, return_pred=False):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    safe_std = max(std, 1e-6)

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        z, pos, batch_idx, y = batch.z.to(device), batch.pos.to(device), batch.batch.to(device), batch.y.to(device)
        pred = model(z, pos, batch_idx)
        
        if torch.isnan(pred).any(): continue

        loss = criterion((pred - mean) / safe_std, (y - mean) / safe_std)
        total_loss += loss.item()

        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    if len(preds) == 0: return {"mae": float('inf'), "loss": float('inf')}

    preds, targets = np.concatenate(preds).flatten(), np.concatenate(targets).flatten()
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / max(1, len(dataloader))
    safe_metrics = {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in metrics.items()}
    
    return (safe_metrics, preds, targets) if return_pred else safe_metrics

def main():
    config = TrainingConfig()
    set_seed(config.seed)
    device = config.device
    out_dir = f"./outputs/{config.label_name}"
    ensure_dir(out_dir)
    logger = get_logger(os.path.join(out_dir, "train.log"))
    save_config(config, os.path.join(out_dir, "config.json"))

    # ---------- 初始化数据流 ----------
    dm = DataModule(config)
    dm.prepare_dataset()
    mean = dm.mean.item() if config.standardize else 0.0
    std = dm.std.item() if config.standardize else 1.0

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # ---------- 初始化 GotenNet ----------
    model = GotenNetWrapper(config=config, mean=mean, std=std, num_tasks=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: s / max(1, config.warmup_steps) if s < config.warmup_steps else 1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience, min_lr=config.min_lr, verbose=True)
    early_stopping = EarlyStopping(patience=config.patience)
    criterion = nn.L1Loss()
    best_path = os.path.join(out_dir, "best_model.pth")
    
    if not check_data_integrity(model, train_loader, device, logger):
        logger.error("数据异常，请检查！")
        return 

    global_step = 0  
    for epoch in range(1, config.epochs + 1):
        # 🌟 修改点 4：接收 train_mae
        train_loss, train_mae, global_step = train_one_epoch(model, train_loader, optimizer, criterion, device, mean, std, config, warmup_scheduler, global_step)
        
        val_metrics = evaluate(model, val_loader, criterion, device, mean, std)
        current_val_mae = val_metrics.get("mae", float('inf'))
        test_metrics = evaluate(model, test_loader, criterion, device, mean, std)

        scheduler.step(current_val_mae)
        
        # 🌟 修改点 5：在终端日志中打印 TrainMAE
        logger.info(f"Epoch {epoch} | Loss={train_loss:.4f} | TrainMAE={train_mae:.4f} | ValMAE={current_val_mae:.4f} | TestMAE={test_metrics.get('mae', float('inf')):.4f} | LR={optimizer.param_groups[0]['lr']:.2e}")

        save_model(model, optimizer, epoch, val_metrics, config, best_path)
        early_stopping(current_val_mae, model)
        if early_stopping.early_stop:
            logger.info("触发 Early Stopping")
            break

    logger.info("模型训练结束，执行最终测试绘制图表...")
    test_metrics, preds, targets = evaluate(model, test_loader, criterion, device, mean, std, True)
    plot_predictions_vs_targets(preds, targets, os.path.join(out_dir, "pred_vs_true.png"))
    plot_error_distribution(preds, targets, os.path.join(out_dir, "error_dist.png"))
    with open(os.path.join(out_dir, "test_results.json"), "w") as f: 
        json.dump(test_metrics, f, indent=2)
    logger.info(f"最终预测 MAE = {test_metrics['mae']:.4f}")

if __name__ == "__main__":
    main()