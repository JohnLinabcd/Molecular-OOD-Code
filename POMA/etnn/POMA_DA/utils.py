import os
import json
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional

def get_logger(logfile: str, name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_config(config, path):
    """保存配置到 JSON 文件 (解决报错的关键)"""
    ensure_dir(os.path.dirname(path))
    config_dict = config.__dict__.copy() if hasattr(config, '__dict__') else dict(config)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def compute_metrics(predictions, targets) -> Dict[str, float]:
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    metrics = {
        'mae': np.mean(np.abs(targets - predictions)),
        'rmse': np.sqrt(np.mean((targets - predictions) ** 2)),
        'r2': 1 - (np.sum((targets - predictions) ** 2) / (np.sum((targets - np.mean(targets)) ** 2) + 1e-10))
    }
    if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0:
        metrics['pcc'], _ = pearsonr(targets, predictions)
    else: 
        metrics['pcc'] = 0.0
    return metrics

def plot_training_curves(train_losses, val_maes, save_path):
    ensure_dir(os.path.dirname(save_path))
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, color='tab:blue', label='Train Loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Val MAE', color='tab:red')
    ax2.plot(epochs, val_maes, color='tab:red', label='Val MAE')
    plt.savefig(save_path)
    plt.close()

def plot_predictions_vs_targets(predictions, targets, save_path):
    """绘制预测值与真实值的散点图"""
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    if len(targets) > 0 and len(predictions) > 0:
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(predictions, targets, save_path):
    """绘制误差分布直方图"""
    ensure_dir(os.path.dirname(save_path))
    errors = predictions - targets
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def save_model(model, optimizer, epoch, metrics, config, save_path):
    ensure_dir(os.path.dirname(save_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

class EarlyStopping:
    def __init__(self, patience=10, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_score, model):
        score = -val_score if self.mode == 'min' else val_score
        if self.best_score is None: self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0