# utils.py (拒绝阉割的完全体版)
import os
import sys
import json
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime
import pytz

def get_logger(logfile: str, name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def compute_metrics(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    r_val, _ = pearsonr(preds, targets)
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearsonr": r_val}

class Timer:
    """原版中非常优雅的上下文计时器"""
    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completed in {duration:.2f} seconds")
    
    def elapsed(self) -> float:
        if self.start_time is None: return 0.0
        if self.end_time is None:
            return (datetime.datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

def format_time(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.2f}s"
    elif seconds < 3600: return f"{seconds / 60:.1f}min"
    else: return f"{seconds / 3600:.2f}h"