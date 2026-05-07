# utils.py
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
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# -----------------------------
# 日志和文件工具
# -----------------------------

def get_logger(logfile: str, name: str = __name__) -> logging.Logger:
    """创建logger
    
    Args:
        logfile: 日志文件路径
        name: logger名称
        
    Returns:
        logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 文件handler
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_china_time_str() -> str:
    """获取中国时间字符串
    
    Returns:
        格式化的时间字符串: YYYYMMDD_HHMMSS
    """
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(tz)
    return now.strftime("%Y%m%d_%H%M%S")

def ensure_dir(directory: str) -> None:
    """确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_config(config: Any, path: str) -> None:
    """保存配置到JSON文件
    
    Args:
        config: 配置对象（数据类或字典）
        path: 保存路径
    """
    # 转换为字典
    if hasattr(config, '__dict__'):
        # 数据类或对象
        config_dict = config.__dict__.copy()
    elif hasattr(config, 'items'):
        # 字典
        config_dict = dict(config)
    else:
        # 其他类型
        config_dict = {"config": str(config)}
    
    # 处理特殊类型
    for key, value in config_dict.items():
        if isinstance(value, torch.Tensor):
            config_dict[key] = value.tolist()
        elif isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()
        elif callable(value):
            config_dict[key] = str(value)
    
    # 确保目录存在
    ensure_dir(os.path.dirname(path))
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config(path: str) -> Dict:
    """从JSON文件加载配置
    
    Args:
        path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config_dict

def save_pickle(obj: Any, path: str) -> None:
    """保存对象到pickle文件
    
    Args:
        obj: 要保存的对象
        path: 保存路径
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """从pickle文件加载对象
    
    Args:
        path: pickle文件路径
        
    Returns:
        加载的对象
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# -----------------------------
# 绘图工具
# -----------------------------

def plot_training_curves(train_losses: List[float], 
                        val_metrics_list: List[Dict[str, float]], 
                        save_path: str,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_metrics_list: 验证指标列表
        save_path: 保存路径
        figsize: 图形大小
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 确定子图数量
    n_plots = 6 if 'lr' in val_metrics_list[0] else 5
    n_rows = 2
    n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # 1. 训练损失
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. 验证MAE
    if len(val_metrics_list) > 0:
        val_maes = [m.get('mae', 0) for m in val_metrics_list]
        axes[1].plot(epochs, val_maes, 'r-', linewidth=2, label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # 3. 验证RMSE
    if len(val_metrics_list) > 0:
        val_rmses = [m.get('rmse', 0) for m in val_metrics_list]
        axes[2].plot(epochs, val_rmses, 'g-', linewidth=2, label='Val RMSE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('Validation RMSE')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    # 4. 验证PCC
    if len(val_metrics_list) > 0:
        val_pccs = [m.get('pcc', 0) for m in val_metrics_list]
        axes[3].plot(epochs, val_pccs, 'm-', linewidth=2, label='Val PCC')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('PCC')
        axes[3].set_title('Validation Pearson Correlation')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
    
    # 5. 验证R²
    if len(val_metrics_list) > 0:
        val_r2s = [m.get('r2', 0) for m in val_metrics_list]
        axes[4].plot(epochs, val_r2s, 'c-', linewidth=2, label='Val R²')
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('R²')
        axes[4].set_title('Validation R-squared')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()
    
    # 6. 学习率（如果有）
    if n_plots == 6 and 'lr' in val_metrics_list[0]:
        lrs = [m.get('lr', 0) for m in val_metrics_list]
        axes[5].plot(epochs, lrs, 'y-', linewidth=2, label='Learning Rate')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Learning Rate')
        axes[5].set_title('Learning Rate Schedule')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()
        axes[5].set_yscale('log')
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_predictions_vs_targets(predictions: np.ndarray, 
                               targets: np.ndarray, 
                               save_path: str, 
                               title: str = "Predictions vs Targets",
                               figsize: Tuple[int, int] = (10, 8)) -> Dict[str, float]:
    """绘制预测值与真实值的散点图
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        save_path: 保存路径
        title: 图形标题
        figsize: 图形大小
        
    Returns:
        计算得到的指标
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    
    # 计算指标
    metrics = compute_metrics(predictions, targets)
    
    plt.figure(figsize=figsize)
    
    # 散点图
    scatter = plt.scatter(targets, predictions, alpha=0.6, s=20, 
                         c=np.abs(predictions - targets), 
                         cmap='viridis')
    
    # 颜色条
    plt.colorbar(scatter, label='Absolute Error')
    
    # 对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    margin = 0.05 * (max_val - min_val)
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', linewidth=2, label='Perfect Prediction')
    
    # 添加指标文本
    text_str = (f"MAE: {metrics['mae']:.4f}\n"
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"R²: {metrics['r2']:.4f}\n"
                f"PCC: {metrics['pcc']:.4f}\n"
                f"SMAPE: {metrics['smape']:.2f}%")
    
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return metrics

def plot_error_distribution(predictions: np.ndarray, 
                           targets: np.ndarray, 
                           save_path: str,
                           figsize: Tuple[int, int] = (12, 4)) -> None:
    """绘制误差分布图
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        save_path: 保存路径
        figsize: 图形大小
    """
    errors = predictions - targets
    
    plt.figure(figsize=figsize)
    
    # 1. 直方图
    plt.subplot(1, 3, 1)
    n, bins, patches = plt.hist(errors, bins=50, alpha=0.7, 
                               color='blue', edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 添加正态分布曲线
    from scipy.stats import norm
    mu, std = norm.fit(errors)
    x = np.linspace(min(errors), max(errors), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p * len(errors) * (bins[1] - bins[0]), 'r-', linewidth=2)
    plt.text(0.05, 0.95, f'μ={mu:.4f}\nσ={std:.4f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    
    # 2. Q-Q图
    plt.subplot(1, 3, 2)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # 3. 箱线图
    plt.subplot(1, 3, 3)
    box = plt.boxplot(errors, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    plt.ylabel('Error')
    plt.title('Error Box Plot')
    
    # 添加统计信息
    q1, q3 = np.percentile(errors, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
    
    plt.text(0.05, 0.95, f'Median: {np.median(errors):.4f}\n'
             f'IQR: {iqr:.4f}\n'
             f'Outliers: {len(outliers)}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_residuals(predictions: np.ndarray, 
                  targets: np.ndarray, 
                  save_path: str,
                  figsize: Tuple[int, int] = (10, 8)) -> None:
    """绘制残差图
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        save_path: 保存路径
        figsize: 图形大小
    """
    residuals = predictions - targets
    
    plt.figure(figsize=figsize)
    
    # 残差vs预测值
    plt.subplot(2, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.grid(True, alpha=0.3)
    
    # 残差vs真实值
    plt.subplot(2, 2, 2)
    plt.scatter(targets, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs True Values')
    plt.grid(True, alpha=0.3)
    
    # 残差分布
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    # 残差正态概率图
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residuals Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=150)
    plt.close()

# -----------------------------
# 评估指标工具
# -----------------------------

def compute_metrics(predictions: Union[np.ndarray, List], 
                   targets: Union[np.ndarray, List]) -> Dict[str, float]:
    """计算各种评估指标
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        包含各项指标的字典
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    
    # 确保长度一致
    if len(predictions) != len(targets):
        warnings.warn(f"Predictions and targets have different lengths: "
                     f"{len(predictions)} vs {len(targets)}")
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
    
    metrics = {}
    
    # MAE
    metrics['mae'] = np.mean(np.abs(targets - predictions))
    
    # MSE
    metrics['mse'] = np.mean((targets - predictions) ** 2)
    
    # RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Pearson相关系数
    if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0:
        pcc, _ = pearsonr(targets, predictions)
        metrics['pcc'] = pcc
    else:
        metrics['pcc'] = 0.0
    
    # 对称平均绝对百分比误差
    epsilon = 1e-10
    smape = 100 * np.mean(2 * np.abs(predictions - targets) / 
                         (np.abs(predictions) + np.abs(targets) + epsilon))
    metrics['smape'] = smape
    
    # 最大绝对误差
    metrics['max_error'] = np.max(np.abs(targets - predictions))
    
    # 解释方差
    metrics['explained_variance'] = 1 - np.var(targets - predictions) / (np.var(targets) + 1e-10)
    
    return metrics

def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """打印指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"{prefix}Model Performance Metrics:")
    print(f"{prefix}  MAE: {metrics['mae']:.6f}")
    print(f"{prefix}  MSE: {metrics['mse']:.6f}")
    print(f"{prefix}  RMSE: {metrics['rmse']:.6f}")
    print(f"{prefix}  R²: {metrics['r2']:.4f}")
    print(f"{prefix}  PCC: {metrics['pcc']:.4f}")
    print(f"{prefix}  SMAPE: {metrics['smape']:.2f}%")
    print(f"{prefix}  Max Error: {metrics['max_error']:.6f}")
    print(f"{prefix}  Explained Variance: {metrics['explained_variance']:.4f}")

def format_metrics(metrics: Dict[str, float]) -> str:
    """将指标格式化为字符串
    
    Args:
        metrics: 指标字典
        
    Returns:
        格式化的字符串
    """
    lines = [
        "Model Performance Metrics:",
        f"  MAE: {metrics['mae']:.6f}",
        f"  MSE: {metrics['mse']:.6f}", 
        f"  RMSE: {metrics['rmse']:.6f}",
        f"  R²: {metrics['r2']:.4f}",
        f"  PCC: {metrics['pcc']:.4f}",
        f"  SMAPE: {metrics['smape']:.2f}%",
        f"  Max Error: {metrics['max_error']:.6f}",
        f"  Explained Variance: {metrics['explained_variance']:.4f}"
    ]
    return "\n".join(lines)

# -----------------------------
# 数据处理工具
# -----------------------------

def normalize_data(data: np.ndarray, 
                  mean: Optional[float] = None, 
                  std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """标准化数据
    
    Args:
        data: 输入数据
        mean: 指定均值（如果为None则从数据计算）
        std: 指定标准差（如果为None则从数据计算）
        
    Returns:
        (标准化后的数据, 均值, 标准差)
    """
    data = np.asarray(data)
    
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    
    # 避免除零
    if std < 1e-10:
        std = 1.0
        warnings.warn(f"Standard deviation is too small ({std}), setting to 1.0")
    
    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize_data(normalized_data: np.ndarray, 
                    mean: float, 
                    std: float) -> np.ndarray:
    """反标准化数据
    
    Args:
        normalized_data: 标准化后的数据
        mean: 均值
        std: 标准差
        
    Returns:
        反标准化后的数据
    """
    return normalized_data * std + mean

def split_data_indices(total_size: int, 
                      train_ratio: float = 0.7, 
                      val_ratio: float = 0.15, 
                      test_ratio: float = 0.15, 
                      seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """划分数据索引
    
    Args:
        total_size: 总数据大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (训练索引, 验证索引, 测试索引)
    """
    # 检查比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    indices = list(range(total_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices

def check_data_consistency(data: np.ndarray, 
                          name: str = "data",
                          check_nan: bool = True,
                          check_inf: bool = True) -> Dict[str, Any]:
    """检查数据一致性
    
    Args:
        data: 输入数据
        name: 数据名称
        check_nan: 是否检查NaN
        check_inf: 是否检查无穷大
        
    Returns:
        检查结果字典
    """
    data = np.asarray(data)
    results = {
        'name': name,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'has_nan': False,
        'has_inf': False,
        'is_finite': True
    }
    
    if check_nan:
        results['has_nan'] = np.any(np.isnan(data))
        results['nan_count'] = int(np.sum(np.isnan(data)))
    
    if check_inf:
        results['has_inf'] = np.any(np.isinf(data))
        results['inf_count'] = int(np.sum(np.isinf(data)))
    
    results['is_finite'] = not (results['has_nan'] or results['has_inf'])
    
    return results

def print_data_info(data_info: Dict[str, Any]) -> None:
    """打印数据信息
    
    Args:
        data_info: 数据信息字典
    """
    print(f"Data: {data_info['name']}")
    print(f"  Shape: {data_info['shape']}")
    print(f"  Dtype: {data_info['dtype']}")
    print(f"  Range: [{data_info['min']:.4f}, {data_info['max']:.4f}]")
    print(f"  Mean: {data_info['mean']:.4f}, Std: {data_info['std']:.4f}")
    
    if 'has_nan' in data_info:
        if data_info['has_nan']:
            print(f"  Warning: Contains {data_info['nan_count']} NaN values")
        else:
            print(f"  No NaN values")
    
    if 'has_inf' in data_info:
        if data_info['has_inf']:
            print(f"  Warning: Contains {data_info['inf_count']} infinite values")
        else:
            print(f"  No infinite values")

# -----------------------------
# 模型工具
# -----------------------------

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        (总参数数量, 可训练参数数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_model(model: torch.nn.Module, 
              optimizer: Optional[torch.optim.Optimizer] = None,
              epoch: int = 0,
              metrics: Optional[Dict] = None,
              config: Optional[Any] = None,
              save_path: str = "model.pth") -> None:
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 指标字典
        config: 配置
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': config.__dict__ if hasattr(config, '__dict__') else config,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    ensure_dir(os.path.dirname(save_path))
    torch.save(checkpoint, save_path)

def load_model(model: torch.nn.Module, 
              checkpoint_path: str, 
              device: str = 'cpu',
              load_optimizer: bool = False) -> Dict[str, Any]:
    """加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备
        load_optimizer: 是否加载优化器
        
    Returns:
        加载的信息字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    result = {
        'model': model,
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('model_config', {})
    }
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    return result

def freeze_model_layers(model: torch.nn.Module, freeze: bool = True) -> torch.nn.Module:
    """冻结或解冻模型层
    
    Args:
        model: 模型
        freeze: True表示冻结，False表示解冻
        
    Returns:
        修改后的模型
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    return model

def get_model_summary(model: torch.nn.Module, 
                     input_size: Optional[Tuple] = None) -> str:
    """获取模型摘要
    
    Args:
        model: 模型
        input_size: 输入大小
        
    Returns:
        模型摘要字符串
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model: {model.__class__.__name__}")
    summary_lines.append("=" * 80)
    
    # 参数统计
    total_params, trainable_params = count_parameters(model)
    summary_lines.append(f"Total parameters: {total_params:,}")
    summary_lines.append(f"Trainable parameters: {trainable_params:,}")
    summary_lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # 层信息
    summary_lines.append("\nLayers:")
    summary_lines.append("-" * 80)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        trainable = any(p.requires_grad for p in module.parameters())
        summary_lines.append(f"{name:20s} {str(module.__class__.__name__):30s} "
                           f"Params: {num_params:9,} Trainable: {trainable}")
    
    summary_lines.append("=" * 80)
    return "\n".join(summary_lines)

# -----------------------------
# 训练工具类
# -----------------------------

class EarlyStopping:
    """早停类"""
    
    def __init__(self, patience: int = 10, delta: float = 0, 
                 verbose: bool = False, mode: str = 'min'):
        """
        Args:
            patience: 耐心值
            delta: 最小改善量
            verbose: 是否打印信息
            mode: 'min'或'max'，表示监控的指标是最小化还是最大化
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """检查是否早停
        
        Args:
            val_score: 验证分数
            model: 模型
            
        Returns:
            是否早停
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_score: float, model: torch.nn.Module) -> None:
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation score improved ({val_score:.6f}). Saving model...')
        self.best_state_dict = model.state_dict().copy()
    
    def load_best_model(self, model: torch.nn.Module) -> None:
        """加载最佳模型参数"""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        else:
            warnings.warn("No best model state dict available")

class ExponentialMovingAverage:
    """指数移动平均"""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """
        Args:
            model: 模型
            decay: 衰减率
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self) -> None:
        """注册参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """更新EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """更新指标
        
        Args:
            metrics: 指标字典
            prefix: 前缀
        """
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key not in self.metrics:
                self.metrics[full_key] = []
                self.history[full_key] = []
            
            self.metrics[full_key].append(value)
    
    def reset(self) -> None:
        """重置当前epoch的指标"""
        self.metrics = {}
    
    def get_average(self, prefix: str = "") -> Dict[str, float]:
        """获取平均值
        
        Args:
            prefix: 前缀
            
        Returns:
            平均指标字典
        """
        avg_metrics = {}
        for key, values in self.metrics.items():
            if prefix and not key.startswith(prefix):
                continue
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def save_history(self) -> None:
        """保存历史记录"""
        for key, values in self.metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].extend(values)
    
    def get_history(self) -> Dict[str, List[float]]:
        """获取历史记录"""
        return self.history.copy()
    
    def plot_history(self, save_path: str, keys: Optional[List[str]] = None) -> None:
        """绘制历史记录
        
        Args:
            save_path: 保存路径
            keys: 要绘制的键列表
        """
        if keys is None:
            keys = list(self.history.keys())
        
        n_keys = len(keys)
        if n_keys == 0:
            return
        
        n_cols = min(3, n_keys)
        n_rows = (n_keys + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        
        for idx, key in enumerate(keys):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            values = self.history[key]
            ax.plot(range(1, len(values) + 1), values, 'b-', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel(key)
            ax.set_title(f'{key} History')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(keys), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
        plt.close()

# -----------------------------
# 设备工具
# -----------------------------

def get_device(device_str: str = None) -> torch.device:
    """获取设备
    
    Args:
        device_str: 设备字符串，如 'cuda', 'cuda:0', 'cpu'
        
    Returns:
        torch设备
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(f"CUDA requested but not available, using CPU instead")
            return torch.device("cpu")
        
        # 检查GPU编号
        if ":" in device_str:
            gpu_id = int(device_str.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                warnings.warn(f"GPU {gpu_id} not available, using GPU 0")
                return torch.device("cuda:0")
    
    return torch.device(device_str)

def print_device_info(device: torch.device) -> None:
    """打印设备信息
    
    Args:
        device: torch设备
    """
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"  Device name: {torch.cuda.get_device_name(device)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB")
        print(f"  CUDA version: {torch.version.cuda}")

def clear_gpu_memory() -> None:
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# -----------------------------
# 时间工具
# -----------------------------

class Timer:
    """计时器类"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.datetime.now()
        print(f"{self.name} started at {self.start_time.strftime('%H:%M:%S')}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} completed in {duration:.2f} seconds")
    
    def elapsed(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return (datetime.datetime.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()

def format_time(seconds: float) -> str:
    """格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

# -----------------------------
# 主程序入口检查
# -----------------------------

if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("Available functions:")
    
    # 列出所有函数
    import inspect
    functions = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            functions.append(name)
    
    functions.sort()
    for i, func in enumerate(functions):
        if i % 4 == 0:
            print()
        print(f"{func:25s}", end=" ")
    
    print(f"\n\nTotal {len(functions)} utility functions available.")