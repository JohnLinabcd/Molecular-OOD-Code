import os
import copy
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Bernoulli
import shutil
import logging
import random
from datetime import datetime
import gc
import time

# RDKit 与 GraKel 核心组件
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph

# 导入配置与模型组件
from config import TrainingConfig
from etnn.model import ETNN
from etnn.lifter import get_adjacency_types
# lookfor.py 顶部导入部分
from data import DADataModule, SimpleSMILESDataset # 加上 SimpleSMILESDataset

# 强制使用 spawn 模式：物理隔离 GPU 的最强底线
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# =============================================================================
# 1. 真实目标域定义 (北极星锚点)
# =============================================================================
REAL_TARGETS = [
    "O=C1CCCN1", "O=C1CC=CC1", "c1c[nH]nn1", "O=C1CC=CN1", 
    "c1cnon1", "O=C1CCOCC1=O", "c1cnncn1", "O=C1CCOC1=O",
    "c1ncon1", "c1conn1", "c1nc[nH]n1", "C1=NCCCO1",
    "O=c1ccoc(=O)[nH]1", "O=c1cc[nH]cc1", "O=C1CCCCN1", "c1cnc[nH]1"
]

# 在 lookfor.py 靠前的位置添加这个函数
def _precompute_da_worker(args):
    """
    专门给 CPU 进程池调用的预处理函数
    """
    path, cache_dir, label_name, config = args
    from data import SimpleSMILESDataset
    # 触发 Lifting 并保存 pkl
    SimpleSMILESDataset([path], cache_dir, label_name, "da", config, calc_brics=False)
    return True

def setup_grpo_logger():
    """
    初始化 GRPO 专用日志系统，完整记录每一轮演习的 IDs 和 改进分
    """
    os.makedirs("./grpo_logs", exist_ok=True)
    log_file = f"./grpo_logs/grpo_ETNN_ULTIMATE_{datetime.now().strftime('%m%d_%H%M')}.log"
    logger = logging.getLogger("GRPO")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 文件输出
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        # 控制台实时输出
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(ch)
    return logger

def get_fingerprint(smiles, n_bits=128):
    """
    获取分子的 Morgan 指纹，用于 Selector 的状态表征
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return np.zeros(n_bits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    except:
        return np.zeros(n_bits)

def mol_to_grakel_graph(smiles):
    """
    将 SMILES 转换为 GraKel 图对象，用于 WL 图核计算
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 2:
            return None 
        node_labels = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        return Graph(adj.tolist(), node_labels=node_labels)
    except:
        return None

# =============================================================================
# 2. 枢纽替身选取逻辑 (Hub Selection)
# =============================================================================

def select_targeted_proxies(target_pool, real_smiles, total_unique=20):
    """
    在海量骨架池中，为 16 个真实考题匹配最具代表性的“替身”骨架
    """
    print(f">>> 正在为 {len(real_smiles)} 个真实目标分析最优代表性替身 (Hub Selection)...")
    real_fps = [get_fingerprint(s) for s in real_smiles]
    proxy_candidates = []
    
    for scaf in tqdm(target_pool, desc="分析骨架池得分", leave=False):
        s_fp = get_fingerprint(scaf)
        sims = [np.dot(s_fp, r_fp)/(np.linalg.norm(s_fp)*np.linalg.norm(r_fp)+1e-8) for r_fp in real_fps]
        # 枢纽分 = 最大相似度 + 覆盖广度奖励 (针对多个考题的适应性)
        hub_score = max(sims) + 0.1 * sum([1 for s in sims if s > 0.45])
        proxy_candidates.append({'scaf': scaf, 'hub_score': hub_score, 'sims': sims})

    selected_set = set()
    # 第一步：确保 16 个真题各有一个最相似的骨架入选
    for i in range(len(real_smiles)):
        best_scaf = sorted(proxy_candidates, key=lambda x: x['sims'][i], reverse=True)[0]['scaf']
        selected_set.add(best_scaf)
    
    # 第二步：按枢纽分补齐到 20 个独特的枢纽骨架
    remaining = sorted(proxy_candidates, key=lambda x: x['hub_score'], reverse=True)
    for item in remaining:
        if len(selected_set) >= total_unique:
            break
        selected_set.add(item['scaf'])
    
    final_proxies = list(selected_set)
    random.shuffle(final_proxies)
    return final_proxies

# =============================================================================
# 3. 动态任务构建 (基于 WL 图核寻找当前替身的相似数据源)
# =============================================================================

def prepare_dynamic_task(target_scaf, scaf_to_data, target_fp):
    """
    为当前的“替身”考题，从全量源域中挑出 Top-50 个相似骨架作为 Selector 的候选 ID
    """
    target_graph = mol_to_grakel_graph(target_scaf)
    if not target_graph:
        return None, None, None
    
    cand_keys = [k for k in scaf_to_data.keys() if k != target_scaf and len(scaf_to_data[k]) >= 60]
    active_keys, graphs = [], []
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g:
            graphs.append(g)
            active_keys.append(k)
    
    if not graphs:
        return None, None, None

    # 执行高效的 Weisfeiler-Lehman 图核计算
    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph])
    sims = wl.transform(graphs).flatten()
    
    # 结合结构相似度与数据量进行筛选
    results = sorted([{'scaf': k, 'sim': sims[i], 'cnt': len(scaf_to_data[k])} for i, k in enumerate(active_keys)], 
                     key=lambda x: x['sim'] * np.log1p(x['cnt']), reverse=True)[:50]
    
    pool_dir = f"./meta_pools/pool_{os.getpid()}_{int(time.time())}"
    os.makedirs(pool_dir, exist_ok=True)
    
    state_feats, paths = [], []
    for i, c in enumerate(results):
        p = os.path.join(pool_dir, f"c_{i}.csv")
        pd.DataFrame(scaf_to_data[c['scaf']]).to_csv(p, index=False)
        paths.append(p)
        # 状态特征：当前替身指纹 + 候选指纹 + 相似度 + 数据量对数
        state_feats.append(np.concatenate([target_fp, get_fingerprint(c['scaf']), [c['sim'], np.log1p(c['cnt'])/10.0]]))
    
    state_feats_np = np.array([state_feats], dtype=np.float32)
    return torch.from_numpy(state_feats_np), paths, results

# =============================================================================
# 4. ETNN 动态维度推断 (核心修复逻辑)
# =============================================================================

def get_actual_dims(dm, cfg):
    """
    Determines architecture based on configuration, avoiding crash on dimension mismatch.
    """
    sample = dm.source_sup_dataset[0]
    v_dims = list(sample.num_features_per_rank.keys())
    
    base_hetero_dims = {}
    for r in v_dims:
        attr_name = f"x_{r}"
        if hasattr(sample, attr_name):
            base_hetero_dims[r] = getattr(sample, attr_name).shape[1]
        else:
            base_hetero_dims[r] = 0

    node_feat_dim = sample.x.shape[1] if hasattr(sample, 'x') else 0
    num_lifters = len(cfg.lifters)

    actual_dims = {}
    for r in v_dims:
        total_d = 0
        for f_type in cfg.initial_features:
            if f_type == "hetero":
                total_d += base_hetero_dims.get(r, 0)
            elif f_type == "node":
                total_d += node_feat_dim
            elif f_type == "mem":
                total_d += num_lifters
        actual_dims[r] = total_d
        
    return actual_dims, v_dims

# =============================================================================
# 5. 隔离进程组件 (完全对齐 ViSNet ULTIMATE 训练计划)
# =============================================================================

def _isolated_base_worker(target_csv, sup_paths, gpu_id, g_mean, g_std, return_dict):
    """
    在该独立进程中运行 Scheme-00 之前的“基准模型”。
    逻辑：10 Epochs 纯监督训练，返回最后三轮 MAE 的均值。
    """
    import torch
    from etnn.model import ETNN
    from etnn.lifter import get_adjacency_types
    from data import DADataModule
    from config import TrainingConfig
    
    device = torch.device(f"cuda:{gpu_id}")
    try:
        # 严格隔离配置
        target_df = pd.read_csv(target_csv)
        target_smi_set = set(target_df['smiles'].tolist())
        
        cfg = TrainingConfig()
        
        # 将 cfg 转化为 dict 传入 DADataModule
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
        cfg_dict['device'] = f"cuda:{gpu_id}"
        cfg_dict['batch_size'] = 40
        cfg_dict['sup_source_paths'] = sup_paths
        cfg_dict['target_paths'] = [target_csv]
        
        # dm = DADataModule(cfg_dict)
        # dm._mean, dm._std = g_mean.to(device), g_std.to(device)
        # dm.prepare_dataset(filter_smiles=target_smi_set) 
        
        # actual_dims, v_dims = get_actual_dims(dm, cfg)
        dm = DADataModule(cfg_dict)
        # 1. 先准备数据（内部会产生 CPU 版本的 _mean 和 _std）
        dm.prepare_dataset(filter_smiles=target_smi_set) 
        
        # 2. 后强制覆盖为 GPU 版本的全局统计量
        # 这样即便内部计算了，也会被你这行代码修正到正确的设备上
        dm._mean = g_mean.to(device)
        dm._std = g_std.to(device)
        
        actual_dims, v_dims = get_actual_dims(dm, cfg)
        adjs = get_adjacency_types(max(v_dims), cfg.connectivity, cfg.neighbor_types)
        
        model = ETNN(
            num_features_per_rank=actual_dims, 
            num_hidden=cfg.num_hidden, 
            num_out=1, 
            num_layers=cfg.num_layers,
            adjacencies=adjs, 
            initial_features=cfg.initial_features,
            visible_dims=v_dims, 
            normalize_invariants=True, 
            global_pool=True
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        mae_history = []
        for epoch in range(10):
            model.train()
            loader = dm.source_sup_loader()
            step_count = 0
            for batch in loader:
                if step_count >= 300: 
                    break
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # 解构 ETNN 三元组输出
                pred, _, _ = model(batch)
                if pred.ndim == 1: 
                    pred = pred.unsqueeze(-1)
                
                loss = torch.nn.functional.l1_loss((pred - dm.mean)/dm.std, (batch.y - dm.mean)/dm.std)
                loss.backward()
                
                # 梯度清理
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, 0.0, 0.0, 0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                
                optimizer.step()
                step_count += 1
            
            # 每一轮评估一次物理尺度 MAE
            model.eval()
            ps, ts = [], []
            with torch.no_grad():
                for eb in dm.target_test_loader():
                    eb = eb.to(device)
                    o, _, _ = model(eb)
                    ps.append(o.detach().cpu().numpy())
                    ts.append(eb.y.cpu().numpy())
            epoch_mae = np.mean(np.abs(np.concatenate(ps).flatten() - np.concatenate(ts).flatten()))
            mae_history.append(epoch_mae)
            
        save_path = f"./base_checkpoints/base_{os.getpid()}.pt"
        torch.save(model.state_dict(), save_path)
        
        return_dict['path'] = save_path
        # 返回最后三轮均值，降低评价噪声
        return_dict['mae'] = np.mean(mae_history[-3:])
        
    except Exception as e:
        return_dict['err'] = str(e)

def worker_fast_proxy(s_idx, action_vec, gpu_id, paths, target_csv, sup_paths, base_model_path, mae_ref, g_mean, g_std, return_dict):
    """
    独立子进程执行“演习方案”。
    """
    import torch
    from etnn.model import ETNN
    from etnn.lifter import get_adjacency_types
    from data import DADataModule
    from config import TrainingConfig

    # ✨ 状态打卡：进程已启动
    return_dict[f"trace_{s_idx}"] = "Process Launched"
    
    def coral_loss(source, target):
        if source.size(0) < 2 or target.size(0) < 2: return torch.tensor(0.0, device=source.device)
        d = source.size(1)
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = torch.matmul(xm.t(), xm) / (source.size(0) - 1)
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = torch.matmul(xmt.t(), xmt) / (target.size(0) - 1)
        return torch.norm(xc - xct, p='fro').pow(2) / (4 * d * d)

    device = torch.device(f"cuda:{gpu_id}")
    tmp_dir = f"./.tmp_w_{os.getpid()}_{s_idx}"
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # ✨ 状态打卡：开始加载数据
        return_dict[f"trace_{s_idx}"] = "Loading Data (90k PKL)"
        # 1. 获取目标分子的 SMILES 集合（仅用于后续判断，不再在这里过滤文件）
        target_smi_set = set(pd.read_csv(target_csv)['smiles'])
        cfg = TrainingConfig()
        
        # 2. ✨✨ 核心修改：直接映射路径 ✨✨
        da_paths = []
        if np.sum(action_vec) > 0:
            selected_indices = np.where(action_vec == 1.0)[0]
            for p_idx in selected_indices:
                # 直接指向主进程生成的原始 c_i.csv 路径
                # 因为主进程是用这个路径生成的 pkl，文件名才能对上
                da_paths.append(paths[p_idx])
        
        # 3. 准备 DataModule
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
        cfg_dict['device'] = f"cuda:{gpu_id}"
        cfg_dict['batch_size'] = 40
        cfg_dict['da_source_paths'] = da_paths # 这里存的是 c_i.csv
        cfg_dict['sup_source_paths'] = sup_paths
        cfg_dict['target_paths'] = [target_csv]
        
        dm = DADataModule(cfg_dict)
        # 预处理数据
        # 这里的 filter_smiles 只是二次保险，核心是 pkl 会被直接加载
        dm.prepare_dataset(filter_smiles=target_smi_set) 
        
        # 修正设备
        dm._mean = g_mean.to(device)
        dm._std = g_std.to(device)

        # ✨ 状态打卡：数据加载成功，开始初始化模型
        return_dict[f"trace_{s_idx}"] = "Data Loaded, Initializing Model"
        
        # ... (后续训练逻辑保持不变) ...
        
        actual_dims, v_dims = get_actual_dims(dm, cfg)
        adjs = get_adjacency_types(max(v_dims), cfg.connectivity, cfg.neighbor_types)
        
        model = ETNN(num_features_per_rank=actual_dims, num_hidden=cfg.num_hidden, num_out=1, num_layers=cfg.num_layers,
                     adjacencies=adjs, initial_features=cfg.initial_features, visible_dims=v_dims, normalize_invariants=True, global_pool=True).to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        
        # ✨ 强化学习火候：5e-5 低学习率 ✨
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        s_iter = iter(dm.source_sup_loader())
        t_iter = iter(dm.target_train_loader())
        da_loaders = dm.source_da_loaders()
        d_iters = [iter(l) for l in da_loaders]
        
        # ✨ 状态打卡：开始训练循环
        return_dict[f"trace_{s_idx}"] = "Training Cycles Started"

        mae_history = []
        for epoch in range(5):
            model.train()
            for _ in range(300):
                try:
                    batch_s = next(s_iter)
                except:
                    s_iter = iter(dm.source_sup_loader())
                    batch_s = next(s_iter)
                
                batch_s = batch_s.to(device)
                optimizer.zero_grad()
                
                # 前向传播：预测值 + 全局拓扑特征
                out_s, mol_feat_s, _ = model(batch_s)
                if out_s.ndim == 1: out_s = out_s.unsqueeze(-1)
                
                loss = torch.nn.functional.l1_loss((out_s - dm.mean)/dm.std, (batch_s.y - dm.mean)/dm.std)
                
                if da_paths and d_iters:
                    try:
                        batch_t = next(t_iter)
                    except:
                        t_iter = iter(dm.target_train_loader())
                        batch_t = next(t_iter)
                    
                    with torch.no_grad():
                        _, mol_feat_t, _ = model(batch_t.to(device))
                    
                    l_da_total = torch.tensor(0.0, device=device)
                    for i, d_it in enumerate(d_iters):
                        try:
                            batch_d = next(d_it)
                        except:
                            d_iters[i] = iter(da_loaders[i])
                            batch_d = next(d_iters[i])
                        
                        _, mol_feat_d, _ = model(batch_d.to(device))
                        l_da_total += coral_loss(mol_feat_d, mol_feat_t.detach())
                    
                    loss += 0.1 * (l_da_total / len(d_iters))
                
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, 0.0, 0.0, 0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
            
            model.eval()
            ps, ts = [], []
            with torch.no_grad():
                for eb in dm.target_test_loader():
                    eb = eb.to(device)
                    o, _, _ = model(eb)
                    ps.append(o.cpu().numpy())
                    ts.append(eb.y.cpu().numpy())
            mae_history.append(np.mean(np.abs(np.concatenate(ps).flatten() - np.concatenate(ts).flatten())))
        
        return_dict[s_idx] = np.mean(mae_history[-3:])
        del model, dm, optimizer
        gc.collect()

        # ✨ 状态打卡：任务圆满完成
        return_dict[f"trace_{s_idx}"] = "Task Completed Successfully"


    except Exception as e:
        return_dict[f"err_{s_idx}"] = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# =============================================================================
# 6. Selector 类定义 (GRPO 决策核心)
# =============================================================================

class ScaffoldSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        logits = torch.clamp(logits, -10.0, 10.0) 
        return torch.sigmoid(logits), logits
    
    def sample_actions(self, state, group_size=33):
        probs, logits = self.forward(state)
        dist = Bernoulli(probs.expand(group_size, -1))
        actions = dist.sample()
        
        for g in range(group_size):
            if g == 0:
                actions[g].zero_()
                continue 
            
            sel_indices = (actions[g] == 1.0).nonzero(as_tuple=True)[0]
            if len(sel_indices) > 5:
                row_logits = logits[0][sel_indices]
                _, top5_sub_idx = torch.topk(row_logits, 5)
                actions[g].zero_()
                actions[g][sel_indices[top5_sub_idx]] = 1.0
                
        return actions, dist.log_prob(actions).sum(-1), probs

# =============================================================================
# 7. 主循环：GRPO 强化学习全流程
# =============================================================================

def run_meta_grpo():
    logger = setup_grpo_logger()
    cfg = TrainingConfig()
    main_device = torch.device("cpu") 
    
    os.makedirs("./base_checkpoints", exist_ok=True)
    os.makedirs("./virtual_targets", exist_ok=True)
    os.makedirs("./meta_pools", exist_ok=True)

    print(">>> 正在预计算全量源域统计量 (基于你的 Config 规范)...")
    cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
    temp_dm = DADataModule(cfg_dict)
    temp_dm.prepare_dataset()
    g_mean, g_std = temp_dm.mean.cpu(), temp_dm.std.cpu()
    
    full_data_list = temp_dm.source_sup_dataset.data_list
    scaf_to_data = {MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False):[] for d in full_data_list}
    for d in full_data_list:
        s = MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False)
        scaf_to_data[s].append({'smiles': d.smiles, 'homo': d.y.item()})
    del temp_dm
    gc.collect()

    target_pool = [k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 100]
    unique_proxies = select_targeted_proxies(target_pool, REAL_TARGETS)
    task_sequence = unique_proxies + unique_proxies # 跑两轮
    
    selector = ScaffoldSelector(258).to(main_device)
    ref_model = copy.deepcopy(selector).eval().to(main_device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=5e-4)
    manager = mp.Manager()

    for s_idx in range(40):
        if s_idx >= len(task_sequence): break
        
        step_start_time = time.time()
        target_scaf = task_sequence[s_idx]
        target_fp = get_fingerprint(target_scaf)
        
        state, paths, meta_info = prepare_dynamic_task(target_scaf, scaf_to_data, target_fp)
        if state is None: continue
        state = state.to(main_device)
        
        tmp_target_csv = f"./virtual_targets/v_t_{s_idx}.csv"
        pd.DataFrame(scaf_to_data[target_scaf]).to_csv(tmp_target_csv, index=False)
        
        logger.info("\n" + "="*95 + f"\n🚩 Step {s_idx+1}/40 | 替身: {target_scaf[:40]}\n" + "="*95)
        
        # A. 训练基座模型
        ret_base = manager.dict()
        p_base = mp.Process(target=_isolated_base_worker, args=(tmp_target_csv, cfg.sup_source_paths, 0, g_mean, g_std, ret_base))
        p_base.start()
        p_base.join()
        
        if 'err' in ret_base:
            logger.error(f"  [Base Model] 训练失败: {ret_base['err']}")
            continue
            
        base_model_path, mae_ref = ret_base['path'], ret_base['mae']
        logger.info(f"  [Base Model] 基准 MAE: {mae_ref:.4f}")

        # ✨✨ 【新增：主进程炊事员模式】 ✨✨
        # 1. 预处理本轮的目标域 (Target) pkl
        logger.info(">>> [Pre-process] 正在预处理目标域 pkl...")
        SimpleSMILESDataset([tmp_target_csv], os.path.join(cfg.cache_dir, 'target'), cfg.label_name, "target", cfg)

        # # 2. 顺序预处理 50 个候选数据源的 da pkl
        # logger.info(f">>> [Pre-process] 正在顺序生成 50 个候选源的 pkl 缓存 (防止子进程死锁)...")
        # for p in tqdm(paths, desc="Lifting Candidates", leave=False):
        #     # 这一步会执行 data.py 里的 Lifting 并保存 pkl
        #     SimpleSMILESDataset([p], os.path.join(cfg.cache_dir, 'da_subset'), cfg.label_name, "da", cfg, calc_brics=False)
        # 2. 【并行优化版】并行生成 50 个候选源的 pkl 缓存
        logger.info(f">>> [Pre-process] 正在使用多进程 CPU 预热 50 个候选源 (Parallel Mode)...")
        da_cache_dir = os.path.join(cfg.cache_dir, 'da_subset')
        
        # 准备进程池参数
        pool_args = [(p, da_cache_dir, cfg.label_name, cfg) for p in paths]
        
        # 建议开启 8-12 个 CPU 核心，不要全部占满，留一点给系统
        num_cpu_cores = min(os.cpu_count() - 2, 12) 
        
        with mp.Pool(processes=num_cpu_cores) as pool:
            # 使用 imap 或 map 运行，并带上进度条
            list(tqdm(pool.imap(_precompute_da_worker, pool_args), total=len(pool_args), desc="Parallel Pre-Lifting", leave=False))
            
        logger.info(">>> [Pre-process] 并行预处理完毕。")
        
        

        # B. 采样 33 个演习方案
        with torch.no_grad():
            actions, old_log_p, probs = selector.sample_actions(state, 33)

        probs_np = probs.squeeze().cpu().numpy()
        results_map = manager.dict()
        actions_np_main = actions.numpy() 
        
        p0 = mp.Process(target=worker_fast_proxy, args=(0, actions_np_main[0], 0, paths, tmp_target_csv, cfg.sup_source_paths, base_model_path, mae_ref, g_mean, g_std, results_map))
        p0.start()
        p0.join()
        if 0 in results_map:
            logger.info(f"  [Scheme-00] IDs: [] | Probs: [] | MAE: {results_map[0]:.4f} (Avg) | 改进: {mae_ref-results_map[0]:+.4f} | GPU 0")

        # # C. 并行演习 32 个带数据的方案
        # pending_tasks = list(range(1, 33))
        # gpu_slots = {0: None, 1: None} 
        
        # while pending_tasks or any(gpu_slots.values()):
        #     for gid in [0, 1]:
        #         if gpu_slots[gid] is None and pending_tasks:
        #             tid = pending_tasks.pop(0)
        #             p = mp.Process(target=worker_fast_proxy, args=(tid, actions_np_main[tid], gid, paths, tmp_target_csv, cfg.sup_source_paths, base_model_path, mae_ref, g_mean, g_std, results_map))
        #             p.start()
        #             gpu_slots[gid] = (tid, p)
            
        #     for gid in [0, 1]:
        #         if gpu_slots[gid] is not None:
        #             tid, p_item = gpu_slots[gid]
        #             if not p_item.is_alive():
        #                 p_item.join()
        #                 gpu_slots[gid] = None
        #                 if tid in results_map:
        #                     v = results_map[tid]
        #                     delta = mae_ref - v
        #                     ids_list = np.where(actions_np_main[tid]==1.0)[0].tolist()
        #                     probs_list = [round(float(probs_np[idx]), 3) for idx in ids_list]
        #                     logger.info(f"[Scheme-{tid:02d}] IDs: {ids_list} | Probs: {probs_list} | MAE: {v:.4f} (Avg) | 改进: {delta:+.4f} | GPU {gid}")
        #     time.sleep(0.5)
        # # C. 并行演习 32 个带数据的方案 (ULTIMATE 并行版：1卡3人)
        # pending_tasks = list(range(1, 33))
        # workers_per_gpu = 3  # 每张卡分配 3 个 worker
        # # 追踪当前活跃的进程：[(tid, process, gid), ...]
        # active_processes = []
        
        # # 只要还有任务没跑，或者还有进程在运行，就继续循环
        # while pending_tasks or active_processes:
        #     # 1. 尝试填充空位
        #     for gid in [0, 1]:
        #         # 计算当前 GPU 上有多少个 worker 正在跑
        #         current_gpu_load = sum(1 for _, _, g in active_processes if g == gid)
                
        #         while current_gpu_load < workers_per_gpu and pending_tasks:
        #             tid = pending_tasks.pop(0)
        #             p = mp.Process(
        #                 target=worker_fast_proxy, 
        #                 args=(tid, actions_np_main[tid], gid, paths, tmp_target_csv, 
        #                       cfg.sup_source_paths, base_model_path, mae_ref, g_mean, g_std, results_map)
        #             )
        #             p.start()
        #             active_processes.append((tid, p, gid))
        #             current_gpu_load += 1
            
        #     # # 2. 检查并清理已完成的进程
        #     # still_active = []
        #     # for tid, p_item, gid in active_processes:
        #     #     if not p_item.is_alive():
        #     #         p_item.join()
        #     #         # 记录日志 (保持你原有的日志格式)
        #     #         if tid in results_map:
        #     #             v = results_map[tid]
        #     #             delta = mae_ref - v
        #     #             ids_list = np.where(actions_np_main[tid]==1.0)[0].tolist()
        #     #             probs_list = [round(float(probs_np[idx]), 3) for idx in ids_list]
        #     #             logger.info(f"[Scheme-{tid:02d}] IDs: {ids_list} | Probs: {probs_list} | MAE: {v:.4f} (Avg) | 改进: {delta:+.4f} | GPU {gid}")
        #     #     else:
        #     #         still_active.append((tid, p_item, gid))
        #     # 2. 检查并清理已完成的进程
        #     still_active = []
        #     for tid, p_item, gid in active_processes:
        #         if not p_item.is_alive():
        #             p_item.join()
        #             # ✨ 修复：全面捕获并打印日志
        #             if tid in results_map:
        #                 v = results_map[tid]
        #                 delta = mae_ref - v
        #                 ids_list = np.where(actions_np_main[tid]==1.0)[0].tolist()
        #                 probs_list = [round(float(probs_np[idx]), 3) for idx in ids_list]
        #                 logger.info(f"[Scheme-{tid:02d}] IDs: {ids_list} | Probs: {probs_list} | MAE: {v:.4f} (Avg) | 改进: {delta:+.4f} | GPU {gid}")
        #             elif f"err_{tid}" in results_map:
        #                 # 捕捉代码层面的报错
        #                 logger.error(f"❌ [Scheme-{tid:02d}] 代码报错: {results_map[f'err_{tid}']}")
        #             else:
        #                 # 捕捉系统层面的强杀 (OOM/SegFault)
        #                 logger.error(f"⚠️ [Scheme-{tid:02d}] 进程被系统强杀 (疑似内存不足)，无返回结果！")
        #         else:
        #             still_active.append((tid, p_item, gid))
            
        #     active_processes = still_active
        #     time.sleep(1.0) # 稍微增加休眠时间，降低 CPU 轮询压力
        # C. 【全能监控版】并行演习：数据详情 + 死亡诊断
        pending_tasks = list(range(1, 33))
        workers_per_gpu = 8 # 你可以根据内存情况调大到 5
        active_processes = []
        
        while pending_tasks or active_processes:
            # 1. 填充空位
            for gid in [0, 1]:
                current_gpu_load = sum(1 for _, _, g in active_processes if g == gid)
                while current_gpu_load < workers_per_gpu and pending_tasks:
                    tid = pending_tasks.pop(0)
                    p = mp.Process(target=worker_fast_proxy, args=(tid, actions_np_main[tid], gid, paths, tmp_target_csv, cfg.sup_source_paths, base_model_path, mae_ref, g_mean, g_std, results_map))
                    p.start()
                    active_processes.append((tid, p, gid))
                    current_gpu_load += 1
            
            # 2. 检查完成情况
            still_active = []
            for tid, p_item, gid in active_processes:
                if not p_item.is_alive():
                    p_item.join()
                    
                    # 获取系统退出状态和打卡追踪
                    exit_code = p_item.exitcode 
                    trace = results_map.get(f"trace_{tid}", "Unknown")
                    
                    if tid in results_map:
                        # ✨ 方案 A：成功运行（恢复你想看的详细 ID 和 概率列表）
                        v = results_map[tid]
                        delta = mae_ref - v
                        # 解析选中的数据源 ID 列表
                        ids_list = np.where(actions_np_main[tid] == 1.0)[0].tolist()
                        # 获取对应的概率值
                        probs_list = [round(float(probs_np[idx]), 3) for idx in ids_list]
                        
                        logger.info(
                            f"✅ [Scheme-{tid:02d}] 成功 | GPU {gid} | "
                            f"IDs: {ids_list} | Probs: {probs_list} | "
                            f"MAE: {v:.4f} (Avg) | 改进: {delta:+.4f} | 状态: {trace}"
                        )
                    
                    elif f"err_{tid}" in results_map:
                        # 方案 B：代码内部报错
                        logger.error(f"❌ [Scheme-{tid:02d}] 崩溃 | GPU {gid} | 错误: {results_map[f'err_{tid}']} | 最后死于: {trace}")
                    
                    else:
                        # 方案 C：静默猝死（系统强杀）
                        if exit_code == -9: diag = "SIGKILL (内存 OOM)"
                        elif exit_code == -11: diag = "SIGSEGV (段错误)"
                        else: diag = f"ExitCode {exit_code}"
                        
                        logger.error(f"💀 [Scheme-{tid:02d}] 猝死 | GPU {gid} | 诊断: {diag} | 最后痕迹: {trace}")
                else:
                    still_active.append((tid, p_item, gid))
            
            active_processes = still_active
            time.sleep(1.0)

        # D. 强化学习梯度更新 (ULTIMATE 20倍暴击奖励)
        valid_indices = np.array([k for k in results_map.keys() if isinstance(k, int)])
        if 0 not in valid_indices or len(valid_indices) < 2: continue
        
        v0_anchor = results_map[0] 
        all_rewards_raw = np.array([results_map[i] for i in valid_indices])
        deltas_to_v0 = v0_anchor - all_rewards_raw
        
        # ✨ ✨ 核心策略：赢了 Scheme-00 给予 20 倍重赏，输了保持 1 倍 ✨ ✨
        final_rewards = np.where(deltas_to_v0 > 0, deltas_to_v0 * 20.0, deltas_to_v0 * 1.0)
        
        rew_tensor = torch.tensor(final_rewards, dtype=torch.float32).to(main_device)
        advantage = (rew_tensor - rew_tensor.mean()) / (rew_tensor.std() + 0.1)

        for _ in range(4): 
            optimizer.zero_grad()
            curr_p, _ = selector(state)
            curr_p = torch.clamp(curr_p, 1e-6, 1.0 - 1e-6)
            
            new_log_p = Bernoulli(curr_p.expand(33, -1)).log_prob(actions).sum(-1)[valid_indices]
            ratio = torch.exp(new_log_p - old_log_p[valid_indices].detach())
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            with torch.no_grad(): 
                ref_lp = Bernoulli(ref_model(state)[0].expand(33, -1)).log_prob(actions).sum(-1)[valid_indices]
            kl_div = (torch.exp(ref_lp - new_log_p) - (ref_lp - new_log_p) - 1).mean()
            
            total_loss = policy_loss + 0.01 * kl_div - 0.01 * Bernoulli(curr_p).entropy().mean()
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 0.5)
            optimizer.step()

        if os.path.exists(base_model_path): os.remove(base_model_path)
        shutil.rmtree(os.path.dirname(paths[0]), ignore_errors=True)
        # ✨ 核心删除操作：清理掉本轮产生的 pkl 缓存，防止下一轮 Step 错误读取
        # 注意：不要删除全量源域的 sup 缓存，只删 da_subset 和 target
        shutil.rmtree(os.path.join(cfg.cache_dir, 'da_subset'), ignore_errors=True)
        
        if (s_idx + 1) % 5 == 0:
            torch.save(selector.state_dict(), "universal_selector_etnn.pt")
            ref_model.load_state_dict(selector.state_dict())

        logger.info(f"✅ Step {s_idx+1} 完成 | 耗时: {(time.time()-step_start_time)/60:.1f}min | 最高改进: {deltas_to_v0.max():+.4f}")

    torch.save(selector.state_dict(), "universal_selector_etnn_final.pt")

if __name__ == "__main__":
    run_meta_grpo()