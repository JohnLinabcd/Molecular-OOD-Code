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
import pickle
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

# 导入配置与模型 (这里假设 model.py 和 data.py 在 gotennet 目录下)
from config import TrainingConfig
from gotennet.model import GotenNetWrapper
from gotennet.data import DADataModule, MultiTaskDataset

# 强制使用 spawn 模式：物理隔离 GPU 的最强底线
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
# ================= 补充从 train.py 独立出来的函数 =================
def coral_loss(source, target):
    """
    100% 完美复刻你刚才提供的【特征标准化】版本
    针对 GotenNet 高动态特征专门优化
    """
    if source.size(0) < 2 or target.size(0) < 2: 
        return torch.tensor(0.0, device=source.device)
    
    # 🌟 第一步：特征轴标准化 (Feature-wise Normalization)
    # 这步是防止 GotenNet 梯度爆炸的核心！
    s_std = source.std(0, keepdim=True) + 1e-6
    t_std = target.std(0, keepdim=True) + 1e-6
    
    source = (source - source.mean(0, keepdim=True)) / s_std
    target = (target - target.mean(0, keepdim=True)) / t_std
    
    d = source.size(1) # 特征维度
    
    # 🌟 第二步：计算协方差矩阵 (此时其实是相关性矩阵)
    # 因为已经标准化了，所以直接相乘即可，xm 逻辑已经包含在标准化里了
    xc = torch.matmul(source.t(), source) / (source.size(0) - 1)
    xct = torch.matmul(target.t(), target) / (target.size(0) - 1)
    
    # 🌟 第三步：计算 Frobenius 范数
    # 标准化后的矩阵元素都在合理区间，平方后不会爆炸
    loss = torch.norm(xc - xct, p='fro').pow(2) / (4 * d * d)
    return loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    """原版的三参数 evaluate 函数"""
    model.eval()
    total_mae = 0.0
    total_cnt = 0
    for batch in dataloader:
        batch = batch.to(device)
        # 解包三个返回值，只取预测值 out 计算 MAE
        out, _, _ = model(batch.z, batch.pos, batch.batch, batch.sub_batch)
        total_mae += (out - batch.y.view(out.shape)).abs().sum().item()
        total_cnt += batch.y.size(0)
    return total_mae / total_cnt if total_cnt > 0 else 0.0
# ===============================================================
# ================= 1. 真实目标域定义 (北极星锚点) =================
REAL_TARGETS =[
    "O=C1CCCN1", "O=C1CC=CC1", "c1c[nH]nn1", "O=C1CC=CN1", 
    "c1cnon1", "O=C1CCOCC1=O", "c1cnncn1", "O=C1CCOC1=O",
    "c1ncon1", "c1conn1", "c1nc[nH]n1", "C1=NCCCO1",
    "O=c1ccoc(=O)[nH]1", "O=c1cc[nH]cc1", "O=C1CCCCN1", "c1cnc[nH]1"
]

def setup_grpo_logger():
    os.makedirs("./grpo_logs", exist_ok=True)
    log_file = f"./grpo_logs/grpo_V3_ULTIMATE_{datetime.now().strftime('%m%d_%H%M')}.log"
    logger = logging.getLogger("GRPO")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        logger.addHandler(ch)
    return logger

def get_fingerprint(smiles, n_bits=128):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros(n_bits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    except:
        return np.zeros(n_bits)

def mol_to_grakel_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 2: return None 
        node_labels = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        return Graph(adj.tolist(), node_labels=node_labels)
    except:
        return None

# ================= 2. 枢纽替身选取逻辑 (Hub Selection) =================

def select_targeted_proxies(target_pool, real_smiles, total_unique=20):
    print(">>> 正在为 16 个真实目标分析最优代表性替身 (Hub Selection)...")
    real_fps =[get_fingerprint(s) for s in real_smiles]
    proxy_candidates =[]
    
    for scaf in tqdm(target_pool, desc="分析骨架池得分"):
        s_fp = get_fingerprint(scaf)
        sims =[np.dot(s_fp, r_fp)/(np.linalg.norm(s_fp)*np.linalg.norm(r_fp)+1e-8) for r_fp in real_fps]
        # 枢纽分 = 最大相似度 + 覆盖广度奖励
        hub_score = max(sims) + 0.1 * sum([1 for s in sims if s > 0.45])
        proxy_candidates.append({'scaf': scaf, 'hub_score': hub_score, 'sims': sims})

    selected_set = set()
    # 第一步：确保 16 个真题各有一个最像的入选
    for i in range(len(real_smiles)):
        best_scaf = sorted(proxy_candidates, key=lambda x: x['sims'][i], reverse=True)[0]['scaf']
        selected_set.add(best_scaf)
    
    # 第二步：补齐到 20 个独特的枢纽骨架
    remaining = sorted(proxy_candidates, key=lambda x: x['hub_score'], reverse=True)
    for item in remaining:
        if len(selected_set) >= total_unique: break
        selected_set.add(item['scaf'])
    
    final_proxies = list(selected_set)
    random.shuffle(final_proxies)
    return final_proxies

# ================= 3. 动态任务构建 (WL 图核) =================

def prepare_dynamic_task(target_scaf, scaf_to_data, target_fp):
    target_graph = mol_to_grakel_graph(target_scaf)
    if not target_graph: return None, None, None
    
    cand_keys =[k for k in scaf_to_data.keys() if k != target_scaf and len(scaf_to_data[k]) >= 60]
    active_keys, graphs = [],[]
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g:
            graphs.append(g)
            active_keys.append(k)
    
    if not graphs: return None, None, None

    # WL 核计算
    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph])
    sims = wl.transform(graphs).flatten()
    
    # 筛选 Top-50 作为本轮 Selector 的候选 ID
    results = sorted([{'scaf': k, 'sim': sims[i], 'cnt': len(scaf_to_data[k])} for i, k in enumerate(active_keys)], 
                     key=lambda x: x['sim'] * np.log1p(x['cnt']), reverse=True)[:50]
    
    pool_dir = f"./meta_pools/pool_{os.getpid()}_{int(time.time())}"
    os.makedirs(pool_dir, exist_ok=True)
    
    state_feats, paths = [],[]
    for i, c in enumerate(results):
        p = os.path.join(pool_dir, f"c_{i}.csv")
        pd.DataFrame(scaf_to_data[c['scaf']]).to_csv(p, index=False)
        paths.append(p)
        # 拼接特征：Target_FP + Cand_FP + Sim + log(Count)
        state_feats.append(np.concatenate([target_fp, get_fingerprint(c['scaf']), [c['sim'], np.log1p(c['cnt'])/10.0]]))
    
    state_feats_np = np.array([state_feats], dtype=np.float32)
    return torch.from_numpy(state_feats_np), paths, results

# ================= 4. 隔离进程组件 (核心修正：均值反馈、对齐测试集、修正迭代器) =================

def _isolated_base_worker(target_csv, sup_paths, gpu_id, g_mean, g_std, return_dict):
    """在该独立进程中跑基座，返回最后三轮 MAE 的均值"""
    import torch
    from gotennet.model import GotenNetWrapper 
    from gotennet.data import DADataModule
    from config import TrainingConfig
    
    
    device = torch.device(f"cuda:{gpu_id}")
    try:
        # 严格排除逻辑
        target_df = pd.read_csv(target_csv)
        target_smi_set = set(target_df['smiles'].tolist())
        
        cfg = TrainingConfig()
        cfg.device = f"cuda:{gpu_id}"
        cfg.num_workers = 0
        cfg.batch_size = 40 
        cfg.sup_source_paths = sup_paths
        # --- 核心修正：基座评估必须针对当前考题，确保尺子对齐 ---
        cfg.target_paths = [target_csv]
        
        dm = DADataModule(vars(cfg))
        dm._mean, dm._std = g_mean, g_std
        cfg.standardize = False 
        dm.prepare_dataset(filter_smiles=target_smi_set) 
        
        # 把原本的 model = ViSNet(...) 改成：
        model = GotenNetWrapper(
            config=cfg, mean=dm.mean.to(device), std=dm.std.to(device), num_tasks=1
        ).to(device)
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        m_list = []
        for ep in range(10):
            model.train()
            loader = dm.source_sup_loader()
            steps = 0
            for b in loader:
                if steps >= 150: break
                b = b.to(device)
                opt.zero_grad()
                out, _, _ = model(b.z, b.pos, b.batch, b.sub_batch)
                loss = torch.nn.functional.l1_loss(out, b.y.view(out.shape))
                loss.backward()
                opt.step()
                steps += 1
            # 每一轮记录一次
            m_list.append(evaluate(model, dm.target_test_loader(), device))
            
        save_path = f"./base_checkpoints/base_{os.getpid()}.pt"
        torch.save(model.state_dict(), save_path)
        return_dict['path'] = save_path
        # --- 核心修正：返回最后 3 轮的平均值作为稳健基准线 ---
        return_dict['mae'] = np.mean(m_list[-3:])
    except Exception as e:
        return_dict['err'] = str(e)

def worker_fast_proxy(s_idx, action_vec, gpu_id, paths, target_csv, sup_paths, base_model_path, mae_ref, g_mean, g_std, return_dict):
    """独立子进程执行方案演习，返回最后三轮 MAE 的均值"""
    import torch
    from gotennet.model import GotenNetWrapper 
    from gotennet.data import DADataModule
    from config import TrainingConfig
    

    device = torch.device(f"cuda:{gpu_id}")
    tmp_dir = f"./.tmp_w_{os.getpid()}_{s_idx}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        target_df = pd.read_csv(target_csv)
        target_smi_set = set(target_df['smiles'].tolist())
        
        cfg = TrainingConfig()
        cfg.device = f"cuda:{gpu_id}"
        cfg.num_workers = 0
        cfg.batch_size = 40
        
        # 准备选中的精英数据 CSV
        da_paths =[]
        if np.sum(action_vec) > 0:
            for p_idx in np.where(action_vec == 1.0)[0]:
                df = pd.read_csv(paths[p_idx])
                df = df[~df['smiles'].isin(target_smi_set)]
                tp = os.path.join(tmp_dir, f"da_{p_idx}.csv")
                df.to_csv(tp, index=False)
                da_paths.append(tp)
        
        cfg.da_source_paths = da_paths
        cfg.sup_source_paths = sup_paths
        cfg.target_paths =[target_csv]
        
        dm = DADataModule(vars(cfg))
        dm._mean, dm._std = g_mean, g_std
        cfg.standardize = False 
        dm.prepare_dataset(filter_smiles=target_smi_set)
        
        # 把原本的 model = ViSNet(...) 改成：
        model = GotenNetWrapper(
            config=cfg, mean=dm.mean.to(device), std=dm.std.to(device), num_tasks=1
        ).to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # --- 核心修正：将迭代器定义在 Epoch 循环外，确保模型不重复读取同一批 150 个样本 ---
        s_it = iter(dm.source_sup_loader())
        t_it = iter(dm.target_train_loader())
        da_loaders = dm.source_da_loaders()
        d_its =[iter(l) for l in da_loaders]
        
        m_list = []
        for ep in range(5):
            model.train()
            for _ in range(150):
                try: b_s = next(s_it).to(device)
                except: s_it = iter(dm.source_sup_loader()); b_s = next(s_it).to(device)
                
                opt.zero_grad()
                out_s, _, _ = model(b_s.z, b_s.pos, b_s.batch, b_s.sub_batch)
                loss = torch.nn.functional.l1_loss(out_s, b_s.y.view(out_s.shape))
                
                if da_paths and d_its:
                    try: b_t = next(t_it).to(device)
                    except: t_it = iter(dm.target_train_loader()); b_t = next(t_it).to(device)
                    
                    with torch.no_grad():
                        _, mol_t, _ = model(b_t.z, b_t.pos, b_t.batch, b_t.sub_batch)
                    mol_t = mol_t.detach()
                    
                    l_da_batch = torch.tensor(0.0, device=device)
                    for i, d_it in enumerate(d_its):
                        try: b_d = next(d_it).to(device)
                        except: d_its[i] = iter(da_loaders[i]); b_d = next(d_its[i]).to(device)
                        _, m_s, _ = model(b_d.z, b_d.pos, b_d.batch, b_d.sub_batch)
                        l_da_batch += coral_loss(m_s, mol_t)
                    loss += 0.1 * (l_da_batch / len(d_its))
                
                loss.backward()
                opt.step()
            # 每一轮演习都评估 MAE
            m_list.append(evaluate(model, dm.target_test_loader(), device))
        
        # 返回演习后三轮平均值，作为最终反馈
        return_dict[s_idx] = np.mean(m_list[-3:])
        del model, dm, opt
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"!!! Worker-{s_idx:02d} Error: {e}")
        return_dict[f"err_{s_idx}"] = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ================= 5. Selector 类定义 (稳定性加固) =================

class ScaffoldSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 1)
        )
    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        # --- 核心修正：截断 logits，防止 Sigmoid 极其接近 1 导致梯度消失或爆炸 ---
        logits = torch.clamp(logits, -10.0, 10.0) 
        return torch.sigmoid(logits), logits
    
    def sample_actions(self, state, group_size=33):
        probs, logits = self.forward(state)
        dist = Bernoulli(probs.expand(group_size, -1))
        actions = dist.sample()
        for g in range(group_size):
            if g == 0: actions[g].zero_(); continue 
            sel_idx = (actions[g] == 1.0).nonzero(as_tuple=True)[0]
            if len(sel_idx) > 5:
                row_l = logits[0][sel_idx]
                _, top5 = torch.topk(row_l, 5)
                actions[g].zero_()
                actions[g][sel_idx[top5]] = 1.0
        return actions, dist.log_prob(actions).sum(-1), probs

# ================= 6. 主循环：双巡回计划 (绝对锚定 + 宁缺毋滥 + 防爆版) =================

def run_meta_grpo():
    from config import TrainingConfig
    
    from gotennet.data import DADataModule

    # 🌟🌟🌟 GPU 与 进程池控制中枢 🌟🌟🌟
    AVAILABLE_GPUS = [0, 1]         # 你想用的显卡 ID 列表，单卡就写 [0]
    PROCESSES_PER_GPU = 2           # 每个 GPU 上同时跑几个子进程？(比如双卡，这里填2，就是一共并行 4 个进程)
    
    logger = setup_grpo_logger()
    cfg = TrainingConfig()
    main_device = torch.device("cpu") 
    
    os.makedirs("./base_checkpoints", exist_ok=True)
    os.makedirs("./virtual_targets", exist_ok=True)
    os.makedirs("./meta_pools", exist_ok=True)

    print(">>> 正在预计算全量源域统计量 (用于标准化尺度统一)...")
    temp_dm = DADataModule(vars(cfg))
    temp_dm.prepare_dataset()
    g_mean, g_std = temp_dm.mean.cpu(), temp_dm.std.cpu()
    del temp_dm; gc.collect()

    with open(os.path.join(cfg.cache_dir, "full_source_data_homo.pkl"), 'rb') as f:
        full_list = pickle.load(f)
    scaf_to_data = {MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False):[] for d in full_list}
    for d in full_list:
        s = MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False)
        scaf_to_data[s].append({'smiles': d.smiles, 'homo': d.y.item()})
    
    target_pool =[k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 200]
    unique_proxies = select_targeted_proxies(target_pool, REAL_TARGETS)
    task_sequence = unique_proxies + unique_proxies 
    
    selector = ScaffoldSelector(258).to(main_device)
    ref_model = copy.deepcopy(selector).eval().to(main_device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=5e-4)
    manager = mp.Manager()

    for s_idx in range(40):
        step_start_time = time.time()
        target_scaf = task_sequence[s_idx]
        target_fp = get_fingerprint(target_scaf)
        
        state, paths, meta_results = prepare_dynamic_task(target_scaf, scaf_to_data, target_fp)
        if state is None: continue
        state = state.to(main_device)
        
        tmp_target_csv = os.path.join("./virtual_targets", f"v_t_{s_idx}.csv")
        pd.DataFrame(scaf_to_data[target_scaf]).to_csv(tmp_target_csv, index=False)
        
        logger.info(f"\n" + "="*95 + f"\n🚩 Step {s_idx+1}/40 | 替身: {target_scaf[:40]}\n" + "="*95)
        ret_base = manager.dict()
        p_base = mp.Process(target=_isolated_base_worker, args=(tmp_target_csv, cfg.sup_source_paths, 0, g_mean, g_std, ret_base))
        p_base.start()
        p_base.join()
        
        if 'err' in ret_base:
            logger.error(f"[Base Model] 训练失败，跳过本轮: {ret_base['err']}")
            continue
            
        base_path, mae_ref = ret_base['path'], ret_base['mae']
        logger.info(f"  [Base Model] 均值反馈训练完成 | 基准 MAE: {mae_ref:.4f}")

        # --- B. 采样 33 个方案 ---
        with torch.no_grad():
            actions, old_log_p, probs = selector.sample_actions(state, 33)

        probs_np = probs.squeeze().cpu().numpy()
        results_map = manager.dict()
        actions_np_main = actions.numpy() 
        
        # 首先跑 Scheme-00 作为绝对锚点裁判
        logger.info(">>> 正在启动 Scheme-00 (基准锚点)...")
        p0 = mp.Process(target=worker_fast_proxy, args=(0, actions_np_main[0], 0, paths, tmp_target_csv, cfg.sup_source_paths, base_path, mae_ref, g_mean, g_std, results_map))
        p0.start(); p0.join()
        if 0 in results_map:
            v0 = results_map[0]
            logger.info(f"  [Scheme-00] IDs: [] | Probs: [] | MAE: {v0:.4f} (Avg) | 改进: {mae_ref-v0:+.4f} | GPU 0")
        elif 'err_0' in results_map:
            logger.error(f"  [Scheme-00] 运行失败: {results_map['err_0']}")

        pending_tasks = list(range(1, 33))
        
        # 🌟 初始化槽位：现在每个 GPU 对应一个【列表】，用来装载正在运行的进程
        gpu_slots = {gid: [] for gid in AVAILABLE_GPUS} 
        
        # 只要还有任务没发完，或者还有进程在跑，就继续循环
        while pending_tasks or any(len(slots) > 0 for slots in gpu_slots.values()):
            
            # 1. 尝试派发新任务 (填满所有 GPU 的空闲槽位)
            for gid in AVAILABLE_GPUS:
                while len(gpu_slots[gid]) < PROCESSES_PER_GPU and pending_tasks:
                    tid = pending_tasks.pop(0)  # 直接取下一个任务
                    p = mp.Process(
                        target=worker_fast_proxy, 
                        args=(tid, actions_np_main[tid], gid, paths, tmp_target_csv, cfg.sup_source_paths, base_path, mae_ref, g_mean, g_std, results_map)
                    )
                    p.start()
                    gpu_slots[gid].append((tid, p))
            
            # 2. 回收完成的任务
            for gid in AVAILABLE_GPUS:
                alive_processes = []
                for tid, p_item in gpu_slots[gid]:
                    if not p_item.is_alive():
                        p_item.join() # 进程结束，回收资源
                        
                        # 处理结果日志
                        if tid in results_map:
                            v = results_map[tid]
                            delta = mae_ref - v
                            ids_list = np.where(actions_np_main[tid]==1.0)[0].tolist()
                            probs_list = [round(float(probs_np[idx]), 3) for idx in ids_list]
                            logger.info(f"[Scheme-{tid:02d}] IDs: {ids_list} | Probs: {probs_list} | MAE: {v:.4f} (Avg) | 改进: {delta:+.4f} | GPU {gid}")
                        elif f"err_{tid}" in results_map:
                            logger.error(f"[Scheme-{tid:02d}] 运行失败: {results_map[f'err_{tid}']} | GPU {gid}")
                    else:
                        # 还没运行完的，继续放回存活列表
                        alive_processes.append((tid, p_item))
                
                # 更新当前 GPU 的活动进程列表
                gpu_slots[gid] = alive_processes
                
            time.sleep(0.5)

        # --- D. 梯度更新 (NaN 防护与均值中心化版) ---
        valid_keys = np.array([k for k in results_map.keys() if isinstance(k, int)])
        if 0 not in valid_keys or len(valid_keys) < 2: continue
        
        # 使用当前批次的方案 0 作为绝对锚点
        v0 = results_map[0] 
        rewards_raw = np.array([results_map[i] for i in valid_keys])
        deltas_v0 = v0 - rewards_raw
        
        # --- 核心策略：赢了方案 0 给予 20 倍赞誉，输了保持 1 倍差异化 ---
        final_rewards = np.where(deltas_v0 > 0, deltas_v0 * 20.0, deltas_v0 * 1.0)
        
        rew_t = torch.tensor(final_rewards, dtype=torch.float32).to(main_device)
        
        # --- 核心修正：回归 GRPO 均值中心化稳定算法 ---
        # 解释：减去均值保证梯度有升有降，std+0.1 防止数值爆炸
        adv = (rew_t - rew_t.mean()) / (rew_t.std() + 0.1)

        for _ in range(4): 
            optimizer.zero_grad()
            curr_p, _ = selector(state)
            # Clamp 概率防止 log(0) 产生 NaN
            curr_p = torch.clamp(curr_p, 1e-6, 1.0 - 1e-6)
            
            new_log_p = Bernoulli(curr_p.expand(33, -1)).log_prob(actions).sum(-1)[valid_keys]
            ratio = torch.exp(new_log_p - old_log_p[valid_keys].detach())
            
            surr1 = ratio * adv; surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 使用更稳健的 Reverse KL 公式进行约束
            with torch.no_grad(): 
                rlp = Bernoulli(ref_model(state)[0].expand(33, -1)).log_prob(actions).sum(-1)[valid_keys]
            kl_div = (torch.exp(rlp - new_log_p) - (rlp - new_log_p) - 1).mean()
            
            # 保持熵约束在 0.01，允许选率在后期爬向两极
            loss = policy_loss + 0.01 * kl_div - 0.01 * Bernoulli(curr_p).entropy().mean()
            
            loss.backward()
            # --- 核心修正：梯度裁剪，防止权重跳飞 ---
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 0.5)
            optimizer.step()

        if os.path.exists(base_path): os.remove(base_path)
        if os.path.exists(tmp_target_csv): os.remove(tmp_target_csv)
        shutil.rmtree(os.path.dirname(paths[0]), ignore_errors=True)
        
        if (s_idx + 1) % 5 == 0:
            torch.save(selector.state_dict(), "universal_selector.pt")
            ref_model.load_state_dict(selector.state_dict())

        logger.info(f"✅ Step {s_idx+1} 完成 | 耗时: {(time.time()-step_start_time)/60:.1f}min | 最高与方案0差值: {deltas_v0.max():+.4f}")

    torch.save(selector.state_dict(), "universal_selector_final.pt")

if __name__ == "__main__": run_meta_grpo()