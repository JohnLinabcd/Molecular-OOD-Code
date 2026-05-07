import os
import copy
import re
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from torch.distributions import Bernoulli
import shutil
import pickle
import logging
import random
from datetime import datetime

# RDKit 修正导入
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph

from config import TrainingConfig
from visnet.model import ViSNet
from visnet.data import DADataModule, MultiTaskDataset
from train import evaluate, coral_loss

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ================= 1. 日志与工具 =================

def setup_grpo_logger():
    os.makedirs("./grpo_logs", exist_ok=True)
    log_file = f"./grpo_logs/grpo_probs_v26_{datetime.now().strftime('%m%d_%H%M')}.log"
    logger = logging.getLogger("GRPO")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file); fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh); ch = logging.StreamHandler(); logger.addHandler(ch)
    return logger

def get_fingerprint(smiles, n_bits=128):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)) if mol else np.zeros(n_bits)
    except: return np.zeros(n_bits)

def mol_to_grakel_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 2: return None 
        node_labels = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        return Graph(adj.tolist(), node_labels=node_labels)
    except: return None

# ================= 2. 动态元考题构建 =================

def prepare_dynamic_task(target_scaf, scaf_to_data, target_fp):
    target_graph = mol_to_grakel_graph(target_scaf)
    if not target_graph: return None, None, None
    cand_keys =[k for k in scaf_to_data.keys() if k != target_scaf and len(scaf_to_data[k]) >= 60]
    active_keys, graphs = [],[]
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g: graphs.append(g); active_keys.append(k)
    if not graphs: return None, None, None
    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph]); sims = wl.transform(graphs).flatten()
    results = sorted([{'scaf': k, 'sim': sims[i], 'cnt': len(scaf_to_data[k])} for i, k in enumerate(active_keys)], key=lambda x: x['sim']*np.log1p(x['cnt']), reverse=True)[:50]
    pool_dir = "./.meta_pool"
    if os.path.exists(pool_dir): shutil.rmtree(pool_dir)
    os.makedirs(pool_dir)
    state_feats, paths = [],[]
    for i, c in enumerate(results):
        p = os.path.join(pool_dir, f"c_{i}.csv"); pd.DataFrame(scaf_to_data[c['scaf']]).to_csv(p, index=False); paths.append(p)
        state_feats.append(np.concatenate([target_fp, get_fingerprint(c['scaf']), [c['sim'], np.log1p(c['cnt'])/10.0]]))
    return torch.tensor([state_feats], dtype=torch.float32), paths, results

def check_combination_validity(action_vec, paths, target_smi_set):
    idx_list = np.where(action_vec == 1.0)[0]
    if len(idx_list) == 0: return True 
    for p_idx in idx_list:
        df = pd.read_csv(paths[p_idx], usecols=['smiles'])
        if ( ~df['smiles'].isin(target_smi_set) ).sum() < 50: return False
    return True

# ================= 3. 基座训练 (150步) =================

def train_shared_base(target_csv, sup_paths, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}")
    save_path = "./.shared_base.pt"
    target_smi_set = set(pd.read_csv(target_csv)['smiles'].tolist())
    cfg = TrainingConfig(); cfg.device = str(device); cfg.num_workers = 0; cfg.batch_size = 28; cfg.sup_source_paths = sup_paths
    dm = DADataModule(vars(cfg)); dm.prepare_dataset(filter_smiles=target_smi_set)
    loader = dm.source_sup_loader()
    model = ViSNet(hidden_channels=128, num_layers=6, num_heads=8, num_rbf=32, mean=dm.mean.to(device), std=dm.std.to(device), num_tasks=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4); m_ref =[]
    for ep in range(10):
        model.train(); s_iter = iter(loader)
        for _ in range(150):
            try: b = next(s_iter).to(device)
            except StopIteration: s_iter = iter(loader); b = next(s_iter).to(device)
            opt.zero_grad(); out, _, _ = model(b.z, b.pos, b.batch, b.sub_batch); torch.nn.functional.l1_loss(out, b.y.view(out.shape)).backward(); opt.step()
        if ep >= 7: m_ref.append(evaluate(model, dm.target_test_loader(), device))
    torch.save(model.state_dict(), save_path); return save_path, np.mean(m_ref)

# ================= 4. Worker (150步) =================

def worker_fast_proxy(s_idx, action_vec, gpu_id, paths, target_csv, sup_paths, base_model_path, mae_ref):
    torch.cuda.set_device(gpu_id); device = torch.device(f"cuda:{gpu_id}")
    is_empty = (np.sum(action_vec) == 0)
    target_smi_set = set(pd.read_csv(target_csv)['smiles'].tolist())
    tmp_dir = f"./.tmp_gpu{gpu_id}_{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        sampled_da_paths =[]
        if not is_empty:
            for p_idx in np.where(action_vec == 1.0)[0]:
                df = pd.read_csv(paths[p_idx]); df = df[~df['smiles'].isin(target_smi_set)]
                tp = os.path.join(tmp_dir, f"da_{p_idx}.csv"); df.sample(min(len(df), 512)).to_csv(tp, index=False); sampled_da_paths.append(tp)
        
        cfg = TrainingConfig(); cfg.device = str(device); cfg.num_workers = 0; cfg.batch_size = 28
        cfg.sup_source_paths = sup_paths; cfg.da_source_paths = sampled_da_paths; cfg.target_paths = [target_csv]
        dm = DADataModule(vars(cfg)); dm.prepare_dataset(filter_smiles=target_smi_set)
        
        model = ViSNet(hidden_channels=128, num_layers=6, num_heads=8, num_rbf=32, mean=dm.mean.to(device), std=dm.std.to(device), num_tasks=1).to(device)
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5); m_hist =[]
        
        s_loader = dm.source_sup_loader()
        t_loader = dm.target_train_loader()
        da_loaders = dm.source_da_loaders()
        
        s_iter = iter(s_loader)
        t_iter = iter(t_loader)
        d_iters =[iter(l) for l in da_loaders]

        for ep in range(5):
            model.train()
            for _ in range(150):
                try: b_s = next(s_iter).to(device)
                except StopIteration: s_iter = iter(s_loader); b_s = next(s_iter).to(device)
                
                opt.zero_grad()
                out_s, _, _ = model(b_s.z, b_s.pos, b_s.batch, b_s.sub_batch)
                loss = torch.nn.functional.l1_loss(out_s, b_s.y.view(out_s.shape))
                
                if not is_empty and d_iters:
                    try: b_t = next(t_iter).to(device)
                    except StopIteration: t_iter = iter(t_loader); b_t = next(t_iter).to(device)
                    _, mol_t, _ = model(b_t.z, b_t.pos, b_t.batch, b_t.sub_batch)
                    
                    l_da = torch.tensor(0.0, device=device)
                    for i, d_it in enumerate(d_iters):
                        try: b_da = next(d_it).to(device)
                        except StopIteration: 
                            d_iters[i] = iter(da_loaders[i])
                            b_da = next(d_iters[i]).to(device)
                        _, m_s, _ = model(b_da.z, b_da.pos, b_da.batch, b_da.sub_batch)
                        l_da += coral_loss(m_s, mol_t)
                    loss += 0.1 * (l_da / len(d_iters))
                
                loss.backward(); opt.step()
            m_hist.append(evaluate(model, dm.target_test_loader(), device))
        
        return s_idx, np.mean(m_hist[-3:]), "SUCCESS"
    except Exception as e: 
        # 【修改点1】：将报错返回值改为 None，方便主进程识别并实施“采样屏蔽”
        # 原来是 return s_idx, 9.0, ... 
        err_msg = str(e)[:10]
        if "CUDA out of memory" in str(e): err_msg = "OOM"
        return s_idx, None, f"ERR:{err_msg}"
    finally:
        if 'model' in locals(): del model
        shutil.rmtree(tmp_dir, ignore_errors=True); torch.cuda.empty_cache()

# ================= 5. Selector & 主循环 =================

class ScaffoldSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits), logits
    def sample_actions(self, state, group_size=33):
        probs, logits = self.forward(state)
        dist = Bernoulli(probs.expand(group_size, -1))
        actions = dist.sample()
        for g in range(group_size):
            sel_idx = (actions[g] == 1).nonzero(as_tuple=True)[0]
            if len(sel_idx) > 5:
                row_l = logits[0][sel_idx]
                _, top_k = torch.topk(row_l, 5)
                actions[g].zero_(); actions[g][sel_idx[top_k]] = 1
        return actions, dist.log_prob(actions).sum(-1), probs

def run_meta_grpo():
    cfg = TrainingConfig(); logger = setup_grpo_logger(); sup_csvs = cfg.sup_source_paths
    with open(os.path.join(cfg.cache_dir, "full_source_data_homo.pkl"), 'rb') as f: full_list = pickle.load(f)
    scaf_to_data = {}
    for d in full_list:
        s = MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False)
        if s not in scaf_to_data: scaf_to_data[s] = []
        scaf_to_data[s].append({'smiles': d.smiles, 'homo': d.y.item()})
    del full_list

    target_pool =[k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 200 and mol_to_grakel_graph(k) is not None]
    selector = ScaffoldSelector(258).cuda(); ref = copy.deepcopy(selector).eval().cuda()
    optimizer = torch.optim.AdamW(selector.parameters(), lr=5e-4); STEPS, G, WORKERS = 40, 33, 2
    
    for s in range(STEPS):
        target_scaf = random.choice(target_pool); target_fp = get_fingerprint(target_scaf)
        state, paths, elite = prepare_dynamic_task(target_scaf, scaf_to_data, target_fp)
        if state is None: continue
        target_smi_set = set([d['smiles'] for d in scaf_to_data[target_scaf]])
        tmp_target = "./.virtual_target.csv"; pd.DataFrame(scaf_to_data[target_scaf]).to_csv(tmp_target, index=False)
        
        base_path, mae_ref = train_shared_base(tmp_target, sup_csvs, 0)
        
        with torch.no_grad():
            _, _, current_probs_tensor = selector.sample_actions(state.cuda(), 1)
            prob_map = current_probs_tensor[0].cpu().numpy()

        final_actions =[]
        while len(final_actions) < G:
            with torch.no_grad(): batch_act, _, _ = selector.sample_actions(state.cuda(), G)
            for i in range(G):
                act = batch_act[i].cpu().numpy()
                if len(final_actions) == 0: act[:] = 0.0 # 强制锚点
                if check_combination_validity(act, paths, target_smi_set):
                    final_actions.append(act)
                if len(final_actions) == G: break
        
        actions_np = np.array(final_actions); actions_tensor = torch.tensor(actions_np).cuda()
        with torch.no_grad(): old_log_p = Bernoulli(selector(state.cuda())[0].expand(G, -1)).log_prob(actions_tensor).sum(-1)
        
        # 使用 list 动态收集有效的 rewards 和对应的 indices
        valid_indices = []
        rewards_list =[]
        
        logger.info(f"\n🚀 Step {s+1}/{STEPS} | 目标: {target_scaf[:20]} | 基准MAE: {mae_ref:.4f}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as exe:
            futures =[exe.submit(worker_fast_proxy, i, actions_np[i], i % 2, paths, tmp_target, sup_csvs, base_path, mae_ref) for i in range(G)]
            for f in tqdm(concurrent.futures.as_completed(futures), total=G, desc="Battle"):
                idx, val, status = f.result()
                
                # 【修改点2】：检测是否返回了 None (代表崩溃或 OOM)
                if val is not None:
                    valid_indices.append(idx)
                    rewards_list.append(val)
                    sel_ids = np.where(actions_np[idx] == 1.0)[0].tolist()
                    sel_probs = [round(float(prob_map[i]), 3) for i in sel_ids]
                    logger.info(f"  [Live] 方案 {idx+1:02} | IDs: {str(sel_ids)} | Probs: {str(sel_probs)} | MAE: {val:.4f} | 改进: {mae_ref-val:+.4f} | {status}")
                else:
                    logger.warning(f"  [Skip] 方案 {idx+1:02} 发生异常 ({status})，已从本次梯度更新中隔离。")

        # 确保至少有 2 个有效样本（锚点 + 至少一个其他样本）来计算均值和方差
        if len(valid_indices) < 2 or 0 not in valid_indices:
            logger.warning(f"⚠️ 本轮有效样本过少或锚点崩溃，跳过参数更新。")
            continue

        # --- 【修改点3】：重构奖惩逻辑与更新 (采样屏蔽 + 软性奖励) ---
        
        # 将无序的 valid 列表根据 idx 重新排序，并剥离出来
        valid_indices = np.array(valid_indices)
        sort_order = np.argsort(valid_indices)
        valid_indices = valid_indices[sort_order]
        rewards_raw = np.array(rewards_list)[sort_order]
        
        # 找到锚点在 valid_indices 中的位置 (肯定是第 0 个，因为排过序)
        anchor_val = rewards_raw[0] 
        final_rewards = np.zeros(len(valid_indices))
        
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            # 改进值 = 基准 MAE (锚点是负数，所以改进 = 锚点 - 当前)
            delta = anchor_val - rewards_raw[idx_in_valid]
            
            # 【取消 -1.0 硬惩罚，使用软性真实的 delta】
            # 现在：做得好是正 delta，做差了是负 delta，不再是断崖式的 -1.0
            final_rewards[idx_in_valid] = delta 
        
        rew_t = torch.tensor(final_rewards, dtype=torch.float32).cuda()
        # 仅基于有效样本计算 Advantage
        adv = (rew_t - rew_t.mean()) / (rew_t.std() + 1e-8)
        
        # 绝对底线控制：表现比不选差的(delta<0)，Adv 强行变为微小负数，拒绝矮子里拔将军
        mask_bad_perf = (rew_t < 0) & (adv > 0)
        adv[mask_bad_perf] = -0.1
        
        # 锚点现在也参与更新了 (无 adv[0]=0.0)，它会作为标杆把大家的选率往下拉
        
        # 过滤出有效的 actions 和 old_log_p 用于计算 Loss
        valid_actions_tensor = actions_tensor[valid_indices]
        valid_old_log_p = old_log_p[valid_indices]
        
        for _ in range(2):
            optimizer.zero_grad()
            curr_p, _ = selector(state.cuda())
            # 计算当前 valid 部分的 log_prob
            new_log_p = Bernoulli(curr_p.expand(len(valid_indices), -1)).log_prob(valid_actions_tensor).sum(-1)
            ratio = torch.exp(new_log_p - valid_old_log_p.detach())
            
            rlp = Bernoulli(ref(state.cuda())[0].expand(len(valid_indices), -1)).log_prob(valid_actions_tensor).sum(-1)
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean() + 0.01 * (torch.exp(rlp - new_log_p) - (rlp - new_log_p) - 1).mean()
            loss.backward(); optimizer.step()
        
        # ================= 【每步强制存盘逻辑】 =================
        torch.save(selector.state_dict(), "universal_selector.pt")
        logger.info(f"💾 Step {s+1} 更新完成，pt 文件已同步。")

        logger.info(f"📈 Step {s+1} | 最佳改进: {final_rewards.max():+.4f} | 选率均值: {curr_p.mean():.3f} | 有效样本: {len(valid_indices)}/{G}")

    torch.save(selector.state_dict(), "universal_selector_final.pt")

if __name__ == "__main__": run_meta_grpo()