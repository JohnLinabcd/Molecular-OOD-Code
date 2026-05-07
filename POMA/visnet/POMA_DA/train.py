import numpy as np  # 确保有这一行
import os, pickle, shutil, torch, time
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from visnet.model import ViSNet
from visnet.data import DADataModule  
from config import TrainingConfig
from utils import set_seed, ensure_dir, get_logger
from controller import AdaptiveDAWeightController
from selector_utils import ScaffoldSelector, prepare_dynamic_task, get_fingerprint, MurckoScaffold

# --- 损失函数 ---
def coral_loss(source, target):
    if source.size(0) < 2 or target.size(0) < 2: return torch.tensor(0.0, device=source.device)
    d = source.size(1)
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(xm.t(), xm) / (source.size(0) - 1)
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(xmt.t(), xmt) / (target.size(0) - 1)
    return torch.norm(xc - xct, p='fro').pow(2) / (4 * d * d)

def sample_substructures(feat, sample_size):
    if feat.size(0) <= sample_size: return feat
    return feat[torch.randperm(feat.size(0), device=feat.device)[:sample_size]]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae, cnt = 0, 0
    for b in loader:
        b = b.to(device)
        res = model(b.z, b.pos, b.batch, b.sub_batch)
        mae += (res[0] - b.y.view(res[0].shape)).abs().sum().item()
        cnt += b.y.size(0)
    return mae / cnt if cnt > 0 else 0.0

# --- 智能筛选 ---
def run_smart_selection(config):
    device = torch.device(config.device)
    selector = ScaffoldSelector(258).to(device)
    selector.load_state_dict(torch.load("universal_selector.pt", map_location=device))
    selector.eval()

    cache_file = os.path.join(config.cache_dir, "data_brics_homo.pkl")
    with open(cache_file, 'rb') as f:
        full_list = pickle.load(f)
    
    scaf_to_data = {}
    for d in full_list:
        s = MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False)
        if s not in scaf_to_data: scaf_to_data[s] = []
        scaf_to_data[s].append({'smiles': d.smiles, 'homo': d.y.item()})
    
    target_df = pd.read_csv(config.target_paths[0])
    target_scaf = MurckoScaffold.MurckoScaffoldSmiles(target_df['smiles'].iloc[0], False)
    target_fp = get_fingerprint(target_scaf)
    
    state, _, results = prepare_dynamic_task(target_scaf, scaf_to_data, target_fp)
    with torch.no_grad():
        probs, _ = selector(state.to(device))
        probs = probs[0].cpu().numpy()
    
    top_idx = probs.argsort()[-5:][::-1]
    # 替换原来的 top_idx = probs.argsort()[-5:][::-1]
    # valid_indices = np.where(probs >= 0.33)[0]
    # sorted_valid_indices = valid_indices[probs[valid_indices].argsort()[::-1]]
    # top_idx = sorted_valid_indices[:5]
    out_dir = "./smart_selected_csvs"
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    for i in top_idx:
        res = results[i]
        score = probs[i] # 获取该骨架对应的 Selector 评分
        # 在文件名中加入 score_{:.4f} 以供 config 解析权重
        fname = os.path.join(out_dir, f"score_{score:.4f}_sim_{res['sim']:.3f}_cnt_{res['cnt']}.csv")
        pd.DataFrame(scaf_to_data[res['scaf']]).to_csv(fname, index=False)
    print(f">>> Selector 已成功挑选 Top 5 骨架并生成 CSV。")

# --- 训练主循环 ---
def train_curriculum(config: TrainingConfig):
    set_seed(config.seed)
    run_smart_selection(config)
    config.__post_init__() # 重新触发权重解析

    out_dir = f"./outputs/final_train_{os.path.basename(config.target_paths[0])}"
    ensure_dir(out_dir); logger = get_logger(os.path.join(out_dir, "train.log"))
    
    # 详尽日志：训练启动信息
    logger.info(f"🚀 开始训练 | 目标: {os.path.basename(config.target_paths[0])} | 设备: {config.device}")
    logger.info(f"🧬 精英源权重: {[round(w, 4) for w in config.da_source_weights]}")

    dm = DADataModule(vars(config))
    dm.prepare_dataset()
    
    source_sup_loader = dm.source_sup_loader()
    source_da_loaders = dm.source_da_loaders()
    da_weights = torch.tensor(config.da_source_weights, device=config.device)
    
    target_train_loader = dm.target_train_loader()
    target_test_loader = dm.target_test_loader()
    
    mean, std = dm.mean.to(config.device), dm.std.to(config.device)
    model = ViSNet(hidden_channels=config.hidden_channels, num_layers=config.num_layers, num_heads=config.num_heads, 
                   num_rbf=config.num_rbf, mean=mean, std=std, num_tasks=1).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 110, 140, 170, 200], gamma=0.5)
    controller = AdaptiveDAWeightController(config.init_w_mol, config.init_w_sub)
    
    w_mol, w_sub, w_reg = (config.init_w_mol, config.init_w_sub, 1.0)
    best_mae = float('inf')

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_reg, epoch_mol, epoch_sub = 0, 0, 0
        active_w_mol = w_mol if epoch > 10 else 0.0
        active_w_sub = w_sub if epoch > 10 else 0.0

        da_iters = [iter(l) for l in source_da_loaders]
        target_iter = iter(target_train_loader)
        
        # 详细日志：Epoch 进度条
        pbar = tqdm(source_sup_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch_s_sup in pbar:
            optimizer.zero_grad()
            batch_s_sup = batch_s_sup.to(config.device)
            out_s, _, _ = model(batch_s_sup.z, batch_s_sup.pos, batch_s_sup.batch, batch_s_sup.sub_batch)
            loss_reg = F.l1_loss((out_s - mean)/std, (batch_s_sup.y - mean)/std)
            
            loss_da = torch.tensor(0.0, device=config.device)
            current_mol_raw = 0.0
            current_sub_raw = 0.0

            if active_w_mol > 0 or active_w_sub > 0:
                try: batch_t = next(target_iter).to(config.device)
                except StopIteration: target_iter = iter(target_train_loader); batch_t = next(target_iter).to(config.device)
                _, mol_t, sub_t = model(batch_t.z, batch_t.pos, batch_t.batch, batch_t.sub_batch)
                
                # --- 修正点：将初始化从整数 0 改为张量，防止 .item() 报错 ---
                l_m_total = torch.tensor(0.0, device=config.device)
                l_s_total = torch.tensor(0.0, device=config.device)
                
                for i, s_iter in enumerate(da_iters):
                    try: batch_s = next(s_iter).to(config.device)
                    except StopIteration: continue
                    _, mol_s, sub_s = model(batch_s.z, batch_s.pos, batch_s.batch, batch_s.sub_batch)
                    
                    alpha_i = da_weights[i]
                    if active_w_mol > 0: 
                        l_m_total += alpha_i * coral_loss(mol_s, mol_t)
                    if active_w_sub > 0: 
                        l_s_total += alpha_i * coral_loss(sample_substructures(sub_s, 128), sample_substructures(sub_t, 128))
                
                # 记录原始损失数值用于统计
                current_mol_raw = l_m_total.item()
                current_sub_raw = l_s_total.item()
                # 计算加权后的 DA 损失并截断
                loss_da = (active_w_mol * torch.clamp(l_m_total, max=10.0)) + (active_w_sub * torch.clamp(l_s_total, max=5.0))
                epoch_mol += current_mol_raw
                epoch_sub += current_sub_raw

            loss = (w_reg * loss_reg) + loss_da
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_reg += loss_reg.item()

        # 结算本轮统计
        num_batches = len(source_sup_loader)
        avg_reg = epoch_reg / num_batches
        avg_mol = epoch_mol / num_batches
        avg_sub = epoch_sub / num_batches
        epoch_duration = time.time() - epoch_start_time
        curr_lr = optimizer.param_groups[0]['lr']

        # 权重控制器更新逻辑 (输入放大1000倍显示)
        if epoch >= 10: 
            w_mol, w_sub, w_reg = controller.update(epoch, avg_reg, avg_mol , avg_sub )
        
        # 目标域评估
        tgt_mae = evaluate(model, target_test_loader, config.device)
        
        # 详尽日志行：构造格式化输出
        log_line = (
            f"Ep {epoch:03d}/{config.epochs} | Time: {epoch_duration:.1f}s | LR: {curr_lr:.1e} | "
            f"MAE: {tgt_mae:.4f} "
        )
        if tgt_mae < best_mae:
            best_mae = tgt_mae
            log_line += "⭐ | " # 记录更优模型
            torch.save({'model_state_dict': model.state_dict(), 'mae': tgt_mae, 'epoch': epoch}, 
                       os.path.join(out_dir, "best_model.pt"))
        else:
            log_line += "   | "
        
        log_line += (
            f"Reg: {avg_reg:.4f} | molDA: {avg_mol*1000:.3f} | subDA: {avg_sub*1000:.3f} | "
            f"W(r,m,s): ({w_reg:.2f}, {active_w_mol:.1e}, {active_w_sub:.1e})"
        )
        logger.info(log_line)

        # 保存最后一轮模型 (pth 格式)
        if epoch == config.epochs:
            torch.save({'model_state_dict': model.state_dict(), 'mae': tgt_mae, 'epoch': epoch}, 
                       os.path.join(out_dir, "last_model.pth"))
            logger.info(f">>> Final Epoch {epoch} Model saved as last_model.pth")

        scheduler.step()

if __name__ == "__main__":
    train_curriculum(TrainingConfig())
