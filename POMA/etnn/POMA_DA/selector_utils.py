# selector_utils.py
import os
import shutil
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# 必须安装 grakel 库
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph as GrakelGraph

# ==========================================
# 1. 强化学习模型 (Selector) 定义
# ==========================================
class ScaffoldSelector(nn.Module):
    """
    强化学习训练出的决策大脑。
    输入维度 = 128(Target_FP) + 128(Cand_FP) + 1(Sim) + 1(Count) = 258
    """
    def __init__(self, input_dim=258):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits), logits

# ==========================================
# 2. 化学与图核辅助函数
# ==========================================
def get_fingerprint(smiles, n_bits=128):
    """获取分子的 Morgan 指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)) if mol else np.zeros(n_bits)
    except: return np.zeros(n_bits)

def mol_to_grakel_graph(smiles):
    """将 SMILES 转换为 GraKel 图，用于高精度拓扑相似度匹配"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 1: return None 
        node_labels = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        return GrakelGraph(adj.tolist(), node_labels=node_labels)
    except: return None

def prepare_dynamic_task(target_scaf_smi, scaf_to_data, target_fp):
    """
    为 Target 骨架计算全库中其它骨架的 Weisfeiler-Lehman (WL) 图核相似度。
    返回相似度最高的 Top-50 候选状态（State）。
    """
    target_graph = mol_to_grakel_graph(target_scaf_smi)
    if not target_graph: return None, None, None

    cand_keys = [k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 60]
    active_keys, graphs = [], []
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g:
            graphs.append(g)
            active_keys.append(k)
    
    if not graphs: return None, None, None

    # WL 核相似度计算
    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph])
    sims = wl.transform(graphs).flatten() 

    results_all = []
    for i, k in enumerate(active_keys):
        results_all.append({'scaf': k, 'sim': sims[i], 'cnt': len(scaf_to_data[k])})
    
    results_top50 = sorted(results_all, key=lambda x: x['sim'] * np.log1p(x['cnt']), reverse=True)[:50]

    state_feats = []
    for c in results_top50:
        c_fp = get_fingerprint(c['scaf'])
        feat = np.concatenate([target_fp, c_fp, [c['sim']], [np.log1p(c['cnt'])/10.0]])
        state_feats.append(feat)
        
    return torch.tensor([state_feats], dtype=torch.float32), None, results_top50

# ==========================================
# 3. 核心功能：加载 .pt 模型执行选源 (Pipeline)
# ==========================================
def run_smart_selection(config):
    """
    加载强化学习结果 (universal_selector_etnn_final.pt)，
    在全量源域中挑出得分最高的前 5 个辅助骨架，
    并把它们的数据存为 CSV，供 config.py 读取。
    """
    device = torch.device(config.device)
    
    # 1. 加载强化学习得到的模型权重
    model_path = "universal_selector_etnn_final.pt"
    if not os.path.exists(model_path):
        # 兼容备用名
        model_path = "universal_selector_etnn.pt"
    
    if not os.path.exists(model_path):
        print(f">>> [Error] 找不到强化学习模型 {model_path}，无法进行智能筛选！")
        return

    print(f">>> [Selector] 正在加载强化学习模型: {model_path}")
    selector = ScaffoldSelector(input_dim=258).to(device)
    selector.load_state_dict(torch.load(model_path, map_location=device))
    selector.eval()

    # 2. 读取全量监督数据的 pkl 缓存
    # 我们假设全量数据已经由 data.py 预处理好了
    c_base = os.path.basename(config.sup_source_paths[0]).replace(".csv", "")
    sup_cache_file = os.path.join(config.cache_dir, f"sup_{c_base}_{config.label_name}_full.pkl")
    
    if not os.path.exists(sup_cache_file):
        print(f">>> [Error] 源域缓存不存在: {sup_cache_file}。请先运行 data.py 预处理！")
        return

    print(">>> [Selector] 加载源域全量数据构建骨架池...")
    with open(sup_cache_file, 'rb') as f:
        full_list = pickle.load(f)
    
    # 构建骨架映射字典
    scaf_to_data = {}
    for d in full_list:
        s = MurckoScaffold.MurckoScaffoldSmiles(d.smiles, False)
        if s not in scaf_to_data: 
            scaf_to_data[s] = []
        scaf_to_data[s].append({'smiles': d.smiles, config.label_name: d.y.item()})
    
    # 3. 解析当前 Target 域信息
    target_df = pd.read_csv(config.target_paths[0])
    # 获取 Target 中最典型的骨架作为锚点
    target_scaf = MurckoScaffold.MurckoScaffoldSmiles(target_df['smiles'].iloc[0], False)
    target_fp = get_fingerprint(target_scaf)
    
    print(f">>> [Selector] 当前 Target 骨架: {target_scaf[:40]}...")
    
    # 4. 让 Selector 打分
    state, _, results = prepare_dynamic_task(target_scaf, scaf_to_data, target_fp)
    if state is None:
        print(">>> [Error] 无法为该 Target 构建候选池。")
        return

    with torch.no_grad():
        # 推断每个候选源成为“最佳替身”的概率
        probs, _ = selector(state.to(device))
        probs = probs[0].cpu().numpy()
    
    # 5. 严格选取概率最高的前 5 名
    top_idx = probs.argsort()[-5:][::-1]
    
    # 6. 导出为 CSV，供 config.py __post_init__ 读取
    out_dir = "./smart_selected_csvs"
    if os.path.exists(out_dir): 
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    print(f">>> [Selector] 开始导出 Top 5 辅助源...")
    for i in top_idx:
        res = results[i]
        score = probs[i] 
        # 文件名埋点，config.py 会用正则提取 sim 和 score
        fname = os.path.join(out_dir, f"score_{score:.4f}_sim_{res['sim']:.3f}_cnt_{res['cnt']}.csv")
        # 将该骨架下的所有分子保存为一个独立的 csv
        pd.DataFrame(scaf_to_data[res['scaf']]).to_csv(fname, index=False)
        
    print(f">>> [Selector] 成功生成 {len(top_idx)} 个精选源数据文件！")