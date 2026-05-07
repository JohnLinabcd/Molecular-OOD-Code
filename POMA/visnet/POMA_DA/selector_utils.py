import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# --- 新增 GraKel 导入 ---
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph as GrakelGraph

class ScaffoldSelector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 1)
        )
    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits), logits

def get_fingerprint(smiles, n_bits=128):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)) if mol else np.zeros(n_bits)
    except: return np.zeros(n_bits)

# --- 新增：将 SMILES 转为 GraKel 图格式的辅助函数 ---
def mol_to_grakel_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 1: return None 
        node_labels = {i: atom.GetAtomicNum() for i, atom in enumerate(mol.GetAtoms())}
        return GrakelGraph(adj.tolist(), node_labels=node_labels)
    except: return None

# --- 修改后的准备函数 (改动点集中在此) ---
def prepare_dynamic_task(target_scaf_smi, scaf_to_data, target_fp):
    # 1. 转换目标骨架为图
    target_graph = mol_to_grakel_graph(target_scaf_smi)
    if not target_graph: return None, None, None

    # 2. 准备候选池（过滤掉数据量太小的）
    cand_keys = [k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 60]
    
    active_keys = []
    graphs = []
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g:
            graphs.append(g)
            active_keys.append(k)
    
    if not graphs: return None, None, None

    # 3. 【核心改动】使用 Weisfeiler-Lehman 图核计算相似度
    # 注意：n_iter 必须与你 lookfor 阶段训练时保持一致
    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph])
    # 一次性计算出所有候选相对于目标的 WL 相似度
    sims = wl.transform(graphs).flatten() 

    # 4. 组装结果
    results_all = []
    for i, k in enumerate(active_keys):
        results_all.append({
            'scaf': k, 
            'sim': sims[i], # 这里的 sim 已经是图核相似度了
            'cnt': len(scaf_to_data[k])
        })
    
    # 按照 图核相似度 * log(样本量) 排序初筛前50名
    results_top50 = sorted(results_all, key=lambda x: x['sim'] * np.log1p(x['cnt']), reverse=True)[:50]

    # 5. 构建特征向量 (258维)
    state_feats = []
    for c in results_top50:
        c_fp = get_fingerprint(c['scaf'])
        # 拼接：Target_FP (128) + Cand_FP (128) + WL_Sim (1) + Norm_Count (1)
        feat = np.concatenate([
            target_fp, 
            c_fp, 
            [c['sim']], 
            [np.log1p(c['cnt'])/10.0]
        ])
        state_feats.append(feat)
        
    return torch.tensor([state_feats], dtype=torch.float32), None, results_top50