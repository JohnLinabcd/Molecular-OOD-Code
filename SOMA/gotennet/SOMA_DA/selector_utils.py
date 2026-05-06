import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

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

def mol_to_grakel_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        adj = Chem.GetAdjacencyMatrix(mol)
        if adj.shape[0] < 1: return None 
        node_labels = {i: mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())}
        return GrakelGraph(adj, node_labels=node_labels)
    except:
        return None

def prepare_dynamic_task(target_scaf_smi, scaf_to_data, target_fp):
    target_graph = mol_to_grakel_graph(target_scaf_smi)
    if not target_graph: return None, None, None

    cand_keys = [k for k in scaf_to_data.keys() if len(scaf_to_data[k]) >= 60]
    
    active_keys = []
    graphs = []
    for k in cand_keys:
        g = mol_to_grakel_graph(k)
        if g:
            graphs.append(g)
            active_keys.append(k)
    
    if not graphs: return None, None, None

    wl = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    wl.fit([target_graph])
    sims = wl.transform(graphs).flatten() 

    results_all = []
    for i, k in enumerate(active_keys):
        results_all.append({
            'scaf': k, 
            'sim': sims[i], 
            'cnt': len(scaf_to_data[k])
        })
    
    results_all.sort(key=lambda x: x['sim'] * np.log1p(x['cnt']), reverse=True)
    top_cands = results_all[:100]

    states = []
    for res in top_cands:
        s_fp = get_fingerprint(res['scaf'])
        state = np.concatenate([target_fp, s_fp, [res['sim'], np.log1p(res['cnt'])]])
        states.append(state)
    
    return torch.tensor(np.array(states), dtype=torch.float32), None, top_cands