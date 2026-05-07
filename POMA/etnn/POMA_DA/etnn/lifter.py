# etnn/lifter.py
from collections import defaultdict
from functools import partial
import numpy as np
import torch
from scipy.sparse import csc_matrix
from toponetx.classes import CombinatorialComplex
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from etnn.combinatorial_data import Cell, CombinatorialComplexData

class Lifter:
    def __init__(self, lifter_names, lifter_registry, lifter_dim=None, **kwargs):
        self.lifters = []
        for l_str in lifter_names:
            parts = l_str.split(":")
            method = parts[0]
            l_func = partial(lifter_registry[method], **kwargs) if method == "rips" else lifter_registry[method]
            rank = "c" if (len(parts) == 1 or parts[1] == "c") else int(parts[1])
            self.lifters.append((l_func, rank))
        self.num_features_dict = defaultdict(int)
        for lf, r in self.lifters:
            if isinstance(r, int): self.num_features_dict[r] += getattr(lf, 'num_features', 0)
        self.dim = lifter_dim if lifter_dim is not None else max([r for _, r in self.lifters if isinstance(r, int)] + [1])

    def lift(self, graph: Data):
        cell_lifter_map = {}
        for idx, (lf, _) in enumerate(self.lifters):
            out = lf(graph)
            for node_idc, f_vec in out:
                if node_idc not in cell_lifter_map: cell_lifter_map[node_idc] = [None] * len(self.lifters)
                cell_lifter_map[node_idc][idx] = f_vec
        cell_dict = {r: {} for r in range(self.dim + 1)}
        for node_idc, f_vecs in cell_lifter_map.items():
            ranks = []
            for idx, fv in enumerate(f_vecs):
                if fv is not None:
                    lr = self.lifters[idx][1]
                    ranks.append(len(node_idc)-1 if lr=="c" else lr)
            rank = min(ranks) if ranks else 0
            if rank <= self.dim:
                comb_f = []
                for idx, (lf, lr) in enumerate(self.lifters):
                    if lr == rank:
                        comb_f.extend(f_vecs[idx] if f_vecs[idx] is not None else [0.0]*getattr(lf,'num_features',0))
                cell_dict[rank][(node_idc, tuple(comb_f))] = [(fv is not None) for fv in f_vecs]
        return cell_dict

def get_adjacency_types(max_dim, connectivity, neighbor_types):
    adj_types = []
    for i in range(max_dim + 1):
        adj_types.append(f"{i}_{i}")
        if connectivity == "self_and_next" and i < max_dim: adj_types.append(f"{i}_{i+1}")
    new_adjs = []
    for at in adj_types:
        r = list(map(int, at.split("_")))
        if r[0] == r[1]:
            for nt in neighbor_types:
                if nt=="+1" and r[0]<max_dim: new_adjs.append(f"{r[0]}_{r[0]}_{r[0]+1}")
                elif nt=="-1" and r[0]>0: new_adjs.append(f"{r[0]}_{r[0]}_{r[0]-1}")
        else: new_adjs.append(at)
    # return list(set(new_adjs))//改动林卓豪
    return sorted(list(set(new_adjs)))

# 安全矩阵生成，防止空维度引发 Core Dump
# 安全矩阵保护函数
def incidence_matrix(cc, rank, to_rank):
    from scipy.sparse import csc_matrix
    ni, nj = len(cc.skeleton(rank=rank)), len(cc.skeleton(rank=to_rank))
    if ni == 0 or nj == 0: return csc_matrix((ni, nj), dtype=np.float64)
    try:
        if rank < to_rank: return cc.incidence_matrix(rank=rank, to_rank=to_rank)
        return cc.incidence_matrix(rank=to_rank, to_rank=rank).T
    except: return csc_matrix((ni, nj), dtype=np.float64)

def adjacency_matrix(cc, rank, via_rank):
    from scipy.sparse import csc_matrix
    ni = len(cc.skeleton(rank=rank))
    if ni == 0: return csc_matrix((ni, ni), dtype=np.float64)
    try:
        kwargs = dict(rank=rank, via_rank=via_rank, index=False)
        return cc.coadjacency_matrix(**kwargs) if via_rank < rank else cc.adjacency_matrix(**kwargs)
    except: return csc_matrix((ni, ni), dtype=np.float64)

class CombinatorialComplexTransform(BaseTransform):
    def __init__(self, lifter, adjacencies):
        super().__init__(); self.lifter = lifter; self.adjacencies = adjacencies
    def forward(self, graph: Data) -> CombinatorialComplexData:
        cc_cells = self.lifter.lift(graph)
        cc = CombinatorialComplex()
        for r, cells in cc_cells.items():
            if len(cells) > 0: cc.add_cells_from([c[0] for c in cells.keys()], ranks=r)
        
        adj_d = {}
        for at in self.adjacencies:
            r = [int(x) for x in at.split("_")]
            sm = adjacency_matrix(cc, r[0], r[2]) if len(r)==3 else incidence_matrix(cc, r[0], r[1])
            r_idx, c_idx = sm.nonzero()
            adj_d[at] = torch.from_numpy(np.array([r_idx, c_idx])).long()
            
        res = graph.to_dict()
        for r, cells in cc_cells.items():
            res[f"cell_{r}"] = [sorted(c[0]) for c in cells.keys()]
            res[f"x_{r}"] = [c[1] for c in cells.keys()]
            res[f"mem_{r}"] = list(cells.values())
        for k, v in adj_d.items(): res[f"adj_{k}"] = v
        res["num_features_dict"] = {r: self.lifter.num_features_dict[r] for r in range(self.lifter.dim + 1)}
        for k, v in res.items():
            if torch.is_tensor(v): res[k] = v.tolist()
        if "mol" in res: res.pop("mol")
        return CombinatorialComplexData.from_ccdict(res)