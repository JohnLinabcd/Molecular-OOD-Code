# data.py
import os, gc, sys, pickle
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import one_hot, scatter
from torch.utils.data import Dataset, random_split

# ETNN 组件
from etnn.lifter import Lifter, CombinatorialComplexTransform, get_adjacency_types
from etnn.qm9.lifts.registry import LIFTER_REGISTRY
from etnn.combinatorial_data import CombinatorialComplexData

# 适配 PyTorch 2.6+ 加载安全策略
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([CombinatorialComplexData])

def optimized_brics_labeling(mol):
    """虚拟图遍历打标，不修改分子结构，防止 C++ 崩溃"""
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 2: return torch.zeros(num_atoms, dtype=torch.long), 1
    b_bonds = list(BRICS.FindBRICSBonds(mol))
    broken_indices = {mol.GetBondBetweenAtoms(b[0][0], b[0][1]).GetIdx() 
                      for b in b_bonds if mol.GetBondBetweenAtoms(b[0][0], b[0][1])}
    adj = [[] for _ in range(num_atoms)]
    for b in mol.GetBonds():
        if b.GetIdx() not in broken_indices:
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[u].append(v); adj[v].append(u)
    sb = [-1] * num_atoms; ns = 0
    for i in range(num_atoms):
        if sb[i] == -1:
            q = [i]; sb[i] = ns
            while q:
                curr = q.pop(0)
                for n in adj[curr]:
                    if sb[n] == -1: sb[n] = ns; q.append(n)
            ns += 1
    return torch.tensor(sb, dtype=torch.long), ns

class SimpleSMILESDataset(Dataset):
    def __init__(self, csv_paths, cache_dir, label_name, stage, config, filter_smiles=None, calc_brics=True):
        self.csv_paths = csv_paths
        self.label_name = label_name
        self.config = config
        self.stage = stage
        self.filter_smiles = filter_smiles if filter_smiles else set()
        self.calc_brics = calc_brics
        
        # 🎯 核心契约：强制对齐模型 sup200.pth 训练时的标准特征维度 (Hetero部分)
        # 对应总维度: Rank0=26(15+11), Rank1=32(21+11), Rank2=15(4+11)
        self.STANDARD_HETERO_DIMS = {0: 15, 1: 21, 2: 4}

        lifters_list = getattr(self.config, 'lifters', [])
        ranks = [int(l.split(":")[1]) for l in lifters_list if ":" in l and l.split(":")[1].isdigit()]
        self.v_dim = max(ranks) if ranks else 1
        self.expected_ranks = ranks
        
        self.lifter = Lifter(lifters_list, LIFTER_REGISTRY, self.v_dim)
        self.adjacencies = get_adjacency_types(self.v_dim, self.config.connectivity, self.config.neighbor_types)
        self.transform = CombinatorialComplexTransform(self.lifter, self.adjacencies)
        
        # 统一缓存文件名逻辑
        c_base = os.path.basename(csv_paths[0]).replace(".csv", "") if csv_paths else "data"
        self.cache_file = os.path.join(cache_dir, f"{stage}_{c_base}_{label_name}_full.pkl")
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(self.cache_file):
            print(f">>> [Data] 加载已有缓存: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data_list = pickle.load(f)
            # 维度校验：防止加载了旧的、没补齐的缓存
            if len(self.data_list) > 0:
                sample = self.data_list[0]
                if hasattr(sample, 'x_1') and sample.x_1.shape[1] != self.STANDARD_HETERO_DIMS[1]:
                    print(f">>> [Warning] 缓存维度不匹配！正在重新执行全量 Lifting...")
                    self.data_list = self._direct_process_to_pkl(csv_paths)
        else:
            print(f">>> [Data] 缓存不存在，正在执行全量 Lifting 计算 (直接存为 pkl)...")
            self.data_list = self._direct_process_to_pkl(csv_paths)

    def _get_official_features(self, mol):
        """官方 11 维特征提取 (Node Features)"""
        types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        N = mol.GetNumAtoms()
        t_idx, a_num, arom, sp, sp2, sp3 = [], [], [], [], [], []
        for a in mol.GetAtoms():
            t_idx.append(types.get(a.GetSymbol(), 1)); a_num.append(a.GetAtomicNum()); arom.append(1 if a.GetIsAromatic() else 0)
            h = a.GetHybridization(); sp.append(1 if h == Chem.rdchem.HybridizationType.SP else 0)
            sp2.append(1 if h == Chem.rdchem.HybridizationType.SP2 else 0); sp3.append(1 if h == Chem.rdchem.HybridizationType.SP3 else 0)
        z = torch.tensor(a_num, dtype=torch.long)
        ei = torch.tensor(Chem.GetAdjacencyMatrix(mol)).nonzero().t().contiguous()
        num_hs = scatter((z == 1).to(torch.float)[ei[0]], ei[1], dim_size=N, reduce="sum").tolist()
        return torch.cat([one_hot(torch.tensor(t_idx), 5), torch.tensor([a_num, arom, sp, sp2, sp3, num_hs], dtype=torch.float).t()], dim=-1), z

    def _direct_process_to_pkl(self, csv_paths):
        df = pd.concat([pd.read_csv(p) for p in csv_paths if os.path.exists(p)], ignore_index=True)
        final_list = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Lifting {self.stage}"):
            if row['smiles'] in self.filter_smiles: continue
            try:
                mol = Chem.MolFromSmiles(row['smiles']); mol = Chem.AddHs(mol)
                # if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1: continue
                # ✨ 核心修改：强制给 RDKit 传入固定的 randomSeed=42
                embed_params = AllChem.ETKDGv3()
                embed_params.randomSeed = 42
                if AllChem.EmbedMolecule(mol, embed_params) == -1: continue
                
                # 1. 提取官方 Node 特征 (11维)
                x_node, z = self._get_official_features(mol)
                base = Data(x=x_node, pos=torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32), z=z, 
                            edge_index=torch.tensor(Chem.GetAdjacencyMatrix(mol)).nonzero().t().contiguous())
                base.mol = mol
                
                # 2. 执行核心变换 (Hetero 特征)
                cc = self.transform(base)
                
                # 3. ✨ 维度补齐逻辑：强制对齐到训练 Giant 集时的标准 Hetero 维度
                for r in self.expected_ranks:
                    target_h_dim = self.STANDARD_HETERO_DIMS.get(int(r), 0)
                    attr_name = f'x_{r}'
                    
                    if hasattr(cc, attr_name):
                        feat = getattr(cc, attr_name)
                        curr_dim = feat.shape[1]
                        if curr_dim < target_h_dim:
                            # 补齐缺失的化学环境位
                            pad = torch.zeros((feat.shape[0], target_h_dim - curr_dim), dtype=torch.float32)
                            setattr(cc, attr_name, torch.cat([feat, pad], dim=1))
                        elif curr_dim > target_h_dim:
                            # 截断（保险机制）
                            setattr(cc, attr_name, feat[:, :target_h_dim])
                    else:
                        # 如果该阶完全没提取出实体（例如完全没环），补齐一个空矩阵占位
                        setattr(cc, attr_name, torch.zeros((0, target_h_dim), dtype=torch.float32))
                        setattr(cc, f'cells_{r}', torch.tensor([], dtype=torch.long))
                        setattr(cc, f'slices_{r}', torch.tensor([], dtype=torch.long))

                # 4. BRICS 标签
                sb, ns = optimized_brics_labeling(mol) if self.calc_brics else (torch.zeros(mol.GetNumAtoms(), dtype=torch.long), 1)
                
                # 5. 挂载基础属性
                cc.x, cc.sub_batch, cc.num_subs, cc.y, cc.smiles = x_node, sb, ns, torch.tensor([[row[self.label_name]]]), row['smiles']
                
                # 6. 挂载空邻接表补齐（防止 DataLoader 报错）
                for at in self.adjacencies:
                    if not hasattr(cc, f'adj_{at}'): setattr(cc, f'adj_{at}', torch.tensor([[], []], dtype=torch.long))
                
                final_list.append(cc)
                del mol
                if i % 200 == 0: gc.collect()
            except: continue

        if final_list:
            print(f">>> [Data] 正在写入缓存文件: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(final_list, f)
        else:
            raise RuntimeError(f"{self.stage} 处理失败，结果集为空！")
        return final_list

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

# ================== DataModule (用于监督学习) ==================
class DataModule:
    def __init__(self, config):
        self.config = config
        self._mean, self._std = None, None

    def prepare_dataset(self, filter_smiles=None):
        self.train_ds = SimpleSMILESDataset(self.config.train_paths, self.config.cache_dir, self.config.label_name, "train", self.config, filter_smiles)
        self.val_ds = SimpleSMILESDataset(self.config.val_paths, self.config.cache_dir, self.config.label_name, "val", self.config)
        self.test_ds = SimpleSMILESDataset(self.config.test_paths, self.config.cache_dir, self.config.label_name, "test", self.config)
        
        if getattr(self.config, 'standardize', False) and len(self.train_ds) > 0:
            print(">>> [DataModule] 计算源域标准化参数...")
            ys = torch.cat([d.y for d in self.train_ds.data_list])
            self._mean, self._std = ys.mean(dim=0), ys.std(dim=0) + 1e-6

    @property
    def mean(self): return self._mean
    @property
    def std(self): return self._std
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.config.inference_batch_size)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.config.inference_batch_size)

# ================== DADataModule (用于域适应) ==================
class DADataModule:
    def __init__(self, config_dict):
        self.config = config_dict
        self._mean, self._std = None, None

    def prepare_dataset(self, filter_smiles=None):
        from argparse import Namespace
        cfg_obj = Namespace(**self.config) if isinstance(self.config, dict) else self.config
        label_key = self.config.get('label_name', 'homo')
        cache_root = self.config.get('cache_dir', './cache')
        
        self.source_sup_dataset = SimpleSMILESDataset(self.config['sup_source_paths'], cache_root, label_key, "sup", cfg_obj, filter_smiles, calc_brics=False)
        self.source_da_datasets = [SimpleSMILESDataset([p], os.path.join(cache_root, 'da_subset'), label_key, "da", cfg_obj, calc_brics=True) for p in self.config.get('da_source_paths', [])]
        full_target = SimpleSMILESDataset(self.config['target_paths'], os.path.join(cache_root, 'target'), label_key, "target", cfg_obj, calc_brics=True)
        
        n_total = len(full_target); n_unlabeled = int(n_total * 0.8)
        if n_total > 1:
            self.target_unlabeled, self.target_test = random_split(full_target, [n_unlabeled, n_total - n_unlabeled], generator=torch.Generator().manual_seed(42))
        else:
            self.target_unlabeled, self.target_test = full_target, full_target
            
        if len(self.source_sup_dataset) > 0: self._compute_stats()

    def _compute_stats(self):
        print(">>> [DADataModule] 计算源域标准化参数...")
        ys = torch.cat([d.y for d in self.source_sup_dataset.data_list])
        self._mean, self._std = ys.mean(dim=0), ys.std(dim=0) + 1e-6

    @property
    def mean(self): return self._mean
    @property
    def std(self): return self._std
    def source_sup_loader(self): return DataLoader(self.source_sup_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def source_da_loaders(self): return [DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True) for ds in self.source_da_datasets if len(ds) >= 2]
    def target_train_loader(self): return DataLoader(self.target_unlabeled, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def target_test_loader(self): return DataLoader(self.target_test, batch_size=self.config.get('inference_batch_size', 1), shuffle=False)