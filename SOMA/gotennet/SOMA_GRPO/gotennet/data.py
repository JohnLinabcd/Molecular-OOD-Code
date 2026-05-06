# gotennet/data.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'sub_batch': return self.num_subs
        return super().__inc__(key, value, *args, **kwargs)

class MultiTaskDataset(Dataset):
    def __init__(self, csv_paths, cache_dir, label_names=['homo'], filter_smiles=None):
        self.csv_paths = csv_paths
        self.label_names = label_names
        # 🌟 1. 确保过滤名单是一个集合，提高查找效率
        self.filter_smiles = set(filter_smiles) if filter_smiles else set()
        
        os.makedirs(cache_dir, exist_ok=True)
        
        labels_str = "_".join(label_names)
        if len(csv_paths) > 1:
            self.new_cache = os.path.join(cache_dir, f"full_source_data_{labels_str}.pkl")
        else:
            base = os.path.basename(csv_paths[0]).replace(".csv", "")
            self.new_cache = os.path.join(cache_dir, f"cache_{base}_{labels_str}.pkl")

        # 加载数据（从缓存或从源文件）
        if os.path.exists(self.new_cache):
            with open(self.new_cache, 'rb') as f: 
                self.data_list = pickle.load(f)
        else:
            self.data_list = self._load_from_csv()
            with open(self.new_cache, 'wb') as f: 
                pickle.dump(self.data_list, f)

        # 🌟 2. 核心过滤逻辑：不论是从缓存读还是从CSV读，加载后立即根据名单剔除重复分子
        if self.filter_smiles:
            initial_count = len(self.data_list)
            self.data_list = [d for d in self.data_list if d.smiles not in self.filter_smiles]
            # 只有当确实剔除了分子时才打印信息（避免子进程日志太乱）
            diff = initial_count - len(self.data_list)
            if diff > 0:
                print(f">>> 数据清洗：已从当前数据集中剔除 {diff} 个目标域重合分子")

    def _get_brics_substructures(self, mol):
        num_atoms = mol.GetNumAtoms()
        atom_to_sub_id = torch.zeros(num_atoms, dtype=torch.long)
        try:
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            if not brics_bonds: return atom_to_sub_id, 1
            bond_indices = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for (a1, a2), _ in brics_bonds if mol.GetBondBetweenAtoms(a1, a2)]
            mol_broken = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
            frags = Chem.GetMolFrags(mol_broken)
            for sub_id, atom_indices in enumerate(frags):
                for idx in atom_indices:
                    if idx < num_atoms: atom_to_sub_id[idx] = sub_id
            return atom_to_sub_id, len(frags)
        except: return atom_to_sub_id, 1

    def _load_from_csv(self):
        data_list = []
        dfs = [pd.read_csv(p) for p in self.csv_paths if os.path.exists(p)]
        if not dfs: return []
        for _, row in tqdm(pd.concat(dfs, ignore_index=True).iterrows(), total=len(pd.concat(dfs)), desc="CSV->3D"):
            smiles, target = row.get('smiles', ''), row[self.label_names[0]]
            if "." in smiles or pd.isna(target): continue
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1: continue
                conf = mol.GetConformer()
                pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float)
                z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())], dtype=torch.long)
                sub_batch, num_subs = self._get_brics_substructures(mol)
                data_list.append(MolData(z=z, pos=pos, y=torch.tensor([[target]], dtype=torch.float), sub_batch=sub_batch, num_subs=num_subs, smiles=smiles))
            except: continue
        return data_list
    
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class DADataModule(LightningDataModule):
    def __init__(self, config_dict):
        super().__init__()
        self.config = config_dict
        self._mean, self._std = None, None

    # 🌟 3. 开启参数接收口，允许 lookfor.py 传入 filter_smiles
    def prepare_dataset(self, filter_smiles=None):
        ln = self.config.get('label_name', 'homo')
        
        # 将名单分发给源域训练集
        self.source_sup_dataset = MultiTaskDataset(
            self.config['sup_source_paths'], 
            self.config['cache_dir'], 
            [ln],
            filter_smiles=filter_smiles
        )
        
        # 将名单分发给各个 DA 候选源域
        self.source_da_datasets = [
            MultiTaskDataset(
                [p], 
                os.path.join(self.config['cache_dir'], 'da_subset'), 
                [ln],
                filter_smiles=filter_smiles
            ) for p in self.config.get('da_source_paths', [])
        ]
        
        # 目标域（考试题）本身不需要过滤自己
        full_target = MultiTaskDataset(self.config['target_paths'], os.path.join(self.config['cache_dir'], 'target'), [ln])
        
        n_total = len(full_target)
        n_unlabeled = int(n_total * 0.8) if n_total > 1 else 0
        self.target_unlabeled, self.target_test = random_split(full_target, [n_unlabeled, n_total - n_unlabeled], generator=torch.Generator().manual_seed(42))
        
        if len(self.source_sup_dataset) > 0:
            loader = DataLoader(self.source_sup_dataset, batch_size=256)
            all_y = torch.cat([b.y for b in loader], dim=0)
            self._mean, self._std = all_y.mean(dim=0), all_y.std(dim=0) + 1e-6

    @property
    def mean(self): return self._mean
    @property
    def std(self): return self._std
    def source_sup_loader(self): return DataLoader(self.source_sup_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def source_da_loaders(self): return [DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True) for ds in self.source_da_datasets if len(ds) > 0]
    def target_train_loader(self): return DataLoader(self.target_unlabeled, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def target_test_loader(self): return DataLoader(self.target_test, batch_size=self.config.get('inference_batch_size', 28), shuffle=False, drop_last=False)