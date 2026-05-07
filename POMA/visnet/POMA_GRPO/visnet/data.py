import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
import os
import pickle
import numpy as np

class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'sub_batch': return self.num_subs
        return super().__inc__(key, value, *args, **kwargs)

class MultiTaskDataset(Dataset):
    def __init__(self, csv_paths, cache_dir, label_names=['homo'], filter_smiles=None):
        self.csv_paths = csv_paths
        self.label_names = label_names
        self.filter_smiles = filter_smiles if filter_smiles else set()
        labels_str = "_".join(label_names)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 修正变量名一致性
        if len(csv_paths) > 1:
            self.new_cache = os.path.join(cache_dir, f"full_source_data_{labels_str}.pkl")
        else:
            base = os.path.basename(csv_paths[0]).replace(".csv", "")
            self.new_cache = os.path.join(cache_dir, f"cache_{base}_{labels_str}.pkl")

        if os.path.exists(self.new_cache):
            with open(self.new_cache, 'rb') as f:
                raw_data_list = pickle.load(f)
        else:
            raw_data_list = self._load_from_csv()
            with open(self.new_cache, 'wb') as f:
                pickle.dump(raw_data_list, f)
        
        self.data_list = [d for d in raw_data_list if d.smiles not in self.filter_smiles]

    def _load_from_csv(self):
        data_list = []
        all_dfs = [pd.read_csv(p) for p in self.csv_paths if os.path.exists(p)]
        if not all_dfs: return []
        df = pd.concat(all_dfs, ignore_index=True)
        for _, row in df.iterrows():
            smiles = str(row.get('smiles', ''))
            try:
                if "." in smiles or not smiles: continue
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1: continue
                conf = mol.GetConformer()
                pos = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
                z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
                labels = [float(row[n]) for n in self.label_names]
                data_list.append(MolData(z=torch.tensor(z), pos=torch.tensor(pos, dtype=torch.float), 
                                        num_nodes=len(z), y=torch.tensor([labels], dtype=torch.float), 
                                        smiles=smiles, sub_batch=torch.zeros(len(z), dtype=torch.long), num_subs=1))
            except: continue
        return data_list

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class DADataModule(LightningDataModule):
    def __init__(self, config_dict):
        super().__init__()
        self.config = config_dict
        self._mean, self._std = None, None

    def prepare_dataset(self, filter_smiles=None):
        self.source_sup_dataset = MultiTaskDataset(self.config['sup_source_paths'], self.config['cache_dir'], self.config['label_names'], filter_smiles=filter_smiles)
        self.source_da_datasets = [MultiTaskDataset([p], os.path.join(self.config['cache_dir'], 'da_subset'), self.config['label_names']) for p in self.config.get('da_source_paths', [])]
        full_target = MultiTaskDataset(self.config['target_paths'], os.path.join(self.config['cache_dir'], 'target'), self.config['label_names'])
        n_total = len(full_target)
        n_unlabeled = int(n_total * 0.8)
        self.target_unlabeled, self.target_test = random_split(full_target, [n_unlabeled, n_total - n_unlabeled], generator=torch.Generator().manual_seed(42))
        if len(self.source_sup_dataset) > 0: self._compute_stats()

    def _compute_stats(self):
        loader = DataLoader(self.source_sup_dataset, batch_size=min(128, len(self.source_sup_dataset)))
        all_y = torch.cat([b.y for b in loader], dim=0)
        self._mean, self._std = all_y.mean(dim=0), all_y.std(dim=0) + 1e-6

    @property
    def mean(self): return self._mean
    @property
    def std(self): return self._std
    def source_sup_loader(self): return DataLoader(self.source_sup_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def source_da_loaders(self): 
        return [DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True) for ds in self.source_da_datasets if len(ds) >= 50]
    def target_train_loader(self): return DataLoader(self.target_unlabeled, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def target_test_loader(self): return DataLoader(self.target_test, batch_size=self.config['inference_batch_size'], shuffle=False)