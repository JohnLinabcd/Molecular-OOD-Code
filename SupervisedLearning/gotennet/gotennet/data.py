# data.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

class MissingLabelException(Exception): pass

class MolData(Data):
    """支持域适应框架子图切分的专属 Data"""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'sub_batch':
            return self.num_subs
        return super().__inc__(key, value, *args, **kwargs)

class StandardDataset(Dataset):
    def __init__(self, csv_paths, cache_dir="./cache", label_name="gap"):
        self.csv_paths = csv_paths
        self.label_name = label_name
        self.cache_file = os.path.join(cache_dir, f"data_cache_{label_name}_brics.pkl")
        
        if os.path.exists(self.cache_file):
            print(f">>> 加载包含 BRICS 的缓存数据: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data_list = pickle.load(f)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(">>> 提取 3D 构象并挂载 BRICS 标签...")
            self.data_list = self._load_data()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data_list, f)
            print(f">>> 缓存已保存: {self.cache_file}")

    def _get_brics_substructures(self, mol):
        try:
            num_atoms = mol.GetNumAtoms()
            atom_to_sub_id = torch.zeros(num_atoms, dtype=torch.long)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            
            if not brics_bonds: return atom_to_sub_id, 1
                
            bond_indices = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() 
                            for (a1, a2), _ in brics_bonds if mol.GetBondBetweenAtoms(a1, a2)]
            if not bond_indices: return atom_to_sub_id, 1
                
            mol_broken = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
            frags = Chem.GetMolFrags(mol_broken)
            for sub_id, atom_indices in enumerate(frags):
                for atom_idx in atom_indices:
                    if atom_idx < num_atoms: atom_to_sub_id[atom_idx] = sub_id
            return atom_to_sub_id, len(frags)
        except Exception:
            return torch.zeros(mol.GetNumAtoms(), dtype=torch.long), 1

    def _load_data(self):
        data_list = []
        all_dfs = [pd.read_csv(p) for p in self.csv_paths if os.path.exists(p)]
        if not all_dfs: return []
            
        df = pd.concat(all_dfs, ignore_index=True)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Molecules"):
            smiles, target_val = row.get('smiles', ''), row.get(self.label_name)
            if "." in smiles or pd.isna(target_val) or np.isinf(target_val): continue
                
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1: continue
                try: AllChem.MMFFOptimizeMolecule(mol)
                except: pass 
                
                conf = mol.GetConformer()
                num_atoms = mol.GetNumAtoms()
                pos = [list(conf.GetAtomPosition(i)) for i in range(num_atoms)]
                z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)]
                
                sub_batch, num_subs = self._get_brics_substructures(mol)
                data_list.append(MolData(
                    z=torch.tensor(z, dtype=torch.long), 
                    pos=torch.tensor(pos, dtype=torch.float32), 
                    num_nodes=num_atoms, 
                    y=torch.tensor([[target_val]], dtype=torch.float32), 
                    smiles=smiles, sub_batch=sub_batch, num_subs=num_subs
                ))
            except: continue
        return data_list
    
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()

    def prepare_dataset(self):
        label_name = self.hparams.get("label_name", "gap")
        cache_base = self.hparams.get("cache_dir", "./cache")
        
        self.train_dataset = StandardDataset(self.hparams.get('train_paths'), os.path.join(cache_base, "train"), label_name)
        self.val_dataset = StandardDataset(self.hparams.get('val_paths'), os.path.join(cache_base, "val"), label_name)
        self.test_dataset = StandardDataset(self.hparams.get('test_paths'), os.path.join(cache_base, "test"), label_name)
        
        print(f"Datasets => Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        if self.hparams.get("standardize", True): self._standardize()
    
    def train_dataloader(self): return self._get_dataloader(self.train_dataset, "train")
    def val_dataloader(self): return self._get_dataloader(self.val_dataset, "val")
    def test_dataloader(self): return self._get_dataloader(self.test_dataset, "test")

    @property
    def mean(self): return self._mean
    @property
    def std(self): return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage in self._saved_dataloaders and store_dataloader and not self.hparams.get("reload"): return self._saved_dataloaders[stage]
        batch_size = self.hparams["batch_size"] if stage == "train" else self.hparams["inference_batch_size"]
        shuffle = stage == "train"
        dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.hparams["num_workers"], pin_memory=True)
        if store_dataloader: self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        print("计算 Z-Score 参数 (eV/Hartree 量纲无关化)...")
        if len(self.train_dataset) == 0: return 
        loader = DataLoader(self.train_dataset, batch_size=256, shuffle=False, num_workers=0)
        all_y = torch.cat([b.y for b in loader if b.y is not None], dim=0)
        self._mean = all_y.mean(dim=0)
        self._std = all_y.std(dim=0) + 1e-6
        print(f"均值(Mean): {self._mean.item():.4f}, 标准差(Std): {self._std.item():.4f}")