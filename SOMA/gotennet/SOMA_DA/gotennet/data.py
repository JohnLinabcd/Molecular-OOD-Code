import os
import pickle
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'sub_batch':
            return self.num_subs
        return super().__inc__(key, value, *args, **kwargs)

class MultiTaskDataset(Dataset):
    def __init__(self, csv_paths, cache_dir, label_names=['homo']):
        self.csv_paths = csv_paths
        self.label_names = label_names
        labels_str = "_".join(label_names)
        os.makedirs(cache_dir, exist_ok=True)
        
        if len(csv_paths) > 1:
            self.new_cache = os.path.join(cache_dir, f"data_brics_{labels_str}.pkl")
        else:
            base = os.path.basename(csv_paths[0]).replace(".csv", "")
            self.new_cache = os.path.join(cache_dir, f"cache_{base}_{labels_str}.pkl")
        
        if os.path.exists(self.new_cache):
            with open(self.new_cache, 'rb') as f:
                self.data_list = pickle.load(f)
        else:
            self.data_list = self._load_from_csv()
            with open(self.new_cache, 'wb') as f:
                pickle.dump(self.data_list, f)

    def _get_brics_substructures(self, mol):
        try:
            num_atoms = mol.GetNumAtoms()
            atom_to_sub_id = torch.zeros(num_atoms, dtype=torch.long)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            
            if not brics_bonds: 
                return atom_to_sub_id, 1
                
            bond_indices = [mol.GetBondBetweenAtoms(a1, a2).GetIdx() for (a1, a2), _ in brics_bonds if mol.GetBondBetweenAtoms(a1, a2)]
            if not bond_indices: 
                return atom_to_sub_id, 1
                
            mol_broken = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
            frags = Chem.GetMolFrags(mol_broken)
            
            for sub_id, atom_indices in enumerate(frags):
                for atom_idx in atom_indices:
                    if atom_idx < num_atoms: 
                        atom_to_sub_id[atom_idx] = sub_id
            return atom_to_sub_id, len(frags)
        except:
            return torch.zeros(mol.GetNumAtoms(), dtype=torch.long), 1

    def _load_from_csv(self):
        data_list = []
        all_dfs = [pd.read_csv(p) for p in self.csv_paths if os.path.exists(p)]
        if not all_dfs: return []
        
        df = pd.concat(all_dfs, ignore_index=True)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV->3D"):
            smiles = row.get('smiles', '')
            try:
                if "." in smiles: continue
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == -1: continue
                
                conf = mol.GetConformer()
                pos = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
                z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
                labels = [float(row[n]) for n in self.label_names]
                
                sub_batch, num_subs = self._get_brics_substructures(mol)
                
                data_list.append(MolData(
                    z=torch.tensor(z), pos=torch.tensor(pos, dtype=torch.float), 
                    num_nodes=len(z), y=torch.tensor([labels], dtype=torch.float), 
                    smiles=smiles, sub_batch=sub_batch, num_subs=num_subs
                ))
            except: 
                continue
        return data_list

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class DADataModule(LightningDataModule):
    def __init__(self, config_dict):
        super().__init__()
        self.config = config_dict
        self._mean, self._std = None, None

    def prepare_dataset(self):
        ln = self.config.get('label_names', ['homo'])
        c_dir = self.config.get('cache_dir', './cache_substructure')
        
        self.source_sup = MultiTaskDataset(self.config['sup_source_paths'], c_dir, ln)
        self.source_da = [MultiTaskDataset([p], os.path.join(c_dir, 'da_subset'), ln) for p in self.config.get('da_source_paths', [])]
        full_target = MultiTaskDataset(self.config['target_paths'], os.path.join(c_dir, 'target'), ln)
        
        n_tot = len(full_target)
        n_train = int(n_tot * 0.8)
        if n_train == n_tot and n_tot > 0: n_train = n_tot - 1
        
        self.target_train, self.target_test = random_split(
            full_target, [n_train, n_tot - n_train], 
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        self._compute_stats()

    def _compute_stats(self):
        if len(self.source_sup) > 0:
            loader = DataLoader(self.source_sup, batch_size=256)
            all_y = torch.cat([b.y for b in loader], dim=0)
            self._mean, self._std = all_y.mean(dim=0), all_y.std(dim=0) + 1e-6

    @property
    def mean(self): return self._mean if self._mean is not None else torch.tensor([0.0])
    @property
    def std(self): return self._std if self._std is not None else torch.tensor([1.0])

    def source_sup_loader(self): return DataLoader(self.source_sup, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def source_da_loaders(self): return [DataLoader(ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True) for ds in self.source_da if len(ds) > 0]
    def target_train_loader(self): return DataLoader(self.target_train, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
    def target_test_loader(self): return DataLoader(self.target_test, batch_size=self.config.get('inference_batch_size', 128), shuffle=False, drop_last=False)