# data.py
from os.path import join
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem  # ##### 必须导入这个 #####
import os
import pickle
import numpy as np
from visnet.utils import MissingLabelException


class SimpleSMILESDataset(Dataset):
    """简单的SMILES数据集，支持缓存"""
    def __init__(self, csv_paths, cache_dir="./cache", label_name="gap"):
        self.csv_paths = csv_paths
        self.cache_dir = cache_dir
        self.label_name = label_name
        self.cache_file = os.path.join(cache_dir, f"data_cache_{label_name}.pkl")
        
        # 检查缓存
        if os.path.exists(self.cache_file):
            print(f"加载缓存数据: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data_list = pickle.load(f)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print("处理SMILES数据 (这可能需要几分钟)...")
            self.data_list = self._load_data()
            print(f"保存缓存: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data_list, f)
        
        print(f"数据集大小: {len(self.data_list)} 个分子")
    
    def _load_data(self):
        data_list = []
        all_dfs = []
        for path in self.csv_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_dfs.append(df)
            else:
                print(f"警告: 文件不存在 {path}")
        
        if not all_dfs:
            raise ValueError("没有找到数据文件")
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # 处理每个分子
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理分子"):
            try:
                data = self._create_molecule_data(row)
                data_list.append(data)
            except Exception as e:
                # ##### 修改: 打印跳过的分子信息，方便排查 #####
                # print(f"跳过分子 {row.get('smiles', 'N/A')}: {e}")
                continue
        
        return data_list
    
    def _create_molecule_data(self, row):
        smiles = row['smiles']
        
        # ##### 修改: 过滤非连通分子（例如 "Na+.Cl-"） #####
        if "." in smiles:
            raise ValueError(f"包含断开结构: {smiles}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        
        # ##### 修改: 严格的 3D 生成逻辑 #####
        # 使用 ETKDGv3，成功率更高
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True # 对于刚性分子很有用
        params.maxAttempts = 50
        
        res = AllChem.EmbedMolecule(mol, params)
        
        # ##### 关键修改: 如果生成失败，直接报错丢弃，不要用 2D 坐标！#####
        if res == -1:
            raise ValueError(f"无法生成 3D 构象: {smiles}")
            
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass # 优化失败通常可以接受，只要坐标不是重叠的
        
        conformer = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        
        atomic_numbers = []
        positions = []
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conformer.GetAtomPosition(i)
            atomic_numbers.append(atom.GetAtomicNum())
            positions.append([pos.x, pos.y, pos.z])
        
        # 转换为tensor
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        # 检查边 (虽然 ViSNet 主要基于距离，但保留结构)
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(z=z, pos=pos, edge_index=edge_index, num_nodes=num_atoms)
        
        # 标签处理
        # ##### 修改: 简单的数值检查 #####
        target_val = row[self.label_name]
        if pd.isna(target_val) or np.isinf(target_val):
            raise ValueError("标签值为 NaN 或 Inf")
        if abs(target_val) > 10000: # 假设阈值，防止极端异常值
             raise ValueError(f"标签值异常大: {target_val}")

        data.y = torch.tensor([[target_val]], dtype=torch.float32)
        data.smiles = smiles
        
        return data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        
        # 保持你的原始路径配置
        self.train_paths = hparams.get('train_paths')
        self.val_paths = hparams.get('val_paths')
        self.test_paths = hparams.get('test_paths')
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_dataset(self):
        self._prepare_custom_dataset()
        print(f"训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}, 测试集: {len(self.test_dataset)}")
        if self.hparams["standardize"]:
            self._standardize()
    
    def _prepare_custom_dataset(self):
        label_name = self.hparams.get("label_name", "gap")
        cache_base = self.hparams.get("cache_dir", "./cache")
        
        # 保持你要求的按文件划分逻辑
        self.train_dataset = SimpleSMILESDataset(
            self.train_paths, 
            cache_dir=os.path.join(cache_base, "train"),
            label_name=label_name
        )
        self.val_dataset = SimpleSMILESDataset(
            self.val_paths,
            cache_dir=os.path.join(cache_base, "val"),
            label_name=label_name
        )
        self.test_dataset = SimpleSMILESDataset(
            self.test_paths,
            cache_dir=os.path.join(cache_base, "test"),
            label_name=label_name
        )
    
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        def get_label(batch):
            if batch.y is None:
                raise MissingLabelException()
            return batch.y.clone()

        print("计算标准化参数（使用训练集）...")
        if len(self.train_dataset) == 0:
            return 

        # 简单采样计算，避免遍历整个大Dataset太慢
        sample_loader = DataLoader(self.train_dataset, batch_size=256, shuffle=False, num_workers=0)
        all_ys = []
        for batch in tqdm(sample_loader, desc="Statistics"):
            all_ys.append(batch.y)
        
        ys = torch.cat(all_ys)
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
        print(f"标准化参数 - 均值: {self._mean.item():.4f}, 标准差: {self._std.item():.4f}")
    
    
   