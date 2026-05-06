import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
from tqdm import tqdm
import joblib
from functools import lru_cache
import hashlib
import time
import warnings

warnings.filterwarnings('ignore')

class QM9BalancedSplitterOptimized:
    def __init__(self, data_path='qm9.csv', output_dir='data', cache_dir='.cache'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.df = None
        self.HARTREE_TO_MEV = 27211.386245988
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化缓存字典
        self.scaffold_cache = {}
        self.features_cache = {}
        self.stratum_cache = {}
    
    def get_cache_key(self, smiles):
        """生成缓存的key"""
        return hashlib.md5(smiles.encode()).hexdigest()
    
    @lru_cache(maxsize=10000)
    def get_mol_from_smiles(self, smiles):
        """带缓存的分子解析"""
        return Chem.MolFromSmiles(smiles)
    
    def load_data(self):
        """加载数据"""
        print("加载QM9数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"总分子数: {len(self.df)}")
        return self.df
    
    def extract_features_with_cache(self):
        """带缓存的特征提取"""
        print("提取分子特征（使用缓存）...")
        
        # 如果缓存存在，直接加载
        cache_file = os.path.join(self.cache_dir, 'features_cache.pkl')
        if os.path.exists(cache_file):
            print("从缓存加载特征...")
            with open(cache_file, 'rb') as f:
                cache_data = joblib.load(f)
                return cache_data
        
        # 否则重新计算
        results = []
        
        # 预计算所有需要的中间结果
        smiles_list = self.df['smiles'].tolist()
        
        for smiles in tqdm(smiles_list, desc="提取特征"):
            mol = self.get_mol_from_smiles(smiles)
            if mol:
                # 获取骨架
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                
                # 分类
                stratum = self.pre_classify_by_rings(mol)
                
                results.append({
                    'smiles': smiles,
                    'scaffold_smiles': scaffold_smiles,
                    'stratum': stratum,
                })
        
        # 保存到缓存
        cache_data = pd.DataFrame(results)
        with open(cache_file, 'wb') as f:
            joblib.dump(cache_data, f, compress=3)
        
        return cache_data
    
    def pre_classify_by_rings(self, mol):
        """预分类"""
        if mol is None:
            return 'other'
        
        # 检查缓存
        mol_str = mol.ToBinary()
        if mol_str in self.stratum_cache:
            return self.stratum_cache[mol_str]
        
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        if num_rings == 0:
            result = 'acyclic'
        elif num_rings == 1:
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() > 0:
                ring_size = len(ring_info.AtomRings()[0])
                result = 'small_monocyclic' if ring_size <= 6 else 'large_monocyclic'
            else:
                result = 'other'
        elif num_rings == 2:
            result = 'bicyclic'
        else:
            result = 'polycyclic'
        
        # 存入缓存
        self.stratum_cache[mol_str] = result
        return result
    
    def calculate_scaffold_features(self, scaffold_smiles):
        """计算骨架特征（带缓存）"""
        # 检查缓存
        if scaffold_smiles in self.features_cache:
            return self.features_cache[scaffold_smiles]
        
        # 重新计算
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is None:
            return None
        
        features = []
        
        # 基础特征
        features.append(float(mol.GetNumAtoms()))
        features.append(float(mol.GetNumHeavyAtoms()))
        features.append(float(rdMolDescriptors.CalcNumRings(mol)))
        
        # 元素组成
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for element in ['C', 'O', 'N', 'F']:
            features.append(float(atom_symbols.count(element)))
        
        # 键类型
        bond_types = [bond.GetBondType().name for bond in mol.GetBonds()]
        for bond_type in ['SINGLE', 'DOUBLE', 'TRIPLE']:
            features.append(float(bond_types.count(bond_type)))
        
        # 拓扑特征
        features.append(float(rdMolDescriptors.CalcNumRotatableBonds(mol)))
        features.append(float(rdMolDescriptors.CalcNumHeteroatoms(mol)))
        
        # 存入缓存
        self.features_cache[scaffold_smiles] = features
        return features
    
    def prepare_stratified_data(self, feature_df):
        """准备分层数据"""
        print("准备分层数据...")
        
        strata_data = {
            'acyclic': {'indices': [], 'scaffolds': set(), 'scaffold_features': {}},
            'small_monocyclic': {'indices': [], 'scaffolds': set(), 'scaffold_features': {}},
            'large_monocyclic': {'indices': [], 'scaffolds': set(), 'scaffold_features': {}},
            'bicyclic': {'indices': [], 'scaffolds': set(), 'scaffold_features': {}},
            'polycyclic': {'indices': [], 'scaffolds': set(), 'scaffold_features': {}}
        }
        
        scaffold_to_stratum = {}
        
        # 收集所有骨架和索引
        for idx, row in tqdm(feature_df.iterrows(), total=len(feature_df), desc="收集骨架"):
            stratum = row['stratum']
            scaffold = row['scaffold_smiles']
            
            strata_data[stratum]['indices'].append(idx)
            strata_data[stratum]['scaffolds'].add(scaffold)
            scaffold_to_stratum[scaffold] = stratum
        
        # 计算骨架特征（每个骨架只算一次）
        print("计算骨架特征（每个骨架只算一次）...")
        for stratum, data in strata_data.items():
            scaffolds = list(data['scaffolds'])
            for scaffold in tqdm(scaffolds, desc=f"{stratum}特征", leave=False):
                features = self.calculate_scaffold_features(scaffold)
                if features:
                    data['scaffold_features'][scaffold] = features
        
        # 转换格式以便后续使用
        for stratum, data in strata_data.items():
            data['features'] = list(data['scaffold_features'].values())
        
        return strata_data, scaffold_to_stratum
    
    def balance_strata_clusters(self, strata_data, scaffold_to_stratum, total_clusters=8):
        """平衡分层聚类分配"""
        print("\n平衡分层聚类分配...")
        
        stratum_sizes = {}
        for stratum, data in strata_data.items():
            stratum_sizes[stratum] = len(data['indices'])
        
        total_molecules = sum(stratum_sizes.values())
        
        # 按比例分配聚类数
        clusters_per_stratum = {}
        for stratum, size in stratum_sizes.items():
            if size == 0:
                clusters_per_stratum[stratum] = 0
                continue
                
            proportion = size / total_molecules
            clusters = max(1, min(4, int(round(proportion * total_clusters * 1.5))))
            clusters_per_stratum[stratum] = clusters
        
        # 调整总数
        current_total = sum(clusters_per_stratum.values())
        adjustment = total_clusters - current_total
        
        if adjustment > 0:
            # 增加聚类数
            for _ in range(adjustment):
                # 找分子数最多但聚类数未达上限的层
                candidate = max(
                    [(stratum, size) for stratum, size in stratum_sizes.items() 
                     if clusters_per_stratum[stratum] < 4],
                    key=lambda x: x[1],
                    default=(None, None)
                )
                if candidate[0]:
                    clusters_per_stratum[candidate[0]] += 1
        
        print("各层分配的聚类数:")
        for stratum, clusters in clusters_per_stratum.items():
            if stratum_sizes.get(stratum, 0) > 0:
                print(f"  {stratum:20s}: {clusters:2d} 个聚类, {stratum_sizes[stratum]:6d} 个分子")
        
        return clusters_per_stratum
    
    def cluster_within_strata(self, strata_data, clusters_per_stratum):
        """在各层内部进行聚类"""
        print("\n在各层内部进行聚类...")
        
        all_scaffold_clusters = {}
        next_cluster_id = 0
        
        for stratum, stratum_data in strata_data.items():
            if not stratum_data['indices']:
                continue
            
            num_clusters = clusters_per_stratum.get(stratum, 0)
            if num_clusters == 0:
                continue
            
            features = stratum_data['features']
            scaffolds = list(stratum_data['scaffold_features'].keys())
            
            if len(features) <= num_clusters or len(features) <= 1:
                # 骨架太少，直接分配
                for i, scaffold in enumerate(scaffolds):
                    all_scaffold_clusters[scaffold] = next_cluster_id + (i % num_clusters)
                next_cluster_id += num_clusters
            else:
                # 进行K-Means聚类
                try:
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    kmeans = KMeans(
                        n_clusters=num_clusters,
                        random_state=42,
                        n_init='auto',
                        max_iter=300
                    )
                    
                    cluster_labels = kmeans.fit_predict(features_scaled)
                    
                    # 分配聚类ID
                    for scaffold, label in zip(scaffolds, cluster_labels):
                        all_scaffold_clusters[scaffold] = next_cluster_id + label
                    
                    next_cluster_id += num_clusters
                    
                except Exception as e:
                    print(f"聚类 {stratum} 时出错: {e}")
                    # 出错时均匀分配
                    for i, scaffold in enumerate(scaffolds):
                        all_scaffold_clusters[scaffold] = next_cluster_id + (i % num_clusters)
                    next_cluster_id += num_clusters
        
        return all_scaffold_clusters
    
    def create_balanced_datasets(self, all_scaffold_clusters, scaffold_to_stratum):
        """创建平衡的数据集"""
        print("\n创建平衡的数据集...")
        
        # 构建骨架到分子的映射
        scaffold_to_indices = defaultdict(list)
        smiles_list = self.df['smiles'].tolist()
        
        for idx, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="构建骨架映射"):
            mol = self.get_mol_from_smiles(smiles)
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
                scaffold_to_indices[scaffold_smiles].append(idx)
        
        # 分配分子到聚类
        cluster_to_indices = defaultdict(list)
        skipped = 0
        
        for scaffold_smiles, indices in tqdm(scaffold_to_indices.items(), 
                                            desc="分配分子到聚类"):
            cluster_id = all_scaffold_clusters.get(scaffold_smiles)
            if cluster_id is not None:
                cluster_to_indices[cluster_id].extend(indices)
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"警告: {skipped} 个骨架没有分配到聚类")
        
        # 生成数据集文件
        print("\n生成数据集文件...")
        cluster_stats = []
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        for cluster_id, indices in tqdm(cluster_to_indices.items(), desc="保存数据集"):
            if indices:
                cluster_df = self.df.iloc[indices].copy()
                cluster_df['lumo_meV'] = cluster_df['lumo'] * self.HARTREE_TO_MEV
                output_df = cluster_df[['smiles', 'lumo_meV']].copy()
                output_df.columns = ['smiles', 'lumo']
                
                output_path = os.path.join(self.output_dir, f'balanced_cluster_{cluster_id}.csv')
                output_df.to_csv(output_path, index=False)
                
                size = len(output_df)
                percentage = size / len(self.df) * 100
                cluster_stats.append((cluster_id, size, percentage))
        
        # 生成报告
        self.generate_balance_report(cluster_stats, all_scaffold_clusters, scaffold_to_stratum)
        
        return cluster_to_indices
    
    def generate_balance_report(self, cluster_stats, all_scaffold_clusters, scaffold_to_stratum):
        """生成平衡性报告"""
        report_path = os.path.join(self.output_dir, 'balance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("QM9平衡数据集划分报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"总分子数: {len(self.df)}\n")
            f.write(f"生成数据集数: {len(cluster_stats)}\n")
            f.write(f"独特骨架数: {len(all_scaffold_clusters)}\n")
            f.write("\n数据集大小分布:\n")
            
            for cluster_id, size, percentage in sorted(cluster_stats, key=lambda x: x[0]):
                f.write(f"  数据集 {cluster_id}: {size:6d} 分子 ({percentage:5.1f}%)\n")
            
            # 计算均衡度
            sizes = [size for _, size, _ in cluster_stats]
            if sizes:
                avg_size = np.mean(sizes)
                std_size = np.std(sizes)
                balance_ratio = std_size / avg_size if avg_size > 0 else 0
                
                f.write(f"\n均衡度统计:\n")
                f.write(f"  平均大小: {avg_size:.0f}\n")
                f.write(f"  标准差: {std_size:.0f}\n")
                f.write(f"  变异系数: {balance_ratio:.3f} (越接近0越均衡)\n")
            
            # 各层分布
            f.write("\n各化学层在数据集中的分布:\n")
            stratum_in_clusters = defaultdict(set)
            for scaffold, stratum in scaffold_to_stratum.items():
                cluster_id = all_scaffold_clusters.get(scaffold)
                if cluster_id is not None:
                    stratum_in_clusters[stratum].add(cluster_id)
            
            for stratum, clusters in stratum_in_clusters.items():
                f.write(f"  {stratum}: 分布在 {len(clusters)} 个数据集中\n")
    
    def run_balanced_split(self):
        """执行平衡划分"""
        print("=" * 60)
        print("QM9平衡分层聚类划分 (优化版)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 提取特征（使用缓存）
            feature_df = self.extract_features_with_cache()
            
            # 3. 准备分层数据
            strata_data, scaffold_to_stratum = self.prepare_stratified_data(feature_df)
            
            # 4. 平衡分配聚类数
            clusters_per_stratum = self.balance_strata_clusters(strata_data, scaffold_to_stratum, total_clusters=8)
            
            # 5. 各层内部聚类
            all_scaffold_clusters = self.cluster_within_strata(strata_data, clusters_per_stratum)
            
            # 6. 创建数据集
            cluster_to_indices = self.create_balanced_datasets(all_scaffold_clusters, scaffold_to_stratum)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print("\n" + "=" * 60)
            print(f"✅ 平衡划分完成！耗时: {elapsed:.1f} 秒")
            print(f"数据集保存在: {self.output_dir}")
            print("=" * 60)
            
            return cluster_to_indices
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return None

# 主程序
if __name__ == "__main__":
    # 使用优化版本
    splitter = QM9BalancedSplitterOptimized(
        data_path='qm9.csv',
        output_dir='data_optimized',
        cache_dir='.qm9_cache'
    )
    
    result = splitter.run_balanced_split()
    
    # 显示结果
    if result:
        import glob
        csv_files = glob.glob('data_optimized/balanced_cluster_*.csv')
        print("\n生成的数据集:")
        for file in sorted(csv_files):
            df = pd.read_csv(file)
            print(f"{os.path.basename(file)}: {len(df)} 分子")