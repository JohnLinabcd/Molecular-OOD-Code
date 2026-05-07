# gotennet/model.py
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph

# 相对导入你的原版 core
from .gotennet_core import GotenNetCore

class GotenNetWrapper(nn.Module):
    def __init__(self, config, mean: float = 0.0, std: float = 1.0, num_tasks: int = 1):
        super().__init__()
        self.cutoff = config.cutoff
        self.reduce_op = config.reduce_op

        # 1. 完美复刻原版名字与参数
        self.goten_core = GotenNetCore(
            hidden_channels=config.hidden_channels, # 256
            num_layers=config.num_layers,           # 4
            num_rbf=50,                             # 原版写死的 50
            cutoff=config.cutoff
        )

        # 2. 完美复刻原版的原子级降维层
        self.node_regressor = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 2)
        )
        
        # 3. 完美复刻原版的分子级预测头
        self.graph_regressor = nn.Sequential(
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 4),
            nn.SiLU(),
            nn.Linear(config.hidden_channels // 4, num_tasks)
        )

        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).view(1, -1))

    # ✨ 核心修改：在输入里加上 sub_batch
    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, sub_batch: torch.Tensor):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_diff = torch.norm(edge_vec, dim=-1)

        # 提取底层物理特征
        s_feat, v_feat = self.goten_core(
            z=z, pos=pos, edge_index=edge_index, edge_vec=edge_vec, edge_diff=edge_diff
        )
        
        # 过原子级降维层
        h_node = self.node_regressor(s_feat)
        
        # ✨ 为 SOMA 添加的聚合逻辑：分别聚合出分子级和子结构级特征
        h_graph = scatter(h_node, batch, dim=0, reduce=self.reduce_op)
        h_sub = scatter(h_node, sub_batch, dim=0, reduce=self.reduce_op)
        
        # 预测并反标准化
        pred = self.graph_regressor(h_graph)
        pred = pred * self.std + self.mean
        
        # 完美返回三元组，供 train.py 和 DA 损失使用
        return pred, h_graph, h_sub