# gotennet/model.py
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph

# 🌟 修改点：使用点号(.)相对导入同一子文件夹下的 core 模块
from .gotennet_core import GotenNetCore

class GotenNetWrapper(nn.Module):
    def __init__(self, config, mean: float = 0.0, std: float = 1.0, num_tasks: int = 1):
        super().__init__()
        self.cutoff = config.cutoff
        self.reduce_op = config.reduce_op

        self.goten_core = GotenNetCore(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            num_rbf=50,             
            cutoff=config.cutoff
        )

        self.node_regressor = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 2)
        )
        
        self.graph_regressor = nn.Sequential(
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 4),
            nn.SiLU(),
            nn.Linear(config.hidden_channels // 4, num_tasks)
        )

        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).view(1, -1))

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_diff = torch.norm(edge_vec, dim=-1)

        s_feat, v_feat = self.goten_core(
            z=z, pos=pos, edge_index=edge_index, edge_vec=edge_vec, edge_diff=edge_diff
        )
        
        h_node = self.node_regressor(s_feat)
        h_graph = scatter(h_node, batch, dim=0, reduce=self.reduce_op)
        
        pred = self.graph_regressor(h_graph)
        pred = pred * self.std + self.mean
        return pred