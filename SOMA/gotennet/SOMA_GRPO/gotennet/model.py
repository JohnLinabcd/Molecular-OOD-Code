import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import radius_graph
from .gotennet_core import GotenNetCore

class GotenNetWrapper(nn.Module):
    def __init__(self, config, mean=0.0, std=1.0, num_tasks=1):
        super().__init__()
        self.cutoff = config.cutoff
        self.reduce_op = config.reduce_op
        
        self.goten = GotenNetCore(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            num_rbf=config.num_rbf,
            cutoff=config.cutoff
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.SiLU()
        )
        self.regressor = nn.Linear(config.hidden_channels, num_tasks)

        self.register_buffer('mean', torch.as_tensor(mean).view(1, -1))
        self.register_buffer('std', torch.as_tensor(std).view(1, -1))

    def forward(self, z, pos, batch, sub_batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_diff = torch.norm(edge_vec, dim=-1)

        s_node, _ = self.goten(z, pos, edge_index, edge_vec, edge_diff)
        x_atom_feat = self.feature_extractor(s_node)

        mol_feat = scatter(x_atom_feat, batch, dim=0, reduce=self.reduce_op)
        sub_feat = scatter(x_atom_feat, sub_batch, dim=0, reduce=self.reduce_op)

        pred = self.regressor(mol_feat) * self.std + self.mean
        
        # 完美输出：预测值, 分子特征, BRICS子图特征 (被 lookfor.py 里的 coral_loss 使用)
        return pred, mol_feat, sub_feat