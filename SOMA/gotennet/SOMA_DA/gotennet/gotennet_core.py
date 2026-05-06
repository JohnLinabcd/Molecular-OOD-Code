# gotennet/gotennet_core.py
import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff
    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GeometricInteraction(MessagePassing):
    def __init__(self, hidden_channels, num_rbf):
        super().__init__(aggr='add', node_dim=0)
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
    def forward(self, s, v, edge_index, edge_attr, edge_vec, edge_dist):
        W = self.mlp(edge_attr)
        return self.propagate(edge_index, s=s, v=v, W=W, edge_vec=edge_vec, edge_dist=edge_dist)
    def message(self, s_j, v_j, W, edge_vec, edge_dist):
        W_s, W_v1, W_v2 = torch.split(W, W.shape[-1] // 3, dim=-1)
        msg_s = s_j * W_s
        edge_vec_norm = edge_vec / (edge_dist.unsqueeze(-1) + 1e-6)
        msg_v = v_j * W_v1.unsqueeze(-1) + (s_j * W_v2).unsqueeze(-1) * edge_vec_norm.unsqueeze(1)
        return msg_s, msg_v
    def aggregate(self, inputs, index, dim_size=None):
        msg_s, msg_v = inputs
        out_s = scatter(msg_s, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        out_v = scatter(msg_v, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return out_s, out_v

class GeometricUpdate(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.scalar_proj = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, s, v):
        v_proj = self.vec_proj(v.transpose(1, 2)).transpose(1, 2)
        v_u, v_v = torch.split(v_proj, v_proj.shape[1] // 2, dim=1)
        v_norm = torch.norm(v_v, dim=-1)
        s_cat = torch.cat([s, v_norm], dim=-1)
        s_out = self.mlp(s_cat)
        a_vv, a_sv, a_ss = torch.split(s_out, s_out.shape[-1] // 3, dim=-1)
        dv = v_u * a_vv.unsqueeze(-1)
        ds = a_ss + torch.sum(v_u * v_v, dim=-1) * a_sv
        return s + self.scalar_proj(ds), v + dv

class GotenNetCore(nn.Module):
    def __init__(self, hidden_channels=256, num_layers=4, num_rbf=50, cutoff=5.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.embedding = nn.Embedding(100, hidden_channels) 
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_rbf)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.interactions = nn.ModuleList([GeometricInteraction(hidden_channels, num_rbf) for _ in range(num_layers)])
        self.updates = nn.ModuleList([GeometricUpdate(hidden_channels) for _ in range(num_layers)])
    def forward(self, z, pos, edge_index, edge_vec, edge_diff):
        s = self.embedding(z)
        v = torch.zeros((s.size(0), self.hidden_channels, 3), device=s.device)
        edge_attr = self.distance_expansion(edge_diff) * self.cutoff_fn(edge_diff).unsqueeze(-1)
        for interaction, update in zip(self.interactions, self.updates):
            ds, dv = interaction(s, v, edge_index, edge_attr, edge_vec, edge_diff)
            s = s + ds
            v = v + dv
            s, v = update(s, v)
        return s, v