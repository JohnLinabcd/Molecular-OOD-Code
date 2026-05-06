import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from etnn.layers import ETNNLayer
from etnn import utils, invariants

class ETNN(nn.Module):
    """
    The E(n)-Equivariant Topological Neural Network (ETNN) model.
    """
    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        visible_dims: list[int] | None,
        normalize_invariants: bool,
        hausdorff_dists: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.0,
        lean: bool = True,
        global_pool: bool = False,  # whether or not to use global pooling
        sparse_invariant_computation: bool = False,
        sparse_agg_max_cells: int = 100,  # maximum size to consider for diameter and hausdorff dists
        pos_update: bool = False,  # performs the equivariant position update, optional
    ) -> None:
        super().__init__()

        self.initial_features = initial_features

        # make inv_fts_map for backward compatibility
        self.num_invariants = 5 if hausdorff_dists else 3
        self.num_inv_fts_map = {k: self.num_invariants for k in adjacencies}
        self.adjacencies = adjacencies
        self.normalize_invariants = normalize_invariants
        self.batch_norm = batch_norm
        self.lean = lean
        max_dim = max(num_features_per_rank.keys())
        self.global_pool = global_pool
        self.visible_dims = visible_dims
        self.pos_update = pos_update
        self.dropout = dropout

        # params for invariant computation
        self.sparse_invariant_computation = sparse_invariant_computation
        self.sparse_agg_max_cells = sparse_agg_max_cells
        self.hausdorff = hausdorff_dists
        self.cell_list_fmt = "list" if sparse_invariant_computation else "padded"

        if sparse_invariant_computation:
            self.inv_fun = invariants.compute_invariants_sparse
        else:
            self.inv_fun = invariants.compute_invariants

        # keep only adjacencies that are compatible with visible_dims
        if visible_dims is not None:
            self.adjacencies = []
            for adj in adjacencies:
                max_rank = max(int(rank) for rank in adj.split("_")[:2])
                if max_rank in visible_dims:
                    self.adjacencies.append(adj)
        else:
            self.visible_dims = list(range(max_dim + 1))
            self.adjacencies = adjacencies

        # layers
        if self.normalize_invariants:
            self.inv_normalizer = nn.ModuleDict(
                {
                    adj: nn.BatchNorm1d(self.num_inv_fts_map[adj], affine=False)
                    for adj in self.adjacencies
                }
            )

        embedders = {}
        for dim in self.visible_dims:
            embedder_layers = [nn.Linear(num_features_per_rank[dim], num_hidden)]
            if self.batch_norm:
                embedder_layers.append(nn.BatchNorm1d(num_hidden))
            embedders[str(dim)] = nn.Sequential(*embedder_layers)
        self.feature_embedding = nn.ModuleDict(embedders)

        self.layers = nn.ModuleList(
            [
                ETNNLayer(
                    self.adjacencies,
                    self.visible_dims,
                    num_hidden,
                    self.num_inv_fts_map,
                    self.batch_norm,
                    self.lean,
                    self.pos_update,
                )
                for _ in range(num_layers)
            ]
        )

        self.pre_pool = nn.ModuleDict()

        for dim in visible_dims:
            if self.global_pool:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_hidden),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_hidden)
            else:
                if not self.lean:
                    self.pre_pool[str(dim)] = nn.Sequential(
                        nn.Linear(num_hidden, num_hidden),
                        nn.SiLU(),
                        nn.Linear(num_hidden, num_out),
                    )
                else:
                    self.pre_pool[str(dim)] = nn.Linear(num_hidden, num_out)

        if self.global_pool:
            self.post_pool = nn.Sequential(
                nn.Linear(len(self.visible_dims) * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_out),
            )

    def forward(self, graph: Data) -> tuple[Tensor, Tensor, Tensor]:
        device = graph.pos.device

        cell_ind = {
            str(i): graph.cell_list(i, format=self.cell_list_fmt).to(device)
            for i in self.visible_dims
        }

        adj = {
            adj_type: getattr(graph, f"adj_{adj_type}")
            for adj_type in self.adjacencies
            if hasattr(graph, f"adj_{adj_type}")
        }

        # compute initial features
        features = {}
        for feature_type in self.initial_features:
            features[feature_type] = {}
            for i in self.visible_dims:
                if feature_type == "node":
                    features[feature_type][str(i)] = invariants.compute_centroids(
                        cell_ind[str(i)], graph.x
                    )
                elif feature_type == "mem":
                    mem = {j: getattr(graph, f"mem_{j}") for j in self.visible_dims}
                    features[feature_type][str(i)] = mem[i].float()
                elif feature_type == "hetero":
                    features[feature_type][str(i)] = getattr(graph, f"x_{i}")

        x = {
            str(i): torch.cat(
                [
                    features[feature_type][str(i)]
                    for feature_type in self.initial_features
                ],
                dim=1,
            )
            for i in self.visible_dims
        }

        pos = graph.pos
        x = {dim: self.feature_embedding[dim](feature) for dim, feature in x.items()}
        inv = self.inv_fun(pos, cell_ind=cell_ind, adj=adj, hausdorff=True)

        if self.normalize_invariants:
            inv = {
                adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()
            }

        # message passing
        for layer in self.layers:
            x, pos = layer(x, adj, inv, pos)
            if self.dropout > 0:
                x = {
                    dim: nn.functional.dropout(feature, p=self.dropout, training=self.training)
                    for dim, feature in x.items()
                }

        # read out
        out_node_level = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}

        # ✨ 子结构特征提取 (用于域适应)
        if hasattr(graph, 'sub_batch'):
            x_sub_feat = scatter(out_node_level['0'], graph.sub_batch.to(device), dim=0, reduce='mean')
        else:
            x_sub_feat = torch.zeros((1, out_node_level['0'].shape[-1]), device=device)

        if self.global_pool:
            batch_size = getattr(graph, 'num_graphs', 1)
            cell_batch = {
                str(i): utils.slices_to_pointer(graph._slice_dict[f"slices_{i}"]).to(device)
                for i in self.visible_dims
            }
            pooled = {
                dim: global_add_pool(out_node_level[dim], cell_batch[dim], size=batch_size)
                for dim, feature in out_node_level.items()
            }
            state = torch.cat(tuple([feature for dim, feature in pooled.items()]), dim=1)
            
            x_mol_feat = state # ✨ 全局表征
            
            out = self.post_pool(state)
            out = torch.squeeze(out, -1)
        else:
            out = out_node_level['0']
            x_mol_feat = out_node_level['0']

        return out, x_mol_feat, x_sub_feat

    def __str__(self):
        return f"ETNN ({self.type})"