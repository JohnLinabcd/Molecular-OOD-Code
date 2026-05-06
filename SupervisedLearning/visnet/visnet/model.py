import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Parameter
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import scatter

# =============================================================================
# PART 1: 基础数学与几何组件 (Basic Math & Geometry)
# =============================================================================

class CosineCutoff(torch.nn.Module):
    r"""Applies a cosine cutoff to the input distances."""
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(torch.nn.Module):
    r"""Applies exponential normal smearing to the input distances."""
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', Parameter(means))
            self.register_parameter('betas', Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value))**-2] *
                             self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        smeared_dist = self.cutoff_fn(dist) * (-self.betas * (
            (self.alpha * (-dist)).exp() - self.means)**2).exp()
        return smeared_dist


class Sphere(torch.nn.Module):
    r"""Computes spherical harmonics of the input data."""
    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        self.lmax = lmax

    def forward(self, edge_vec: Tensor) -> Tensor:
        return self._spherical_harmonics(
            self.lmax,
            edge_vec[..., 0],
            edge_vec[..., 1],
            edge_vec[..., 2],
        )

    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([
                sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
            ], dim=-1)

        raise ValueError(f"'lmax' needs to be 1 or 2 (got {lmax})")


class VecLayerNorm(torch.nn.Module):
    r"""Applies layer normalization to the input vectors."""
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool,
        norm_type: Optional[str] = 'max_min',
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter('weight', Parameter(weight))
        else:
            self.register_buffer('weight', weight)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)
        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = dist.max(dim=-1)
        min_val, _ = dist.min(dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return dist.relu() * direct

    def forward(self, vec: Tensor) -> Tensor:
        if vec.size(1) == 3:
            if self.norm_type == 'max_min':
                vec = self.max_min_norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.size(1) == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            if self.norm_type == 'max_min':
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)

        raise ValueError(f"'{self.__class__.__name__}' only support 3 or 8 "
                         f"channels (got {vec.size(1)})")


class Distance(torch.nn.Module):
    r"""Computes the pairwise distances between atoms."""
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec

# =============================================================================
# PART 2: 嵌入与消息传递层 (Embeddings & Message Passing)
# =============================================================================

class NeighborEmbedding(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
    ) -> None:
        super().__init__(aggr='add')
        self.embedding = Embedding(max_z, hidden_channels)
        self.distance_proj = Linear(num_rbf, hidden_channels)
        self.combine = Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        torch.nn.init.xavier_uniform_(self.distance_proj.weight)
        torch.nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.zero_()
        self.combine.bias.data.zero_()

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, num_rbf: int, hidden_channels: int) -> None:
        super().__init__()
        self.edge_proj = Linear(num_rbf, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.zero_()

    def forward(
        self,
        edge_index: Tensor,
        edge_attr: Tensor,
        x: Tensor,
    ) -> Tensor:
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]
        return (x_i + x_j) * self.edge_proj(edge_attr)


class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(aggr='add', node_dim=0)

        if hidden_channels % num_heads != 0:
            raise ValueError("Hidden channels must be divisible by num_heads")

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = torch.nn.SiLU()
        self.attn_activation = torch.nn.SiLU()
        self.cutoff = CosineCutoff(cutoff)
        self.vec_proj = Linear(hidden_channels, hidden_channels * 3, False)

        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.dk_proj = Linear(hidden_channels, hidden_channels)
        self.dv_proj = Linear(hidden_channels, hidden_channels)
        self.s_proj = Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels)
            self.w_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.w_trg_proj = Linear(hidden_channels, hidden_channels, False)
        self.o_proj = Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.zero_()
        if not self.last_layer:
            torch.nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.w_src_proj.weight)
            torch.nn.init.xavier_uniform_(self.w_trg_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec_proj.weight)
        torch.nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        x, vec_out = self.propagate(edge_index, q=q, k=k, v=v, dk=dk, dv=dv,
                                    vec=vec, r_ij=r_ij, d_ij=d_ij)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, vec_j: Tensor,
                dk: Tensor, dv: Tensor, r_ij: Tensor,
                d_ij: Tensor) -> Tuple[Tensor, Tensor]:

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class ViS_MP_Vertex(ViS_MP):
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str],
        trainable_vecnorm: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__(num_heads, hidden_channels, cutoff, vecnorm_type,
                         trainable_vecnorm, last_layer)

        if not self.last_layer:
            self.f_proj = Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = Linear(hidden_channels, hidden_channels, False)
            self.t_trg_proj = Linear(hidden_channels, hidden_channels, False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if not self.last_layer:
            if hasattr(self, 't_src_proj'):
                torch.nn.init.xavier_uniform_(self.t_src_proj.weight)
            if hasattr(self, 't_trg_proj'):
                torch.nn.init.xavier_uniform_(self.t_trg_proj.weight)

    def edge_update(self, vec_i: Tensor, vec_j: Tensor, d_ij: Tensor,
                    f_ij: Tensor) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)

        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels,
                             dim=-1)
        return f1 * w_dot + f2 * t_dot


# =============================================================================
# PART 3: 核心 ViSNetBlock
# =============================================================================

class ViSNetBlock(torch.nn.Module):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.embedding = Embedding(max_z, hidden_channels)
        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.sphere = Sphere(lmax=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf,
                                                    trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf,
                                                    cutoff, max_z)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.vis_mp_layers = torch.nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
        )
        vis_mp_class = ViS_MP if not vertex else ViS_MP_Vertex
        for _ in range(num_layers - 1):
            layer = vis_mp_class(last_layer=False, **vis_mp_kwargs)
            self.vis_mp_layers.append(layer)
        self.vis_mp_layers.append(
            vis_mp_class(last_layer=True, **vis_mp_kwargs))

        self.out_norm = LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask],
                                                     dim=1).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1)**2) - 1, x.size(1),
                          dtype=x.dtype, device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight,
                                        edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight,
                                             edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)

        return x, vec


# =============================================================================
# PART 4: 特征提取与输出头 (Feature Heads & Output)
# =============================================================================

class GatedEquivariantBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = Linear(hidden_channels, out_channels, bias=False)

        self.update_net = torch.nn.Sequential(
            Linear(hidden_channels * 2, intermediate_channels),
            torch.nn.SiLU(),
            Linear(intermediate_channels, out_channels * 2),
        )

        self.act = torch.nn.SiLU() if scalar_activation else None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.zero_()

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantFeatureHead(torch.nn.Module):
    r"""Extracts high-dimensional atomic features using gated equivariant blocks."""
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.output_network = torch.nn.ModuleList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels,
                scalar_activation=True,
            ),
        ])

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        for layer in self.output_network:
            x, v = layer(x, v)
        return x


# =============================================================================
# PART 5: 主模型 ViSNet (Upgraded Architecture)
# =============================================================================

class ViSNet(torch.nn.Module):
    r"""
    The upgraded ViSNet model designed for molecular property prediction.
    Features:
    - Uses Feature Aggregation + MLP architecture (replacing atomic energy summation).
    - Removes Atomref and force derivative calculations (not needed for properties like HOMO/LUMO).
    - Supports multi-task prediction.
    """
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        reduce_op: str = "sum",
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        num_tasks: int = 1,
        derivative: bool = False, # Kept for API compatibility, but ignored internally
    ) -> None:
        super().__init__()
        
        self.reduce_op = reduce_op
        self.num_tasks = num_tasks

        # 1. Representation Backbone
        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )

        # 2. Feature Extractor Head
        self.feature_extractor = EquivariantFeatureHead(hidden_channels=hidden_channels)

        # 3. Global Regressor (MLP)
        self.global_regressor = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            Linear(hidden_channels, num_tasks)
        )

        # 4. Standardization Buffers
        if mean is None:
            mean = torch.zeros(1, num_tasks)
        if std is None:
            std = torch.ones(1, num_tasks)
        
        # Ensure correct shape [1, num_tasks] for broadcasting
        self.register_buffer('mean', torch.as_tensor(mean).view(1, -1))
        self.register_buffer('std', torch.as_tensor(std).view(1, -1))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.feature_extractor.reset_parameters()
        for layer in self.global_regressor:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r"""
        Computes the property prediction for a batch of molecules.
        """
        # 1. Extract atomic features
        x_atom, v_atom = self.representation_model(z, pos, batch)

        # 2. Process features (fuse vector info into scalar)
        x_atom_feat = self.feature_extractor(x_atom, v_atom)

        # 3. Aggregate to molecular level
        x_mol_feat = scatter(x_atom_feat, batch, dim=0, reduce=self.reduce_op)

        # 4. Regression MLP
        out = self.global_regressor(x_mol_feat)

        # 5. Denormalize
        out = out * self.std + self.mean

        return out