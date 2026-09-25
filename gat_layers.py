import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class HeterogeneousGATv2Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels

        self.gat_layers = nn.ModuleDict()
        self.edge_types = [
            ('source', 'excites', 'filter'),
            ('filter', 'filters', 'resonator'),
            ('source', 'pressures', 'resonator'),
            ('source', 'inertia', 'source'),
            ('filter', 'inertia', 'filter'),
            ('resonator', 'inertia', 'resonator')
        ]

        for edge_type in self.edge_types:
            key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            self.gat_layers[key] = GATv2Conv(
                in_channels,
                out_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=0.1,
                add_self_loops=True
            )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        out_dict = {}

        for edge_type in self.edge_types:
            key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"

            if edge_type not in edge_index_dict:
                continue

            gat_layer = self.gat_layers[key]
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]

            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]

            out = gat_layer((x_src, x_dst), edge_index)

            if dst_type not in out_dict:
                out_dict[dst_type] = out
            else:
                out_dict[dst_type] = out_dict[dst_type] + out

        for node_type in out_dict:
            out_dict[node_type] = F.relu(out_dict[node_type])

        return out_dict


class JumpingKnowledge(nn.Module):
    def __init__(self, mode: str = 'cat'):
        super().__init__()
        self.mode = mode

    def forward(self, xs: list) -> torch.Tensor:
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=0).max(dim=0)[0]
        else:
            raise ValueError(f"Unsupported JK mode: {self.mode}")