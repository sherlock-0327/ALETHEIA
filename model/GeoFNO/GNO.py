import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv

from .base_model import BaseModel
from .net_utils import MLP


class GNO(BaseModel):
    """Graph operator on values aligned with the query positions.

    The shared training interface is forward(values, pos, surf_pos=None). The
    optional third argument is accepted for compatibility but is not used.
    """

    def __init__(self, in_channel=3, width=32, mid_width=64,
                 out_channel=3, r=0.2, max_neighbors=8):
        super().__init__()
        self.input_proj = nn.Linear(in_channel + 3, width)
        self.convs = nn.ModuleList([
            NNConv(
                width, width,
                MLP([6, mid_width // 2, mid_width, width * width], nn.GELU),
                aggr="mean",
            )
            for _ in range(4)
        ])
        self.to_output = nn.Linear(width, out_channel)
        self.r = r
        self.max_neighbors = max_neighbors

    def get_graph(self, pos):
        """Build a bounded radius graph without allocating an N-by-N matrix."""
        batch_size, n_points, _ = pos.shape
        k = min(self.max_neighbors, n_points)
        sources = []
        targets = []
        for batch in range(batch_size):
            coords = pos[batch]
            for start in range(0, n_points, 256):
                stop = min(start + 256, n_points)
                distances = torch.cdist(coords[start:stop], coords)
                rows = torch.arange(stop - start, device=pos.device)
                distances[rows, start + rows] = 0  # always retain self edges
                nearest_dist, nearest_idx = distances.topk(k, dim=-1, largest=False)
                nearest_idx[:, 0] = start + rows
                nearest_dist[:, 0] = 0
                keep = nearest_dist <= self.r
                target_idx = torch.arange(start, stop, device=pos.device)
                target_idx = target_idx[:, None].expand_as(nearest_idx)
                sources.append(nearest_idx[keep] + batch * n_points)
                targets.append(target_idx[keep] + batch * n_points)

        source = torch.cat(sources)
        target = torch.cat(targets)
        edge_index = torch.stack((source, target))
        flat_pos = pos.reshape(batch_size * n_points, 3)
        edge_attr = torch.cat((flat_pos[source], flat_pos[target]), dim=-1)
        return edge_index, edge_attr

    def forward(self, u_in, pos, surf_pos=None):
        if u_in.ndim != 3 or pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError("GNO expects values [B,N,C] and positions [B,N,3]")
        if u_in.shape[:2] != pos.shape[:2]:
            raise ValueError("GNO requires input values aligned with target positions")

        edge_index, edge_attr = self.get_graph(pos)
        batch_size, n_points, _ = u_in.shape
        features = torch.cat((u_in, pos), dim=-1)
        features = self.input_proj(features).reshape(batch_size * n_points, -1)
        for conv in self.convs:
            features = F.gelu(conv(features, edge_index, edge_attr))
        return self.to_output(features).reshape(batch_size, n_points, -1)
