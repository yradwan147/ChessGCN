"""
ResGATv2 model for chess position evaluation (WDL classification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm


class ResGATv2Block(nn.Module):
    """Residual block with two GATv2Conv layers."""

    def __init__(self, channels, heads, edge_dim, dropout=0.1):
        super().__init__()
        self.conv1 = GATv2Conv(
            channels, channels // heads, heads=heads,
            edge_dim=edge_dim, concat=True, dropout=dropout,
        )
        self.bn1 = BatchNorm(channels)

        self.conv2 = GATv2Conv(
            channels, channels // heads, heads=heads,
            edge_dim=edge_dim, concat=True, dropout=dropout,
        )
        self.bn2 = BatchNorm(channels)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = x + residual  # skip connection
        x = F.relu(x)
        return x


class ChessGATv2(nn.Module):
    """
    Graph Attention Network for chess position evaluation.

    Architecture:
        Input projection (node_dim → hidden)
        → N ResGATv2Blocks (each with 2 GATv2Conv layers + residual)
        → Global mean pool
        → FC head → 3-class WDL output
    """

    def __init__(
        self,
        node_dim=21,
        edge_dim=6,
        hidden=128,
        heads=4,
        num_blocks=4,
        dropout=0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            ResGATv2Block(hidden, heads, edge_dim, dropout=0.1)
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # Win, Draw, Loss
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)
        logits = self.head(x)
        return logits
