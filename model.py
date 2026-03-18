"""
ResGATv2 model for chess position evaluation (WDL + optional policy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, GlobalAttention, BatchNorm

from data import EDGE_DIM


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
        → Value head: global mean pool → FC → 3-class WDL
        → Policy head (optional): per-edge MLP(h_src, h_dst, edge_attr) → 1 logit per move
    """

    def __init__(
        self,
        node_dim=22,
        edge_dim=EDGE_DIM,
        hidden=128,
        heads=4,
        num_blocks=4,
        dropout=0.2,
        policy_head=True,
        attn_pool=False,
        moves_left_head=False,
        score_dist_bins=0,
    ):
        super().__init__()
        self.has_policy_head = policy_head
        self.has_moves_left_head = moves_left_head
        self.score_dist_bins = score_dist_bins

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            ResGATv2Block(hidden, heads, edge_dim, dropout=0.1)
            for _ in range(num_blocks)
        ])

        # Pooling: attention-based or mean
        if attn_pool:
            gate_nn = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            self.pool = GlobalAttention(gate_nn)
        else:
            self.pool = None  # use global_mean_pool

        # Value head: 3-class WDL or N-bin score distribution
        if score_dist_bins > 0:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, score_dist_bins),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3),  # Win, Draw, Loss
            )

        if policy_head:
            self.policy_mlp = nn.Sequential(
                nn.Linear(hidden * 2 + edge_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        if moves_left_head:
            self.moves_left_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        # Pooling
        if self.pool is not None:
            x_pool = self.pool(x, batch)
        else:
            x_pool = global_mean_pool(x, batch)

        # Value head
        value_logits = self.value_head(x_pool)

        # Policy head (only on legal-move edges, not self-edges)
        policy_logits = None
        if self.has_policy_head and edge_index.size(1) > 0:
            move_mask = edge_index[0] != edge_index[1]
            if move_mask.any():
                move_idx = edge_index[:, move_mask]
                move_attr = edge_attr[move_mask]
                src_emb = x[move_idx[0]]
                dst_emb = x[move_idx[1]]
                edge_input = torch.cat([src_emb, dst_emb, move_attr], dim=1)
                policy_logits = self.policy_mlp(edge_input).squeeze(-1)

        # Moves-left head
        moves_left = None
        if self.has_moves_left_head:
            moves_left = self.moves_left_head(x_pool)

        return value_logits, policy_logits, moves_left


def load_v1_checkpoint(model, checkpoint_path, device):
    """Load a value-only (v1) checkpoint into a dual-head model.

    Renames 'head.*' → 'value_head.*' and skips layers with shape
    mismatches so different architectures don't crash.
    """
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_state = model.state_dict()
    migrated = {}
    skipped = []
    for k, v in state_dict.items():
        new_key = k.replace("head.", "value_head.") if k.startswith("head.") else k
        if new_key in model_state and model_state[new_key].shape == v.shape:
            migrated[new_key] = v
        else:
            skipped.append(new_key)
    model.load_state_dict(migrated, strict=False)
    if skipped:
        print(f"  Skipped {len(skipped)} layers with shape mismatch (expected for different arch)")
