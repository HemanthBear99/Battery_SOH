"""
Model Architecture Module for Battery SoH Estimation
Multi-Scale CNN - Transformer - GNN Hybrid Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    Adds positional information to sequence embeddings
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class MultiScaleCNN1D(nn.Module):
    """
    Multi-Scale 1D CNN Branch

    Processes voltage and IC curves with dilated convolutions
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 128,
        output_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(
            input_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2_dilated = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv2_dilated_2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4
        )
        self.bn2_2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = nn.Conv1d(
            hidden_channels, hidden_channels * 2, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2)

        # Residual skip from input → after conv3 (improves gradient flow)
        self.residual_proj = nn.Conv1d(input_channels, hidden_channels * 2, kernel_size=1)
        self.bn_res = nn.BatchNorm1d(hidden_channels * 2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale CNN

        Args:
            x: Input tensor [batch_size, seq_len] or [batch_size, 1, seq_len]

        Returns:
            CNN embeddings [batch_size, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Residual shortcut — projected before any transformation
        res = self.activation(self.bn_res(self.residual_proj(x)))

        x1 = self.activation(self.bn1(self.conv1(x)))

        x2 = self.activation(self.bn2(self.conv2_dilated(x1)))
        x2_2 = self.activation(self.bn2_2(self.conv2_dilated_2(x1)))
        x2 = x2 + x2_2

        # Add residual before final activation
        x3 = self.activation(self.bn3(self.conv3(x2)) + res)

        pooled = self.pool(x3).squeeze(-1)

        output = self.fc(pooled)

        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network Layer (Vectorized)

    h_i' = sigma(sum_j(alpha_ij * W * h_j))

    Fully vectorized — no Python loops in forward pass.
    """

    def __init__(
        self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()

        self.heads = heads
        self.out_channels = out_channels

        self.W = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Linear(out_channels * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.output_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT layer (vectorized)

        Args:
            x: Node features [batch_size, num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape

        h = self.W(x)
        h = h.view(batch_size, num_nodes, self.heads, self.out_channels)

        # Gather source and destination features — fully vectorized
        h_src = h[:, edge_index[0], :, :]  # [B, E, H, C]
        h_dst = h[:, edge_index[1], :, :]  # [B, E, H, C]

        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [B, E, H, 2C]

        e = self.att(h_concat)  # [B, E, H, 1]
        e = self.leaky_relu(e).squeeze(-1)  # [B, E, H]

        # Build attention matrix using scatter — vectorized
        # Compute e_avg first so we can match its dtype (AMP may cast to float16)
        e_avg = e.mean(dim=-1)  # [B, E]
        dst_idx = edge_index[1]  # destination nodes
        src_idx = edge_index[0]  # source nodes

        attention = torch.full(
            (batch_size, num_nodes, num_nodes),
            float("-inf"),
            device=x.device,
            dtype=e_avg.dtype,
        )
        attention[:, dst_idx, src_idx] = e_avg

        attention = F.softmax(attention, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)  # Guard against all-inf rows
        attention = self.dropout(attention)

        # Aggregate: [B, N, N] x [B, N, H*C] -> [B, N, H*C]
        h_flat = h.view(batch_size, num_nodes, self.heads * self.out_channels)
        h_out = torch.bmm(attention, h_flat)  # [B, N, H*C]

        # Average over heads
        h_out = h_out.view(batch_size, num_nodes, self.heads, self.out_channels)
        h_out = h_out.mean(dim=2)  # [B, N, C]

        # BatchNorm
        h_out = h_out.transpose(1, 2)
        h_out = self.output_bn(h_out)
        h_out = h_out.transpose(1, 2)

        h_out = F.elu(h_out)

        return h_out


class GNNBranch(nn.Module):
    """
    Graph Attention Network Branch

    2-layer GAT with multi-node graph input
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.gat1 = GraphAttentionLayer(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout
        )

        self.gat2 = GraphAttentionLayer(
            hidden_dim, output_dim, heads=1, dropout=dropout
        )

        # Cache fully-connected edge_index — computed once, reused every forward pass
        self._cached_edges: Optional[torch.Tensor] = None
        self._cached_num_nodes: Optional[int] = None

    def forward(
        self, node_features: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN

        Args:
            node_features: [batch_size, num_nodes, input_dim]
            edge_index: [2, num_edges]

        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        batch_size, num_nodes, _ = node_features.shape

        if edge_index is None:
            edge_index = self._get_default_edges(num_nodes, node_features.device)

        h = self.gat1(node_features, edge_index)

        h = self.gat2(h, edge_index)

        h = h.mean(dim=1)

        return h

    def _get_default_edges(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Return fully connected edge_index, computed once and cached."""
        if self._cached_edges is None or self._cached_num_nodes != num_nodes:
            src = torch.arange(num_nodes).repeat_interleave(num_nodes - 1)
            dst = torch.cat([
                torch.cat([torch.arange(i), torch.arange(i + 1, num_nodes)])
                for i in range(num_nodes)
            ])
            self._cached_edges = torch.stack([src, dst], dim=0)
            self._cached_num_nodes = num_nodes
        return self._cached_edges.to(device)


class TransformerBranch(nn.Module):
    """
    Transformer Encoder Branch

    Processes multi-token sequences (voltage curve patches)
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.positional_encoding = PositionalEncoding(input_dim, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(input_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(
        self, sequence_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer

        Args:
            sequence_embeddings: [batch_size, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            Transformer embeddings [batch_size, output_dim]
        """
        x = self.positional_encoding(sequence_embeddings)

        x = self.transformer(x, mask=mask)

        x = self.layer_norm(x)

        output = x.mean(dim=1)

        output = self.output_projection(output)

        return output


class FusionNetwork(nn.Module):
    """
    Fusion Network for combining CNN, GNN, and Transformer branches

    z = [z_cnn | z_gnn | z_transformer]

    FC(256) -> GELU -> Dropout
    FC(64) -> GELU
    FC(1) -> SoH prediction (no Sigmoid — allows predictions > 1.0)
    """

    def __init__(
        self,
        cnn_dim: int = 128,
        gnn_dim: int = 64,
        transformer_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fusion_dim = cnn_dim + gnn_dim + transformer_dim

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(
        self,
        cnn_embedding: torch.Tensor,
        gnn_embedding: torch.Tensor,
        transformer_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse embeddings and predict SoH

        Args:
            cnn_embedding: [batch_size, cnn_dim]
            gnn_embedding: [batch_size, gnn_dim]
            transformer_embedding: [batch_size, transformer_dim]

        Returns:
            SoH predictions [batch_size, 1]
        """
        fused = torch.cat([cnn_embedding, gnn_embedding, transformer_embedding], dim=1)

        output = self.fusion(fused)

        return output


class BatterySoHModel(nn.Module):
    """
    Complete Battery SoH Estimation Model

    Multi-Scale CNN – Transformer – GNN Hybrid Architecture

    Architecture:
    1. Multi-Scale CNN Branch (voltage + IC curves)
    2. GNN Branch (multi-node feature graph)
    3. Transformer Branch (patched voltage sequence)
    4. Fusion Network -> SoH prediction
    """

    PATCH_SIZE = 25  # Voltage curve is split into patches for the Transformer

    def __init__(
        self,
        voltage_seq_len: int = 500,
        cnn_hidden: int = 64,
        cnn_output: int = 128,
        gnn_input: int = 1,
        gnn_hidden: int = 32,
        gnn_output: int = 64,
        transformer_hidden: int = 128,
        transformer_output: int = 128,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.voltage_seq_len = voltage_seq_len
        self.patch_size = self.PATCH_SIZE
        self.num_patches = voltage_seq_len // self.patch_size

        # CNN branches for voltage and IC curves
        self.cnn_branch = MultiScaleCNN1D(
            input_channels=1,
            hidden_channels=cnn_hidden,
            output_dim=cnn_output,
            dropout=dropout,
        )

        self.ic_cnn_branch = MultiScaleCNN1D(
            input_channels=1,
            hidden_channels=cnn_hidden,
            output_dim=cnn_output,
            dropout=dropout,
        )

        # GNN branch: input_dim=1 (each node has 1 scalar feature)
        self.gnn_branch = GNNBranch(
            input_dim=gnn_input,
            hidden_dim=gnn_hidden,
            output_dim=gnn_output,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Transformer branch: processes patched voltage sequences
        self.patch_projection = nn.Linear(self.patch_size, transformer_hidden)
        self.transformer_branch = TransformerBranch(
            input_dim=transformer_hidden,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            output_dim=transformer_output,
            dropout=dropout,
            max_seq_len=self.num_patches + 1,  # +1 for optional CLS token
        )

        # Fusion head — no Sigmoid
        self.fusion = FusionNetwork(
            cnn_dim=cnn_output * 2,
            gnn_dim=gnn_output,
            transformer_dim=transformer_output,
            hidden_dim=256,
            output_dim=1,
            dropout=dropout,
        )

        self._init_weights()

        # Stored so checkpoints can recreate the exact model architecture
        self.model_config = {
            "voltage_seq_len": voltage_seq_len,
            "cnn_hidden": cnn_hidden,
            "cnn_output": cnn_output,
            "gnn_input": gnn_input,
            "gnn_hidden": gnn_hidden,
            "gnn_output": gnn_output,
            "transformer_hidden": transformer_hidden,
            "transformer_output": transformer_output,
            "num_transformer_layers": num_transformer_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
        }

    def _init_weights(self):
        """Initialize model weights (matched to GELU activations)"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split a 1D sequence into non-overlapping patches and project.

        Args:
            x: [batch_size, seq_len]

        Returns:
            [batch_size, num_patches, transformer_hidden]
        """
        batch_size, seq_len = x.shape
        # Truncate to fit exact patches
        usable_len = (seq_len // self.patch_size) * self.patch_size
        x = x[:, :usable_len]
        # Reshape into patches: [B, num_patches, patch_size]
        patches = x.view(batch_size, -1, self.patch_size)
        # Project each patch
        return self.patch_projection(patches)

    def forward(
        self,
        voltage_curves: Optional[torch.Tensor] = None,
        ic_curves: Optional[torch.Tensor] = None,
        cycle_indices: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict] = None,
        graph_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model

        Args:
            voltage_curves: [batch_size, seq_len]
            ic_curves: [batch_size, seq_len]
            cycle_indices: [batch_size] (unused directly but kept for API compatibility)
            graph_data: Optional dict with 'x' and 'edge_index'
            graph_features: Optional [batch_size, num_nodes, feature_dim] tensor
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with predictions and optionally embeddings
        """
        device = next(self.parameters()).device

        # --- CNN branch ---
        cnn_embedding = None
        ic_embedding = None

        if voltage_curves is not None:
            cnn_embedding = self.cnn_branch(voltage_curves)

        if ic_curves is not None:
            ic_embedding = self.ic_cnn_branch(ic_curves)

        if cnn_embedding is not None and ic_embedding is not None:
            cnn_combined = torch.cat([cnn_embedding, ic_embedding], dim=1)
        elif cnn_embedding is not None:
            cnn_combined = torch.cat([cnn_embedding, cnn_embedding], dim=1)
        elif ic_embedding is not None:
            cnn_combined = torch.cat([ic_embedding, ic_embedding], dim=1)
        else:
            cnn_combined = torch.zeros(
                1, self.cnn_branch.output_dim * 2, device=device
            )

        batch_size = cnn_combined.size(0)

        # --- GNN branch ---
        gnn_embedding = None
        if graph_features is not None:
            gnn_embedding = self.gnn_branch(graph_features)
        elif graph_data is not None:
            node_features = graph_data.get("x")
            edge_index = graph_data.get("edge_index")
            if node_features is not None:
                gnn_embedding = self.gnn_branch(node_features, edge_index)

        if gnn_embedding is None:
            gnn_embedding = torch.zeros(
                batch_size, self.gnn_branch.output_dim, device=device
            )

        # --- Transformer branch: process voltage patches ---
        transformer_embedding = None
        if voltage_curves is not None and voltage_curves.size(-1) >= self.patch_size:
            patch_embeddings = self._patchify(voltage_curves)  # [B, num_patches, D]
            transformer_embedding = self.transformer_branch(patch_embeddings)

        if transformer_embedding is None:
            transformer_embedding = torch.zeros(
                batch_size, self.transformer_branch.output_dim, device=device
            )

        # --- Fusion ---
        soh_prediction = self.fusion(cnn_combined, gnn_embedding, transformer_embedding)

        output = {"soh_prediction": soh_prediction}

        if return_embeddings:
            output["cnn_embedding"] = cnn_combined
            output["gnn_embedding"] = gnn_embedding
            output["transformer_embedding"] = transformer_embedding

        return output

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_model(config: Optional[Dict] = None) -> BatterySoHModel:
    """
    Create model from configuration

    Args:
        config: Optional model configuration

    Returns:
        BatterySoHModel instance
    """
    if config is None:
        config = {}

    return BatterySoHModel(
        voltage_seq_len=config.get("voltage_seq_len", 500),
        cnn_hidden=config.get("cnn_hidden", 64),
        cnn_output=config.get("cnn_output", 128),
        gnn_input=config.get("gnn_input", 1),
        gnn_hidden=config.get("gnn_hidden", 32),
        gnn_output=config.get("gnn_output", 64),
        transformer_hidden=config.get("transformer_hidden", 128),
        transformer_output=config.get("transformer_output", 128),
        num_transformer_layers=config.get("num_transformer_layers", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 500),
    )
