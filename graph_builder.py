"""
Graph Builder Module for Battery SoH Estimation
Builds feature interaction graphs for the GNN branch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class FeatureGraphBuilder:
    """
    Builds feature interaction graphs for battery cycles
    Nodes represent features, edges represent feature relationships
    """

    NODE_FEATURES = [
        "cycle_age",        # Normalized cycle index
        "temperature",      # Battery temperature
        "ic_peak_height",   # IC peak height
        "ic_peak_location", # IC peak voltage location
        "ic_peak_width",    # IC peak width
        "ic_area",          # IC area under curve
        "ic_num_peaks",     # Number of IC peaks
        "dod",              # Depth of discharge
    ]

    def __init__(self, num_nodes: int = 8):
        self.num_nodes = num_nodes
        self.feature_to_idx = {feat: i for i, feat in enumerate(self.NODE_FEATURES)}
        self.edge_index = self._build_fully_connected_edges()

    def _build_fully_connected_edges(self) -> torch.Tensor:
        """Build fully connected edge indices (vectorized)"""
        src = torch.arange(self.num_nodes).repeat_interleave(self.num_nodes - 1)
        dst = torch.cat(
            [
                torch.cat(
                    [torch.arange(i), torch.arange(i + 1, self.num_nodes)]
                )
                for i in range(self.num_nodes)
            ]
        )
        return torch.stack([src, dst], dim=0)

    def _extract_node_features(self, features: Dict) -> List[float]:
        """Extract features for each node"""
        node_values = []

        for feat_name in self.NODE_FEATURES:
            if feat_name == "cycle_age":
                cycle_idx = features.get("cycle_index", 0)
                max_cycles = features.get("max_cycles", 1000)
                val = cycle_idx / max_cycles if max_cycles > 0 else 0
            elif feat_name == "temperature":
                temp = features.get("temperature", 25)
                val = (temp - 4) / 20 if temp else 0.5
            elif feat_name == "ic_peak_height":
                val = features.get("ic_peak_height", 0)
            elif feat_name == "ic_peak_location":
                val = features.get("ic_peak_location", 3.7) / 4.2
            elif feat_name == "ic_peak_width":
                val = features.get("ic_peak_width", 0)
            elif feat_name == "ic_area":
                val = features.get("ic_area", 0)
            elif feat_name == "ic_num_peaks":
                val = features.get("ic_num_peaks", 0) / 5.0
            elif feat_name == "dod":
                val = features.get("dod", 80) / 100.0
            else:
                val = 0.0

            val = 0.0 if val is None or np.isnan(val) else float(val)
            node_values.append(val)

        return node_values

    def create_graph_from_features(self, features: Dict) -> Dict:
        """
        Create graph data from feature dictionary

        Args:
            features: Dictionary of features

        Returns:
            Dictionary with 'x' and 'edge_index'
        """
        node_features = self._extract_node_features(features)

        return {
            "x": torch.tensor([node_features], dtype=torch.float32).unsqueeze(-1),
            "edge_index": self.edge_index,
            "num_nodes": self.num_nodes,
        }

    def create_batch_graphs(self, features_list: List[Dict]) -> Dict:
        """
        Create a batch of graphs from multiple cycles

        Args:
            features_list: List of feature dictionaries

        Returns:
            Batched graph data dictionary
        """
        batch_x = []

        for features in features_list:
            node_features = self._extract_node_features(features)
            batch_x.append(node_features)

        # Shape: [batch_size, num_nodes, 1]
        x = torch.tensor(batch_x, dtype=torch.float32).unsqueeze(-1)

        return {
            "x": x,
            "edge_index": self.edge_index,
            "num_nodes": self.num_nodes,
        }
