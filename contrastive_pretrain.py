"""
Contrastive Pretraining Module for Battery SoH Estimation
Uses InfoNCE loss to pretrain encoders on cycle embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class InfoNCE(nn.Module):
    """
    InfoNCE Contrastive Loss

    L = -log(exp(sim(z_i, z_j)/tau) / sum_k(exp(sim(z_i, z_k)/tau)))

    Positive pairs: Adjacent cycles from same battery
    Negative pairs: Different batteries or different temperatures
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss

        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] - battery IDs or cycle group IDs

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        normalized_emb = F.normalize(embeddings, p=2, dim=1)

        similarity_matrix = (
            torch.matmul(normalized_emb, normalized_emb.t()) / self.temperature
        )

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        mask = mask - torch.eye(batch_size, device=mask.device)

        exp_sim = torch.exp(similarity_matrix)

        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        mask_positive = mask.sum(dim=1)
        mask_positive = torch.where(
            mask_positive > 0, mask_positive, torch.ones_like(mask_positive)
        )

        loss = -(mask * log_prob).sum(dim=1) / mask_positive

        return loss.mean()


class ContrastivePretrainer:
    """
    Contrastive pretraining for battery cycle embeddings

    Positive pairs: Adjacent cycles from same battery
    Negative pairs: Different batteries or far apart cycles
    """

    def __init__(
        self,
        encoder: nn.Module,
        temperature: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.encoder = encoder
        self.temperature = temperature
        self.device = device

        self.loss_fn = InfoNCE(temperature=temperature)

        self.encoder.to(device)

    def create_contrastive_labels(
        self, battery_ids: List[str], cycle_indices: List[int]
    ) -> torch.Tensor:
        """
        Create labels for contrastive learning

        Positive pairs: Same battery, adjacent cycles
        Negative pairs: Different batteries

        Args:
            battery_ids: List of battery IDs
            cycle_indices: List of cycle indices

        Returns:
            Tensor of labels
        """
        battery_to_idx = {}
        label_counter = 0
        labels = []

        for battery_id, cycle_idx in zip(battery_ids, cycle_indices):
            if battery_id not in battery_to_idx:
                battery_to_idx[battery_id] = label_counter
                label_counter += 1

            base_label = battery_to_idx[battery_id]
            labels.append(base_label)

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def pretrain(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Pretrain encoder using contrastive learning

        Args:
            dataloader: Training data loader
            num_epochs: Number of pretraining epochs
            lr: Learning rate
            weight_decay: Weight decay
            log_interval: Logging interval

        Returns:
            Dictionary of training metrics
        """
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        metrics = {"loss": [], "learning_rate": []}

        self.encoder.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(
                dataloader, desc=f"Pretrain Epoch {epoch + 1}/{num_epochs}"
            ):
                battery_ids = batch["battery_ids"]
                cycle_indices = batch["cycle_indices"]

                labels = self.create_contrastive_labels(
                    battery_ids,
                    cycle_indices.tolist()
                    if hasattr(cycle_indices, "tolist")
                    else list(cycle_indices),
                )

                voltage_curves = batch.get("voltage_curves")

                if voltage_curves is not None:
                    voltage_curves = voltage_curves.to(self.device)
                else:
                    continue

                optimizer.zero_grad()

                # Pass voltage curves through the CNN encoder
                embeddings = self.encoder(voltage_curves)

                unique_labels = torch.unique(labels)
                if len(unique_labels) < 2:
                    continue

                loss = self.loss_fn(embeddings, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(num_batches, 1)
            metrics["loss"].append(avg_loss)
            metrics["learning_rate"].append(scheduler.get_last_lr()[0])

            if (epoch + 1) % log_interval == 0:
                print(f"Pretrain Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        return metrics

    def get_embeddings(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract embeddings from pretrained encoder

        Args:
            dataloader: Data loader

        Returns:
            Embeddings and battery IDs
        """
        self.encoder.eval()

        all_embeddings = []
        all_battery_ids = []

        with torch.no_grad():
            for batch in dataloader:
                battery_ids = batch["battery_ids"]

                voltage_curves = batch.get("voltage_curves")

                if voltage_curves is not None:
                    voltage_curves = voltage_curves.to(self.device)
                else:
                    continue

                embeddings = self.encoder(voltage_curves)

                all_embeddings.append(embeddings.cpu())
                all_battery_ids.extend(battery_ids)

        embeddings = torch.cat(all_embeddings, dim=0)

        return embeddings, all_battery_ids


def create_positive_pairs(
    battery_ids: List[str], cycle_indices: List[int], window_size: int = 5
) -> List[Tuple[int, int]]:
    """
    Create positive pair indices

    Args:
        battery_ids: List of battery IDs
        cycle_indices: List of cycle indices
        window_size: Window for positive pairs

    Returns:
        List of (i, j) positive pair indices
    """
    positive_pairs = []

    battery_cycles = {}
    for idx, (battery_id, cycle_idx) in enumerate(zip(battery_ids, cycle_indices)):
        if battery_id not in battery_cycles:
            battery_cycles[battery_id] = []
        battery_cycles[battery_id].append((idx, cycle_idx))

    for battery_id, cycles in battery_cycles.items():
        cycles_sorted = sorted(cycles, key=lambda x: x[1])

        for i in range(len(cycles_sorted)):
            for j in range(i + 1, min(i + window_size + 1, len(cycles_sorted))):
                idx_i = cycles_sorted[i][0]
                idx_j = cycles_sorted[j][0]
                positive_pairs.append((idx_i, idx_j))

    return positive_pairs
