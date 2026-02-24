"""
Physics-Informed Loss Module for Battery SoH Estimation

Total loss:
L = L_MSE + λ1 * L_mono + λ2 * L_res

Where:
1. L_MSE - Mean Squared Error
2. L_mono - Monotonicity Constraint (SoH should decrease with cycling)
3. L_res - Resistance Consistency (Higher Rct -> Lower SoH)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss for battery SoH estimation

    Combines MSE loss with physics-based constraints:
    - Monotonicity: SoH should decrease over cycles
    - Resistance consistency: Higher resistance -> Lower SoH
    """

    def __init__(
        self,
        lambda_mono: float = 0.1,
        lambda_res: float = 0.1,
        use_resistance: bool = True,
    ):
        super().__init__()

        self.lambda_mono = lambda_mono
        self.lambda_res = lambda_res
        self.use_resistance = use_resistance

    def mse_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss

        Args:
            predictions: Predicted SoH values
            targets: True SoH values

        Returns:
            MSE loss
        """
        return F.mse_loss(predictions, targets)

    def monotonicity_loss(
        self,
        predictions: torch.Tensor,
        cycle_indices: torch.Tensor,
        battery_ids: list = None,
    ) -> torch.Tensor:
        """
        Monotonicity constraint

        SoH should generally decrease over cycles (degradation)
        L_mono = sum_t max(0, SoH_hat_t - SoH_hat_{t-1})

        Args:
            predictions: Predicted SoH values
            cycle_indices: Cycle indices for ordering
            battery_ids: Battery IDs for grouping

        Returns:
            Monotonicity loss
        """
        if battery_ids is None:
            unique_batteries = [None]
        else:
            unique_batteries = list(set(battery_ids))

        total_loss = 0.0
        count = 0

        for battery_id in unique_batteries:
            if battery_id is not None:
                mask = [b == battery_id for b in battery_ids]
                indices = torch.tensor(
                    [i for i, m in enumerate(mask) if m], device=predictions.device
                )
            else:
                indices = torch.arange(len(predictions), device=predictions.device)

            if len(indices) < 2:
                continue

            sorted_indices = indices[torch.argsort(cycle_indices[indices])]

            preds_sorted = predictions[sorted_indices].squeeze()

            if preds_sorted.dim() > 1:
                preds_sorted = preds_sorted.squeeze(-1)

            if preds_sorted.size(0) < 2:
                continue

            diffs = preds_sorted[1:] - preds_sorted[:-1]

            # Allow small fluctuations (0.5% SoH) without penalizing
            _MONO_TOLERANCE = 0.005
            monotonicity_violations = F.relu(diffs - _MONO_TOLERANCE)

            total_loss += monotonicity_violations.sum()
            count += len(diffs)

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=predictions.device)

    def resistance_consistency_loss(
        self,
        predictions: torch.Tensor,
        resistance: Optional[torch.Tensor] = None,
        rct: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Resistance consistency constraint

        Higher resistance should correlate with lower SoH
        Higher Rct -> Lower SoH

        L_res = |∂SoH/∂Rct| (gradient penalty)

        Args:
            predictions: Predicted SoH values
            resistance: Internal resistance values
            rct: Charge transfer resistance values

        Returns:
            Resistance consistency loss
        """
        if resistance is None and rct is None:
            return torch.tensor(0.0, device=predictions.device)

        if not predictions.requires_grad:
            predictions.requires_grad_(True)

        total_loss = 0.0
        count = 0

        if resistance is not None and resistance.numel() > 1:
            if resistance.requires_grad:
                resistance = resistance.detach()

            res_norm = (resistance - resistance.mean()) / (resistance.std() + 1e-8)

            grad_pred_res = torch.autograd.grad(
                outputs=predictions.sum(),
                inputs=res_norm,
                create_graph=True,
                retain_graph=True,
            )[0]

            if grad_pred_res is not None:
                correlation = (res_norm * grad_pred_res).mean()

                total_loss += F.relu(-correlation)
                count += 1

        if rct is not None and rct.numel() > 1:
            if rct.requires_grad:
                rct = rct.detach()

            rct_norm = (rct - rct.mean()) / (rct.std() + 1e-8)

            grad_pred_rct = torch.autograd.grad(
                outputs=predictions.sum(),
                inputs=rct_norm,
                create_graph=True,
                retain_graph=True,
            )[0]

            if grad_pred_rct is not None:
                correlation = (rct_norm * grad_pred_rct).mean()

                total_loss += F.relu(-correlation)
                count += 1

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=predictions.device)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        cycle_indices: Optional[torch.Tensor] = None,
        battery_ids: Optional[list] = None,
        resistance: Optional[torch.Tensor] = None,
        rct: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total physics-informed loss

        Args:
            predictions: Predicted SoH values
            targets: True SoH values
            cycle_indices: Cycle indices for monotonicity
            battery_ids: Battery IDs for grouping
            resistance: Internal resistance
            rct: Charge transfer resistance

        Returns:
            Dictionary with total loss and individual components
        """
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.squeeze(-1) if targets.dim() > 1 else targets

        loss_mse = self.mse_loss(predictions, targets)

        loss_mono = torch.tensor(0.0, device=predictions.device)
        if cycle_indices is not None:
            loss_mono = self.monotonicity_loss(
                predictions, cycle_indices, battery_ids
            )

        loss_res = torch.tensor(0.0, device=predictions.device)
        if self.use_resistance and (resistance is not None or rct is not None):
            loss_res = self.resistance_consistency_loss(predictions, resistance, rct)

        total_loss = (
            loss_mse + self.lambda_mono * loss_mono + self.lambda_res * loss_res
        )

        return {
            "total_loss": total_loss,
            "loss_mse": loss_mse,
            "loss_mono": loss_mono,
            "loss_res": loss_res,
        }


class SoHConstraintLoss(nn.Module):
    """
    Additional constraints for SoH predictions:
    1. SoH should be in [0, 1.5]
    2. Initial SoH should be close to 1.0
    """

    def __init__(self, boundary_penalty: float = 1.0, initial_soh_weight: float = 0.5):
        super().__init__()

        self.boundary_penalty = boundary_penalty
        self.initial_soh_weight = initial_soh_weight

    def boundary_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Penalize SoH predictions outside [0, 1.5]

        Args:
            predictions: Predicted SoH values

        Returns:
            Boundary violation loss
        """
        below_zero = F.relu(-predictions)
        above_max = F.relu(predictions - 1.5)

        return below_zero.mean() + above_max.mean()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cycle_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SoH constraint loss

        Args:
            predictions: Predicted SoH values
            targets: True SoH values (for initial SoH)
            cycle_indices: Cycle indices

        Returns:
            Constraint loss components
        """
        boundary_loss = self.boundary_loss(predictions)

        initial_loss = torch.tensor(0.0, device=predictions.device)
        if targets is not None and cycle_indices is not None:
            min_cycle_idx = cycle_indices.min()
            initial_mask = cycle_indices == min_cycle_idx

            if initial_mask.any():
                initial_preds = predictions[initial_mask]
                initial_targets = targets[initial_mask]
                initial_loss = F.mse_loss(initial_preds, initial_targets)

        total_constraint = (
            self.boundary_penalty * boundary_loss
            + self.initial_soh_weight * initial_loss
        )

        return {
            "total_constraint": total_constraint,
            "boundary_loss": boundary_loss,
            "initial_loss": initial_loss,
        }
