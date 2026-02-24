"""
Training Module for Battery SoH Estimation
Handles training loop, optimization, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.amp
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import os
from datetime import datetime


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """
        Check if should stop

        Args:
            metric: Current metric value

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Trainer class for Battery SoH Model
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device

        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        else:
            self.scheduler = scheduler

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "learning_rate": []}

        self.best_val_loss = float("inf")

        # AMP support
        self.use_amp = use_amp and device != "cpu" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_mono = 0.0
        total_res = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            battery_ids = batch["battery_ids"]
            cycle_indices = batch["cycle_indices"].to(self.device)
            target_soh = batch["target_soh"].to(self.device)

            voltage_curves = None
            if "voltage_curves" in batch and batch["voltage_curves"] is not None:
                voltage_curves = batch["voltage_curves"].to(self.device)

            ic_curves = None
            if "ic_curves" in batch and batch["ic_curves"] is not None:
                ic_curves = batch["ic_curves"].to(self.device)

            graph_features = None
            if "graph_features" in batch and batch["graph_features"] is not None:
                graph_features = batch["graph_features"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                outputs = self.model(
                    voltage_curves=voltage_curves,
                    ic_curves=ic_curves,
                    cycle_indices=cycle_indices,
                    graph_features=graph_features,
                )

                predictions = outputs["soh_prediction"]

                loss_dict = self.loss_fn(
                    predictions,
                    target_soh,
                    cycle_indices=cycle_indices,
                    battery_ids=battery_ids,
                )

                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_mse += loss_dict["loss_mse"].item()
            total_mono += loss_dict["loss_mono"].item()
            total_res += loss_dict["loss_res"].item()
            num_batches += 1

        return {
            "train_loss": total_loss / num_batches,
            "train_mse": total_mse / num_batches,
            "train_mono": total_mono / num_batches,
            "train_res": total_res / num_batches,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                battery_ids = batch["battery_ids"]
                cycle_indices = batch["cycle_indices"].to(self.device)
                target_soh = batch["target_soh"].to(self.device)

                voltage_curves = None
                if "voltage_curves" in batch and batch["voltage_curves"] is not None:
                    voltage_curves = batch["voltage_curves"].to(self.device)

                ic_curves = None
                if "ic_curves" in batch and batch["ic_curves"] is not None:
                    ic_curves = batch["ic_curves"].to(self.device)

                graph_features = None
                if "graph_features" in batch and batch["graph_features"] is not None:
                    graph_features = batch["graph_features"].to(self.device)

                outputs = self.model(
                    voltage_curves=voltage_curves,
                    ic_curves=ic_curves,
                    cycle_indices=cycle_indices,
                    graph_features=graph_features,
                )

                predictions = outputs["soh_prediction"]

                loss_dict = self.loss_fn(
                    predictions,
                    target_soh,
                    cycle_indices=cycle_indices,
                    battery_ids=battery_ids,
                )

                total_loss += loss_dict["total_loss"].item()
                total_mse += loss_dict["loss_mse"].item()
                num_batches += 1

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target_soh.cpu().numpy().flatten())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        return {
            "val_loss": total_loss / num_batches,
            "val_mse": total_mse / num_batches,
            "val_rmse": rmse,
            "predictions": predictions,
            "targets": targets,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 15,
        save_best: bool = True,
        log_interval: int = 5,
    ) -> Dict:
        """
        Full training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            patience: Early stopping patience
            save_best: Whether to save best model
            log_interval: Logging interval

        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=patience, mode="min")

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            train_metrics = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Handle scheduler step correctly for both types
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["learning_rate"].append(current_lr)

            if (epoch + 1) % log_interval == 0:
                print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
                print(f"Train MSE: {train_metrics['train_mse']:.4f}")
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"Val RMSE: {val_metrics['val_rmse']:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")

            if save_best and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(
                    epoch,
                    {
                        "train_loss": train_metrics["train_loss"],
                        "val_loss": val_metrics["val_loss"],
                        "val_rmse": val_metrics["val_rmse"],
                    },
                    is_best=True,
                )
                print(f"Saved best model (val_loss: {val_metrics['val_loss']:.4f})")

            if early_stopping(val_metrics["val_loss"]):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        return self.history

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "model_config": getattr(self.model, "model_config", {}),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        return checkpoint.get("epoch", 0)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs,
) -> optim.Optimizer:
    """
    Create optimizer

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 10,
    **kwargs,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with optional linear warmup.

    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('cosine' or 'plateau')
        warmup_epochs: Linear warmup epochs before main schedule (0 = disabled)
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "cosine":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-6)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(T_max - warmup_epochs, 1),
            eta_min=eta_min,
        )
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return cosine
    elif scheduler_type.lower() == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
