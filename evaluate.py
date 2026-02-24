"""
Evaluation Module for Battery SoH Estimation
Computes metrics, generates plots, and saves results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BatterySoHEvaluator:
    """
    Evaluator for Battery SoH Model

    Computes:
    - RMSE
    - RMSPE
    - MAPE
    - R²
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def compute_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(targets, predictions))

    def compute_rmspe(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Root Mean Squared Percentage Error"""
        mask = targets != 0
        if not mask.any():
            return 0.0
        percentage_errors = ((targets[mask] - predictions[mask]) / targets[mask]) ** 2
        return np.sqrt(np.mean(percentage_errors))

    def compute_mape(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error"""
        mask = targets != 0
        if not mask.any():
            return 0.0
        return (
            np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        )

    def compute_r2(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute R-squared (coefficient of determination)"""
        return r2_score(targets, predictions)

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader, return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on test set

        Args:
            dataloader: Test data loader
            return_predictions: Whether to return predictions and targets

        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_targets = []
        all_battery_ids = []
        all_cycle_indices = []

        with torch.no_grad():
            for batch in dataloader:
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

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target_soh.cpu().numpy().flatten())
                all_battery_ids.extend(battery_ids)
                all_cycle_indices.extend(cycle_indices.cpu().numpy())

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        rmse = self.compute_rmse(predictions, targets)
        rmspe = self.compute_rmspe(predictions, targets)
        mape = self.compute_mape(predictions, targets)
        r2 = self.compute_r2(predictions, targets)

        results = {
            "rmse": rmse,
            "rmspe": rmspe,
            "mape": mape,
            "r2": r2,
            "num_samples": len(predictions),
        }

        if return_predictions:
            results["predictions"] = predictions
            results["targets"] = targets
            results["battery_ids"] = all_battery_ids
            results["cycle_indices"] = all_cycle_indices
            results["battery_metrics"] = compute_per_battery_metrics(
                predictions, targets, all_battery_ids
            )

        return results

    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Predicted vs True SoH",
    ):
        """Plot predicted vs true SoH"""
        plt.figure(figsize=(10, 8))

        plt.scatter(targets, predictions, alpha=0.5, s=20)

        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        plt.xlabel("True SoH", fontsize=12)
        plt.ylabel("Predicted SoH", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.close()

    def plot_degradation_curves(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        battery_ids: List[str],
        cycle_indices: List[float],
        save_path: Optional[str] = None,
        num_batteries: int = 4,
    ):
        """Plot degradation curves for individual batteries"""
        unique_batteries = list(set(battery_ids))[:num_batteries]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, battery_id in enumerate(unique_batteries):
            if idx >= 4:
                break

            mask = [b == battery_id for b in battery_ids]
            cycles = np.array(cycle_indices)[mask]
            preds = predictions[mask]
            trues = targets[mask]

            sort_idx = np.argsort(cycles)
            cycles = cycles[sort_idx]
            preds = preds[sort_idx]
            trues = trues[sort_idx]

            axes[idx].plot(cycles, trues, "b-", label="True SoH", linewidth=2)
            axes[idx].plot(cycles, preds, "r--", label="Predicted SoH", linewidth=2)

            axes[idx].set_xlabel("Cycle Index")
            axes[idx].set_ylabel("SoH")
            axes[idx].set_title(f"Battery: {battery_id}")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Degradation curves saved to {save_path}")

        plt.close()

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Plot error distribution histogram"""
        errors = predictions - targets

        plt.figure(figsize=(10, 6))

        plt.hist(errors, bins=50, edgecolor="black", alpha=0.7)

        plt.axvline(x=0, color="r", linestyle="--", label="Zero Error")
        plt.axvline(
            x=np.mean(errors),
            color="g",
            linestyle="--",
            label=f"Mean Error: {np.mean(errors):.4f}",
        )

        plt.xlabel("Prediction Error (SoH)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Error Distribution", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Error distribution saved to {save_path}")

        plt.close()

    def save_results(
        self,
        results: Dict,
        output_dir: str = "evaluation_results",
        experiment_name: str = None,
    ):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or "battery_soh"

        results_path = os.path.join(output_dir, f"{exp_name}_results_{timestamp}.json")

        results_to_save = {
            "rmse": float(results["rmse"]),
            "rmspe": float(results["rmspe"]),
            "mape": float(results["mape"]),
            "r2": float(results["r2"]),
            "num_samples": int(results["num_samples"]),
        }

        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to {results_path}")

        return results_path


def compute_per_battery_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    battery_ids: List[str],
) -> Dict[str, Dict]:
    """Compute metrics per battery"""
    unique_batteries = list(set(battery_ids))
    battery_metrics = {}

    for battery_id in unique_batteries:
        clean_id = battery_id.split("_")[0] if "_" in battery_id else battery_id

        mask = np.array([b == battery_id for b in battery_ids])
        bat_preds = predictions[mask]
        bat_targets = targets[mask]

        if len(bat_preds) < 2:
            continue

        rmse = np.sqrt(mean_squared_error(bat_targets, bat_preds))

        mask_nonzero = bat_targets != 0
        if mask_nonzero.any():
            rmspe = np.sqrt(
                np.mean(
                    (
                        (bat_targets[mask_nonzero] - bat_preds[mask_nonzero])
                        / bat_targets[mask_nonzero]
                    )
                    ** 2
                )
            )
            mape = (
                np.mean(
                    np.abs(
                        (bat_targets[mask_nonzero] - bat_preds[mask_nonzero])
                        / bat_targets[mask_nonzero]
                    )
                )
                * 100
            )
        else:
            rmspe = 0.0
            mape = 0.0

        r2 = r2_score(bat_targets, bat_preds)

        battery_metrics[clean_id] = {
            "rmse": rmse,
            "rmspe": rmspe,
            "mape": mape,
            "r2": r2,
            "num_samples": len(bat_preds),
        }

    return battery_metrics


def print_evaluation_metrics(results: Dict, model_name: str = "Proposed"):
    """Print evaluation metrics in a formatted table"""
    if "battery_metrics" in results:
        battery_metrics = results["battery_metrics"]

        print("\n" + "=" * 95)
        print(
            f"| Model        | Battery     | RMSE (%)   | RMSPE (%)  | MAPE (%)   | R2       |"
        )
        print(
            "|"
            + "-" * 11
            + "|"
            + "-" * 11
            + "|"
            + "-" * 11
            + "|"
            + "-" * 10
            + "|"
            + "-" * 10
            + "|"
            + "-" * 8
            + "|"
        )

        sorted_batteries = sorted(battery_metrics.keys())

        for i, battery_id in enumerate(sorted_batteries):
            m = battery_metrics[battery_id]
            model_col = model_name if i == 0 else ""
            print(
                f"| {model_col:<11} | {battery_id:<11} | {m['rmse'] * 100:10.4f} | {m['rmspe'] * 100:10.4f} | {m['mape']:10.4f} | {m['r2']:8.4f} |"
            )

        print("=" * 95)

        avg_rmse = np.mean([m["rmse"] for m in battery_metrics.values()])
        avg_rmspe = np.mean([m["rmspe"] for m in battery_metrics.values()])
        avg_mape = np.mean([m["mape"] for m in battery_metrics.values()])
        avg_r2 = np.mean([m["r2"] for m in battery_metrics.values()])

        print(
            f"| {'Average':<11} | {'':<11} | {avg_rmse * 100:10.4f} | {avg_rmspe * 100:10.4f} | {avg_mape:10.4f} | {avg_r2:8.4f} |"
        )
        print("=" * 95 + "\n")
    else:
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)
        print(f"RMSE:   {results['rmse']:.6f}")
        print(f"RMSPE:  {results['rmspe']:.6f}")
        print(f"MAPE:   {results['mape']:.6f}%")
        print(f"R2:     {results['r2']:.6f}")
        print(f"Samples: {results['num_samples']}")
        print("=" * 50 + "\n")
