"""
Bayesian Optimization Module for Battery SoH Estimation
Uses Optuna for hyperparameter tuning
"""

import torch
import optuna
from optuna.samplers import TPESampler
import numpy as np
from typing import Dict, Optional, Callable
import os
import json
from datetime import datetime


class BayesianOptimizer:
    """
    Bayesian Optimization using Optuna

    Tunes:
    - Learning rate
    - Transformer layers
    - CNN filters
    - GAT heads
    - Dropout
    - Lambda weights (λ1, λ2)
    """

    def __init__(
        self,
        objective_fn: Callable,
        n_trials: int = 50,
        study_name: str = "battery_soh_optimization",
        storage: Optional[str] = None,
        direction: str = "minimize",
    ):
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.study_name = study_name
        self.direction = direction

        self.sampler = TPESampler(seed=42)

        self.storage = storage

        self.study = None
        self.best_params = None

    @staticmethod
    def create_search_space(trial: optuna.Trial) -> Dict:
        """
        Define hyperparameter search space

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of hyperparameters
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "num_transformer_layers": trial.suggest_int("num_transformer_layers", 1, 4),
            "num_heads": trial.suggest_int("num_heads", 2, 8),
            "cnn_hidden": trial.suggest_categorical("cnn_hidden", [32, 64, 128]),
            "gnn_hidden": trial.suggest_categorical("gnn_hidden", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lambda_mono": trial.suggest_float("lambda_mono", 0.0, 1.0),
            "lambda_res": trial.suggest_float("lambda_res", 0.0, 1.0),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "transformer_hidden": trial.suggest_categorical(
                "transformer_hidden", [64, 128, 256]
            ),
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna

        Args:
            trial: Optuna trial

        Returns:
            Validation loss (to minimize)
        """
        params = self.create_search_space(trial)

        try:
            val_loss = self.objective_fn(params)
            return val_loss

        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float("inf")

    def optimize(
        self, callbacks: Optional[list] = None, n_jobs: int = 1
    ) -> optuna.Study:
        """
        Run Bayesian optimization

        Args:
            callbacks: Optuna callbacks
            n_jobs: Number of parallel jobs

        Returns:
            Optuna study
        """
        self.study = optuna.create_study(
            study_name=self.study_name,
            sampler=self.sampler,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=callbacks,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params

        return self.study

    def get_best_params(self) -> Dict:
        """Get best hyperparameters"""
        if self.best_params is None and self.study is not None:
            self.best_params = self.study.best_params
        return self.best_params

    def save_results(self, output_dir: str = "optimization_results"):
        """Save optimization results"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "study_name": self.study_name,
            "n_trials": self.n_trials,
            "best_params": self.get_best_params(),
            "best_value": self.study.best_value if self.study else None,
            "all_trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": str(trial.state),
                }
                for trial in self.study.trials
                if trial.value is not None
            ],
        }

        results_path = os.path.join(
            output_dir, f"optimization_results_{timestamp}.json"
        )

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")

        return results_path
