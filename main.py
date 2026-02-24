"""
Main Entry Point for Battery SoH Estimation Project
Physics-Informed Multi-Scale CNN–Transformer–GNN Hybrid with Contrastive Pretraining
"""

import torch
import os
import json
import argparse
import numpy as np
import random
import logging
from datetime import datetime

from data_loader import DataLoaderFactory
from model import BatterySoHModel, create_model
from physics_loss import PhysicsInformedLoss
from train import Trainer, create_optimizer, create_scheduler
from evaluate import BatterySoHEvaluator, print_evaluation_metrics
from contrastive_pretrain import ContrastivePretrainer
from optimize import BayesianOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Must be False for deterministic behavior


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Battery SoH Estimation with Physics-Informed Deep Learning"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "optimize"],
        help="Mode to run",
    )

    parser.add_argument(
        "--dataset_path", type=str, default="Dataset", help="Path to dataset directory"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to model checkpoint"
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    parser.add_argument(
        "--num_epochs", type=int, default=200, help="Number of training epochs"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    parser.add_argument(
        "--lambda_mono", type=float, default=0.1, help="Monotonicity loss weight"
    )

    parser.add_argument(
        "--lambda_res",
        type=float,
        default=0.1,
        help="Resistance consistency loss weight",
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=50,
        help="Number of Bayesian optimization trials",
    )

    parser.add_argument(
        "--pretrain_epochs", type=int, default=50, help="Number of pretraining epochs"
    )

    parser.add_argument(
        "--use_pretraining", action="store_true", help="Use contrastive pretraining"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )

    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
        choices=["nasa", "calce"],
        help="Target dataset: 'nasa' or 'calce'. Datasets are trained/tested separately for comparison.",
    )

    return parser.parse_args()


def train_mode(args):
    """
    Training mode

    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("BATTERY SOH ESTIMATION - TRAINING MODE")
    print("=" * 60 + "\n")

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from: {args.dataset_path}")
    factory = DataLoaderFactory(args.dataset_path)

    print("Creating dataloaders...")
    train_loader, test_loader, test_batteries = factory.create_dataloaders(
        batch_size=args.batch_size, train_ratio=0.8, seed=args.seed,
        dataset=args.target_dataset.lower(),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Test batteries: {len(test_batteries)}")

    print("\nCreating model...")
    model = create_model(
        {
            "voltage_seq_len": 500,
            "cnn_hidden": 128,
            "cnn_output": 256,
            "gnn_input": 1,
            "gnn_hidden": 64,
            "gnn_output": 128,
            "transformer_hidden": 256,
            "transformer_output": 256,
            "num_transformer_layers": 4,
            "num_heads": 8,
            "dropout": 0.05,
        }
    )

    print(f"Model parameters: {model.get_num_params():,}")

    # Use consistent lambdas — only from args, no hardcoded overrides
    loss_fn = PhysicsInformedLoss(
        lambda_mono=args.lambda_mono, lambda_res=args.lambda_res
    )

    optimizer = create_optimizer(
        model,
        optimizer_type="adamw",
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 10-epoch linear warmup then cosine annealing for stable early training
    scheduler = create_scheduler(optimizer, "cosine", warmup_epochs=10, T_max=args.num_epochs)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
    )

    # Contrastive pretraining (if enabled)
    if args.use_pretraining:
        print("\nStarting contrastive pretraining...")
        pretrainer = ContrastivePretrainer(
            encoder=model.cnn_branch,
            temperature=0.1,
            device=args.device,
        )
        pretrain_metrics = pretrainer.pretrain(
            dataloader=train_loader,
            num_epochs=args.pretrain_epochs,
            lr=args.learning_rate * 0.1,
        )
        print(f"Pretraining complete. Final loss: {pretrain_metrics['loss'][-1]:.4f}")

    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.num_epochs,
        patience=50,
        save_best=True,
        log_interval=5,
    )

    print("\nLoading best model for evaluation...")
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
    trainer.load_checkpoint(best_checkpoint)

    print("\nEvaluating on test set...")
    evaluator = BatterySoHEvaluator(model, device=args.device)
    results = evaluator.evaluate(test_loader, return_predictions=True)

    print_evaluation_metrics(results)

    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    evaluator.plot_predictions(
        results["predictions"],
        results["targets"],
        save_path=os.path.join(plots_dir, "predictions_vs_true.png"),
        title="Predicted vs True SoH",
    )

    evaluator.plot_degradation_curves(
        results["predictions"],
        results["targets"],
        results["battery_ids"],
        results["cycle_indices"],
        save_path=os.path.join(plots_dir, "degradation_curves.png"),
        num_batteries=4,
    )

    evaluator.plot_error_distribution(
        results["predictions"],
        results["targets"],
        save_path=os.path.join(plots_dir, "error_distribution.png"),
    )

    evaluator.save_results(
        results,
        output_dir=os.path.join(args.output_dir, "results"),
        experiment_name="battery_soh",
    )

    print(f"\nResults saved to {args.output_dir}")

    return results


def evaluate_mode(args):
    """
    Evaluation mode

    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("BATTERY SOH ESTIMATION - EVALUATION MODE")
    print("=" * 60 + "\n")

    set_seed(args.seed)

    if args.checkpoint_path is None:
        print("Error: --checkpoint_path required for evaluation mode")
        return

    print(f"Loading data from: {args.dataset_path}")
    factory = DataLoaderFactory(args.dataset_path)

    print("Creating dataloaders...")
    _, test_loader, _ = factory.create_dataloaders(
        batch_size=args.batch_size, train_ratio=0.8, seed=args.seed,
        dataset=args.target_dataset.lower(),
    )

    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)

    # Reconstruct exact architecture from saved config (avoids state_dict mismatch)
    model_config = checkpoint.get("model_config", {})
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    print("\nEvaluating...")
    evaluator = BatterySoHEvaluator(model, device=args.device)
    results = evaluator.evaluate(test_loader, return_predictions=True)

    print_evaluation_metrics(results)

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator.plot_predictions(
        results["predictions"],
        results["targets"],
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )

    return results


def optimize_mode(args):
    """
    Bayesian optimization mode

    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("BATTERY SOH ESTIMATION - BAYESIAN OPTIMIZATION")
    print("=" * 60 + "\n")

    set_seed(args.seed)

    print(f"Loading data from: {args.dataset_path}")
    factory = DataLoaderFactory(args.dataset_path)

    print("Creating dataloaders...")
    train_loader, val_loader, _ = factory.create_dataloaders(
        batch_size=args.batch_size, train_ratio=0.8, seed=args.seed,
        dataset=args.target_dataset.lower(),
    )

    def objective_fn(params):
        model = create_model(
            {
                "voltage_seq_len": 500,
                "cnn_hidden": params.get("cnn_hidden", 64),
                "cnn_output": 128,
                "gnn_input": 1,
                "gnn_hidden": params.get("gnn_hidden", 32),
                "gnn_output": 64,
                "transformer_hidden": params.get("transformer_hidden", 128),
                "transformer_output": 128,
                "num_transformer_layers": params.get("num_transformer_layers", 2),
                "num_heads": params.get("num_heads", 4),
                "dropout": params.get("dropout", 0.1),
            }
        )

        loss_fn = PhysicsInformedLoss(
            lambda_mono=params.get("lambda_mono", 0.1),
            lambda_res=params.get("lambda_res", 0.1),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get("learning_rate", 1e-4),
            weight_decay=params.get("weight_decay", 1e-5),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
        )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            patience=5,
            save_best=False,
            log_interval=100,
        )

        return history["val_loss"][-1]

    bo = BayesianOptimizer(
        objective_fn=objective_fn,
        n_trials=args.num_trials,
        study_name="battery_soh_optimization",
        direction="minimize",
    )

    print(f"\nStarting Bayesian optimization with {args.num_trials} trials...")

    study = bo.optimize()

    print("\nBest hyperparameters:")
    for key, value in bo.get_best_params().items():
        print(f"  {key}: {value}")

    os.makedirs(args.output_dir, exist_ok=True)
    bo.save_results(args.output_dir)


def main():
    """Main entry point"""
    args = parse_args()

    print(f"Using device: {args.device}")

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "optimize":
        optimize_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
