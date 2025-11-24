"""
gat_pipeline/train.py

Training logic for fine-tuning pre-trained ALIGNN on superconductor Tc prediction.

This module implements:
- Transfer learning from Materials Project formation energy → superconductor Tc
- Differential learning rates: backbone (1e-5) vs head (1e-3)
- Early stopping based on validation MAE
- Model checkpointing
- Metrics logging (MAE, RMSE, R²)

Training Strategy:
1. Load pre-trained ALIGNN (trained on MP formation energy)
2. Replace final layer for Tc prediction
3. Fine-tune with slow backbone updates, fast head updates
4. Use same train/val/test splits as DINOv3 pipeline for fair comparison
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import get_dataloaders
from model import create_alignn_model


class ALIGNNTrainer:
    """
    Trainer for fine-tuning pre-trained ALIGNN on superconductor Tc prediction.
    """

    def __init__(
        self,
        model,
        device="cpu",
        backbone_lr=1e-5,
        head_lr=1e-3,
        weight_decay=1e-5
    ):
        """
        Initialize trainer with differential learning rates.

        Args:
            model: PretrainedALIGNN model
            device (str): Device ('cpu' or 'cuda')
            backbone_lr (float): Learning rate for pre-trained backbone
            head_lr (float): Learning rate for new Tc prediction head
            weight_decay (float): L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()

        # Get parameter groups with different learning rates
        print(f"\nSetting up optimizer with differential learning rates:")
        param_groups = model.get_optimizer_params(
            backbone_lr=backbone_lr,
            head_lr=head_lr
        )

        self.optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        print(f"  Weight decay: {weight_decay}")
        print(f"  LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with (g, lg, target) batches

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for g, lg, targets in tqdm(train_loader, desc="Training", leave=False):
            g = g.to(self.device)
            lg = lg.to(self.device)
            targets = targets.to(self.device).squeeze()

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(g, lg).squeeze()
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, loader, split_name="validation"):
        """
        Evaluate model on a dataset.

        Args:
            loader: DataLoader
            split_name (str): Name of split for logging

        Returns:
            tuple: (metrics dict, predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for g, lg, targets in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
                g = g.to(self.device)
                lg = lg.to(self.device)
                targets = targets.to(self.device).squeeze()

                predictions = self.model(g, lg).squeeze()
                loss = self.criterion(predictions, targets)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        metrics = {
            "loss": total_loss / num_batches,
            "mae": mean_absolute_error(all_targets, all_predictions),
            "rmse": np.sqrt(mean_squared_error(all_targets, all_predictions)),
            "r2": r2_score(all_targets, all_predictions)
        }

        return metrics, all_predictions, all_targets

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="models"
    ):
        """
        Fine-tune model with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs (int): Maximum epochs
            early_stopping_patience (int): Patience for early stopping
            checkpoint_dir (str): Directory for checkpoints

        Returns:
            dict: Training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        best_val_mae = float("inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_r2": [],
            "learning_rates": []
        }

        print(f"\nFine-tuning ALIGNN for up to {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {early_stopping_patience}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics, _, _ = self.evaluate(val_loader, "validation")

            # Get current learning rates
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            history["learning_rates"].append(current_lrs)

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_r2"].append(val_metrics["r2"])

            # Learning rate scheduling (based on val MAE, not loss)
            self.scheduler.step(val_metrics["mae"])

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val MAE:    {val_metrics['mae']:.4f} K")
            print(f"  Val RMSE:   {val_metrics['rmse']:.4f} K")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  LR (backbone/head): {current_lrs[0]:.2e} / {current_lrs[1]:.2e}")

            # Early stopping based on MAE (not loss)
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0

                # Save best model
                checkpoint_path = checkpoint_dir / "alignn_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_mae": val_metrics["mae"],
                    "val_loss": val_metrics["loss"]
                }, checkpoint_path)
                print(f"  → Saved best model (Val MAE: {best_val_mae:.4f} K)")
            else:
                patience_counter += 1
                print(f"  → No improvement (patience: {patience_counter}/{early_stopping_patience})")

            print()

            # Check early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        checkpoint_path = checkpoint_dir / "alignn_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

        return history


def main():
    """Main training function for ALIGNN fine-tuning."""
    print("=" * 70)
    print("ALIGNN Pipeline: Fine-tuning Pre-trained ALIGNN on Superconductor Tc")
    print("=" * 70)

    # Configuration
    config = {
        "data_root": "../data/processed",
        "batch_size": 32,
        "backbone_lr": 1e-5,  # Slow updates for pre-trained backbone
        "head_lr": 1e-3,      # Fast updates for new Tc head
        "weight_decay": 1e-5,
        "num_epochs": 50,
        "early_stopping_patience": 10,
        "pretrained_model_name": "mp_e_form_alignn",  # Pre-trained on Materials Project formation energy
        "freeze_backbone": False,  # Use differential LR, not freezing
        "hidden_dim": 256,
        "max_samples": None  # Set to small number for debugging (e.g., 100)
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Load data
    print("Loading superconductor datasets...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=0,
            max_samples=config["max_samples"]
        )
    except Exception as e:
        print(f"\nError loading datasets: {e}")
        print("\nMake sure:")
        print("  1. Data files exist in data/processed/")
        print("  2. CIF files are accessible")
        print("  3. ALIGNN is installed: pip install alignn dgl pymatgen")
        return

    # Create model
    print("\n" + "=" * 70)
    print("Creating Pre-trained ALIGNN Model")
    print("=" * 70)
    try:
        model = create_alignn_model(config)
    except Exception as e:
        print(f"\nError creating model: {e}")
        print("\nMake sure ALIGNN is installed:")
        print("  pip install alignn")
        return

    # Create trainer
    print("\n" + "=" * 70)
    print("Initializing Trainer")
    print("=" * 70)
    trainer = ALIGNNTrainer(
        model=model,
        device=device,
        backbone_lr=config["backbone_lr"],
        head_lr=config["head_lr"],
        weight_decay=config["weight_decay"]
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Fine-Tuning")
    print("=" * 70)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_dir="../models"
    )

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_metrics, test_predictions, test_targets = trainer.evaluate(test_loader, "test")

    print(f"\nTest Results:")
    print(f"  MAE:  {test_metrics['mae']:.4f} K")
    print(f"  RMSE: {test_metrics['rmse']:.4f} K")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    # Save results
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Save metrics
    with open(results_dir / "alignn_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        "actual_tc": test_targets,
        "predicted_tc": test_predictions
    })
    predictions_df.to_csv(results_dir / "alignn_predictions.csv", index=False)

    # Save config
    with open(results_dir / "alignn_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    history_df = pd.DataFrame({
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_mae": history["val_mae"],
        "val_rmse": history["val_rmse"],
        "val_r2": history["val_r2"]
    })
    history_df.to_csv(results_dir / "alignn_training_history.csv", index=False)

    print(f"\nResults saved to {results_dir}/")
    print("\n" + "=" * 70)
    print("ALIGNN Fine-Tuning Complete!")
    print("=" * 70)

    # Print comparison instructions
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Compare with DINOv3 results:")
    print("   - ALIGNN: results/alignn_predictions.csv")
    print("   - DINOv3: results/dino_predictions.csv")
    print("\n2. Both models use same train/val/test splits")
    print("3. Both use transfer learning from different domains:")
    print("   - ALIGNN: Materials Project (3D structures → formation energy)")
    print("   - DINOv3: ImageNet (2D images → classification)")
    print("=" * 70)


if __name__ == "__main__":
    main()
