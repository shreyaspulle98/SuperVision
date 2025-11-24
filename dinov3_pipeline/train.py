"""
dino_pipeline/train.py

Training and evaluation logic for DINO vision transformer model.

This module:
- Loads image datasets
- Initializes DINO model with pre-trained weights
- Implements training loop with validation
- Supports different fine-tuning strategies
- Saves trained models and predictions
- Generates attention map visualizations

Training strategies:
1. Linear probe: Freeze backbone, train only regression head
2. Fine-tuning: Train entire model end-to-end
3. Progressive unfreezing: Start frozen, gradually unfreeze layers

Features:
- Early stopping based on validation loss
- Learning rate scheduling
- Gradient clipping
- Model checkpointing
- Mixed precision training (optional)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import get_dataloader, load_metadata
from model import create_dino_model, get_model_size


class DINOTrainer:
    """
    Trainer class for DINO vision transformer model.
    """

    def __init__(
        self,
        model,
        device="cpu",
        learning_rate=1e-4,
        weight_decay=1e-5,
        use_amp=False
    ):
        """
        Initialize trainer.

        Args:
            model: DINO model
            device (str): Device to train on ('cpu' or 'cuda')
            learning_rate (float): Initial learning rate
            weight_decay (float): L2 regularization weight
            use_amp (bool): Use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10
        )

        # Gradient scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using automatic mixed precision (AMP)")

    def train_epoch(self, train_loader):
        """
        Trains for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions.squeeze(), labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions = self.model(images)
                loss = self.criterion(predictions.squeeze(), labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, loader, split_name="validation"):
        """
        Evaluates model on a dataset.

        Args:
            loader: DataLoader
            split_name (str): Name of the split

        Returns:
            tuple: (metrics dict, predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(images)
                loss = self.criterion(predictions.squeeze(), labels)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        # Explicitly free memory to prevent accumulation
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
        early_stopping_patience=15,
        checkpoint_dir="models"
    ):
        """
        Trains the model with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs (int): Maximum number of epochs
            early_stopping_patience (int): Patience for early stopping
            checkpoint_dir (str): Directory to save checkpoints

        Returns:
            dict: Training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_r2": []
        }

        # Check for existing checkpoint to resume from
        # Prefer "last" checkpoint (most recent) over "best" checkpoint
        last_checkpoint_path = checkpoint_dir / "dino_last.pth"
        best_checkpoint_path = checkpoint_dir / "dino_best.pth"

        checkpoint_path = None
        if last_checkpoint_path.exists():
            checkpoint_path = last_checkpoint_path
            print(f"Found last checkpoint at {checkpoint_path}")
        elif best_checkpoint_path.exists():
            checkpoint_path = best_checkpoint_path
            print(f"Found best checkpoint at {checkpoint_path}")

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model and optimizer state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Resume from the next epoch after the checkpoint
            start_epoch = checkpoint["epoch"] + 1

            # Load best validation loss and patience counter if available
            best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
            patience_counter = checkpoint.get("patience_counter", 0)

            print(f"Resuming from epoch {start_epoch}")
            print(f"Current validation loss: {checkpoint['val_loss']:.4f}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
            print(f"Patience counter: {patience_counter}/{early_stopping_patience}")
            print()

        print(f"\nTraining DINO model for up to {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {early_stopping_patience}")
        if start_epoch > 0:
            print(f"Starting from epoch {start_epoch + 1}\n")
        else:
            print()

        for epoch in range(start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics, _, _ = self.evaluate(val_loader, "validation")

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_r2"].append(val_metrics["r2"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val MAE:    {val_metrics['mae']:.4f} K")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")

            # Early stopping and checkpointing
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0

                # Save best model
                checkpoint_path = checkpoint_dir / "dino_best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "patience_counter": patience_counter
                }, checkpoint_path)
                print(f"  → Saved best model (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  → No improvement (patience: {patience_counter}/{early_stopping_patience})")

            # Always save last checkpoint for resuming
            last_checkpoint_path = checkpoint_dir / "dino_last.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter
            }, last_checkpoint_path)

            print()

            # Check early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        checkpoint_path = checkpoint_dir / "dino_best.pth"
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return history


def main():
    """Main training function."""
    print("=" * 70)
    print("DINO Pipeline: Training Vision Transformer")
    print("=" * 70)

    # Configuration
    config = {
        "metadata_path": "../data/images/image_metadata.csv",
        "batch_size": 8,  # Reduced from 32 to prevent memory swapping
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs": 50,
        "early_stopping_patience": 15,
        "model_name": "dinov3_vitb14",
        "freeze_backbone": False,
        "dropout": 0.3,
        "hidden_dim": 512,
        "image_size": 224,
        "num_workers": 0,  # Disabled multiprocessing to reduce memory overhead
        "use_amp": False,  # Disabled - AMP only benefits GPU training
        # LoRA parameters for parameter-efficient fine-tuning
        "use_lora": True,        # Enable LoRA (better than linear probing)
        "lora_rank": 16,         # Rank of LoRA decomposition (4-64 typical)
        "lora_alpha": 32,        # Scaling parameter (usually 2*rank)
        "lora_dropout": 0.1      # Dropout for LoRA layers
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Load data
    print("Loading image datasets...")

    try:
        train_loader = get_dataloader(
            config["metadata_path"],
            split="train",
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            image_size=config["image_size"]
        )

        val_loader = get_dataloader(
            config["metadata_path"],
            split="val",
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            image_size=config["image_size"]
        )

        test_loader = get_dataloader(
            config["metadata_path"],
            split="test",
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            image_size=config["image_size"]
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run 03_render_images.py first to generate images.")
        return

    # Create model
    print("\nCreating DINO model...")
    model = create_dino_model(config)

    stats = get_model_size(model)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Model size: {stats['size_mb']:.2f} MB")

    # Create trainer
    trainer = DINOTrainer(
        model=model,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        use_amp=config["use_amp"]
    )

    # Train
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
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save metrics
    with open(results_dir / "dino_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        "actual_tc": test_targets,
        "predicted_tc": test_predictions
    })
    predictions_df.to_csv(results_dir / "dino_predictions.csv", index=False)

    # Save config
    with open(results_dir / "dino_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "dino_training_history.csv", index=False)

    print(f"\nResults saved to {results_dir}/")
    print("\n" + "=" * 70)
    print("DINO Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
