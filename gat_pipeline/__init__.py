"""
GAT Pipeline - Refactored for ALIGNN Fine-Tuning

Graph neural network pipeline for predicting superconductor critical temperatures
using pre-trained ALIGNN (Atomistic Line Graph Neural Network).

This pipeline uses transfer learning:
- Pre-trained ALIGNN from Materials Project (formation energy prediction)
- Fine-tuned on superconductor Tc prediction
- Uses same train/val/test splits as DINOv3 pipeline for fair comparison

Modules:
- model: Pre-trained ALIGNN with Tc prediction head
- dataset: DGL graph dataset loading from crystal structures
- train: Fine-tuning with differential learning rates

Key Features:
- Transfer learning from Materials Project → superconductors
- Differential learning rates: backbone (1e-5) vs head (1e-3)
- DGL graph representation (atom graph + line graph)
- Early stopping based on validation MAE
- Saves best model to models/alignn_best.pth

Usage:
    from gat_pipeline import create_alignn_model, get_dataloaders, ALIGNNTrainer

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders("data/processed")

    # Create model
    model = create_alignn_model({"pretrained_model_name": "mp_e_form"})

    # Train
    trainer = ALIGNNTrainer(model, device="cuda")
    trainer.fit(train_loader, val_loader)
"""

__version__ = "0.2.0"
__author__ = "SuperVision Team"

# Import only ALIGNN-related functions
from .model import PretrainedALIGNN, create_alignn_model
from .dataset import (
    SuperconductorALIGNNDataset,
    get_dataloader,
    get_dataloaders,
    structure_to_alignn_graphs
)
from .train import ALIGNNTrainer

__all__ = [
    # Model
    "PretrainedALIGNN",
    "create_alignn_model",

    # Dataset
    "SuperconductorALIGNNDataset",
    "get_dataloader",
    "get_dataloaders",
    "structure_to_alignn_graphs",

    # Training
    "ALIGNNTrainer",
]
