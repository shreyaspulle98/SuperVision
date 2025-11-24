# GAT Pipeline - Refactored for ALIGNN Fine-Tuning

## Overview

This pipeline has been **completely refactored** to focus on **transfer learning with pre-trained ALIGNN**, removing the original SimpleGAT implementation.

**Transfer Learning Strategy:**
- Start with ALIGNN pre-trained on Materials Project formation energy (100k+ materials)
- Fine-tune on superconductor Tc prediction with differential learning rates
- Compare against DINOv3 (vision transformer on 2D rendered images)

## What Changed

### ✅ Removed
- `GATModel` class (SimpleGAT trained from scratch)
- Any training-from-scratch logic
- PyTorch Geometric dependencies (replaced with DGL)

### ✅ Added
- `PretrainedALIGNN` class - loads MP-pretrained ALIGNN
- Differential learning rates: backbone (1e-5) vs head (1e-3)
- DGL graph conversion from pymatgen structures
- Same train/val/test splits as DINOv3 for fair comparison

## Architecture

```
ALIGNN (Pre-trained on MP)
├── Atom Graph Network (frozen/slow LR)
├── Line Graph Network (frozen/slow LR)
└── Tc Prediction Head (fast LR) ← NEW, trained on superconductors
```

**Parameter Efficiency:**
- Backbone: ~2M params @ lr=1e-5 (slow updates)
- Head: ~200K params @ lr=1e-3 (fast updates)
- Total: ~2.2M params (vs 86M for DINOv3)

## File Structure

```
gat_pipeline/
├── __init__.py          # Exports only ALIGNN functions
├── model.py             # PretrainedALIGNN class
├── dataset.py           # DGL graph loading from CIF files
├── train.py             # Fine-tuning with differential LRs
└── README.md            # This file
```

## Installation

```bash
# Install ALIGNN and dependencies
pip install alignn dgl dglgo pymatgen torch

# Verify installation
python -m gat_pipeline.model
python -m gat_pipeline.dataset
```

## Usage

### Quick Start

```python
from gat_pipeline import create_alignn_model, get_dataloaders, ALIGNNTrainer

# 1. Load data (same splits as DINOv3)
train_loader, val_loader, test_loader = get_dataloaders(
    data_root="data/processed",
    batch_size=32
)

# 2. Create pre-trained ALIGNN model
model = create_alignn_model({
    "pretrained_model_name": "mp_e_form",  # Materials Project formation energy
    "freeze_backbone": False,              # Use differential LR instead
    "hidden_dim": 256
})

# 3. Train with differential learning rates
trainer = ALIGNNTrainer(
    model=model,
    device="cuda",
    backbone_lr=1e-5,  # Slow updates for pre-trained backbone
    head_lr=1e-3       # Fast updates for new Tc head
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# 4. Evaluate
test_metrics, predictions, targets = trainer.evaluate(test_loader, "test")
print(f"Test MAE: {test_metrics['mae']:.2f} K")
```

### Command-Line Training

```bash
# Train ALIGNN on full dataset
python -m gat_pipeline.train

# Train on subset (for debugging)
# Edit train.py and set max_samples=100
```

## Data Format

**Input:** CSV files with columns:
- `formula_sc`: Chemical formula
- `tc`: Critical temperature (target)
- `cif`: Path to CIF structure file

**Splits:**
- Train: `data/processed/train.csv` (~3600 materials)
- Val: `data/processed/val.csv` (~770 materials)
- Test: `data/processed/test.csv` (~770 materials)

**Graph Representation:**
- **Atom graph**: Nodes=atoms, edges=bonds
- **Line graph**: Nodes=bonds, edges=angle connections
- **Node features**: Atomic number, electronegativity, radius, etc. (from ALIGNN)
- **Edge features**: Bond distances, types (from ALIGNN)

## Training Configuration

**Recommended Settings:**
```python
config = {
    "batch_size": 32,
    "backbone_lr": 1e-5,         # Slow backbone updates
    "head_lr": 1e-3,             # Fast head updates
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "pretrained_model_name": "mp_e_form"
}
```

**Why Differential Learning Rates?**
- Backbone already learned general materials knowledge (MP)
- Only need to adapt slightly for superconductors
- New head needs to learn Tc prediction from scratch
- Result: Better performance + faster convergence

## Outputs

**Saved Files:**
- `models/alignn_best.pth` - Best model checkpoint
- `results/alignn_predictions.csv` - Test predictions
- `results/alignn_metrics.json` - MAE, RMSE, R²
- `results/alignn_training_history.csv` - Training curves
- `results/alignn_config.json` - Hyperparameters

## Comparison with DINOv3

Both pipelines use transfer learning but from different domains:

| Aspect | ALIGNN | DINOv3 |
|--------|--------|--------|
| **Input** | 3D crystal structures | 2D rendered images |
| **Pre-training** | Materials Project (formation energy) | ImageNet (classification) |
| **Architecture** | Graph neural network | Vision transformer |
| **Parameters** | ~2.2M | ~86M |
| **Pre-training data** | 100k materials | 14M images |
| **Inductive bias** | Atomic interactions, bond angles | Spatial patterns, visual features |

**Fair Comparison:**
- Same train/val/test splits (5,773 materials)
- Same evaluation metrics (MAE, RMSE, R²)
- Both use transfer learning (not training from scratch)
- Both save predictions to `results/` for comparison

## Expected Performance

**ALIGNN (based on literature):**
- Formation energy MAE: ~30 meV/atom on MP
- Tc prediction (estimated): MAE ~10-20 K
- Training time: ~30-60 min on GPU, ~2-4 hours on CPU

**Comparison Hypothesis:**
- ALIGNN should perform well on Tc prediction (materials domain)
- DINOv3 may capture visual patterns not obvious in graphs
- Ensemble of both could be optimal

## Troubleshooting

**Issue: ALIGNN not installed**
```bash
pip install alignn dgl
```

**Issue: CIF files not found**
- Check paths in `data/processed/train.csv`
- CIF files should be in `data/final/MP/cifs/`

**Issue: DGL import error**
```bash
# Install DGL with CUDA support (if available)
pip install dgl-cu117 dglgo -f https://data.dgl.ai/wheels/repo.html
```

**Issue: Out of memory**
- Reduce batch_size in config (try 16 or 8)
- Use CPU instead of GPU
- Limit dataset: set `max_samples=100` in train.py

## Next Steps

1. **Train ALIGNN**: Run `python -m gat_pipeline.train`
2. **Compare with DINOv3**: Check `results/` for both models
3. **Analyze results**: Which model performs better? Why?
4. **Ensemble**: Combine predictions from both models

## References

- ALIGNN paper: https://www.nature.com/articles/s41524-021-00650-1
- Materials Project: https://materialsproject.org/
- DGL documentation: https://docs.dgl.ai/
