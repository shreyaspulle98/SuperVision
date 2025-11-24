"""
DINO Pipeline

Vision Transformer pipeline using DINOv3 for predicting superconductor
critical temperatures from physics-informed 2D renderings of crystal structures.

This pipeline:
- Uses pre-trained vision transformers (DINOv3) as feature extractors
- Fine-tunes on physics-informed crystal structure images
- Leverages transfer learning from large-scale vision pretraining
- Captures spatial and structural patterns from visual representations

Modules:
- dataset.py: Image dataset preparation and data loading
- model.py: DINO-based model architecture with regression head
- train.py: Training and evaluation logic
"""

__version__ = "0.1.0"
__author__ = "SuperVision Team"

from . import dataset, model, train

__all__ = ["dataset", "model", "train"]
