---
language: en
license: mit
tags:
- materials-science
- superconductor
- vision-transformer
- transfer-learning
- lora
- critical-temperature
datasets:
- shreyaspulle98/superconductor-3dsc
metrics:
- mae
- rmse
- r2
model-index:
- name: SuperVision-DINOv3
  results:
  - task:
      type: regression
      name: Critical Temperature Prediction
    dataset:
      name: 3DSC Superconductor Dataset
      type: superconductor-3dsc
    metrics:
    - type: mae
      value: 4.85
      name: Mean Absolute Error (K)
    - type: rmse
      value: 9.88
      name: Root Mean Squared Error (K)
    - type: r2
      value: 0.7394
      name: R² Score
---

# SuperVision: DINOv3 + LoRA for Superconductor Tc Prediction

## Model Description

This model predicts superconductor critical temperatures (Tc) using a Vision Transformer (DINOv3) fine-tuned with LoRA on 2D rendered crystal structures.

**Key Features:**
- **Architecture**: DINOv3-base (86M parameters) + LoRA (1.1M trainable)
- **Input**: 224×224 RGB images of crystal structures with physics-informed encoding
- **Pre-training**: ImageNet (14M images) via self-supervised learning
- **Fine-tuning**: 5,773 superconductor materials from 3DSC dataset
- **Performance**: MAE 4.85 K, R² 0.74 (49-60% better than literature)

## Intended Use

**Primary Use Case:**
- Predict critical temperature (Tc) of superconducting materials from crystal structure images
- Screen candidate materials for high-temperature superconductivity
- Accelerate materials discovery in computational materials science

**Users:**
- Materials scientists and computational chemists
- Researchers in condensed matter physics
- Machine learning practitioners working on scientific applications

## Training Data

**Dataset**: 3DSC (3D Superconductor Dataset)
- 5,773 superconductor materials
- Train: 4,041 materials (70%)
- Validation: 866 materials (15%)
- Test: 866 materials (15%)

**Data Sources:**
- Crystal structures: Materials Project
- Critical temperatures: SuperCon database

**Preprocessing:**
- 3D CIF structures rendered to 2D images using ASE
- Physics-informed RGB encoding:
  - R channel: d-electron count
  - G channel: Valence electrons
  - B channel: Atomic mass

## Performance

### Test Set Results

| Metric | Value |
|--------|-------|
| MAE | 4.85 K |
| RMSE | 9.88 K |
| R² | 0.7394 |

### Comparison to Baselines

| Method | MAE (K) | Improvement |
|--------|---------|-------------|
| Random Forest (Stanev et al. 2018) | ~9.5 | **49%** |
| GNN (Konno et al. 2021) | ~12 | **60%** |
| **SuperVision DINOv3 (ours)** | **4.85** | **State-of-the-art** |

### Training Details

- **Best Epoch**: 23/40
- **Training Time**: ~40 hours (CPU)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 8
- **LoRA Config**: rank=16, alpha=32, dropout=0.1

## Usage

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image

# Load model and processor
processor = AutoImageProcessor.from_pretrained("shreyaspulle98/supervision-dinov3-tc-prediction")
model = AutoModel.from_pretrained("shreyaspulle98/supervision-dinov3-tc-prediction")

# Load and process image
image = Image.open("crystal_structure.png")
inputs = processor(images=image, return_tensors="pt")

# Predict critical temperature
with torch.no_grad():
    tc_prediction = model(**inputs).logits

print(f"Predicted Tc: {tc_prediction.item():.2f} K")
```

## Limitations

1. **Training Data Coverage**: Limited to materials in 3DSC dataset (primarily conventional and cuprate superconductors)
2. **High-Tc Underestimation**: Larger errors for materials with Tc > 100 K (limited training samples)
3. **Image Dependency**: Requires crystal structure rendering with specific physics-informed encoding
4. **Computational Cost**: Slower inference than ALIGNN (~120ms vs 50ms per sample)

## Ethical Considerations

- Model is for **research purposes only**, not for production materials design without experimental validation
- Predictions should be verified experimentally before synthesis attempts
- Model may have biases toward material classes well-represented in training data

## Citation

```bibtex
@software{supervision2024,
  title={SuperVision: Transfer Learning for Superconductor Tc Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SuperVision}
}
```

## License

MIT License

## More Information

- **GitHub Repository**: https://github.com/yourusername/SuperVision
- **Paper**: [Coming soon]
- **Contact**: your.email@example.com
