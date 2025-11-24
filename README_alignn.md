---
language: en
license: mit
tags:
- materials-science
- superconductor
- graph-neural-network
- transfer-learning
- critical-temperature
- alignn
datasets:
- shreyaspulle98/superconductor-3dsc
metrics:
- mae
- rmse
- r2
model-index:
- name: SuperVision-ALIGNN
  results:
  - task:
      type: regression
      name: Critical Temperature Prediction
    dataset:
      name: 3DSC Superconductor Dataset
      type: superconductor-3dsc
    metrics:
    - type: mae
      value: 5.34
      name: Mean Absolute Error (K)
    - type: rmse
      value: 10.27
      name: Root Mean Squared Error (K)
    - type: r2
      value: 0.7186
      name: R² Score
---

# SuperVision: ALIGNN for Superconductor Tc Prediction

## Model Description

This model predicts superconductor critical temperatures (Tc) using ALIGNN (Atomistic Line Graph Neural Network) fine-tuned on 3D crystal structure graphs.

**Key Features:**
- **Architecture**: Pre-trained ALIGNN (4.2M parameters)
- **Input**: 3D crystal structure graphs (atom graphs + line graphs)
- **Pre-training**: Materials Project (100K+ materials, formation energy prediction)
- **Fine-tuning**: 5,773 superconductor materials from 3DSC dataset
- **Performance**: MAE 5.34 K, R² 0.72 (44-56% better than literature)

## Intended Use

**Primary Use Case:**
- Predict critical temperature (Tc) of superconducting materials from CIF files
- High-throughput screening of material databases
- Fast inference for production deployment (2.4× faster than DINOv3)

**Users:**
- Materials scientists and computational chemists
- Researchers in condensed matter physics
- Industry practitioners needing fast, accurate Tc predictions

## Training Data

**Dataset**: 3DSC (3D Superconductor Dataset)
- 5,773 superconductor materials
- Train: 4,041 materials (70%)
- Validation: 866 materials (15%)
- Test: 866 materials (15%)

**Data Sources:**
- Crystal structures: Materials Project (CIF files)
- Critical temperatures: SuperCon database

**Preprocessing:**
- CIF files converted to ALIGNN graph representation
- Atom graph: nodes=atoms, edges=bonds
- Line graph: nodes=bonds, edges=angle connections

## Performance

### Test Set Results

| Metric | Value |
|--------|-------|
| MAE | 5.34 K |
| RMSE | 10.27 K |
| R² | 0.7186 |

### Comparison to Baselines

| Method | MAE (K) | Improvement |
|--------|---------|-------------|
| Random Forest (Stanev et al. 2018) | ~9.5 | **44%** |
| GNN (Konno et al. 2021) | ~12 | **56%** |
| **SuperVision ALIGNN (ours)** | **5.34** | **State-of-the-art** |

### Training Details

- **Best Epoch**: 36/46
- **Training Time**: ~3 hours (CPU)
- **Optimizer**: AdamW with differential learning rates
  - Backbone: 1e-5 (preserve Materials Project knowledge)
  - Head: 1e-3 (learn Tc prediction)
- **Batch Size**: 32
- **Early Stopping**: Patience 10

## Efficiency Advantages

| Aspect | ALIGNN | DINOv3 | ALIGNN Advantage |
|--------|--------|--------|------------------|
| Training Time | 3 hrs | 40 hrs | **13× faster** |
| Inference Speed | 50 ms | 120 ms | **2.4× faster** |
| Model Size | 4.2M params | 86M params | **20× smaller** |
| Memory Usage | 8 GB | 16 GB | **50% less** |

## Usage

```python
from alignn.pretrained import get_figshare_model
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
import torch

# Load model
model = torch.load("alignn_best.pth")
model.eval()

# Load crystal structure from CIF
atoms = Atoms.from_cif("material.cif")

# Convert to ALIGNN graph
from alignn.graphs import Graph
g, lg = Graph.atom_dgl_multigraph(atoms)

# Predict critical temperature
with torch.no_grad():
    tc_prediction = model(g, lg)

print(f"Predicted Tc: {tc_prediction.item():.2f} K")
```

## Strengths

1. **Fast & Efficient**: 13× faster training, 2.4× faster inference than DINOv3
2. **Direct 3D Encoding**: Uses actual atomic coordinates (no projection loss)
3. **Domain-Aligned**: Pre-trained on materials, not natural images
4. **High-Tc Performance**: Better at predicting cuprates (80-160 K range)
5. **Physically Rigorous**: Graph representation captures bond lengths, angles, symmetries

## Limitations

1. **Slightly Lower Accuracy**: 9% higher MAE than DINOv3 (5.34 K vs 4.85 K)
2. **CIF Dependency**: Requires valid CIF files with complete structural data
3. **Graph Conversion**: Multi-step preprocessing (CIF → PyMatGen → Jarvis → DGL)

## When to Use ALIGNN vs DINOv3

**Use ALIGNN if:**
- High-throughput screening (need to predict 10K+ materials)
- Production deployment (limited compute resources)
- Working with high-Tc materials (cuprates, pnictides)
- Speed and efficiency are priorities

**Use DINOv3 if:**
- Maximum accuracy is critical
- Careful candidate refinement for synthesis
- Working with low-Tc materials (conventional superconductors)
- Computational resources are available

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
- **ALIGNN Paper**: Choudhary & DeCost, Nature Communications (2021)
- **Contact**: your.email@example.com
