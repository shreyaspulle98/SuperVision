---
language: en
license: mit
tags:
- materials-science
- superconductor
- critical-temperature
- crystal-structure
task_categories:
- tabular-regression
size_categories:
- 1K<n<10K
---

# 3DSC Superconductor Dataset

This dataset contains 5,773 superconducting materials for training ML models to predict critical temperatures.

## Dataset Summary

- **Total Materials**: 5,773
- **Tc Range**: 0-132 K
- **Splits**: Train (4,041) / Val (866) / Test (866)

## Data Fields

- `material_id`: Unique identifier
- `cif_id`: Materials Project ID
- `Tc`: Critical temperature (K)
- `formula`: Chemical formula
- `cif_path`: Path to CIF file

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("shreyaspulle98/superconductor-3dsc")
train_data = dataset["train"]
```

## Citation

Court, C.J., et al. "3-D Inorganic Crystal Structure Generation" (2020)

## License

MIT License
