#!/usr/bin/env python3
"""Fix and upload dataset card"""

from huggingface_hub import HfApi, upload_file
from pathlib import Path

HF_USERNAME = "shreyaspulle98"
DATASET_REPO = f"{HF_USERNAME}/superconductor-3dsc"
PROJECT_ROOT = Path(__file__).parent

# Simple dataset card without complex formatting
dataset_card = f"""---
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

dataset = load_dataset("{HF_USERNAME}/superconductor-3dsc")
train_data = dataset["train"]
```

## Citation

Court, C.J., et al. "3-D Inorganic Crystal Structure Generation" (2020)

## License

MIT License
"""

# Upload dataset card
card_path = PROJECT_ROOT / "README_dataset_simple.md"
card_path.write_text(dataset_card)

api = HfApi()
upload_file(
    path_or_fileobj=str(card_path),
    path_in_repo="README.md",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print(f"✓ Dataset card uploaded successfully!")
print(f"  View at: https://huggingface.co/datasets/{DATASET_REPO}")
