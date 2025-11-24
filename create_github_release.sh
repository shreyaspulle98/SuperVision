#!/bin/bash

# Script to create GitHub release with model files
# Requires GitHub CLI (gh): brew install gh
# Usage: bash create_github_release.sh

set -e  # Exit on error

echo "=========================================="
echo "SuperVision GitHub Release Creator"
echo "=========================================="

# Configuration
VERSION="v1.0.0"
RELEASE_NAME="SuperVision v1.0.0 - State-of-the-Art Tc Prediction"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ ERROR: GitHub CLI (gh) is not installed!"
    echo "   Install: brew install gh"
    echo "   Then run: gh auth login"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ ERROR: Not authenticated with GitHub!"
    echo "   Run: gh auth login"
    exit 1
fi

echo "✓ GitHub CLI authenticated"
echo ""

# Check if models exist
echo "Checking for model files..."
if [ ! -f "models/dino_best.pth" ]; then
    echo "❌ ERROR: DINOv3 model not found at models/dino_best.pth"
    exit 1
fi

if [ ! -f "models/alignn_best.pth" ]; then
    echo "❌ ERROR: ALIGNN model not found at models/alignn_best.pth"
    exit 1
fi

echo "✓ DINOv3 model found ($(du -h models/dino_best.pth | cut -f1))"
echo "✓ ALIGNN model found ($(du -h models/alignn_best.pth | cut -f1))"
echo ""

# Create release notes
RELEASE_NOTES="## SuperVision v1.0.0: State-of-the-Art Superconductor Tc Prediction

This release contains trained models achieving state-of-the-art performance for predicting superconductor critical temperatures.

### 🏆 Performance Highlights

| Model | Test MAE (K) | Test R² | Parameters | Training Time |
|-------|--------------|---------|------------|---------------|
| **DINOv3 + LoRA** | **4.85** ✓ | **0.7394** ✓ | 86M (1.3% trainable) | ~40 hours (CPU) |
| **ALIGNN** | 5.34 | 0.7186 | 4.2M | ~3 hours (CPU) |

**Improvement over literature**: 49-60% lower MAE than published baselines

### 📦 What's Included

This release contains:

1. **\`dino_best.pth\`** (~340 MB)
   - DINOv3 + LoRA model checkpoint
   - Best epoch: 23/40
   - Use for maximum accuracy

2. **\`alignn_best.pth\`** (~48 MB)
   - ALIGNN model checkpoint
   - Best epoch: 36/46
   - Use for high-throughput screening (13× faster)

3. **\`results_bundle.zip\`** (~1 MB)
   - Training histories, configs, metrics
   - Predictions CSV files
   - Visualization figures

### 🚀 Quick Start

**Download Models:**
\`\`\`bash
# Download DINOv3 model
wget https://github.com/YOUR_USERNAME/SuperVision/releases/download/v1.0.0/dino_best.pth

# Download ALIGNN model
wget https://github.com/YOUR_USERNAME/SuperVision/releases/download/v1.0.0/alignn_best.pth
\`\`\`

**Load and Use:**
\`\`\`python
import torch

# Load DINOv3 model
model = torch.load('dino_best.pth', map_location='cpu')
model.eval()

# Predict Tc from image
# See README for full usage examples
\`\`\`

### 📊 Dataset

The processed dataset is available at:
- **HuggingFace**: \`huggingface.co/datasets/YOUR_USERNAME/superconductor-3dsc\`
- **Size**: 5,773 materials (4,041 train / 866 val / 866 test)

### 🔗 Links

- **Paper**: [Coming soon]
- **HuggingFace Models**:
  - DINOv3: \`huggingface.co/YOUR_USERNAME/supervision-dinov3-tc-prediction\`
  - ALIGNN: \`huggingface.co/YOUR_USERNAME/supervision-alignn-tc-prediction\`
- **Documentation**: See [README.md](../README.md)

### 📝 Citation

\`\`\`bibtex
@software{supervision2024,
  title={SuperVision: Transfer Learning for Superconductor Tc Prediction},
  author={Your Name},
  year={2024},
  version={1.0.0},
  url={https://github.com/YOUR_USERNAME/SuperVision}
}
\`\`\`

### ⚖️ License

MIT License - See [LICENSE](../LICENSE) for details

### 🙏 Acknowledgments

- 3DSC Dataset: Court et al.
- ALIGNN: Choudhary & DeCost (NIST)
- DINOv2/v3: Meta AI Research
- LoRA: Microsoft Research
"

echo "Release Notes Preview:"
echo "----------------------"
echo "$RELEASE_NOTES"
echo "----------------------"
echo ""

# Confirm release
read -p "Create release $VERSION? (yes/no): " response
if [ "$response" != "yes" ]; then
    echo "Release cancelled."
    exit 0
fi

# Create results bundle
echo ""
echo "Creating results bundle..."
mkdir -p release_temp
cd release_temp

# Copy all results
cp -r ../results .
cp ../models/alignn_best.pth .
cp ../models/dino_best.pth .

# Create zip
zip -r results_bundle.zip results/

# Move back
mv results_bundle.zip ..
cd ..
rm -rf release_temp

echo "✓ Results bundle created"

# Create GitHub release
echo ""
echo "Creating GitHub release..."

# Save release notes to temporary file
echo "$RELEASE_NOTES" > release_notes.tmp

# Create release with assets
gh release create "$VERSION" \
    --title "$RELEASE_NAME" \
    --notes-file release_notes.tmp \
    models/dino_best.pth \
    models/alignn_best.pth \
    results_bundle.zip

# Clean up
rm -f release_notes.tmp
rm -f results_bundle.zip

echo ""
echo "=========================================="
echo "✓ GitHub release created successfully!"
echo "=========================================="
echo ""
echo "Release URL: $(gh release view $VERSION --json url -q .url)"
echo ""
echo "Users can now download models with:"
echo "  wget https://github.com/YOUR_USERNAME/SuperVision/releases/download/$VERSION/dino_best.pth"
echo "  wget https://github.com/YOUR_USERNAME/SuperVision/releases/download/$VERSION/alignn_best.pth"
echo ""
