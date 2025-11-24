# SuperVision Upload Guide

This guide walks you through uploading your models and datasets to HuggingFace and GitHub.

## 📋 Prerequisites

### 1. Install Required Tools

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Install GitHub CLI (recommended)
brew install gh

# Or download from: https://cli.github.com/
```

### 2. Create Accounts

- **HuggingFace**: https://huggingface.co/join
- **GitHub**: https://github.com/join (if you don't have one)

### 3. Authenticate

```bash
# HuggingFace login
huggingface-cli login
# Paste your HuggingFace token from: https://huggingface.co/settings/tokens

# GitHub login
gh auth login
# Follow the prompts to authenticate
```

---

## 🚀 Step 1: Upload to HuggingFace

### Configure Upload Script

1. Open `upload_to_huggingface.py`
2. Replace `YOUR_HF_USERNAME` with your HuggingFace username (line 15):
   ```python
   HF_USERNAME = "john-doe"  # Replace with your actual username
   ```

### Run Upload

```bash
cd "/Users/shrey/Semantic Search Project/SuperVision"
python3 upload_to_huggingface.py
```

The script will:
1. ✓ Create 3 repositories on your HuggingFace account:
   - `{username}/supervision-dinov3-tc-prediction` (model)
   - `{username}/supervision-alignn-tc-prediction` (model)
   - `{username}/superconductor-3dsc` (dataset)

2. ✓ Upload models with metadata:
   - Model checkpoints (.pth files)
   - Configs (JSON)
   - Metrics (JSON)
   - Training histories (CSV)
   - Model cards (README.md)

3. ✓ Upload dataset:
   - Train/val/test CSV files
   - Dataset statistics
   - Dataset card (README.md)

### Expected Output

```
==========================================================
SuperVision Model & Dataset Upload to HuggingFace
==========================================================

✓ Logged in as: john-doe

This script will upload:
  1. DINOv3 model (~340 MB) to john-doe/supervision-dinov3-tc-prediction
  2. ALIGNN model (~48 MB) to john-doe/supervision-alignn-tc-prediction
  3. Dataset CSVs (~9 MB) to john-doe/superconductor-3dsc

Total upload size: ~397 MB

Proceed with upload? (yes/no): yes

==========================================================
Uploading DINOv3 Model to HuggingFace
==========================================================

✓ Created repository: john-doe/supervision-dinov3-tc-prediction

Uploading model checkpoint (340.0 MB)...
✓ Model checkpoint uploaded
✓ Config uploaded
✓ Metrics uploaded
✓ Training history uploaded
✓ Model card uploaded

✓ DINOv3 model successfully uploaded to: https://huggingface.co/john-doe/supervision-dinov3-tc-prediction

[... ALIGNN and dataset uploads follow similar pattern ...]

==========================================================
✓ All uploads completed successfully!
==========================================================

Your models and dataset are now available at:
  • DINOv3: https://huggingface.co/john-doe/supervision-dinov3-tc-prediction
  • ALIGNN: https://huggingface.co/john-doe/supervision-alignn-tc-prediction
  • Dataset: https://huggingface.co/datasets/john-doe/superconductor-3dsc
```

### Verify Upload

Visit your HuggingFace profile: `https://huggingface.co/{your-username}`

You should see:
- 2 new models
- 1 new dataset
- Each with complete model/dataset cards

---

## 🐙 Step 2: Set Up GitHub Repository

### Configure GitHub Script

1. Open `setup_github.sh`
2. Replace `YOUR_GITHUB_USERNAME` with your GitHub username (line 10):
   ```bash
   GITHUB_USERNAME="john-doe"  # Replace with your actual username
   ```

### Run Setup

```bash
bash setup_github.sh
```

The script will:
1. ✓ Initialize Git repository
2. ✓ Configure Git user (name and email)
3. ✓ Create `.gitignore` (excludes large files)
4. ✓ Stage all files
5. ✓ Create initial commit
6. ✓ Create GitHub repository (if using `gh` CLI)
7. ✓ Push code to GitHub

### Expected Output

```
==========================================
SuperVision GitHub Setup
==========================================

Repository: john-doe/SuperVision

Initializing Git repository...
✓ Git repository initialized
✓ Git configured:
  Name: John Doe
  Email: john@example.com

Creating .gitignore...
✓ .gitignore created

Staging files for commit...

Git status:
A  .gitignore
A  README.md
A  dinov3_pipeline/
A  gat_pipeline/
[... more files ...]

Create initial commit? (yes/no): yes
✓ Initial commit created

GitHub Repository Setup
----------------------

Create GitHub repository using 'gh' CLI? (yes/no): yes

Creating GitHub repository...
✓ Repository created and pushed to GitHub

Repository URL: https://github.com/john-doe/SuperVision

==========================================
✓ GitHub setup complete!
==========================================
```

### Manual GitHub Setup (if not using `gh` CLI)

If you don't have GitHub CLI:

1. Go to https://github.com/new
2. Repository name: **SuperVision**
3. Description: **Transfer learning for superconductor critical temperature prediction**
4. Make it **public**
5. Do NOT initialize with README (we already have one)
6. Click **Create repository**

Then run:
```bash
git remote add origin https://github.com/YOUR_USERNAME/SuperVision.git
git branch -M main
git push -u origin main
```

---

## 📦 Step 3: Create GitHub Release with Models

### Why Create a Release?

GitHub releases provide:
- Permanent download links for model files
- Version control for models
- Easy discovery for users
- Integration with DOI providers (Zenodo)

### Run Release Script

```bash
bash create_github_release.sh
```

The script will:
1. ✓ Check for model files (dino_best.pth, alignn_best.pth)
2. ✓ Create results bundle (training curves, metrics, figures)
3. ✓ Generate release notes
4. ✓ Create GitHub release v1.0.0
5. ✓ Upload model files as assets

### Expected Output

```
==========================================
SuperVision GitHub Release Creator
==========================================

✓ GitHub CLI authenticated

Checking for model files...
✓ DINOv3 model found (340M)
✓ ALIGNN model found (48M)

Release Notes Preview:
----------------------
## SuperVision v1.0.0: State-of-the-Art Superconductor Tc Prediction

This release contains trained models achieving state-of-the-art performance...
[... release notes ...]
----------------------

Create release v1.0.0? (yes/no): yes

Creating results bundle...
✓ Results bundle created

Creating GitHub release...

==========================================
✓ GitHub release created successfully!
==========================================

Release URL: https://github.com/john-doe/SuperVision/releases/tag/v1.0.0

Users can now download models with:
  wget https://github.com/john-doe/SuperVision/releases/download/v1.0.0/dino_best.pth
  wget https://github.com/john-doe/SuperVision/releases/download/v1.0.0/alignn_best.pth
```

---

## 📝 Step 4: Update README with Links

After uploading to both platforms, update the README with actual links.

### Edit README.md

Find and replace placeholder usernames:

1. **HuggingFace Links** (search for `YOUR_HF_USERNAME`):
   ```markdown
   ## Pre-trained Models

   Download our pre-trained models from HuggingFace:

   - **DINOv3 + LoRA**: [huggingface.co/john-doe/supervision-dinov3-tc-prediction](https://huggingface.co/john-doe/supervision-dinov3-tc-prediction)
   - **ALIGNN**: [huggingface.co/john-doe/supervision-alignn-tc-prediction](https://huggingface.co/john-doe/supervision-alignn-tc-prediction)

   ## Dataset

   - **3DSC Dataset**: [huggingface.co/datasets/john-doe/superconductor-3dsc](https://huggingface.co/datasets/john-doe/superconductor-3dsc)
   ```

2. **GitHub Release Links** (search for `YOUR_GITHUB_USERNAME`):
   ```markdown
   ## Quick Start - Download Models

   ```bash
   # Download DINOv3 model
   wget https://github.com/john-doe/SuperVision/releases/download/v1.0.0/dino_best.pth

   # Download ALIGNN model
   wget https://github.com/john-doe/SuperVision/releases/download/v1.0.0/alignn_best.pth
   ```
   ```

3. **Citation** (update with your name and email):
   ```bibtex
   @software{supervision2024,
     title={SuperVision: Transfer Learning for Superconductor Tc Prediction},
     author={Your Name},
     year={2024},
     url={https://github.com/john-doe/SuperVision}
   }
   ```

### Commit and Push Updates

```bash
git add README.md
git commit -m "Add download links for models and dataset"
git push
```

---

## ✅ Verification Checklist

After completing all steps, verify:

### HuggingFace
- [ ] DINOv3 model page shows model card and metrics
- [ ] ALIGNN model page shows model card and metrics
- [ ] Dataset page shows train/val/test splits
- [ ] All files are downloadable

### GitHub
- [ ] Repository is public and accessible
- [ ] README displays correctly with all images
- [ ] Release v1.0.0 exists with 3 assets:
  - [ ] dino_best.pth (~340 MB)
  - [ ] alignn_best.pth (~48 MB)
  - [ ] results_bundle.zip (~1 MB)
- [ ] Download links work

### Test Downloads

```bash
# Test HuggingFace download
python3 -c "
from huggingface_hub import hf_hub_download
file = hf_hub_download(repo_id='YOUR_USERNAME/supervision-dinov3-tc-prediction',
                       filename='pytorch_model.bin')
print(f'✓ Downloaded: {file}')
"

# Test GitHub release download
wget https://github.com/YOUR_USERNAME/SuperVision/releases/download/v1.0.0/dino_best.pth
ls -lh dino_best.pth
rm dino_best.pth
```

---

## 🎯 Summary of URLs

After completing this guide, you'll have:

| Resource | URL |
|----------|-----|
| **GitHub Repo** | `https://github.com/{username}/SuperVision` |
| **DINOv3 Model** | `https://huggingface.co/{username}/supervision-dinov3-tc-prediction` |
| **ALIGNN Model** | `https://huggingface.co/{username}/supervision-alignn-tc-prediction` |
| **Dataset** | `https://huggingface.co/datasets/{username}/superconductor-3dsc` |
| **Release** | `https://github.com/{username}/SuperVision/releases/tag/v1.0.0` |

---

## 🆘 Troubleshooting

### HuggingFace Upload Fails

**Error: "Repository not found"**
```bash
# Solution: Create repository manually
huggingface-cli repo create supervision-dinov3-tc-prediction --type model
```

**Error: "Authentication failed"**
```bash
# Solution: Re-login
huggingface-cli logout
huggingface-cli login
```

**Error: "File too large"**
- HuggingFace supports files up to 50GB (you're fine)
- If upload times out, try again (it will resume)

### GitHub Upload Fails

**Error: "not a git repository"**
```bash
# Solution: Initialize Git
git init
```

**Error: "remote origin already exists"**
```bash
# Solution: Update remote
git remote set-url origin https://github.com/YOUR_USERNAME/SuperVision.git
```

**Error: "large files"**
- Model files are in `.gitignore`, so they won't be in Git
- Use GitHub releases for large files (Step 3)

### GitHub Release Fails

**Error: "gh: command not found"**
```bash
# Solution: Install GitHub CLI
brew install gh
# Or download from: https://cli.github.com/
```

**Error: "release already exists"**
```bash
# Solution: Delete old release first
gh release delete v1.0.0
```

---

## 📧 Need Help?

If you encounter issues:

1. Check error messages carefully
2. Verify authentication (`huggingface-cli whoami`, `gh auth status`)
3. Ensure file paths are correct
4. Try manual upload via web interface as backup

---

## 🎉 Success!

Once complete, share your work:

1. **Twitter/X**: "Just released SuperVision - state-of-the-art Tc prediction for superconductors! 🚀"
2. **LinkedIn**: Post about your achievement
3. **Reddit**: r/MachineLearning, r/Physics
4. **HuggingFace**: Your models appear in community feed

Congratulations on sharing your research with the world! 🎊
