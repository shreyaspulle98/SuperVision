# Upload Quick Start Guide

**Time Required**: ~30 minutes
**Prerequisites**: HuggingFace account, GitHub account

---

## 🚀 3-Step Upload Process

### Step 1: Upload to HuggingFace (~15 min)

```bash
# 1. Install and login
pip install huggingface_hub
huggingface-cli login

# 2. Edit upload script (line 15)
# Change: HF_USERNAME = "YOUR_HF_USERNAME"
# To:     HF_USERNAME = "your-actual-username"
nano upload_to_huggingface.py

# 3. Run upload
python3 upload_to_huggingface.py
```

**Result**: Models and dataset on HuggingFace (397 MB uploaded)

---

### Step 2: Push Code to GitHub (~10 min)

```bash
# 1. Install GitHub CLI
brew install gh
gh auth login

# 2. Edit setup script (line 10)
# Change: GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
# To:     GITHUB_USERNAME="your-actual-username"
nano setup_github.sh

# 3. Run setup
bash setup_github.sh
```

**Result**: Code on GitHub, repository created

---

### Step 3: Create Release with Models (~5 min)

```bash
# Run release script
bash create_github_release.sh
```

**Result**: Models downloadable from GitHub releases

---

## ✅ Verification

Check these URLs (replace `username`):

- [ ] `https://huggingface.co/username/supervision-dinov3-tc-prediction`
- [ ] `https://huggingface.co/username/supervision-alignn-tc-prediction`
- [ ] `https://huggingface.co/datasets/username/superconductor-3dsc`
- [ ] `https://github.com/username/SuperVision`
- [ ] `https://github.com/username/SuperVision/releases/tag/v1.0.0`

---

## 🆘 Troubleshooting

**"Not logged in to HuggingFace"**
```bash
huggingface-cli login
# Paste token from: https://huggingface.co/settings/tokens
```

**"Not authenticated with GitHub"**
```bash
gh auth login
# Follow prompts
```

**"gh: command not found"**
```bash
brew install gh
```

---

## 📝 Final Step

Update README.md placeholders:
1. Replace all `YOUR_HF_USERNAME` with your HuggingFace username
2. Replace all `YOUR_GITHUB_USERNAME` with your GitHub username
3. Commit and push:
   ```bash
   git add README.md
   git commit -m "Add download links"
   git push
   ```

---

## 🎉 Done!

Your models are now publicly available! Share:
- HuggingFace: `https://huggingface.co/username`
- GitHub: `https://github.com/username/SuperVision`

For detailed instructions, see **UPLOAD_GUIDE.md**
