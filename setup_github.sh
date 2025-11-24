#!/bin/bash

# Script to initialize Git repository and prepare for GitHub upload
# Usage: bash setup_github.sh

set -e  # Exit on error

echo "=========================================="
echo "SuperVision GitHub Setup"
echo "=========================================="

# Configuration
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"  # Replace with your GitHub username
REPO_NAME="SuperVision"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ ERROR: Git is not installed!"
    echo "   Install Git: brew install git"
    exit 1
fi

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "⚠️  WARNING: GitHub CLI (gh) is not installed."
    echo "   Install it for easier GitHub integration: brew install gh"
    echo "   Or create repository manually at: https://github.com/new"
fi

# Check if username is configured
if [ "$GITHUB_USERNAME" = "YOUR_GITHUB_USERNAME" ]; then
    echo "❌ ERROR: Please edit this script and set GITHUB_USERNAME!"
    echo "   Example: GITHUB_USERNAME='john-doe'"
    exit 1
fi

echo ""
echo "Repository: $GITHUB_USERNAME/$REPO_NAME"
echo ""

# Initialize Git repository
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

# Configure Git if needed
if [ -z "$(git config user.name)" ]; then
    echo ""
    read -p "Enter your name for Git commits: " git_name
    git config user.name "$git_name"
fi

if [ -z "$(git config user.email)" ]; then
    echo ""
    read -p "Enter your email for Git commits: " git_email
    git config user.email "$git_email"
fi

echo "✓ Git configured:"
echo "  Name: $(git config user.name)"
echo "  Email: $(git config user.email)"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo ""
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Model checkpoints (too large for Git)
*.pth
*.pt
*.ckpt

# Data files (host separately)
data/raw/
data/images/
data/final/MP/cifs/

# Logs
*.log
nohup.out

# Temporary files
*.tmp
*.swp
*~

# Keep these directories but ignore contents
models/.gitkeep
results/.gitkeep
EOF
    echo "✓ .gitignore created"
else
    echo "✓ .gitignore already exists"
fi

# Create .gitkeep files for empty directories
mkdir -p models results/figures results/analysis
touch models/.gitkeep
touch results/.gitkeep
touch results/figures/.gitkeep
touch results/analysis/.gitkeep

# Stage all files
echo ""
echo "Staging files for commit..."
git add .

# Show status
echo ""
echo "Git status:"
git status --short

# Create initial commit
echo ""
read -p "Create initial commit? (yes/no): " response
if [ "$response" = "yes" ]; then
    git commit -m "Initial commit: SuperVision project

- DINOv3 + LoRA pipeline for Tc prediction
- ALIGNN pipeline for graph-based prediction
- Complete training, evaluation, and visualization scripts
- State-of-the-art results: MAE 4.85 K (DINOv3), 5.34 K (ALIGNN)
- Comprehensive README with full project documentation
"
    echo "✓ Initial commit created"
else
    echo "Skipping initial commit"
    exit 0
fi

# Set up GitHub remote
echo ""
echo "GitHub Repository Setup"
echo "----------------------"

if command -v gh &> /dev/null; then
    echo ""
    read -p "Create GitHub repository using 'gh' CLI? (yes/no): " use_gh
    if [ "$use_gh" = "yes" ]; then
        echo ""
        echo "Creating GitHub repository..."
        gh repo create "$REPO_NAME" --public --source=. --remote=origin --push \
            --description "Transfer learning for superconductor critical temperature prediction using DINOv3 and ALIGNN"

        echo "✓ Repository created and pushed to GitHub"
        echo ""
        echo "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    fi
else
    echo ""
    echo "Manual GitHub setup required:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Description: Transfer learning for superconductor critical temperature prediction"
    echo "4. Make it public"
    echo "5. Do NOT initialize with README, .gitignore, or license"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo "  git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
fi

echo ""
echo "=========================================="
echo "✓ GitHub setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit upload_to_huggingface.py and set your HF username"
echo "  2. Run: huggingface-cli login"
echo "  3. Run: python3 upload_to_huggingface.py"
echo "  4. Create GitHub release with model files (see create_github_release.sh)"
echo ""
