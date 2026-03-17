#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# migrate-repo.sh — Extract capstone/ into standalone pubmed-rag repo
#
# Usage:
#   chmod +x capstone/scripts/migrate-repo.sh
#   ./capstone/scripts/migrate-repo.sh
#
# Prerequisites:
#   brew install git-filter-repo   (or pip install git-filter-repo)
#
# This script does NOT modify your current working copy.
# It creates a fresh clone at ~/pubmed-rag.
# ============================================================

ORIGIN_URL="https://github.com/o13o/fde-training.git"
TARGET_DIR="$HOME/pubmed-rag"
TEMP_DIR=$(mktemp -d)
BRANCH="main"

echo "=== Step 0: Preflight checks ==="
if ! command -v git-filter-repo &>/dev/null; then
    echo "ERROR: git-filter-repo not found."
    echo "Install with: brew install git-filter-repo"
    exit 1
fi

if [ -d "$TARGET_DIR" ]; then
    echo "ERROR: $TARGET_DIR already exists. Remove or rename it first."
    exit 1
fi

echo "=== Step 1: Fresh clone into temp directory ==="
git clone --no-local "$ORIGIN_URL" "$TEMP_DIR"
cd "$TEMP_DIR"
git checkout "$BRANCH"

echo "=== Step 2: Extract capstone/ subdirectory (rewrite history) ==="
# This rewrites all commits to only include files under capstone/
# and removes the capstone/ prefix from all paths.
git filter-repo --subdirectory-filter capstone --force

echo "=== Step 3: Generate clean .gitignore ==="
cat > .gitignore << 'GITIGNORE'
# Python
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Node
node_modules/

# Environment
.env

# Data files (large, generated)
*.csv
*.jsonl
*.parquet
*.bin
*.pickle
*.db
*.index

# Keep tracked data/config JSON files
!frontend/package.json
!frontend/package-lock.json
!frontend/tsconfig.json
!frontend/tsconfig.app.json
!frontend/tsconfig.node.json
!backend/tests/eval/dataset.json

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store

# Logs
*.log

# DeepEval
backend/.deepeval/

# Archives
*.zip

# PDF
*.pdf

# Milvus data
volumes/
GITIGNORE

git add .gitignore
git commit -m "chore: generate clean .gitignore for standalone repo"

echo "=== Step 4: Move to target directory ==="
cd "$HOME"
mv "$TEMP_DIR" "$TARGET_DIR"

echo ""
echo "=== Done! ==="
echo "Repository ready at: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  cd $TARGET_DIR"
echo "  git log --oneline | head -10   # verify history"
echo "  ls                              # verify structure"
echo ""
echo "To push to GitHub:"
echo "  gh repo create o13o/pubmed-rag --public --source $TARGET_DIR --push"
echo "  # or:"
echo "  git remote add origin git@github.com:o13o/pubmed-rag.git"
echo "  git push -u origin main"
