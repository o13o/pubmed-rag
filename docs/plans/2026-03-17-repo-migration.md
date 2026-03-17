# Repository Migration Plan: capstone/ → pubmed-rag

## Overview

Extract `capstone/` from `fde-training` into a standalone GitHub repository `pubmed-rag`, preserving git history.

## Prerequisites

```bash
# Install git-filter-repo (one-time)
brew install git-filter-repo
# or: pip install git-filter-repo
```

## Migration Script

A migration script is provided at `capstone/scripts/migrate-repo.sh`. Run it from any directory — it creates a fresh clone and does not modify your working copy.

## What the Script Does

1. **Fresh clone** — Clones `fde-training` into a temp directory (avoids touching your working copy)
2. **Extract capstone/** — `git filter-repo --subdirectory-filter capstone` rewrites history to only include commits touching `capstone/`, and removes the `capstone/` prefix from all paths
3. **Fix .gitignore** — The root `.gitignore` in `fde-training` has broad rules (e.g. `*.json`, `*.csv`) that are specific to the training repo. The script generates a clean `.gitignore` suited for the standalone project
4. **Output** — A ready-to-push local repo at `~/pubmed-rag`

## Post-Script Manual Steps

1. **Create the GitHub repo**

   ```bash
   gh repo create o13o/pubmed-rag --public --source ~/pubmed-rag --push
   # or create via github.com UI, then:
   cd ~/pubmed-rag
   git remote add origin git@github.com:o13o/pubmed-rag.git
   git push -u origin main
   ```

2. **Verify**
   - `git log --oneline | wc -l` — should show ~143 commits (capstone-related)
   - Check that paths no longer have `capstone/` prefix
   - `backend/`, `frontend/`, `loadtest/`, `docs/` should be at the root

3. **Update fde-training** (optional)
   - Remove `capstone/` directory from `fde-training`
   - Add a note in `fde-training` README pointing to the new repo

## Directory Structure After Migration

```
pubmed-rag/               # was capstone/
├── backend/
│   ├── src/
│   ├── tests/
│   ├── prompts/
│   └── pyproject.toml
├── frontend/
├── loadtest/
├── docs/
├── docker-compose.yml
├── .gitignore            # clean, project-specific
└── README.md
```

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Working copy corruption | Script uses a fresh clone, never modifies your working directory |
| Lost commits | `filter-repo` only keeps commits that touch `capstone/`; commits that only touch root `.gitignore` may be lost. Acceptable — those are trivial |
| Broken .gitignore | Script generates a new clean `.gitignore` for the standalone project |
| Submodule/CI references | Not applicable (no submodules, no CI currently) |
