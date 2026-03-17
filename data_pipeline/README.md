# PubMed Pipeline (Playground)

Toy implementation of the PubMed data download & sampling pipeline.
See `docs/specs/2026-03-13-pubmed-download-pipeline-design.md` for full design.

## Setup

```bash
pip install pyyaml
```

## Usage

### 1. Download from NLM FTP Baseline

```bash
# Download 1 baseline file from a recent range (file 1200 has 2023 data)
python download_hf.py --start 1199 --limit 1

# Download 5 files from the recent range
python download_hf.py --start 1195 --limit 5

# Full download (warning: 1334 files, very slow)
python download_hf.py
```

Output: `data/raw/pubmed_raw.jsonl`

### 2. Filter & Sample

```bash
python sample.py
```

Output:
- `data/processed/sampled.jsonl` — Sampled records
- `data/processed/audit_log.json` — Audit log with per-year/per-category stats

### Configuration

Edit `config.yaml` to adjust:
- `sampling.n_max` — Total sample size (default: 100,000)
- `sampling.seed` — Random seed for reproducibility
- `sampling.min_coverage.per_category_per_year` — MeSH category minimum

### Running Tests

```bash
python -m pytest tests/ -v
```
