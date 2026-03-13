# PubMed Pipeline (Playground)

Toy implementation of the PubMed data download & sampling pipeline.
See `capstone/docs/specs/2026-03-13-pubmed-download-pipeline-design.md` for full design.

## Setup

```bash
pip install datasets pyyaml
```

## Usage

### 1. Download from HuggingFace

```bash
# Small test (first 10,000 records)
python download_hf.py --limit 10000

# Full download (warning: slow, large output)
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
