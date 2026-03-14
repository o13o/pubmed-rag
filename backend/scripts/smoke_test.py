"""Smoke test: ingest 100 articles → search → verify results.

Usage:
    cd capstone/backend
    uv run python scripts/smoke_test.py
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Step 0: Paths ---
SAMPLED_JSONL = Path(__file__).resolve().parent.parent.parent / "playground/pubmed_pipeline/data/processed/sampled.jsonl"
N = 100

# --- Step 1: Create temp file with first N lines ---
logger.info("=== Step 1: Preparing %d articles from %s ===", N, SAMPLED_JSONL)
with open(SAMPLED_JSONL) as f:
    lines = [next(f) for _ in range(N)]
logger.info("Read %d lines", len(lines))

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
tmp.writelines(lines)
tmp.flush()
tmp_path = Path(tmp.name)
logger.info("Temp file: %s", tmp_path)

# --- Step 2: Create/recreate collection ---
logger.info("=== Step 2: Setting up Milvus collection ===")
from src.ingestion.milvus_setup import create_collection

collection = create_collection(recreate=True)
logger.info("Collection '%s' ready, fields: %s", collection.name, [f.name for f in collection.schema.fields])

# --- Step 3: Ingest ---
logger.info("=== Step 3: Ingesting %d articles ===", N)
from src.ingestion.pipeline import ingest

report = ingest(tmp_path, collection, batch_size=50)
logger.info("Ingest report: %s", report)

# --- Step 4: Verify count ---
collection.flush()
collection.load()
count = collection.num_entities
logger.info("=== Step 4: Collection has %d entities ===", count)
assert count > 0, f"Expected >0 entities, got {count}"

# --- Step 5: Dense search ---
logger.info("=== Step 5: Dense search test ===")
from src.retrieval.search import search
from src.shared.models import SearchFilters

query = "cancer treatment immunotherapy"
results = search(query, collection, SearchFilters(top_k=5, search_mode="dense"))
logger.info("Query: '%s'", query)
logger.info("Results: %d", len(results))
for i, r in enumerate(results):
    logger.info("  [%d] PMID=%s score=%.4f title='%s'", i + 1, r.pmid, r.score, r.title[:80])

assert len(results) > 0, "Dense search returned 0 results"

# --- Step 6: Hybrid search ---
logger.info("=== Step 6: Hybrid search test ===")
results_hybrid = search(query, collection, SearchFilters(top_k=5, search_mode="hybrid"))
logger.info("Hybrid results: %d", len(results_hybrid))
for i, r in enumerate(results_hybrid):
    logger.info("  [%d] PMID=%s score=%.4f title='%s'", i + 1, r.pmid, r.score, r.title[:80])

assert len(results_hybrid) > 0, "Hybrid search returned 0 results"

# --- Step 7: Metadata filter ---
logger.info("=== Step 7: Search with year filter ===")
# Pick a year that exists in our data
first_record = json.loads(lines[0])
sample_year = int(first_record.get("publication_date", "2021")[:4])
results_filtered = search(query, collection, SearchFilters(top_k=5, year_min=sample_year, search_mode="dense"))
logger.info("Filtered (year >= %d): %d results", sample_year, len(results_filtered))
for r in results_filtered:
    logger.info("  PMID=%s year=%d title='%s'", r.pmid, r.year, r.title[:60])

# --- Done ---
logger.info("=== All smoke tests passed! ===")

# Cleanup temp file
tmp_path.unlink()
