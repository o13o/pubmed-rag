"""Bulk ingestion script for 100k PubMed records.

Usage:
    cd backend
    uv run python scripts/ingest_bulk.py ../../data_pipeline/data/processed/sampled.jsonl

Features:
    - Streams JSONL in batches (no full-file memory load)
    - Progress bar with ETA
    - Checkpoint file for resumption on failure
    - Truncates oversized fields to fit Milvus schema
    - Recreates collection with Phase B schema (BM25)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from openai import OpenAI
from pymilvus import Collection, connections

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.milvus_setup import create_collection
from src.shared.models import Article

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_bulk")

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # Records per API call and Milvus upsert
MAX_RETRIES = 5

# Milvus schema max_length limits
FIELD_LIMITS = {
    "title": 2000,
    "abstract_text": 10000,
    "chunk_text": 12000,
    "journal": 500,
    "authors": 5000,
    "mesh_terms": 5000,
    "publication_types": 2000,
    "keywords": 5000,
}


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def make_chunk_text(title: str, abstract: str, mesh_terms: list[str]) -> str:
    parts = [f"Title: {title}", f"Abstract: {abstract}"]
    if mesh_terms:
        parts.append(f"MeSH: {'; '.join(mesh_terms)}")
    return "\n".join(parts)


def parse_record(raw: dict) -> dict | None:
    """Parse a raw JSONL record into a Milvus-ready dict (minus embedding)."""
    abstract = raw.get("abstract", "").strip()
    if not abstract:
        return None

    pmid = str(raw["pmid"])
    title = raw.get("title", "")
    authors = json.dumps(raw.get("authors", []))
    mesh_terms_list = raw.get("mesh_terms", [])
    mesh_terms = json.dumps(mesh_terms_list)
    keywords = json.dumps(raw.get("keywords", []))
    publication_types = json.dumps(raw.get("publication_types", []))
    journal = raw.get("journal", "")
    chunk_text = make_chunk_text(title, abstract, mesh_terms_list)

    pub_date = raw.get("publication_date", "")
    try:
        year = int(pub_date[:4]) if len(pub_date) >= 4 else 0
    except ValueError:
        year = 0

    return {
        "pmid": pmid,
        "title": truncate(title, FIELD_LIMITS["title"]),
        "abstract_text": truncate(abstract, FIELD_LIMITS["abstract_text"]),
        "chunk_text": truncate(chunk_text, FIELD_LIMITS["chunk_text"]),
        "year": year,
        "journal": truncate(journal, FIELD_LIMITS["journal"]),
        "authors": truncate(authors, FIELD_LIMITS["authors"]),
        "mesh_terms": truncate(mesh_terms, FIELD_LIMITS["mesh_terms"]),
        "keywords": truncate(keywords, FIELD_LIMITS["keywords"]),
        "publication_types": truncate(publication_types, FIELD_LIMITS["publication_types"]),
    }


def load_checkpoint(checkpoint_path: Path) -> int:
    if checkpoint_path.exists():
        return int(checkpoint_path.read_text().strip())
    return 0


def save_checkpoint(checkpoint_path: Path, line_num: int):
    checkpoint_path.write_text(str(line_num))


def generate_embeddings_batch(
    client: OpenAI, texts: list[str],
) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = min(2 ** attempt, 30)
                logger.warning("Embedding error (attempt %d/%d): %s. Retry in %ds",
                               attempt + 1, MAX_RETRIES, e, wait)
                time.sleep(wait)
            else:
                raise


def upsert_batch(collection: Collection, records: list[dict], embeddings: list[list[float]]):
    data = []
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb
        data.append(rec)
    collection.upsert(data)


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest PubMed JSONL into Milvus")
    parser.add_argument("source", type=Path, help="Path to JSONL file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--no-recreate", action="store_true",
                        help="Don't recreate collection (append to existing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    source: Path = args.source
    if not source.exists():
        logger.error("File not found: %s", source)
        sys.exit(1)

    checkpoint_path = source.with_suffix(".checkpoint")

    # Count total lines for progress
    total_lines = count_lines(source)
    logger.info("Source: %s (%d lines)", source, total_lines)

    # Resume support
    skip_lines = 0
    if args.resume:
        skip_lines = load_checkpoint(checkpoint_path)
        if skip_lines > 0:
            logger.info("Resuming from line %d", skip_lines)

    # Setup Milvus
    recreate = not args.no_recreate and skip_lines == 0
    if recreate:
        logger.info("Recreating collection with Phase B schema...")
    collection = create_collection(host=args.host, port=args.port, recreate=recreate)
    collection.load()

    # Setup OpenAI
    client = OpenAI()

    batch: list[dict] = []
    processed = skip_lines
    upserted_total = 0
    skipped = 0
    start_time = time.time()

    with open(source, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= skip_lines:
                continue

            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            record = parse_record(raw)
            if record is None:
                skipped += 1
                continue

            batch.append(record)

            if len(batch) >= args.batch_size:
                # Embed and upsert
                texts = [r["chunk_text"] for r in batch]
                embeddings = generate_embeddings_batch(client, texts)
                upsert_batch(collection, batch, embeddings)

                upserted_total += len(batch)
                processed = line_num
                save_checkpoint(checkpoint_path, processed)

                elapsed = time.time() - start_time
                rate = upserted_total / elapsed if elapsed > 0 else 0
                remaining = (total_lines - processed) / rate if rate > 0 else 0
                pct = processed / total_lines * 100

                logger.info(
                    "[%5.1f%%] %d/%d | upserted: %d | %.1f rec/s | ETA: %dm%02ds",
                    pct, processed, total_lines, upserted_total,
                    rate, remaining // 60, remaining % 60,
                )

                batch = []

    # Final batch
    if batch:
        texts = [r["chunk_text"] for r in batch]
        embeddings = generate_embeddings_batch(client, texts)
        upsert_batch(collection, batch, embeddings)
        upserted_total += len(batch)
        processed += len(batch)
        save_checkpoint(checkpoint_path, processed)

    elapsed = time.time() - start_time

    # Flush to ensure data is persisted
    collection.flush()

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("  Total lines: %d", total_lines)
    logger.info("  Upserted:    %d", upserted_total)
    logger.info("  Skipped:     %d", skipped)
    logger.info("  Time:        %.1fs (%.1f rec/s)", elapsed, upserted_total / elapsed if elapsed > 0 else 0)
    logger.info("=" * 60)

    # Clean up checkpoint on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed (ingestion completed successfully)")


if __name__ == "__main__":
    main()
