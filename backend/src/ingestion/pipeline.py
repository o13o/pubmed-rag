"""Ingestion pipeline: JSONL -> Article -> Chunk -> Embed -> Milvus."""

import logging
from pathlib import Path

from pymilvus import Collection

from src.ingestion.chunker import chunk_article
from src.ingestion.embedder import generate_embeddings, upsert_chunks
from src.ingestion.loader import load_articles
from src.shared.models import IngestReport

logger = logging.getLogger(__name__)


def ingest(source_path: Path, collection: Collection, batch_size: int = 100) -> IngestReport:
    logger.info("Starting ingestion from %s", source_path)

    articles = load_articles(source_path)
    chunks = [chunk_article(a) for a in articles]
    texts = [c.chunk_text for c in chunks]
    embeddings = generate_embeddings(texts, batch_size=batch_size)
    upserted = upsert_chunks(collection, chunks, embeddings)

    report = IngestReport(
        total_articles=len(articles),
        total_chunks=len(chunks),
        upserted=upserted,
        source_path=str(source_path),
    )
    logger.info("Ingestion complete: %s", report)
    return report
