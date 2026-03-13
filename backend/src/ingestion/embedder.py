"""Generate embeddings and upsert into Milvus."""

import json
import logging
import time

from openai import OpenAI
from pymilvus import Collection

from src.shared.models import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


def _get_openai_client() -> OpenAI:
    """Lazy OpenAI client initialization."""
    return OpenAI()


def generate_embeddings(
    texts: list[str],
    batch_size: int = 100,
    max_retries: int = 3,
) -> list[list[float]]:
    client = _get_openai_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in response.data])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Embedding API error (attempt %d): %s. Retrying in %ds...", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    raise

    return all_embeddings


def upsert_chunks(
    collection: Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    data = [
        {
            "pmid": chunk.pmid,
            "embedding": emb,
            "title": chunk.title,
            "abstract_text": chunk.abstract_text,
            "chunk_text": chunk.chunk_text,
            "year": chunk.year,
            "journal": chunk.journal,
            "authors": chunk.authors,
            "mesh_terms": chunk.mesh_terms,
            "publication_types": chunk.publication_types,
            "keywords": chunk.keywords,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    result = collection.upsert(data)
    logger.info("Upserted %d chunks into Milvus", len(data))
    return result.upsert_count
