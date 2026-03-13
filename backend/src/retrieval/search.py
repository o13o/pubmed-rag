"""Milvus vector search with metadata filtering.

Supports dense search (Phase A) and hybrid search (Phase B via search_mode parameter).
"""

import json
import logging

from openai import OpenAI
from pymilvus import Collection

from src.shared.config import get_settings
from src.shared.models import SearchFilters, SearchResult

logger = logging.getLogger(__name__)


def _get_openai_client() -> OpenAI:
    """Lazy OpenAI client initialization (avoids import-time OPENAI_API_KEY check)."""
    return OpenAI()


def build_filter_expression(filters: SearchFilters) -> str:
    """Build a Milvus boolean filter expression from SearchFilters."""
    conditions = []

    if filters.year_min is not None:
        conditions.append(f"year >= {filters.year_min}")
    if filters.year_max is not None:
        conditions.append(f"year <= {filters.year_max}")
    if filters.journals:
        journals_str = json.dumps(filters.journals)
        conditions.append(f"journal in {journals_str}")

    return " and ".join(conditions)


def embed_query(query: str) -> list[float]:
    """Embed a query string using the configured embedding model."""
    settings = get_settings()
    client = _get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


def parse_search_results(hits: list) -> list[SearchResult]:
    """Convert raw Milvus hits into SearchResult models.

    Milvus COSINE metric: distance = cosine similarity (0-1, higher = more similar).
    """
    results = []
    for hit in hits:
        mesh_raw = hit.entity.get("mesh_terms")
        mesh_terms = json.loads(mesh_raw) if isinstance(mesh_raw, str) else mesh_raw

        results.append(
            SearchResult(
                pmid=hit.entity.get("pmid"),
                title=hit.entity.get("title"),
                abstract_text=hit.entity.get("abstract_text"),
                score=hit.distance,
                year=hit.entity.get("year"),
                journal=hit.entity.get("journal"),
                mesh_terms=mesh_terms if mesh_terms else [],
            )
        )
    return results


def search(
    query: str,
    collection: Collection,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Execute vector search against Milvus.

    Args:
        query: Natural language query (will be embedded).
        collection: Milvus collection to search.
        filters: Optional metadata filters.

    Returns: List of SearchResult sorted by relevance (cosine similarity).
    """
    if filters is None:
        filters = SearchFilters()

    query_embedding = embed_query(query)
    filter_expr = build_filter_expression(filters)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
        output_fields=["pmid", "title", "abstract_text", "year", "journal", "mesh_terms"],
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])
