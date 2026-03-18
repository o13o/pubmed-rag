"""Milvus vector search with metadata filtering.

Supports dense search (Phase A) and hybrid search (Phase B via search_mode parameter).
"""

import json
import logging

from openai import OpenAI
from pymilvus import AnnSearchRequest, Collection, RRFRanker

from src.shared.config import get_settings
from src.shared.models import SearchFilters, SearchResult

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = ["pmid", "title", "abstract_text", "year", "journal", "mesh_terms", "publication_types"]


def _get_openai_client() -> OpenAI:
    """Lazy OpenAI client initialization (avoids import-time OPENAI_API_KEY check)."""
    return OpenAI()


def _sanitize_like_value(value: str) -> str:
    """Strip characters that could alter like pattern or break expression syntax."""
    return value.replace("%", "").replace('"', "").replace("\\", "")


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
    if filters.publication_types:
        pt_clauses = [
            f'publication_types like "%{_sanitize_like_value(pt)}%"'
            for pt in filters.publication_types
            if _sanitize_like_value(pt).strip()
        ]
        if pt_clauses:
            conditions.append(f"({' or '.join(pt_clauses)})")
    if filters.mesh_categories:
        # mesh_categories filter maps to the mesh_terms Milvus field
        mc_clauses = [
            f'mesh_terms like "%{_sanitize_like_value(mc)}%"'
            for mc in filters.mesh_categories
            if _sanitize_like_value(mc).strip()
        ]
        if mc_clauses:
            conditions.append(f"({' or '.join(mc_clauses)})")

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
    For hybrid search, scores are RRF-fused (higher = more relevant).
    """
    results = []
    for hit in hits:
        mesh_raw = hit.entity.get("mesh_terms")
        mesh_terms = json.loads(mesh_raw) if isinstance(mesh_raw, str) else mesh_raw

        pt_raw = hit.entity.get("publication_types")
        publication_types = json.loads(pt_raw) if isinstance(pt_raw, str) else pt_raw

        results.append(
            SearchResult(
                pmid=hit.entity.get("pmid"),
                title=hit.entity.get("title"),
                abstract_text=hit.entity.get("abstract_text"),
                score=hit.distance,
                year=hit.entity.get("year"),
                journal=hit.entity.get("journal"),
                mesh_terms=mesh_terms if mesh_terms else [],
                publication_types=publication_types if publication_types else [],
            )
        )
    return results


def _resolve_search_mode(filters: SearchFilters) -> str:
    """Resolve search mode: per-query override > config default."""
    if filters.search_mode is not None:
        return filters.search_mode
    return get_settings().search_mode


def _dense_search(
    query_embedding: list[float],
    collection: Collection,
    filters: SearchFilters,
    filter_expr: str,
) -> list[SearchResult]:
    """Execute dense-only vector search."""
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
        output_fields=OUTPUT_FIELDS,
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])


def _hybrid_search(
    query: str,
    query_embedding: list[float],
    collection: Collection,
    filters: SearchFilters,
    filter_expr: str,
) -> list[SearchResult]:
    """Execute hybrid search (dense + BM25 via RRF fusion)."""
    dense_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
    )

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="chunk_text_sparse",
        param={"metric_type": "BM25"},
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
    )

    collection.load()
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(k=60),
        limit=filters.top_k,
        output_fields=OUTPUT_FIELDS,
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])


def search(
    query: str,
    collection: Collection,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Execute vector search against Milvus.

    Args:
        query: Natural language query (will be embedded).
        collection: Milvus collection to search.
        filters: Optional metadata filters. search_mode controls dense vs hybrid.

    Returns: List of SearchResult sorted by relevance.
    """
    if filters is None:
        filters = SearchFilters()

    query_embedding = embed_query(query)
    filter_expr = build_filter_expression(filters)
    mode = _resolve_search_mode(filters)

    if mode == "hybrid":
        logger.info("Executing hybrid search (dense + BM25)")
        return _hybrid_search(query, query_embedding, collection, filters, filter_expr)
    else:
        logger.info("Executing dense search")
        return _dense_search(query_embedding, collection, filters, filter_expr)
