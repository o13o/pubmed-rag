"""Reranker stub for Phase B (cross-encoder)."""

from src.shared.models import SearchResult


def rerank(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Rerank search results. Phase A: passthrough. Phase B: cross-encoder."""
    return results
