"""Search client abstraction for monolith/microservice dual deployment.

LocalSearchClient: direct Milvus call (monolith mode).
RemoteSearchClient: HTTP call to Search Service (microservice mode).
"""

from __future__ import annotations

from typing import Protocol

import httpx
from pymilvus import Collection

from src.shared.models import SearchFilters, SearchResult


class SearchClient(Protocol):
    def search(self, query: str, filters: SearchFilters) -> list[SearchResult]: ...


class LocalSearchClient:
    """Monolith mode — calls Milvus directly."""

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    def search(self, query: str, filters: SearchFilters) -> list[SearchResult]:
        from src.retrieval.search import search

        return search(query, self._collection, filters)


class RemoteSearchClient:
    """Microservice mode — calls Search Service over HTTP."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def search(self, query: str, filters: SearchFilters) -> list[SearchResult]:
        resp = httpx.post(
            f"{self._base_url}/search",
            json={"query": query, **filters.model_dump(exclude_none=True)},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return [SearchResult(**r) for r in resp.json()["results"]]
