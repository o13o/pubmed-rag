"""POST /search — vector search without RAG generation."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_search_client
from src.retrieval.client import SearchClient
from src.shared.models import SearchFilters, SearchResult

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    req: SearchRequest,
    search_client: SearchClient = Depends(get_search_client),
):
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    results = search_client.search(req.query, filters)
    return SearchResponse(results=results, total=len(results))
