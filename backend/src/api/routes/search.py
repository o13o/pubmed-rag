"""POST /search — vector search without RAG generation."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from pymilvus import Collection

from src.api.dependencies import get_collection
from src.retrieval.search import search as search_milvus
from src.shared.models import SearchFilters, SearchResult

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    req: SearchRequest,
    collection: Collection = Depends(get_collection),
):
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    results = search_milvus(req.query, collection, filters)
    return SearchResponse(results=results, total=len(results))
