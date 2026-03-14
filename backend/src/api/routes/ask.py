"""POST /ask — full RAG pipeline endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from pymilvus import Collection

from src.api.dependencies import get_collection, get_llm, get_mesh_db, get_reranker_dep
from src.rag.chain import ask as rag_ask
from src.retrieval.reranker import BaseReranker
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import Citation, GuardrailWarning, SearchFilters

router = APIRouter()


class AskRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
    guardrails_enabled: bool = True


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    query: str
    warnings: list[GuardrailWarning] = Field(default_factory=list)
    disclaimer: str = ""
    is_grounded: bool = True


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(
    req: AskRequest,
    collection: Collection = Depends(get_collection),
    llm: LLMClient = Depends(get_llm),
    mesh_db: MeSHDatabase = Depends(get_mesh_db),
    reranker: BaseReranker = Depends(get_reranker_dep),
):
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    response = rag_ask(
        query=req.query,
        collection=collection,
        llm=llm,
        mesh_db=mesh_db,
        filters=filters,
        reranker=reranker,
        guardrails_enabled=req.guardrails_enabled,
    )
    return AskResponse(**response.model_dump())
