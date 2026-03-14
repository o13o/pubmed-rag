"""POST /ask — full RAG pipeline endpoint with optional SSE streaming."""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pymilvus import Collection

from src.api.dependencies import get_collection, get_llm, get_mesh_db, get_reranker_dep
from src.rag.chain import ask as rag_ask, ask_stream
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
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    query: str
    warnings: list[GuardrailWarning] = Field(default_factory=list)
    disclaimer: str = ""
    is_grounded: bool = True


def _sse_generator(req, collection, llm, mesh_db, reranker):
    """Format ask_stream() events as SSE wire format."""
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    for event in ask_stream(
        query=req.query,
        collection=collection,
        llm=llm,
        mesh_db=mesh_db,
        filters=filters,
        reranker=reranker,
        guardrails_enabled=req.guardrails_enabled,
    ):
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"


@router.post("/ask")
def ask_endpoint(
    req: AskRequest,
    collection: Collection = Depends(get_collection),
    llm: LLMClient = Depends(get_llm),
    mesh_db: MeSHDatabase = Depends(get_mesh_db),
    reranker: BaseReranker = Depends(get_reranker_dep),
):
    if req.stream:
        return StreamingResponse(
            _sse_generator(req, collection, llm, mesh_db, reranker),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

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
