"""POST /review — literature review pipeline endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.agents.pipeline import ReviewPipeline
from src.api.dependencies import get_llm, get_search_client
from src.retrieval.client import SearchClient
from src.shared.llm import LLMClient
from src.shared.models import LiteratureReview, SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter()


class ReviewRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None


@router.post("/review", response_model=LiteratureReview)
def review_endpoint(
    req: ReviewRequest,
    llm: LLMClient = Depends(get_llm),
    search_client: SearchClient = Depends(get_search_client),
):
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    pipeline = ReviewPipeline(search_client=search_client, llm=llm)

    try:
        return pipeline.run(req.query, filters)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Review pipeline failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
