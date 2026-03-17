# Review Pipeline — Plan B: API Route

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /review` endpoint that wires ReviewPipeline to the FastAPI app.

**Architecture:** Single route file following the existing `/analyze` pattern. Uses FastAPI dependency injection for `SearchClient` and `LLMClient`. ReviewPipeline (from Plan A) does all orchestration.

**Tech Stack:** FastAPI, Pydantic, pytest, TestClient

**Depends on:** Plan A (backend core) must be completed first.

**Parallel with:** Plan C (frontend) — Plan C depends on this plan.

---

### Task 1: Create `/review` route and register it

**Files:**
- Create: `backend/src/api/routes/review.py`
- Modify: `backend/src/api/main.py`
- Test: `backend/tests/unit/test_api_review.py`

- [ ] **Step 1: Write the tests**

Create `backend/tests/unit/test_api_review.py`:

```python
"""Tests for POST /review endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import (
    AgentResult,
    Citation,
    LiteratureReview,
    SearchResult,
)


@pytest.fixture
def client():
    with patch("src.api.main.connections"), \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient"), \
         patch("src.api.main.MeSHDatabase"), \
         patch("src.api.main.get_reranker"):

        mock_col.return_value = MagicMock(num_entities=100)

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


def _mock_review():
    return LiteratureReview(
        query="test query",
        overview="Overview text",
        main_findings="Findings text",
        gaps_and_conflicts="Gaps text",
        recommendations="Recs text",
        citations=[Citation(pmid="123", title="Test", journal="J", year=2023, relevance_score=0.9)],
        search_results=[SearchResult(pmid="123", title="Test", abstract_text="Abstract", score=0.9, year=2023, journal="J", mesh_terms=[])],
        agent_results=[AgentResult(agent_name="test", summary="ok", findings=[], confidence=0.8)],
        agents_succeeded=1,
        agents_failed=0,
    )


@patch("src.api.routes.review.ReviewPipeline")
def test_review_returns_literature_review(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _mock_review()
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["overview"] == "Overview text"
    assert len(data["citations"]) == 1
    assert data["agents_succeeded"] == 1
    mock_pipeline.run.assert_called_once()


@patch("src.api.routes.review.ReviewPipeline")
def test_review_passes_filters(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _mock_review()
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={
        "query": "cancer",
        "year_min": 2020,
        "year_max": 2025,
        "top_k": 5,
    })
    assert response.status_code == 200
    call_args = mock_pipeline.run.call_args
    filters = call_args[0][1]  # second positional arg
    assert filters.year_min == 2020
    assert filters.year_max == 2025
    assert filters.top_k == 5


def test_review_requires_query(client):
    response = client.post("/review", json={})
    assert response.status_code == 422


@patch("src.api.routes.review.ReviewPipeline")
def test_review_empty_results_returns_404(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = ValueError("No results found for query: test")
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test"})
    assert response.status_code == 404
    assert "No results" in response.json()["detail"]


@patch("src.api.routes.review.ReviewPipeline")
def test_review_synthesizer_error_returns_502(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = RuntimeError("LLM API error")
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test"})
    assert response.status_code == 502
    assert response.json()["detail"]  # non-empty error detail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_api_review.py -v`
Expected: FAIL — module `src.api.routes.review` not found

- [ ] **Step 3: Create the route file**

Create `backend/src/api/routes/review.py`:

```python
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
```

- [ ] **Step 4: Register the router in main.py**

Add to `backend/src/api/main.py`:

In the imports section (line 10), add `review` to the import:
```python
from src.api.routes import analyze, ask, health, review, search, transcribe
```

After line 81 (`app.include_router(transcribe.router)`), add:
```python
    app.include_router(review.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_api_review.py -v`
Expected: 5 PASS

- [ ] **Step 6: Run all unit tests to check for regressions**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add backend/src/api/routes/review.py backend/src/api/main.py backend/tests/unit/test_api_review.py
git commit -m "feat: add POST /review endpoint for literature review pipeline"
```
