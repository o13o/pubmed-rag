# Phase C-1: FastAPI API Core + Endpoints

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing RAG pipeline through FastAPI endpoints (`POST /ask`, `POST /search`, `GET /health`). Add dependency injection via lifespan, CORS for React frontend.

**Architecture:** `src/api/` contains the FastAPI app. Lifespan manages Milvus connection, LLMClient, MeSHDatabase, Reranker. Routes consume these via `Depends()`. Existing `rag.chain.ask()` and `retrieval.search.search()` are called directly — no new business logic.

**Tech Stack:** FastAPI, uvicorn, pydantic (request/response schemas reuse `shared.models`)

**Spec:** [statements_jp.md](../../statements_jp.md) - 要件1「APIエンドポイントとして中核機能を公開」

**Prerequisites:** Phase A (RAG pipeline) + Phase B (reranker, guardrails) all merged.

---

## Chunk 1: App Factory + Dependencies

### Task 1: Add FastAPI/uvicorn dependencies

**Files:**
- Modify: `capstone/backend/pyproject.toml`

- [ ] **Step 1: Add fastapi and uvicorn to dependencies**

Add to the `dependencies` list in `pyproject.toml`:

```toml
"fastapi>=0.115",
"uvicorn[standard]>=0.34",
```

- [ ] **Step 2: Sync dependencies**

```bash
cd capstone/backend
uv sync
```

Expected: fastapi and uvicorn installed.

- [ ] **Step 3: Commit**

```bash
git add capstone/backend/pyproject.toml capstone/backend/uv.lock
git commit -m "deps: add fastapi and uvicorn"
```

---

### Task 2: Lifespan + App Factory

**Files:**
- Create: `capstone/backend/src/api/main.py`

- [ ] **Step 1: Write failing test for app creation**

Create: `capstone/backend/tests/unit/test_api_health.py`

```python
# tests/unit/test_api_health.py
"""Tests for API health endpoint and app factory."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked services."""
    with patch("src.api.main.connections") as mock_conn, \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient") as mock_llm_cls, \
         patch("src.api.main.MeSHDatabase") as mock_mesh_cls, \
         patch("src.api.main.get_reranker") as mock_reranker:

        mock_col_instance = MagicMock()
        mock_col_instance.num_entities = 1000
        mock_col.return_value = mock_col_instance

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_cors_headers(client):
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-origin") == "*"
```

- [ ] **Step 2: Implement app factory with lifespan**

```python
# src/api/main.py
"""FastAPI application factory with lifespan-managed services."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import Collection, connections

from src.api.routes import ask, health, search
from src.retrieval.reranker import get_reranker
from src.shared.config import get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown shared services."""
    settings = get_settings()

    # Startup
    connections.connect("default", host=settings.milvus_host, port=str(settings.milvus_port))
    collection = Collection(settings.milvus_collection)
    collection.load()

    llm = LLMClient(model=settings.llm_model, timeout=settings.llm_timeout)
    mesh_db = MeSHDatabase(settings.mesh_db_path)
    reranker = get_reranker(
        reranker_type=settings.reranker_type,
        model_name=settings.reranker_model,
        llm=llm if settings.reranker_type == "llm" else None,
    )

    app.state.collection = collection
    app.state.llm = llm
    app.state.mesh_db = mesh_db
    app.state.reranker = reranker
    app.state.settings = settings

    logger.info("API started: collection=%s", settings.milvus_collection)
    yield

    # Shutdown
    mesh_db.close()
    connections.disconnect("default")
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PubMed RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(search.router)

    return app


app = create_app()
```

- [ ] **Step 3: Create dependencies module**

Create: `capstone/backend/src/api/dependencies.py`

```python
# src/api/dependencies.py
"""FastAPI dependencies — extract services from app.state."""

from fastapi import Request
from pymilvus import Collection

from src.retrieval.reranker import BaseReranker
from src.shared.config import Settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase


def get_collection(request: Request) -> Collection:
    return request.app.state.collection


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_mesh_db(request: Request) -> MeSHDatabase:
    return request.app.state.mesh_db


def get_reranker_dep(request: Request) -> BaseReranker:
    return request.app.state.reranker


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
```

- [ ] **Step 4: Create routes `__init__.py`**

Create: `capstone/backend/src/api/routes/__init__.py`

```python
# Empty — routes are imported by main.py
```

- [ ] **Step 5: Run tests**

```bash
cd capstone/backend
uv run pytest tests/unit/test_api_health.py -v
```

Expected: FAIL (routes not yet implemented).

- [ ] **Step 6: Commit**

```bash
git add capstone/backend/src/api/
git commit -m "feat(api): add FastAPI app factory with lifespan and CORS"
```

---

## Chunk 2: Endpoints

### Task 3: GET /health

**Files:**
- Create: `capstone/backend/src/api/routes/health.py`

- [ ] **Step 1: Implement health endpoint**

```python
# src/api/routes/health.py
"""Health check endpoint."""

import logging

from fastapi import APIRouter, Depends
from pymilvus import Collection

from src.api.dependencies import get_collection

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(collection: Collection = Depends(get_collection)):
    try:
        count = collection.num_entities
        return {"status": "ok", "milvus_connected": True, "collection_count": count}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "degraded", "milvus_connected": False, "error": str(e)}
```

- [ ] **Step 2: Run health test**

```bash
cd capstone/backend
uv run pytest tests/unit/test_api_health.py::test_health_returns_ok -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add capstone/backend/src/api/routes/health.py
git commit -m "feat(api): add GET /health endpoint"
```

---

### Task 4: POST /search

**Files:**
- Create: `capstone/backend/src/api/routes/search.py`
- Create: `capstone/backend/tests/unit/test_api_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_api_search.py
"""Tests for POST /search endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import SearchResult


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


@patch("src.api.routes.search.search_milvus")
def test_search_returns_results(mock_search, client):
    mock_search.return_value = [
        SearchResult(
            pmid="123", title="Test", abstract_text="Abstract",
            score=0.95, year=2023, journal="Nature", mesh_terms=[],
        ),
    ]
    response = client.post("/search", json={"query": "cancer treatment"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["pmid"] == "123"


def test_search_requires_query(client):
    response = client.post("/search", json={})
    assert response.status_code == 422
```

- [ ] **Step 2: Implement search endpoint**

```python
# src/api/routes/search.py
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
```

- [ ] **Step 3: Run tests**

```bash
cd capstone/backend
uv run pytest tests/unit/test_api_search.py -v
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add capstone/backend/src/api/routes/search.py capstone/backend/tests/unit/test_api_search.py
git commit -m "feat(api): add POST /search endpoint"
```

---

### Task 5: POST /ask

**Files:**
- Create: `capstone/backend/src/api/routes/ask.py`
- Create: `capstone/backend/tests/unit/test_api_ask.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_api_ask.py
"""Tests for POST /ask endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import Citation, RAGResponse, ValidatedResponse


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


@patch("src.api.routes.ask.rag_ask")
def test_ask_returns_answer(mock_ask, client):
    mock_ask.return_value = ValidatedResponse(
        answer="Test answer with [PMID: 123].",
        citations=[Citation(pmid="123", title="Test", journal="Nature", year=2023, relevance_score=0.95)],
        query="test query",
        warnings=[],
        disclaimer="Disclaimer text.",
        is_grounded=True,
    )
    response = client.post("/ask", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["query"] == "test query"
    assert len(data["citations"]) == 1


def test_ask_requires_query(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422
```

- [ ] **Step 2: Implement ask endpoint**

```python
# src/api/routes/ask.py
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
```

- [ ] **Step 3: Run tests**

```bash
cd capstone/backend
uv run pytest tests/unit/test_api_ask.py -v
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add capstone/backend/src/api/routes/ask.py capstone/backend/tests/unit/test_api_ask.py
git commit -m "feat(api): add POST /ask RAG endpoint"
```

---

### Task 6: Update api `__init__.py` + Full test suite

**Files:**
- Modify: `capstone/backend/src/api/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
# src/api/__init__.py
"""API module — FastAPI application."""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
```

- [ ] **Step 2: Run full test suite**

```bash
cd capstone/backend
uv run pytest tests/unit/test_api_health.py tests/unit/test_api_search.py tests/unit/test_api_ask.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Manual E2E test (requires Milvus running + data ingested)**

```bash
cd capstone/backend
uv run uvicorn src.api.main:app --reload --port 8000
```

Test with curl:

```bash
# Health
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "breast cancer immunotherapy", "top_k": 3}'

# Ask
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are recent advances in breast cancer treatment?", "top_k": 5}'
```

- [ ] **Step 4: Commit**

```bash
git add capstone/backend/src/api/
git commit -m "feat(api): complete FastAPI endpoints (health, search, ask)"
```
