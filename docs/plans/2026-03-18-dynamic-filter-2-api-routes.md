# Dynamic Filtering — Plan 2: Backend API Routes

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `publication_types` and `mesh_categories` fields to all API request models and pass them through to `SearchFilters`.

**Architecture:** Add `list[str]` fields to `AskRequest`, `SearchRequest`, `ReviewRequest`. Pass through to every `SearchFilters()` constructor (6 call sites across 3 routes).

**Tech Stack:** Python, FastAPI, Pydantic, pytest

**Spec:** `docs/specs/2026-03-18-dynamic-filtering-design.md`

**Parallelism:** This plan has NO dependencies on Plans 1 or 3. Can run in parallel. `SearchFilters` already has `publication_types` and `mesh_categories` fields — the route tests mock the search client so they don't test filter expression generation.

---

### Task 1: Add fields to `SearchRequest` and test

**Files:**
- Modify: `backend/src/api/routes/search.py:13-19,32-38`
- Test: `backend/tests/unit/test_api_search.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/test_api_search.py`:

```python
@patch("src.retrieval.client.LocalSearchClient.search")
def test_search_passes_publication_types_filter(mock_search, client):
    mock_search.return_value = []
    response = client.post("/search", json={
        "query": "cancer",
        "publication_types": ["Randomized Controlled Trial", "Meta-Analysis"],
        "mesh_categories": ["Neoplasms"],
    })
    assert response.status_code == 200
    call_args = mock_search.call_args
    filters = call_args[0][1]  # second positional arg to search(query, filters)
    assert filters.publication_types == ["Randomized Controlled Trial", "Meta-Analysis"]
    assert filters.mesh_categories == ["Neoplasms"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_api_search.py::test_search_passes_publication_types_filter -v`
Expected: FAIL — `SearchRequest` does not accept `publication_types`.

- [ ] **Step 3: Add fields to `SearchRequest` and pass through**

Edit `backend/src/api/routes/search.py`:

```python
class SearchRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
```

Update the `SearchFilters` constructor in `search_endpoint`:

```python
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/unit/test_api_search.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/search.py backend/tests/unit/test_api_search.py
git commit -m "feat: add publication_types and mesh_categories to /search endpoint"
```

---

### Task 2: Add fields to `AskRequest` and test (sync + streaming)

**Files:**
- Modify: `backend/src/api/routes/ask.py:20-28,42-48,80-86`
- Test: `backend/tests/unit/test_api_ask.py`

- [ ] **Step 1: Write failing tests**

Add to `backend/tests/unit/test_api_ask.py`:

```python
@patch("src.api.routes.ask.rag_ask")
def test_ask_passes_publication_types_filter(mock_ask, client):
    mock_ask.return_value = ValidatedResponse(
        answer="Answer.", citations=[], query="test",
        warnings=[], disclaimer="Disclaimer.", is_grounded=True,
    )
    response = client.post("/ask", json={
        "query": "cancer",
        "publication_types": ["Review"],
        "mesh_categories": ["Neoplasms"],
    })
    assert response.status_code == 200
    call_kwargs = mock_ask.call_args
    filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
    assert filters.publication_types == ["Review"]
    assert filters.mesh_categories == ["Neoplasms"]


@patch("src.api.routes.ask.ask_stream")
def test_ask_stream_passes_publication_types_filter(mock_ask_stream, client):
    mock_ask_stream.return_value = iter([
        {"event": "done", "data": {"citations": [], "warnings": [], "disclaimer": "", "is_grounded": True}},
    ])
    response = client.post("/ask", json={
        "query": "cancer",
        "publication_types": ["Meta-Analysis"],
        "mesh_categories": ["Cardiovascular Diseases"],
        "stream": True,
    })
    assert response.status_code == 200
    call_kwargs = mock_ask_stream.call_args
    filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
    assert filters.publication_types == ["Meta-Analysis"]
    assert filters.mesh_categories == ["Cardiovascular Diseases"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_api_ask.py -v -k "publication_types"`
Expected: FAIL

- [ ] **Step 3: Add fields to `AskRequest` and update both `SearchFilters` constructors**

Edit `backend/src/api/routes/ask.py`:

```python
class AskRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
    guardrails_enabled: bool = True
    stream: bool = False
```

Update `_sse_generator` (lines 42-48):

```python
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
```

Update `ask_endpoint` (lines 80-86):

```python
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/unit/test_api_ask.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/routes/ask.py backend/tests/unit/test_api_ask.py
git commit -m "feat: add publication_types and mesh_categories to /ask endpoint"
```

---

### Task 3: Add fields to `ReviewRequest` and test

**Files:**
- Modify: `backend/src/api/routes/review.py:19-25,34-40`
- Test: `backend/tests/unit/test_api_review.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/unit/test_api_review.py`:

```python
@patch("src.api.routes.review.ReviewPipeline")
def test_review_passes_publication_types_filter(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _mock_review()
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={
        "query": "cancer",
        "publication_types": ["Systematic Review"],
        "mesh_categories": ["Neoplasms", "Cardiovascular Diseases"],
    })
    assert response.status_code == 200
    call_args = mock_pipeline.run.call_args
    filters = call_args[0][1]  # second positional arg
    assert filters.publication_types == ["Systematic Review"]
    assert filters.mesh_categories == ["Neoplasms", "Cardiovascular Diseases"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_api_review.py::test_review_passes_publication_types_filter -v`
Expected: FAIL

- [ ] **Step 3: Add fields to `ReviewRequest` and update `SearchFilters` constructor**

Edit `backend/src/api/routes/review.py`:

```python
class ReviewRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
```

Update the `SearchFilters` constructor in `review_endpoint`:

```python
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd backend && uv run pytest tests/unit/test_api_review.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full unit test suite**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add backend/src/api/routes/review.py backend/tests/unit/test_api_review.py
git commit -m "feat: add publication_types and mesh_categories to /review endpoint"
```
