# SSE Streaming — Backend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SSE streaming support to the `/ask` endpoint backend so LLM tokens stream incrementally.

**Architecture:** Add `complete_stream()` to `LLMClient`, `ask_stream()` generator to `chain.py`, and a `stream` parameter to the `/ask` route that returns `StreamingResponse` with SSE-formatted events.

**Tech Stack:** LiteLLM (streaming), FastAPI `StreamingResponse`, SSE wire format

**Spec:** `docs/specs/2026-03-15-sse-streaming-design.md`

All file paths relative to `backend/`.

---

## Chunk 1: LLMClient.complete_stream()

### Task 1: Add `complete_stream()` test

**Files:**
- Modify: `tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_llm.py`:

```python
def test_llm_client_complete_stream():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        # Simulate streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]
        chunk3.usage = MagicMock(total_tokens=25)

        mock_completion.return_value = iter([chunk1, chunk2, chunk3])

        client = LLMClient(model="gpt-4o-mini")
        chunks = list(client.complete_stream(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        ))

    assert chunks == ["Hello", " world"]
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}


def test_llm_client_complete_stream_handles_empty_choices():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        chunk_empty = MagicMock()
        chunk_empty.choices = []
        chunk_empty.usage = None

        chunk_normal = MagicMock()
        chunk_normal.choices = [MagicMock(delta=MagicMock(content="ok"))]
        chunk_normal.usage = None

        mock_completion.return_value = iter([chunk_empty, chunk_normal])

        client = LLMClient()
        chunks = list(client.complete_stream(
            system_prompt="sys", user_prompt="usr",
        ))

    assert chunks == ["ok"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_llm.py::test_llm_client_complete_stream -v`
Expected: FAIL — `LLMClient` has no `complete_stream` method

- [ ] **Step 3: Implement `complete_stream()`**

Add to `src/shared/llm.py`, after the existing `complete()` method:

```python
from collections.abc import Generator

# Add Generator import at the top of the file

class LLMClient:
    # ... existing __init__ and complete() ...

    def complete_stream(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Yield text chunks from LLM streaming response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            delta = (
                getattr(chunk.choices[0].delta, "content", None)
                if chunk.choices
                else None
            )
            if delta:
                yield delta
            if hasattr(chunk, "usage") and chunk.usage:
                logger.debug(
                    "LLM stream: model=%s, tokens=%d",
                    self.model,
                    chunk.usage.total_tokens,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_llm.py -v`
Expected: ALL PASS (existing tests + 2 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/shared/llm.py tests/unit/test_llm.py
git commit -m "feat(backend): add LLMClient.complete_stream() for SSE support"
```

---

## Chunk 2: ask_stream() generator

### Task 2: Add `ask_stream()` test and implementation

**Files:**
- Modify: `src/rag/chain.py`
- Modify: `tests/unit/test_chain.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_chain.py`:

```python
from src.rag.chain import ask, ask_stream


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_stream_yields_token_and_done_events(mock_expander_cls, mock_search):
    """ask_stream() should yield token events then a done event."""
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete_stream.return_value = iter(["Based on ", "research..."])
    # Guardrail validation LLM call
    mock_llm.complete.return_value = "[]"

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    events = list(ask_stream(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=True,
    ))

    # Should have 2 token events + 1 done event
    token_events = [e for e in events if e["event"] == "token"]
    done_events = [e for e in events if e["event"] == "done"]

    assert len(token_events) == 2
    assert token_events[0]["data"]["text"] == "Based on "
    assert token_events[1]["data"]["text"] == "research..."

    assert len(done_events) == 1
    done_data = done_events[0]["data"]
    assert "citations" in done_data
    assert "warnings" in done_data
    assert "disclaimer" in done_data
    assert "is_grounded" in done_data
    assert len(done_data["citations"]) == 1


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_stream_without_guardrails(mock_expander_cls, mock_search):
    """ask_stream() without guardrails should still yield done with empty warnings."""
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete_stream.return_value = iter(["answer"])

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    events = list(ask_stream(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=False,
    ))

    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["data"]["warnings"] == []
    assert done_events[0]["data"]["disclaimer"] == ""


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_stream_yields_error_on_exception(mock_expander_cls, mock_search):
    """ask_stream() should yield an error event if an exception occurs."""
    mock_search.side_effect = RuntimeError("Milvus connection lost")
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="test")
    mock_expander_cls.return_value = mock_expander

    events = list(ask_stream(
        query="test",
        collection=MagicMock(),
        llm=MagicMock(),
        mesh_db=MagicMock(),
    ))

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "Milvus connection lost" in events[0]["data"]["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_chain.py::test_ask_stream_yields_token_and_done_events -v`
Expected: FAIL — `ask_stream` cannot be imported

- [ ] **Step 3: Implement `ask_stream()`**

Add to `src/rag/chain.py`, after the existing `ask()` function. Add `Generator` import at top:

```python
from collections.abc import Generator

# ... existing imports ...

def ask_stream(
    query: str,
    collection: Collection,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
    reranker: BaseReranker | None = None,
    guardrails_enabled: bool = True,
) -> Generator[dict, None, None]:
    """Execute the RAG pipeline with streaming LLM output.

    Yields dicts with 'event' and 'data' keys:
    - {"event": "token", "data": {"text": "..."}} for each LLM chunk
    - {"event": "done", "data": {...}} with citations and guardrail results
    - {"event": "error", "data": {"message": "..."}} on failure
    """
    try:
        if filters is None:
            filters = SearchFilters()
        if reranker is None:
            reranker = NoOpReranker()

        # 1. Query expansion
        expander = QueryExpander(llm=llm, mesh_db=mesh_db)
        expanded = expander.expand(query)
        logger.info("Expanded query: '%s' → '%s'", query, expanded.expanded_query)

        # 2. Search
        results = search(expanded.expanded_query, collection, filters)
        logger.info("Retrieved %d results", len(results))

        # 3. Rerank
        results = reranker.rerank(query, results, top_k=filters.top_k)
        logger.info("After reranking: %d results", len(results))

        # 4. Build prompt
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, results)

        # 5. Stream LLM tokens
        full_answer = ""
        for chunk in llm.complete_stream(system_prompt=system_prompt, user_prompt=user_prompt):
            full_answer += chunk
            yield {"event": "token", "data": {"text": chunk}}

        # 6. Build citations
        citations = [
            Citation(
                pmid=r.pmid,
                title=r.title,
                journal=r.journal,
                year=r.year,
                relevance_score=r.score,
            )
            for r in results
        ]

        # 7. Guardrails
        warnings = []
        disclaimer = ""
        is_grounded = True

        if guardrails_enabled:
            rag_response = RAGResponse(answer=full_answer, citations=citations, query=query)
            validator = GuardrailValidator(llm=llm, mesh_db=mesh_db)
            validated = validator.validate(rag_response, results)
            warnings = [w.model_dump() for w in validated.warnings]
            disclaimer = validated.disclaimer
            is_grounded = validated.is_grounded

        yield {
            "event": "done",
            "data": {
                "citations": [c.model_dump() for c in citations],
                "warnings": warnings,
                "disclaimer": disclaimer,
                "is_grounded": is_grounded,
            },
        }

    except Exception as e:
        logger.exception("Error in ask_stream")
        yield {"event": "error", "data": {"message": str(e)}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_chain.py -v`
Expected: ALL PASS (existing 3 tests + 3 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/chain.py tests/unit/test_chain.py
git commit -m "feat(backend): add ask_stream() generator for SSE pipeline"
```

---

## Chunk 3: /ask endpoint SSE support

### Task 3: Add SSE endpoint test and route changes

**Files:**
- Modify: `src/api/routes/ask.py`
- Modify: `tests/unit/test_api_ask.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_api_ask.py`:

```python
@patch("src.api.routes.ask.ask_stream")
def test_ask_stream_returns_sse(mock_ask_stream, client):
    mock_ask_stream.return_value = iter([
        {"event": "token", "data": {"text": "Hello"}},
        {"event": "token", "data": {"text": " world"}},
        {"event": "done", "data": {
            "citations": [{"pmid": "123", "title": "Test", "journal": "Nature", "year": 2023, "relevance_score": 0.95}],
            "warnings": [],
            "disclaimer": "Disclaimer.",
            "is_grounded": True,
        }},
    ])

    response = client.post("/ask", json={"query": "test", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "event: token" in body
    assert '"text": "Hello"' in body or '"text":"Hello"' in body
    assert "event: done" in body
    assert '"citations"' in body


@patch("src.api.routes.ask.ask_stream")
def test_ask_stream_sse_headers(mock_ask_stream, client):
    mock_ask_stream.return_value = iter([
        {"event": "done", "data": {"citations": [], "warnings": [], "disclaimer": "", "is_grounded": True}},
    ])

    response = client.post("/ask", json={"query": "test", "stream": True})
    assert response.headers.get("cache-control") == "no-cache"


@patch("src.api.routes.ask.rag_ask")
def test_ask_stream_false_returns_json(mock_ask, client):
    """stream=false (default) should still return JSON."""
    mock_ask.return_value = ValidatedResponse(
        answer="Test answer.", citations=[], query="test",
        warnings=[], disclaimer="Disclaimer.", is_grounded=True,
    )
    response = client.post("/ask", json={"query": "test", "stream": False})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_api_ask.py::test_ask_stream_returns_sse -v`
Expected: FAIL — `stream` field not in `AskRequest`, `ask_stream` not imported

- [ ] **Step 3: Implement SSE route**

Update `src/api/routes/ask.py`. **Important:** Remove `response_model=AskResponse` from the `@router.post("/ask")` decorator. This is required because the endpoint now returns either `AskResponse` (JSON) or `StreamingResponse` (SSE), and FastAPI cannot validate both with a single response model.

Replace the full file content:

```python
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
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_api_ask.py -v`
Expected: ALL PASS (existing 2 tests + 3 new tests)

- [ ] **Step 5: Run full backend test suite**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/routes/ask.py tests/unit/test_api_ask.py
git commit -m "feat(backend): add SSE streaming support to /ask endpoint"
```
