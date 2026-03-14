# SSE Streaming for `/ask` Endpoint — Design Spec

**Date:** 2026-03-15
**Owner:** Yasuhiro Okamoto
**Status:** Approved
**Parent Spec:** [2026-03-14-pubmed-rag-system-design.md](2026-03-14-pubmed-rag-system-design.md)

All file paths are relative to `capstone/`.

## 1. Goal

Add Server-Sent Events (SSE) streaming to the `/ask` endpoint so that LLM-generated answer tokens are delivered incrementally to the frontend, improving perceived latency and UX.

## 2. Approach

- Add `stream: bool = False` parameter to the existing `/ask` endpoint
- When `stream=false` (default), behavior is unchanged — returns JSON response
- When `stream=true`, returns `text/event-stream` with incremental token delivery
- Pipeline steps 1-3 (query expansion, search, rerank) run synchronously before streaming begins
- LLM answer generation (step 5) streams token-by-token
- Guardrails (step 7) run after full answer is assembled, results sent in final `done` event
- The endpoint remains a synchronous `def`. FastAPI runs the sync generator in a threadpool automatically. Do not convert to `async def` unless the LLM client is made async.

## 3. SSE Event Format

Three event types:

```
event: token
data: {"text": "Based on"}

event: token
data: {"text": " the retrieved"}

...

event: done
data: {"citations": [...], "warnings": [...], "disclaimer": "...", "is_grounded": true}
```

On error during streaming:

```
event: error
data: {"message": "LLM generation failed: timeout"}
```

- `token` — a text chunk from the LLM
- `done` — final metadata (citations, guardrail results) sent once after LLM generation + guardrails complete
- `error` — sent if an exception occurs mid-stream (LLM failure, guardrails error, timeout)

## 4. Backend Changes

### 4.1 `backend/src/shared/llm.py` — Add `complete_stream()`

New generator method on `LLMClient`:

```python
def complete_stream(
    self, system_prompt: str, user_prompt: str, temperature: float = 0.0
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
        delta = getattr(chunk.choices[0].delta, "content", None) if chunk.choices else None
        if delta:
            yield delta
        # Log usage from final chunk if available
        if hasattr(chunk, "usage") and chunk.usage:
            logger.debug("LLM stream: model=%s, tokens=%d", self.model, chunk.usage.total_tokens)
```

Existing `complete()` is unchanged.

**LangFuse compatibility:** LiteLLM's LangFuse integration supports streaming natively (aggregates chunks into a single trace). No additional configuration needed — verify during testing.

### 4.2 `backend/src/rag/chain.py` — Add `ask_stream()`

New generator function alongside existing `ask()`. Signature mirrors `ask()` exactly:

```python
def ask_stream(
    query: str,
    collection: Collection,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
    reranker: BaseReranker | None = None,
    guardrails_enabled: bool = True,
) -> Generator[dict, None, None]:
```

Flow:

- Steps 1-3 (expand, search, rerank) execute synchronously (no yield)
- Step 4 builds prompt
- Step 5 calls `llm.complete_stream()`, yielding each chunk and accumulating full answer
- Step 6 builds citations from search results (retained from step 2/3)
- Step 7 runs guardrails on accumulated answer. Note: `GuardrailValidator.validate()` requires both the `RAGResponse` and the original `search_results: list[SearchResult]` from step 2/3, and internally makes a separate non-streaming LLM call. This adds latency after streaming completes.
- Yields a final `done` dict with citations, warnings, disclaimer, is_grounded

The entire generator body is wrapped in try/except. On exception, yield an `error` event instead of raising.

```python
def ask_stream(...):
    try:
        # Steps 1-4: synchronous (same as ask())
        ...
        results = ...  # retained for guardrails

        # Step 5: stream LLM tokens
        full_answer = ""
        for chunk in llm.complete_stream(system_prompt, user_prompt):
            full_answer += chunk
            yield {"event": "token", "data": {"text": chunk}}

        # Steps 6-7: citations + guardrails (uses results from step 2/3)
        ...
        yield {"event": "done", "data": {
            "citations": [...],  # serialized Citation list
            "warnings": [...],   # serialized GuardrailWarning list
            "disclaimer": "...",
            "is_grounded": True,
        }}
    except Exception as e:
        logger.exception("Error in ask_stream")
        yield {"event": "error", "data": {"message": str(e)}}
```

### 4.3 `backend/src/api/routes/ask.py` — SSE Response

Remove `response_model=AskResponse` from the decorator (required because SSE and JSON are different response types). Non-streaming path returns `AskResponse` explicitly.

```python
from fastapi.responses import StreamingResponse, JSONResponse

class AskRequest(BaseModel):
    # ... existing fields ...
    stream: bool = False

@router.post("/ask")
def ask_endpoint(req: AskRequest, ...):
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
    response = rag_ask(...)
    return AskResponse(**response.model_dump())
```

`_sse_generator()` formats each yielded dict from `ask_stream()` into SSE wire format:
```
event: {event_type}\ndata: {json}\n\n
```

**CORS:** The existing CORS middleware (`allow_origins=["*"]`) is sufficient for SSE over `fetch()` (POST). No changes needed. Note: if `EventSource` (GET-based) is ever used, CORS and endpoint method would need to change.

## 5. Frontend Changes

### 5.1 `frontend/src/types/index.ts` — SSE Event Types and AskRequest Update

```typescript
export interface SSETokenEvent {
  text: string;
}

export interface SSEDoneEvent {
  citations: Citation[];
  warnings: Warning[];
  disclaimer: string;
  is_grounded: boolean;
}

export interface SSEErrorEvent {
  message: string;
}
```

Add `stream?: boolean` to existing `AskRequest` interface.

Field mapping: backend `GuardrailWarning` (check, severity, message, span) maps directly to frontend `Warning` type.

### 5.2 `frontend/src/lib/api.ts` — Add `askQueryStream()`

Uses `fetch()` + `ReadableStream` reader to parse SSE events. Accepts `AbortSignal` for cancellation support.

```typescript
export async function askQueryStream(
  req: AskRequest & { stream: true },
  onToken: (text: string) => void,
  onDone: (data: SSEDoneEvent) => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
    const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal,
    });
    if (!res.ok) {
        onError(new Error(`API error: ${res.status}`));
        return;
    }
    // Read SSE stream from res.body using TextDecoderStream
    // Parse "event:" and "data:" lines
    // Dispatch to onToken / onDone / onError based on event type
}
```

### 5.3 `frontend/src/App.tsx` — Streaming `handleSend`

- Create assistant message immediately with empty content
- On each `token` callback, append text to the message's content via state update
- On `done` callback, set citations, warnings, disclaimer on the message
- On `error` callback, update message role to "error"
- `loading` state is true during streaming, cleared on `done` or error

## 6. File Changes Summary

| File | Change |
|------|--------|
| `backend/src/shared/llm.py` | Add `complete_stream()` method |
| `backend/src/rag/chain.py` | Add `ask_stream()` generator function |
| `backend/src/api/routes/ask.py` | Add `stream` param, remove `response_model`, SSE response path |
| `frontend/src/types/index.ts` | Add `SSETokenEvent`, `SSEDoneEvent`, `SSEErrorEvent`, `stream` to `AskRequest` |
| `frontend/src/lib/api.ts` | Add `askQueryStream()` with `AbortSignal` support |
| `frontend/src/App.tsx` | Update `handleSend` for streaming |
| `backend/tests/unit/test_chain.py` | Add tests for `ask_stream()` — collect yielded events, assert structure |
| `backend/tests/unit/test_api_ask.py` | Add test for SSE response — verify event format and headers |
