# SSE Early Citations Design

## Problem

When a user sends a query via `/ask` with `stream: true`, the entire search + rerank phase completes before any SSE events are emitted. Citations (search results) are only sent in the final `done` event, after all LLM tokens have streamed. This means the user sees nothing for several seconds, then tokens appear, and citations only show at the very end.

## Goal

Send search results (citations) to the frontend as soon as retrieval + reranking completes, **before** the LLM starts generating tokens. This gives the user immediate visual feedback and lets them start reading relevant papers while the answer streams in.

## Design

### Backend: New SSE event type

Add a `citations` event to `ask_stream()` in `src/rag/chain.py`, emitted after reranking (step 3) and before LLM generation (step 4).

**Current SSE event sequence:**
```
[silence during search + rerank + LLM first token]
event: token  (repeated)
event: done   (includes citations, warnings, disclaimer)
```

**New SSE event sequence:**
```
[silence during search + rerank only]
event: citations  <- NEW: {citations: [...]}
event: token      (repeated, starts immediately after)
event: done       (warnings, disclaimer, is_grounded — citations removed from here)
```

The `citations` event payload matches the existing `Citation` model:
```json
{
  "citations": [
    {"pmid": "...", "title": "...", "journal": "...", "year": 2024, "relevance_score": 0.89}
  ]
}
```

### Frontend: Handle `citations` event

- `askQueryStream()` in `lib/api.ts` gets a new `onCitations` callback parameter.
- `App.tsx` wires `onCitations` to update the results panel immediately.
- No changes to `ChatPanel.tsx` (it only handles message display).

### Backward compatibility

- The `done` event **keeps** the `citations` field as well for non-streaming callers and robustness. The frontend just uses whichever arrives first.
- The non-streaming `/ask` endpoint is unchanged.

## Scope

- Backend: ~5 lines changed in `chain.py`
- Frontend: ~15 lines across `api.ts`, `types/index.ts`, `App.tsx`
- Tests: Update existing `ask_stream` unit test to assert `citations` event order

## Out of scope

- Changing the non-streaming `/ask` response format
- Adding progress indicators for query expansion or reranking phases
