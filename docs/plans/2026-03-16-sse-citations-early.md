# SSE Early Citations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit search results (citations) as an early SSE event before LLM streaming begins, so the frontend can display them immediately.

**Architecture:** Add a `citations` SSE event to `ask_stream()` after reranking. Frontend handles the new event via an `onCitations` callback.

**Tech Stack:** Python/FastAPI (backend), TypeScript/React (frontend)

---

## Chunk 1: Backend — emit early citations event

### Task 1: Update `ask_stream` to yield citations event

**Files:**
- Modify: `backend/src/rag/chain.py:130-131` (between rerank and prompt build)

- [ ] **Step 1: Add citations yield after reranking**

In `ask_stream()`, after step 3 (rerank) and before step 4 (build prompt), build the citations list and yield a `citations` event:

```python
        # 3. Rerank
        results = reranker.rerank(query, results, top_k=filters.top_k)
        logger.info("After reranking: %d results", len(results))

        # 3.5 Emit citations early
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
        yield {
            "event": "citations",
            "data": {"citations": [c.model_dump() for c in citations]},
        }

        # 4. Build prompt
```

Remove the duplicate citations list-building that currently exists after step 5 (LLM streaming). Reuse the `citations` variable already built above.

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `cd backend && .venv/bin/pytest tests/unit/ -v -x`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/rag/chain.py
git commit -m "feat(sse): emit citations event before LLM streaming"
```

---

## Chunk 2: Frontend — handle early citations

### Task 2: Add `onCitations` callback to SSE client

**Files:**
- Modify: `frontend/src/lib/api.ts:40-110`
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Add `onCitations` parameter to `askQueryStream`**

```typescript
export async function askQueryStream(
  req: AskRequest & { stream: true },
  onToken: (text: string) => void,
  onDone: (data: SSEDoneEvent) => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
  onCitations?: (citations: Citation[]) => void,  // NEW
): Promise<void> {
```

In the SSE event parsing loop, add a handler for the `citations` event:

```typescript
          } else if (currentEvent === "citations") {
            if (onCitations) {
              onCitations((data as { citations: Citation[] }).citations);
            }
```

- [ ] **Step 2: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/types/index.ts
git commit -m "feat(frontend): handle citations SSE event in stream client"
```

### Task 3: Wire `onCitations` in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Pass `onCitations` callback to `askQueryStream`**

In the function that calls `askQueryStream`, add an `onCitations` handler that updates the results/citations state immediately:

```typescript
onCitations: (citations) => {
  setCitations(citations);
},
```

The exact state variable name depends on the current App.tsx implementation. Read App.tsx to identify the correct state setter.

- [ ] **Step 2: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): display citations immediately on SSE citations event"
```

---

## Chunk 3: Verify end-to-end

### Task 4: Manual E2E verification

- [ ] **Step 1: Start backend and frontend**
- [ ] **Step 2: Send a query with streaming enabled**
- [ ] **Step 3: Verify citations panel populates before LLM tokens start appearing**
