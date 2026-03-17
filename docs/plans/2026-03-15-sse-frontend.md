# SSE Streaming — Frontend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume SSE streaming from the `/ask` endpoint so LLM answer tokens render incrementally in the chat UI.

**Architecture:** Add an `askQueryStream()` function using `fetch()` + `ReadableStream` to parse SSE events, update `App.tsx` to stream tokens into the assistant message in real-time.

**Tech Stack:** React, TypeScript, Fetch API / ReadableStream, SSE parsing

**Spec:** `docs/specs/2026-03-15-sse-streaming-design.md`

**Dependency:** Backend SSE endpoint must be implemented first (see `2026-03-15-sse-backend.md`). However, the frontend code can be written and type-checked independently — integration testing requires the backend.

All file paths relative to `frontend/`.

---

## Chunk 1: Types and SSE API client

### Task 1: Add SSE types

**Files:**
- Modify: `src/types/index.ts`

- [ ] **Step 1: Add SSE event types and `stream` to `AskRequest`**

Add to the end of `src/types/index.ts`, before the `Message` interface:

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

Add `stream?: boolean` to the existing `AskRequest` interface:

```typescript
export interface AskRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
  stream?: boolean;
}
```

- [ ] **Step 2: Verify types compile**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/types/index.ts
git commit -m "feat(frontend): add SSE event types and stream flag to AskRequest"
```

---

### Task 2: Add `askQueryStream()` function

**Files:**
- Modify: `src/lib/api.ts`

- [ ] **Step 1: Implement `askQueryStream()`**

Add `askQueryStream()` to `src/lib/api.ts`. Replace the existing import at the top of the file to include `SSEDoneEvent`:

```typescript
import type {
  AskRequest,
  AskResponse,
  SearchRequest,
  SearchResponse,
  SSEDoneEvent,
} from "../types";
```

Then add after the existing `searchQuery()` function:

```typescript
export async function askQueryStream(
  req: AskRequest & { stream: true },
  onToken: (text: string) => void,
  onDone: (data: SSEDoneEvent) => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!res.ok) {
    onError(new Error(`API error: ${res.status} ${res.statusText}`));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError(new Error("No response body"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          let data: unknown;
          try {
            data = JSON.parse(line.slice(6));
          } catch {
            onError(new Error("Failed to parse SSE data: " + line));
            return;
          }
          if (currentEvent === "token") {
            onToken(data.text);
          } else if (currentEvent === "done") {
            onDone(data);
          } else if (currentEvent === "error") {
            onError(new Error(data.message));
          }
          currentEvent = "";
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
```

- [ ] **Step 2: Verify types compile**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/lib/api.ts
git commit -m "feat(frontend): add askQueryStream() SSE client with AbortSignal"
```

---

## Chunk 2: App integration

### Task 3: Wire up streaming in App.tsx

**Files:**
- Modify: `src/App.tsx`

- [ ] **Step 1: Update handleSend to use streaming**

Replace the full `src/App.tsx` file. Key changes:
1. Import `askQueryStream` instead of `askQuery` (non-streaming `askQuery` remains in `api.ts` as fallback but is no longer used here)
2. Add `useRef` for `AbortController`
3. In ask mode, create an empty assistant message immediately
4. Stream tokens into that message incrementally
5. On `done`, update the message with citations/warnings/disclaimer

Full updated `App.tsx`:

```typescript
import { useState, useRef } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { askQueryStream, searchQuery } from "./lib/api";
import type {
  Citation,
  Filters,
  Message,
  Mode,
  SearchResult,
  SSEDoneEvent,
} from "./types";

function App() {
  const [mode, setMode] = useState<Mode>("ask");
  const [filters, setFilters] = useState<Filters>({
    top_k: 10,
    search_mode: "dense",
  });
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const handleSend = async (query: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      if (mode === "ask") {
        const assistantId = crypto.randomUUID();
        const assistantMsg: Message = {
          id: assistantId,
          role: "assistant",
          content: "",
        };
        setMessages((prev) => [...prev, assistantMsg]);

        abortRef.current = new AbortController();

        await askQueryStream(
          {
            query,
            year_min: filters.year_min,
            year_max: filters.year_max,
            top_k: filters.top_k,
            search_mode: filters.search_mode,
            stream: true,
          },
          // onToken
          (text: string) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + text }
                  : m,
              ),
            );
          },
          // onDone
          (data: SSEDoneEvent) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      citations: data.citations,
                      warnings: data.warnings,
                      disclaimer: data.disclaimer,
                    }
                  : m,
              ),
            );
            setCitations(data.citations);
            setLoading(false);
          },
          // onError
          (error: Error) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, role: "error", content: error.message }
                  : m,
              ),
            );
            setLoading(false);
          },
          abortRef.current.signal,
        );
      } else {
        const res = await searchQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
        });
        const infoMsg: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Found ${res.total} results.`,
        };
        setMessages((prev) => [...prev, infoMsg]);
        setSearchResults(res.results);
      }
    } catch (err) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "error",
        content:
          err instanceof Error ? err.message : "Backend unavailable",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold tracking-tight">
          <span className="text-blue-400">PubMed</span> RAG
        </h1>
        <span className="text-xs text-gray-600">
          {mode === "ask" ? "Ask Mode" : "Search Mode"} ·{" "}
          {filters.search_mode}
        </span>
      </header>
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 flex flex-col min-w-0">
          <ChatPanel
            messages={messages}
            loading={loading}
            onSend={handleSend}
          />
        </main>
        <aside className="w-80 border-l border-gray-800 p-4 overflow-y-auto space-y-4">
          <FilterPanel
            mode={mode}
            filters={filters}
            onModeChange={setMode}
            onFiltersChange={setFilters}
          />
          <ResultsPanel
            citations={citations}
            searchResults={searchResults}
            mode={mode}
          />
        </aside>
      </div>
    </div>
  );
}

export default App;
```

- [ ] **Step 2: Fix "Thinking..." indicator during streaming**

In `src/components/ChatPanel.tsx`, the "Thinking..." indicator shows whenever `loading` is true. During streaming, this means "Thinking..." appears below the assistant message while tokens are already flowing. Fix by checking if the last message is an assistant message with content (streaming active):

Replace the loading indicator block in `ChatPanel.tsx`:

```typescript
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 text-gray-400 rounded-lg px-4 py-2">
              Thinking...
            </div>
          </div>
        )}
```

With:

```typescript
        {loading && messages.length > 0 && messages[messages.length - 1].role !== "assistant" && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 text-gray-400 rounded-lg px-4 py-2">
              Thinking...
            </div>
          </div>
        )}
```

This hides "Thinking..." once the empty assistant message is added (streaming has begun). The "Thinking..." only shows during the synchronous retrieval phase (before the first token).

Note: The `messages` prop already contains the streaming assistant message, so checking `messages[messages.length - 1].role` is sufficient.

- [ ] **Step 3: Verify types compile**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Verify dev server starts**

Run: `cd frontend && npm run dev`
Expected: Compiles without errors, accessible at http://localhost:5173

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/components/ChatPanel.tsx
git commit -m "feat(frontend): wire up SSE streaming in chat UI"
```
