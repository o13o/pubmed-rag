# Phase D: React Frontend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React chat UI with side panel (filters + results) that calls the existing FastAPI backend (`/ask`, `/search`).

**Architecture:** Vite + React + TypeScript + Tailwind CSS. Single page app. `App.tsx` holds all state, child components are presentational. API client isolated in `lib/api.ts`. Vite proxy forwards API calls to backend during development.

**Tech Stack:** Vite, React 18, TypeScript, Tailwind CSS v4

**Spec:** [2026-03-15-frontend-design.md](../specs/2026-03-15-frontend-design.md)

**Prerequisites:** Backend API running on `localhost:8000` with data ingested (see `scripts/smoke_test.py`).

---

## File Structure

```
capstone/frontend/
  index.html
  package.json
  tsconfig.json
  vite.config.ts
  src/
    main.tsx                  # Entry point
    App.tsx                   # Root layout + state
    index.css                 # Tailwind directives
    lib/
      api.ts                  # API_BASE + fetch wrappers
    types/
      index.ts                # TypeScript interfaces (matches backend Pydantic)
    components/
      ChatPanel.tsx           # Message list + input form
      MessageBubble.tsx       # Single Q or A bubble
      FilterPanel.tsx         # Mode toggle + filters
      ResultsPanel.tsx        # Citations / search results
```

---

## Chunk 1: Project Scaffold

### Task 1: Initialize Vite + React + TypeScript project

**Files:**
- Create: `capstone/frontend/package.json` (via `npm create vite`)
- Create: `capstone/frontend/vite.config.ts`
- Create: `capstone/frontend/tsconfig.json`
- Create: `capstone/frontend/index.html`

- [ ] **Step 1: Scaffold Vite project**

```bash
cd capstone
npm create vite@latest frontend -- --template react-ts
```

- [ ] **Step 2: Install dependencies**

```bash
cd capstone/frontend
npm install
```

- [ ] **Step 3: Install Tailwind CSS v4**

```bash
cd capstone/frontend
npm install tailwindcss @tailwindcss/vite
```

- [ ] **Step 4: Configure Vite with Tailwind and API proxy**

Replace the contents of `capstone/frontend/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/ask": "http://localhost:8000",
      "/search": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
```

- [ ] **Step 5: Configure Tailwind CSS**

Replace the contents of `capstone/frontend/src/index.css`:

```css
@import "tailwindcss";
```

- [ ] **Step 6: Verify dev server starts**

```bash
cd capstone/frontend
npm run dev
```

Expected: Vite dev server starts on `http://localhost:5173` with default React page.

- [ ] **Step 7: Commit**

```bash
git add capstone/frontend/
git commit -m "feat(frontend): scaffold Vite + React + TypeScript + Tailwind"
```

---

## Chunk 2: Types + API Client

### Task 2: TypeScript type definitions

**Files:**
- Create: `capstone/frontend/src/types/index.ts`

- [ ] **Step 1: Create type definitions matching backend Pydantic models**

```typescript
// src/types/index.ts

export interface Citation {
  pmid: string;
  title: string;
  journal: string;
  year: number;
  relevance_score: number;
}

export interface Warning {
  check: string;
  severity: string;
  message: string;
  span: string;
}

export interface AskRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  query: string;
  warnings: Warning[];
  disclaimer: string;
  is_grounded: boolean;
}

export interface SearchRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
}

export interface SearchResult {
  pmid: string;
  title: string;
  abstract_text: string;
  score: number;
  year: number;
  journal: string;
  mesh_terms: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
}

export type Mode = "ask" | "search";

export interface Filters {
  year_min?: number;
  year_max?: number;
  top_k: number;
  search_mode: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  citations?: Citation[];
  warnings?: Warning[];
  disclaimer?: string;
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/types/
git commit -m "feat(frontend): add TypeScript type definitions"
```

---

### Task 3: API client

**Files:**
- Create: `capstone/frontend/src/lib/api.ts`

- [ ] **Step 1: Create API client with configurable base URL**

```typescript
// src/lib/api.ts

import type {
  AskRequest,
  AskResponse,
  SearchRequest,
  SearchResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function askQuery(req: AskRequest): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function searchQuery(
  req: SearchRequest
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/lib/
git commit -m "feat(frontend): add API client with configurable base URL"
```

---

## Chunk 3: Components

### Task 4: MessageBubble component

**Files:**
- Create: `capstone/frontend/src/components/MessageBubble.tsx`

- [ ] **Step 1: Implement MessageBubble**

```tsx
// src/components/MessageBubble.tsx

import type { Message } from "../types";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-blue-600 text-white rounded-lg px-4 py-2 max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="flex justify-start mb-4">
        <div className="bg-red-900/50 border border-red-700 text-red-200 rounded-lg px-4 py-2 max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-gray-800 text-gray-100 rounded-lg px-4 py-3 max-w-[80%]">
        <div className="whitespace-pre-wrap">{message.content}</div>
        {message.warnings && message.warnings.length > 0 && (
          <div className="mt-2 border-t border-gray-700 pt-2">
            {message.warnings.map((w, i) => (
              <div
                key={i}
                className={`text-xs mt-1 ${
                  w.severity === "error"
                    ? "text-red-400"
                    : "text-yellow-400"
                }`}
              >
                [{w.check}] {w.message}
              </div>
            ))}
          </div>
        )}
        {message.disclaimer && (
          <div className="mt-2 text-xs text-gray-500 italic border-t border-gray-700 pt-2">
            {message.disclaimer}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/components/MessageBubble.tsx
git commit -m "feat(frontend): add MessageBubble component"
```

---

### Task 5: FilterPanel component

**Files:**
- Create: `capstone/frontend/src/components/FilterPanel.tsx`

- [ ] **Step 1: Implement FilterPanel**

```tsx
// src/components/FilterPanel.tsx

import type { Filters, Mode } from "../types";

interface Props {
  mode: Mode;
  filters: Filters;
  onModeChange: (mode: Mode) => void;
  onFiltersChange: (filters: Filters) => void;
}

export function FilterPanel({
  mode,
  filters,
  onModeChange,
  onFiltersChange,
}: Props) {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Mode
      </h3>
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => onModeChange("ask")}
          className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === "ask"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Ask
        </button>
        <button
          onClick={() => onModeChange("search")}
          className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === "search"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Search
        </button>
      </div>

      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Filters
      </h3>
      <div className="space-y-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Year Min</label>
          <input
            type="number"
            value={filters.year_min ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_min: e.target.value ? Number(e.target.value) : undefined,
              })
            }
            placeholder="e.g. 2020"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 placeholder-gray-600"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Year Max</label>
          <input
            type="number"
            value={filters.year_max ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_max: e.target.value ? Number(e.target.value) : undefined,
              })
            }
            placeholder="e.g. 2024"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 placeholder-gray-600"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Top K</label>
          <input
            type="number"
            value={filters.top_k}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                top_k: Number(e.target.value) || 10,
              })
            }
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Search Mode
          </label>
          <div className="flex gap-2">
            <button
              onClick={() =>
                onFiltersChange({ ...filters, search_mode: "dense" })
              }
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                filters.search_mode === "dense"
                  ? "bg-emerald-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              Dense
            </button>
            <button
              onClick={() =>
                onFiltersChange({ ...filters, search_mode: "hybrid" })
              }
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                filters.search_mode === "hybrid"
                  ? "bg-emerald-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              Hybrid
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/components/FilterPanel.tsx
git commit -m "feat(frontend): add FilterPanel component with mode and search_mode toggle"
```

---

### Task 6: ResultsPanel component

**Files:**
- Create: `capstone/frontend/src/components/ResultsPanel.tsx`

- [ ] **Step 1: Implement ResultsPanel**

```tsx
// src/components/ResultsPanel.tsx

import type { Citation, SearchResult } from "../types";

interface Props {
  citations: Citation[];
  searchResults: SearchResult[];
  mode: "ask" | "search";
}

export function ResultsPanel({ citations, searchResults, mode }: Props) {
  if (mode === "ask") {
    if (citations.length === 0) {
      return (
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Citations
          </h3>
          <p className="text-xs text-gray-600">
            Ask a question to see citations.
          </p>
        </div>
      );
    }

    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Citations ({citations.length})
        </h3>
        <div className="space-y-2">
          {citations.map((c) => (
            <div
              key={c.pmid}
              className="bg-gray-800 rounded p-2 border border-gray-700"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono text-blue-400">
                  PMID: {c.pmid}
                </span>
                <span className="text-xs font-mono text-emerald-400">
                  {c.relevance_score.toFixed(3)}
                </span>
              </div>
              <div className="text-xs text-gray-300 leading-snug">
                {c.title}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {c.journal} ({c.year})
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Search mode
  if (searchResults.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Search Results
        </h3>
        <p className="text-xs text-gray-600">Search to see results.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Search Results ({searchResults.length})
      </h3>
      <div className="space-y-2">
        {searchResults.map((r) => (
          <div
            key={r.pmid}
            className="bg-gray-800 rounded p-2 border border-gray-700"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-blue-400">
                PMID: {r.pmid}
              </span>
              <span className="text-xs font-mono text-emerald-400">
                {r.score.toFixed(4)}
              </span>
            </div>
            <div className="text-xs text-gray-300 leading-snug">{r.title}</div>
            <div className="text-xs text-gray-500 mt-1">
              {r.journal} ({r.year})
            </div>
            <div className="text-xs text-gray-400 mt-1 line-clamp-3">
              {r.abstract_text}
            </div>
            {r.mesh_terms.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {r.mesh_terms.slice(0, 5).map((term) => (
                  <span
                    key={term}
                    className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded"
                  >
                    {term}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/components/ResultsPanel.tsx
git commit -m "feat(frontend): add ResultsPanel component (citations + search results)"
```

---

### Task 7: ChatPanel component

**Files:**
- Create: `capstone/frontend/src/components/ChatPanel.tsx`

- [ ] **Step 1: Implement ChatPanel**

```tsx
// src/components/ChatPanel.tsx

import { useState, useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  loading: boolean;
  onSend: (query: string) => void;
}

export function ChatPanel({ messages, loading, onSend }: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;
    onSend(query);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-600">
            <p>Ask a question about medical research.</p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 text-gray-400 rounded-lg px-4 py-2">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form
        onSubmit={handleSubmit}
        className="border-t border-gray-800 p-4 flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-6 py-2 rounded-lg font-medium transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/components/ChatPanel.tsx
git commit -m "feat(frontend): add ChatPanel component with message list and input"
```

---

## Chunk 4: App Assembly + Verify

### Task 8: App.tsx — wire everything together

**Files:**
- Modify: `capstone/frontend/src/App.tsx`

- [ ] **Step 1: Implement App.tsx with state management**

Replace the contents of `capstone/frontend/src/App.tsx`:

```tsx
// src/App.tsx

import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { askQuery, searchQuery } from "./lib/api";
import type {
  Citation,
  Filters,
  Message,
  Mode,
  SearchResult,
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
        const res = await askQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
        });
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: res.answer,
          citations: res.citations,
          warnings: res.warnings,
          disclaimer: res.disclaimer,
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setCitations(res.citations);
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

- [ ] **Step 2: Clean up main.tsx**

Replace the contents of `capstone/frontend/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 3: Delete default Vite boilerplate files**

```bash
cd capstone/frontend
rm -f src/App.css src/assets/react.svg public/vite.svg
```

- [ ] **Step 4: Verify dev server compiles**

```bash
cd capstone/frontend
npm run dev
```

Expected: Vite compiles with no TypeScript errors. Open `http://localhost:5173` — should see the 2-column layout with chat area and side panel.

- [ ] **Step 5: Verify API integration (requires backend running)**

```bash
# In another terminal, start backend:
cd capstone/backend
set -a && source .env && set +a
uv run uvicorn src.api.main:app --port 8000
```

Then in the browser at `http://localhost:5173`:
1. Type "cancer treatment" and click Send
2. Expect: answer appears in chat, citations in side panel

- [ ] **Step 6: Commit**

```bash
git add capstone/frontend/
git commit -m "feat(frontend): wire up App with ChatPanel, FilterPanel, ResultsPanel"
```

---

### Task 9: Build verification

- [ ] **Step 1: Verify production build works**

```bash
cd capstone/frontend
npm run build
```

Expected: `dist/` directory created with `index.html` and bundled JS/CSS.

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "test: verify frontend production build succeeds"
```
