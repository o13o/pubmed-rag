# PubMed RAG Frontend Design

## Overview

React frontend for the PubMed RAG system. Chat-style UI with a side panel for filters and results. Calls the existing FastAPI backend (`/ask`, `/search`).

## Tech Stack

- Vite + React + TypeScript
- Tailwind CSS
- No routing library (single page)
- No state management library (`useState`/`useReducer` sufficient)

## Layout

Two-column layout: chat (left, main) + side panel (right).

```
+----------------------------------------------------------+
| Header: "PubMed RAG"                                     |
+--------------------------------------+-------------------+
|                                      |  FilterPanel      |
|  ChatPanel                           |  - Ask/Search     |
|  - Message list                      |  - year_min/max   |
|  - Q/A bubbles                       |  - top_k          |
|  - Input + Send                      +-------------------+
|                                      |  ResultsPanel     |
|                                      |  - Citations (Ask)|
|                                      |  - Results(Search)|
+--------------------------------------+-------------------+
```

## Components

| Component | Purpose |
|---|---|
| `App.tsx` | Root layout: header + 2-column grid |
| `ChatPanel` | Message list + input form. Sends query to API |
| `MessageBubble` | Single Q or A. Answer includes disclaimer and warnings |
| `FilterPanel` | Ask/Search mode toggle, year_min, year_max, top_k inputs |
| `ResultsPanel` | Ask mode: citation list. Search mode: search result cards |

## API Client

```typescript
// hooks/useApi.ts — thin wrappers around fetch()

interface AskRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  guardrails_enabled?: boolean;
}

interface AskResponse {
  answer: string;
  citations: Citation[];
  query: string;
  warnings: Warning[];
  disclaimer: string;
  is_grounded: boolean;
}

interface SearchRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
}

interface Citation {
  pmid: string;
  title: string;
  journal: string;
  year: number;
  relevance_score: number;
}

interface SearchResult {
  pmid: string;
  title: string;
  abstract_text: string;
  score: number;
  year: number;
  journal: string;
  mesh_terms: string[];
}

interface Warning {
  check: string;
  severity: string;
  message: string;
  span: string;
}
```

## Data Flow

### Ask Mode (default)
1. User types query + clicks Send
2. Frontend calls `POST /ask` with query + filters
3. Response displayed as chat bubble (answer + disclaimer)
4. Citations shown in ResultsPanel (right)

### Search Mode
1. User toggles to Search mode in FilterPanel
2. User types query + clicks Send
3. Frontend calls `POST /search` with query + filters
4. Results displayed as cards in ResultsPanel (right)
5. Chat area shows query only (no AI-generated answer)

## Error Handling

- API errors: show error message in chat area
- Network errors: show "Backend unavailable" message
- No loading spinner or health polling — simple inline status

## API Base URL

All API calls go through a single `API_BASE` constant so the backend URL is trivially swappable:

```typescript
// lib/api.ts
const API_BASE = import.meta.env.VITE_API_BASE ?? "";
// "" = relative path (same origin) — works with Vite proxy AND when served by FastAPI
// "http://localhost:8000" = explicit backend URL for standalone dev
```

All fetch calls use `${API_BASE}/ask`, `${API_BASE}/search`, etc.

### Development (Vite proxy)

```typescript
// vite.config.ts
server: {
  proxy: {
    '/ask': 'http://localhost:8000',
    '/search': 'http://localhost:8000',
    '/health': 'http://localhost:8000',
  }
}
```

### Production (FastAPI serves built frontend)

```python
# backend: mount dist/ as static files
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")
```

Since `API_BASE` defaults to `""` (relative), the built frontend calls `/ask` on the same origin — no config change needed.

## Style

- Dark theme (Tailwind dark classes)
- Monospace accents for PMIDs and scores
- Minimal, functional — POC quality

## Out of Scope

- Authentication
- Responsive/mobile layout
- Dark/light mode toggle (dark only)
- SSR, routing, complex state management
- Streaming responses
