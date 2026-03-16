# Plan C: Frontend (Types, API Client, AgentResultsPanel, App.tsx)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frontend support for agent analysis — types, API client, results panel, and "Analyze" button wiring.

**Architecture:** New TypeScript types matching backend AgentResult. New `analyzeQuery()` fetch function. New `AgentResultsPanel` component. App.tsx wired with state + button.

**Tech Stack:** React 18, TypeScript, Tailwind CSS

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Prerequisites:** Plan S completed (models exist for type reference). Frontend changes do not depend on Plan A/B agent implementations — they only need the `/analyze` endpoint contract.

---

## Task 1: Add TypeScript types for agent analysis

**Files:**
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Append agent types**

Append to `frontend/src/types/index.ts`:

```typescript
export interface Finding {
  label: string;
  detail: string;
  severity: "info" | "warning" | "critical";
}

export interface AgentResult {
  agent_name: string;
  summary: string;
  findings: Finding[];
  confidence: number;
  score: number | null;
  details: Record<string, unknown> | null;
}

export interface AnalyzeRequest {
  query: string;
  results: SearchResult[];
  agents?: string[];
}

export interface AnalyzeResponse {
  query: string;
  agent_results: AgentResult[];
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "feat(frontend): add agent analysis TypeScript types"
```

---

## Task 2: Add analyzeQuery API function and Vite proxy

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/vite.config.ts`

- [ ] **Step 1: Add import and function to api.ts**

In `frontend/src/lib/api.ts`, add `AnalyzeRequest` and `AnalyzeResponse` to the import from `"../types"`:

```typescript
import type {
  AskRequest,
  AskResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  SearchRequest,
  SearchResponse,
  SSEDoneEvent,
} from "../types";
```

Then append the function:

```typescript
export async function analyzeQuery(
  req: AnalyzeRequest
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
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

- [ ] **Step 2: Add /analyze to Vite proxy**

In `frontend/vite.config.ts`, add to the `server.proxy` object:

```typescript
"/analyze": "http://localhost:8000",
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/vite.config.ts
git commit -m "feat(frontend): add analyzeQuery API function and proxy"
```

---

## Task 3: Create AgentResultsPanel component

**Files:**
- Create: `frontend/src/components/AgentResultsPanel.tsx`

- [ ] **Step 1: Implement component**

Create `frontend/src/components/AgentResultsPanel.tsx`:

```tsx
import type { AgentResult } from "../types";

interface Props {
  agentResults: AgentResult[];
  loading: boolean;
}

const SEVERITY_COLORS = {
  info: "text-blue-400",
  warning: "text-yellow-400",
  critical: "text-red-400",
};

const AGENT_LABELS: Record<string, string> = {
  retrieval: "Retrieval",
  methodology_critic: "Methodology Critic",
  statistical_reviewer: "Statistical Reviewer",
  clinical_applicability: "Clinical Applicability",
  summarization: "Summarization",
};

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 7
      ? "bg-emerald-600"
      : score >= 4
        ? "bg-yellow-600"
        : "bg-red-600";
  return (
    <span
      className={`${color} text-white text-xs font-bold px-2 py-0.5 rounded-full`}
    >
      {score}/10
    </span>
  );
}

export function AgentResultsPanel({ agentResults, loading }: Props) {
  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Agent Analysis
        </h3>
        <p className="text-xs text-gray-500 animate-pulse">Analyzing...</p>
      </div>
    );
  }

  if (agentResults.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Agent Analysis ({agentResults.length})
      </h3>
      <div className="space-y-3">
        {agentResults.map((r) => (
          <div
            key={r.agent_name}
            className="bg-gray-800 rounded p-3 border border-gray-700"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-gray-300">
                {AGENT_LABELS[r.agent_name] ?? r.agent_name}
              </span>
              {r.score !== null && <ScoreBadge score={r.score} />}
            </div>
            <p className="text-xs text-gray-400 mb-2">{r.summary}</p>
            {r.findings.length > 0 && (
              <div className="space-y-1">
                {r.findings.map((f, i) => (
                  <div key={i} className="flex gap-2 text-xs">
                    <span
                      className={
                        SEVERITY_COLORS[
                          f.severity as keyof typeof SEVERITY_COLORS
                        ] ?? "text-gray-400"
                      }
                    >
                      [{f.label}]
                    </span>
                    <span className="text-gray-500">{f.detail}</span>
                  </div>
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
git add frontend/src/components/AgentResultsPanel.tsx
git commit -m "feat(frontend): add AgentResultsPanel component"
```

---

## Task 4: Wire up Analyze button in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Update imports**

In `frontend/src/App.tsx`, update the imports:

```typescript
import { useState, useRef } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { AgentResultsPanel } from "./components/AgentResultsPanel";
import { analyzeQuery, askQueryStream, searchQuery } from "./lib/api";
import type {
  AgentResult,
  Citation,
  Filters,
  Message,
  Mode,
  SearchResult,
  SSEDoneEvent,
} from "./types";
```

- [ ] **Step 2: Add state variables**

After the existing state declarations, add:

```typescript
const [agentResults, setAgentResults] = useState<AgentResult[]>([]);
const [analyzing, setAnalyzing] = useState(false);
```

- [ ] **Step 3: Add handleAnalyze function**

After `handleSend`, add:

```typescript
const handleAnalyze = async () => {
  if (searchResults.length === 0 && citations.length === 0) return;
  setAnalyzing(true);
  setAgentResults([]);
  try {
    // In search mode, use searchResults directly.
    // In ask mode, searchResults may be empty — use citations as a fallback.
    // Note: citations lack abstract_text, so agent analysis will be limited
    // to title/metadata. For full analysis, use search mode first.
    const results: SearchResult[] =
      searchResults.length > 0
        ? searchResults
        : citations.map((c) => ({
            pmid: c.pmid,
            title: c.title,
            abstract_text: "",
            score: c.relevance_score,
            year: c.year,
            journal: c.journal,
            mesh_terms: [],
          }));
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    const query = lastUserMsg?.content ?? "";
    const res = await analyzeQuery({ query, results });
    setAgentResults(res.agent_results);
  } catch (err) {
    const errorMsg: Message = {
      id: crypto.randomUUID(),
      role: "error",
      content: err instanceof Error ? err.message : "Analysis failed",
    };
    setMessages((prev) => [...prev, errorMsg]);
  } finally {
    setAnalyzing(false);
  }
};
```

- [ ] **Step 4: Add Analyze button and AgentResultsPanel to the aside**

In the `<aside>` section, between `<FilterPanel ... />` and `<ResultsPanel ... />`, add:

```tsx
{(searchResults.length > 0 || citations.length > 0) && (
  <button
    onClick={handleAnalyze}
    disabled={analyzing}
    className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
  >
    {analyzing ? "Analyzing..." : "Analyze with Agents"}
  </button>
)}
<AgentResultsPanel agentResults={agentResults} loading={analyzing} />
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd capstone/frontend && npx --package=typescript tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): wire up Analyze button and AgentResultsPanel"
```
