# Review Pipeline — Plan C: Frontend

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Generate Literature Review" button and ReviewPanel component to display structured literature reviews.

**Architecture:** New button in sidebar triggers `POST /review`. New `ReviewPanel.tsx` renders the 4 review sections with agent results in a collapsible detail. Types and API function added following existing patterns.

**Tech Stack:** React 19, TypeScript, Tailwind CSS 4

**Depends on:** Plan B (API route) must be completed first.

---

### Task 1: Add TypeScript types for LiteratureReview

**Files:**
- Modify: `capstone/frontend/src/types/index.ts`

- [ ] **Step 1: Add LiteratureReview types**

Append to `capstone/frontend/src/types/index.ts`:

```typescript
export interface ReviewRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  journals?: string[];
  top_k?: number;
  search_mode?: string;
}

export interface LiteratureReview {
  query: string;
  overview: string;
  main_findings: string;
  gaps_and_conflicts: string;
  recommendations: string;
  citations: Citation[];
  search_results: SearchResult[];
  agent_results: AgentResult[];
  agents_succeeded: number;
  agents_failed: number;
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/types/index.ts
git commit -m "feat: add LiteratureReview TypeScript types"
```

---

### Task 2: Add reviewQuery API function

**Files:**
- Modify: `capstone/frontend/src/lib/api.ts`

- [ ] **Step 1: Add the API function**

Add import to the top of `capstone/frontend/src/lib/api.ts`:
```typescript
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  AskRequest,
  AskResponse,
  Citation,
  LiteratureReview,
  ReviewRequest,
  SearchRequest,
  SearchResponse,
  SearchResult,
  SSEDoneEvent,
  TranscribeResponse,
} from "../types";
```

Append to the end of `capstone/frontend/src/lib/api.ts`:

```typescript
export async function reviewQuery(
  req: ReviewRequest
): Promise<LiteratureReview> {
  const res = await fetch(`${API_BASE}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Review failed: ${res.status} ${detail}`);
  }
  return res.json();
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/lib/api.ts
git commit -m "feat: add reviewQuery API function"
```

---

### Task 3: Create ReviewPanel component

**Files:**
- Create: `capstone/frontend/src/components/ReviewPanel.tsx`

- [ ] **Step 1: Create the component**

Create `capstone/frontend/src/components/ReviewPanel.tsx`:

```tsx
import { useState } from "react";
import type { LiteratureReview } from "../types";

interface Props {
  review: LiteratureReview | null;
  loading: boolean;
}

function Section({ title, content }: { title: string; content: string }) {
  return (
    <div className="mb-4">
      <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
        {title}
      </h4>
      <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
        {content}
      </p>
    </div>
  );
}

export function ReviewPanel({ review, loading }: Props) {
  const [showAgents, setShowAgents] = useState(false);

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Literature Review
        </h3>
        <p className="text-xs text-gray-500 animate-pulse">
          Generating literature review... This may take a minute.
        </p>
      </div>
    );
  }

  if (!review) return null;

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Literature Review
      </h3>

      <Section title="Overview" content={review.overview} />
      <Section title="Main Findings" content={review.main_findings} />
      <Section title="Gaps & Conflicts" content={review.gaps_and_conflicts} />
      <Section title="Recommendations" content={review.recommendations} />

      <div className="border-t border-gray-700 pt-3 mt-3">
        <p className="text-xs text-gray-500">
          {review.citations.length} citations &middot;{" "}
          {review.agents_succeeded} agents succeeded
          {review.agents_failed > 0 && (
            <span className="text-yellow-500">
              {" "}&middot; {review.agents_failed} failed
            </span>
          )}
        </p>
        <button
          onClick={() => setShowAgents(!showAgents)}
          className="text-xs text-blue-400 hover:text-blue-300 mt-1"
        >
          {showAgents ? "Hide" : "Show"} agent details
        </button>
        {showAgents && (
          <div className="mt-2 space-y-2">
            {review.agent_results.map((a) => (
              <div
                key={a.agent_name}
                className="bg-gray-800 rounded p-2 border border-gray-700"
              >
                <span className="text-xs font-semibold text-gray-300">
                  {a.agent_name}
                </span>
                <span className="text-xs text-gray-500 ml-2">
                  confidence: {a.confidence.toFixed(2)}
                </span>
                <p className="text-xs text-gray-400 mt-1">{a.summary}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add capstone/frontend/src/components/ReviewPanel.tsx
git commit -m "feat: add ReviewPanel component"
```

---

### Task 4: Integrate into App.tsx

**Files:**
- Modify: `capstone/frontend/src/App.tsx`

- [ ] **Step 1: Add imports**

Add to the imports in `capstone/frontend/src/App.tsx`:

```typescript
import { ReviewPanel } from "./components/ReviewPanel";
import { analyzeQuery, askQueryStream, reviewQuery, searchQuery } from "./lib/api";
```

Add `LiteratureReview` to the type imports:
```typescript
import type {
  AgentResult,
  Citation,
  Filters,
  LiteratureReview,
  Message,
  Mode,
  SearchResult,
  SSEDoneEvent,
} from "./types";
```

- [ ] **Step 2: Add state variables**

Add after the `analyzing` state (around line 39):

```typescript
const [reviewResult, setReviewResult] = useState<LiteratureReview | null>(null);
const [reviewing, setReviewing] = useState(false);
```

- [ ] **Step 3: Clear review state in handleClear and handleSend**

In the `handleClear` function, add:
```typescript
setReviewResult(null);
setReviewing(false);
```

In the `handleSend` function, add alongside the existing `setAgentResults([])` (around line 89):
```typescript
setReviewResult(null);
```

- [ ] **Step 4: Add handleReview function**

Add after `handleAnalyze` (around line 245):

```typescript
const handleReview = async () => {
  const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
  const query = lastUserMsg?.content ?? "";
  if (!query) return;

  setReviewing(true);
  setReviewResult(null);

  try {
    const result = await reviewQuery({
      query,
      year_min: filters.year_min,
      year_max: filters.year_max,
      top_k: filters.top_k,
      search_mode: filters.search_mode,
    });
    setReviewResult(result);
  } catch (err) {
    const errorMsg: Message = {
      id: crypto.randomUUID(),
      role: "error",
      content: err instanceof Error ? err.message : "Review generation failed",
    };
    setMessages((prev) => [...prev, errorMsg]);
  } finally {
    setReviewing(false);
  }
};
```

- [ ] **Step 5: Add button and ReviewPanel to sidebar**

In the sidebar `<aside>` section, add the "Generate Literature Review" button after the "Analyze with Agents" button, and add `<ReviewPanel>` above `<AgentResultsPanel>`:

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
{(searchResults.length > 0 || citations.length > 0) && (
  <button
    onClick={handleReview}
    disabled={reviewing}
    className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
  >
    {reviewing ? "Generating Review..." : "Generate Literature Review"}
  </button>
)}
<ReviewPanel review={reviewResult} loading={reviewing} />
<AgentResultsPanel agentResults={agentResults} loading={analyzing} totalAgents={AGENT_NAMES.length} />
```

- [ ] **Step 6: Verify the app compiles**

Run: `cd capstone/frontend && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 7: Commit**

```bash
git add capstone/frontend/src/App.tsx
git commit -m "feat: integrate literature review button and panel into App"
```
