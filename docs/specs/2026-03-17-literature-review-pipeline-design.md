# Literature Review Pipeline Design

## Overview

Add a `/review` endpoint that orchestrates a 3-stage agent pipeline to generate a structured literature review. This implements Agent-to-Agent (A2A) handoff within the process — each stage's output feeds directly into the next stage as input.

## Motivation

The existing `/analyze` endpoint runs agents independently with no inter-agent communication. The requirements call for:
- Agent-to-Agent (A2A) communication for systematic review collaboration
- Handoff to full-text retrieval and meta-analysis agents
- Automated literature review generation for complex medical queries

This feature addresses all three by introducing a pipeline where agents build on each other's output.

## Architecture

```
POST /review { query, filters }
  |
  Stage 1: Search (via SearchClient)
  |         query + filters -> list[SearchResult]
  |
  Stage 2: Parallel analysis (6 agents, ThreadPoolExecutor)
  |         query + SearchResult[] -> AgentResult[] (6 results)
  |         - MethodologyCriticAgent
  |         - StatisticalReviewerAgent
  |         - ClinicalApplicabilityAgent
  |         - ConflictingFindingsAgent
  |         - TrendAnalysisAgent
  |         - KnowledgeGraphAgent
  |
  Stage 3: ReviewSynthesizer
            query + SearchResult[] + AgentResult[] -> LiteratureReview
```

### Handoff Model

Each stage receives the accumulated context from prior stages. This is in-process function-call handoff — no HTTP or message queue needed. Data flows via Pydantic models:

- Stage 1 output (`list[SearchResult]`) is passed to Stage 2 and Stage 3
- Stage 2 output (`list[AgentResult]`) is passed to Stage 3
- Stage 3 synthesizes everything into a `LiteratureReview`

## New Components

### 1. `src/shared/models.py` — LiteratureReview model

```python
class LiteratureReview(BaseModel):
    query: str
    overview: str
    main_findings: str
    gaps_and_conflicts: str
    recommendations: str
    citations: list[Citation]
    agent_results: list[AgentResult]
```

### 2. `src/agents/review_synthesizer.py` — ReviewSynthesizer

- Takes query, search results, and all 6 AgentResults
- Uses a dedicated system prompt to produce the 4-section review (overview, main_findings, gaps_and_conflicts, recommendations)
- Returns `LiteratureReview`
- Not registered in `agents/registry.py` (pipeline-only, not available via `/analyze`)

### 3. `src/agents/pipeline.py` — ReviewPipeline

Orchestrates the 3-stage pipeline:

```python
class ReviewPipeline:
    def __init__(self, search_client, llm, mesh_db, reranker):
        ...

    def run(self, query: str, filters: SearchFilters) -> LiteratureReview:
        # Stage 1: Search
        results = self._search(query, filters)

        # Stage 2: Parallel agent analysis
        agent_results = self._analyze(query, results)

        # Stage 3: Synthesize review
        return self._synthesize(query, results, agent_results)
```

- Stage 2 uses `concurrent.futures.ThreadPoolExecutor` for parallel agent execution (6 agents)
- Each agent receives the same query + search results (same interface as `/analyze`)

### 4. `src/api/routes/review.py` — POST /review

Request:
```json
{
  "query": "Latest treatments for early-stage pancreatic cancer",
  "year_min": 2020,
  "top_k": 10,
  "search_mode": "hybrid"
}
```

Response: `LiteratureReview` JSON (synchronous).

### 5. `prompts/agents/review_synthesizer.yaml` — Dedicated prompt

System prompt instructs the LLM to synthesize agent analyses into a 4-section review:
- **Overview**: Brief context and scope
- **Main Findings**: Key results across studies
- **Gaps & Conflicts**: Contradictions and evidence gaps
- **Recommendations**: Research directions and clinical implications

### 6. Frontend

- "Generate Literature Review" button (appears when search results or citations exist)
- Review display panel with the 4 sections rendered as markdown
- Shows agent results in a collapsible section below the review

## Existing Code Changes

- `src/api/main.py`: Add `review` router
- `src/api/dependencies.py`: No changes (reuses existing deps)
- `src/agents/registry.py`: No changes (ReviewSynthesizer is pipeline-only)
- `src/shared/models.py`: Add `LiteratureReview` model

## Testing

- Unit test for `ReviewSynthesizer` (mock LLM, verify output structure)
- Unit test for `ReviewPipeline` (mock search client + agents, verify stage handoff)
- Unit test for `/review` endpoint (mock pipeline, verify request/response)

## Non-Goals

- SSE streaming for `/review` (synchronous is sufficient)
- Modifying existing `/analyze` behavior
- Inter-agent communication via external protocols (Google A2A protocol, etc.)
