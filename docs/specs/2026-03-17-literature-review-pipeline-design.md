# Literature Review Pipeline Design

## Overview

Add a `POST /review` endpoint that orchestrates a 3-stage agent pipeline to generate a structured literature review. This implements Agent-to-Agent (A2A) handoff within the process — each stage's output feeds directly into the next stage as input.

## Motivation

The existing `/analyze` endpoint runs agents independently with no inter-agent communication. The requirements call for:
- Agent-to-Agent (A2A) communication for systematic review collaboration
- Handoff to full-text retrieval and meta-analysis agents
- Automated literature review generation for complex medical queries

This feature addresses all three by introducing a pipeline where agents build on each other's output. In-process handoff was chosen over HTTP-based or message-queue A2A because all agents run in the same process and share the same LLM client — external communication adds latency and complexity with no benefit for this use case.

## Architecture

```
POST /review { query, filters }
  |
  Stage 1: Search (via SearchClient)
  |         query + filters -> list[SearchResult]
  |
  Stage 2: Parallel analysis (6 agents, ThreadPoolExecutor, max_workers=6)
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

### Agent Selection Rationale

Stage 2 uses 6 of the 8 registered agents. The two excluded agents are:
- **RetrievalAgent**: Redundant — Stage 1 already performs retrieval via SearchClient.
- **SummarizationAgent**: Replaced — Stage 3 (ReviewSynthesizer) handles synthesis with richer input (agent analyses + search results).

### Handoff Model

Each stage receives the accumulated context from prior stages. This is in-process function-call handoff — no HTTP or message queue needed. Data flows via Pydantic models:

- Stage 1 output (`list[SearchResult]`) is passed to Stage 2 and Stage 3
- Stage 2 output (`list[AgentResult]`) is passed to Stage 3
- Stage 3 synthesizes everything into a `LiteratureReview`

## New Components

### 1. `backend/src/shared/models.py` — LiteratureReview model

```python
class LiteratureReview(BaseModel):
    query: str
    overview: str
    main_findings: str
    gaps_and_conflicts: str
    recommendations: str
    citations: list[Citation]
    search_results: list[SearchResult]
    agent_results: list[AgentResult]
    agents_succeeded: int
    agents_failed: int
```

### 2. `backend/src/agents/review_synthesizer.py` — ReviewSynthesizer

Does NOT implement `BaseAgent` protocol (different signature — takes agent results as additional input).

```python
class ReviewSynthesizer:
    def __init__(self, llm: LLMClient): ...

    def run(
        self,
        query: str,
        results: list[SearchResult],
        agent_results: list[AgentResult],
    ) -> LiteratureReview: ...
```

- Uses a dedicated system prompt to return JSON with 4 fields: `overview`, `main_findings`, `gaps_and_conflicts`, `recommendations`
- Builds citations from search results
- Not registered in `agents/registry.py` (pipeline-only)

### 3. `backend/src/agents/pipeline.py` — ReviewPipeline

Orchestrates the 3-stage pipeline:

```python
class ReviewPipeline:
    def __init__(self, search_client: SearchClient, llm: LLMClient):
        self.search_client = search_client
        self.llm = llm

    def run(self, query: str, filters: SearchFilters) -> LiteratureReview:
        # Stage 1: Search
        results = self.search_client.search(query, filters)

        # Stage 2: Parallel agent analysis
        agent_results = self._run_agents(query, results)

        # Stage 3: Synthesize review
        return ReviewSynthesizer(self.llm).run(query, results, agent_results)
```

- Stage 2 uses `concurrent.futures.ThreadPoolExecutor(max_workers=6)`
- Each agent has individual error handling: on failure, returns a degraded AgentResult (matching existing agent pattern) rather than aborting the pipeline
- The pipeline continues to Stage 3 with partial results if some agents fail

### Error Handling

| Stage | Failure | Behavior |
|-------|---------|----------|
| Stage 1: Search returns 0 results | HTTP 404 with message "No results found for query" |
| Stage 2: Some agents fail | Pipeline continues; failed agents return degraded AgentResult with `confidence: 0.0` |
| Stage 2: All agents fail | Pipeline continues to Stage 3; ReviewSynthesizer works with search results only |
| Stage 3: ReviewSynthesizer fails | HTTP 502 with error message |

### 4. `backend/src/api/routes/review.py` — POST /review

```python
class ReviewRequest(BaseModel):
    query: str
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
```

Response: `LiteratureReview` JSON (synchronous).

### 5. `backend/prompts/agents/review_synthesizer.yaml` — Dedicated prompt

System prompt instructs the LLM to return JSON with these fields:
- **overview**: Brief context and scope of the review
- **main_findings**: Key results across studies
- **gaps_and_conflicts**: Contradictions and evidence gaps identified
- **recommendations**: Research directions and clinical implications

The prompt includes the agent analysis summaries as structured input so the LLM can synthesize rather than re-analyze.

### 6. Frontend

- **Button**: "Generate Literature Review" in the sidebar, below "Analyze with Agents" button. Appears when search results or citations exist.
- **State**: `reviewResult: LiteratureReview | null`, `reviewing: boolean` in App.tsx
- **API function**: `reviewQuery(req: ReviewRequest): Promise<LiteratureReview>` in `lib/api.ts`
- **Type**: `LiteratureReview` added to `types/index.ts`
- **Display**: New `ReviewPanel.tsx` component renders the 4 sections as markdown. Agent results shown in a collapsible section below. Placed in sidebar above AgentResultsPanel.
- **Mode**: No new mode needed. The button triggers a standalone API call, independent of ask/search mode.

## Existing Code Changes

- `backend/src/api/main.py`: Add `review` router
- `backend/src/api/dependencies.py`: No changes (reuses existing deps)
- `backend/src/agents/registry.py`: No changes
- `backend/src/shared/models.py`: Add `LiteratureReview` model

## Testing

- Unit test for `ReviewSynthesizer` (mock LLM, verify JSON parsing and output structure)
- Unit test for `ReviewPipeline` (mock SearchClient + mock agents, verify 3-stage handoff and partial failure handling)
- Unit test for `/review` endpoint (mock pipeline via dependency override, verify request/response)
- Integration test for full pipeline (mock LLM only, verify real agent orchestration)

## Non-Goals

- SSE streaming for `/review` (synchronous is sufficient)
- Modifying existing `/analyze` behavior
- Inter-agent communication via external protocols (Google A2A protocol, etc.)
- Guardrails on review output (can be added later; the component abstracts exist)
