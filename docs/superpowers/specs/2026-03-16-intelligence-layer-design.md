# Medical Research Intelligence Layer — Design Spec

## Overview

Add three new agents to the existing multi-agent analysis framework to fulfill Requirement 2 "Medical Research Intelligence Layer" from `capstone/statements.md`:

1. **Conflicting Findings Agent** — identifies contradictory conclusions across studies
2. **Trend Analysis Agent** — detects emerging research trends and directions
3. **Knowledge Graph Agent** — extracts entity-relationship graphs (disease, treatment, outcome)

## Approach

All three follow the identical pattern as existing agents (MethodologyCritic, StatisticalReviewer, etc.):
- Specialized `SYSTEM_PROMPT` → LLM call → parse JSON → return `AgentResult`
- Structured data goes in `AgentResult.details` (already `dict[str, Any] | None`)
- No new dependencies, models, or API routes required

## Agent Specifications

### 1. Conflicting Findings Agent

**File:** `backend/src/agents/conflicting_findings.py`

- Input: query + list[SearchResult]
- LLM prompt asks for contradictory pairs across abstracts
- `details` schema: `{"conflicts": [{"pmid_a", "pmid_b", "topic", "description"}]}`
- `score`: None (qualitative analysis)

### 2. Trend Analysis Agent

**File:** `backend/src/agents/trend_analysis.py`

- Input: query + list[SearchResult] (uses title, year, mesh_terms)
- LLM prompt asks for temporal research trends and emerging directions
- `details` schema: `{"trends": [{"topic", "direction": "increasing|decreasing|stable", "period", "evidence_count"}]}`
- `score`: None

### 3. Knowledge Graph Agent

**File:** `backend/src/agents/knowledge_graph.py`

- Input: query + list[SearchResult]
- LLM prompt asks for entities (disease, treatment, outcome, gene, biomarker) and relations (treats, causes, associated_with)
- `details` schema: `{"nodes": [{"id", "label", "type"}], "edges": [{"source", "target", "relation"}]}`
- `score`: None

## Changes to Existing Files

| File | Change |
|------|--------|
| `backend/src/agents/registry.py` | Add 3 agents to registry dict |
| `frontend/src/components/AgentResultsPanel.tsx` | Add 3 entries to `AGENT_LABELS` |

## No Changes Required

- `models.py` — `AgentResult.details` already supports arbitrary dict
- `analyze.py` — uses `get_agents()` which auto-includes registered agents
- `types/index.ts` — `AgentResult.details` already typed as `Record<string, unknown> | null`
- `base.py` — Protocol unchanged

## Tests

Three new unit test files following existing agent test pattern:
- `backend/tests/unit/test_agent_conflicting.py`
- `backend/tests/unit/test_agent_trend.py`
- `backend/tests/unit/test_agent_knowledge_graph.py`

Each test: mock LLM, verify AgentResult structure, verify error handling.
