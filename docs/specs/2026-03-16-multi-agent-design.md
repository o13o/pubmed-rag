# Multi-Agent Research Analysis Layer — Design Spec

**Date:** 2026-03-16
**Status:** Approved
**Parent Spec:** [2026-03-14-pubmed-rag-system-design.md](2026-03-14-pubmed-rag-system-design.md)

All file paths are relative to the repository root.

## 1. Goal

Add a Multi-Agent analysis layer that evaluates retrieved medical research abstracts from multiple expert perspectives (methodology, statistics, clinical applicability) and produces a synthesized research insight. Agents operate independently of the existing `/ask` RAG pipeline and analyze search results directly.

## 2. Design Decisions

### 2.1 Agent Independence

Each agent is independent with no inter-agent dependencies. Agents receive the same input (query + search results) and produce structured output. This keeps the system simple, testable, and parallelizable.

Handoff / Agent-to-Agent communication is not implemented in this version. If needed in the future, a routing/orchestrator agent can be introduced to coordinate agent execution and pass context between agents.

### 2.2 Input: Search Results Only

Agents receive `/search` results (query + list of `SearchResult` with abstracts) as input. They do not depend on `/ask` output. This keeps agents decoupled from the RAG pipeline and allows them to analyze source material directly without bias from the RAG answer.

### 2.3 Output: Common Format with Extension

All agents return a unified `AgentResult` format. Evaluation-oriented agents (Methodology Critic, Statistical Reviewer, Clinical Applicability) additionally return a 1-10 score. An optional `details` dict allows agent-specific extensions without breaking the common contract.

### 2.4 Implementation: LLM + Specialized Prompt

Each agent is a specialized system prompt + LLM call that returns structured JSON. The `BaseAgent` interface is designed so that future implementations can incorporate tools (PubMed API search, MeSH lookup, etc.) without changing the interface contract.

### 2.5 API: Synchronous JSON Endpoint

The `/analyze` endpoint returns a synchronous JSON response. If progressive/streaming UX is needed in the future, a separate `/analyze/stream` SSE endpoint can be added without modifying the synchronous endpoint.

### 2.6 Dual Use: Runtime + Evaluation

Agent logic is reusable as DeepEval custom metrics. The same agent `run()` method is called from both the `/analyze` API endpoint (runtime) and from `tests/eval/metrics/custom.py` (evaluation). This ensures consistency between runtime analysis and RAG quality evaluation, and satisfies the Requirement 2 criteria: custom evaluation for clinical relevance, evidence quality, and study design assessment.

## 3. Models

### 3.1 Finding

```python
class Finding(BaseModel):
    label: str       # e.g. "Weak sample size"
    detail: str      # Explanation
    severity: str    # "info" | "warning" | "critical"
```

### 3.2 AgentResult

```python
class AgentResult(BaseModel):
    agent_name: str
    summary: str                           # 1-2 sentence summary
    findings: list[Finding]
    confidence: float                      # 0.0-1.0
    score: int | None = None               # 1-10, evaluation agents only
    details: dict[str, Any] | None = None  # Agent-specific extensions
```

### 3.3 AnalyzeRequest

```python
class AnalyzeRequest(BaseModel):
    query: str
    results: list[SearchResult]            # From /search response
    agents: list[str] | None = None        # None = run all agents
```

### 3.4 AnalyzeResponse

```python
class AnalyzeResponse(BaseModel):
    query: str
    agent_results: list[AgentResult]
```

## 4. Agent Definitions

### 4.1 BaseAgent Protocol

```python
class BaseAgent(Protocol):
    name: str
    description: str

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        ...
```

The interface is intentionally minimal. Future tool-using agents implement the same `run()` method but internally call external tools.

### 4.2 Five Agents

| Agent | File | Responsibility | Returns Score |
|-------|------|---------------|---------------|
| Retrieval | `retrieval.py` | Evaluate relevance, coverage, and gaps in search results | No |
| Methodology Critic | `methodology_critic.py` | Evaluate study design, bias risk, methodological rigor | Yes (1-10) |
| Statistical Reviewer | `statistical_reviewer.py` | Analyze statistical methods, significance, sample sizes | Yes (1-10) |
| Clinical Applicability | `clinical_applicability.py` | Assess real-world clinical relevance and applicability | Yes (1-10) |
| Summarization | `summarization.py` | Synthesize insights across all retrieved studies | No |

Each agent uses a specialized system prompt instructing the LLM to analyze the provided abstracts and return a JSON response conforming to the `AgentResult` schema.

### 4.3 Registry

```python
# registry.py
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "retrieval": RetrievalAgent,
    "methodology_critic": MethodologyCriticAgent,
    "statistical_reviewer": StatisticalReviewerAgent,
    "clinical_applicability": ClinicalApplicabilityAgent,
    "summarization": SummarizationAgent,
}

def get_agents(names: list[str] | None = None) -> list[BaseAgent]:
    """Return agent instances. If names is None, return all."""
```

## 5. API Endpoint

### 5.1 `POST /analyze`

```
POST /analyze
Content-Type: application/json

{
  "query": "mRNA vaccine efficacy",
  "results": [ ... SearchResult objects from /search ... ],
  "agents": ["methodology_critic", "summarization"]  // optional
}
```

Response:

```json
{
  "query": "mRNA vaccine efficacy",
  "agent_results": [
    {
      "agent_name": "methodology_critic",
      "summary": "3 of 5 studies use randomized controlled trial design...",
      "findings": [
        {"label": "Strong RCT presence", "detail": "3/5 studies are RCTs", "severity": "info"},
        {"label": "Selection bias risk", "detail": "2 observational studies lack matching", "severity": "warning"}
      ],
      "confidence": 0.85,
      "score": 7,
      "details": null
    },
    {
      "agent_name": "summarization",
      "summary": "Consensus across studies suggests mRNA vaccines show high efficacy...",
      "findings": [
        {"label": "Consistent efficacy", "detail": "4/5 studies report >90% efficacy", "severity": "info"}
      ],
      "confidence": 0.9,
      "score": null,
      "details": null
    }
  ]
}
```

### 5.2 Route Registration

Add `analyze` router to `src/api/main.py` alongside existing routes.

## 6. DeepEval Integration

Agents are reused as custom DeepEval metrics in `tests/eval/metrics/custom.py`:

```python
class MethodologyQualityMetric(BaseMetric):
    def measure(self, test_case: LLMTestCase) -> float:
        agent = MethodologyCriticAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=self._parse_context(test_case))
        self.score = result.score / 10  # Normalize to 0-1
        return self.score
```

Similarly for `StatisticalValidityMetric` and `ClinicalRelevanceMetric`.

## 7. File Changes Summary

### New Files

| File | Description |
|------|-------------|
| `backend/src/agents/__init__.py` | Package init |
| `backend/src/agents/base.py` | BaseAgent Protocol, shared models (Finding, AgentResult) |
| `backend/src/agents/retrieval.py` | RetrievalAgent |
| `backend/src/agents/methodology_critic.py` | MethodologyCriticAgent |
| `backend/src/agents/statistical_reviewer.py` | StatisticalReviewerAgent |
| `backend/src/agents/clinical_applicability.py` | ClinicalApplicabilityAgent |
| `backend/src/agents/summarization.py` | SummarizationAgent |
| `backend/src/agents/registry.py` | Agent registry and factory |
| `backend/src/api/routes/analyze.py` | POST /analyze endpoint |
| `backend/tests/unit/test_agents.py` | Unit tests for all agents |
| `backend/tests/unit/test_api_analyze.py` | Endpoint tests |

### Modified Files

| File | Change |
|------|--------|
| `backend/src/shared/models.py` | Add `Finding`, `AgentResult` models |
| `backend/src/api/main.py` | Register analyze router |
| `backend/src/api/routes/__init__.py` | Export analyze router |
| `tests/eval/metrics/custom.py` | Add `MethodologyQualityMetric`, `StatisticalValidityMetric`, `ClinicalRelevanceMetric` |

### Frontend Changes

| File | Change |
|------|--------|
| `frontend/src/types/index.ts` | Add `AnalyzeRequest`, `AgentResult`, `Finding` types |
| `frontend/src/lib/api.ts` | Add `analyzeQuery()` function |
| `frontend/src/App.tsx` | Add `agentResults` state, "Analyze" button |
| `frontend/src/components/AgentResultsPanel.tsx` | New component: agent result cards with score badges |

## 8. Future Extensions (Out of Scope)

| Item | Description |
|------|-------------|
| Routing Agent | Auto-select agents based on query content |
| `/analyze/stream` (SSE) | Progressive delivery as each agent completes |
| Auto-trigger | Automatically run `/analyze` after `/ask` completes |
| Agent-to-Agent handoff | Pass context between agents for deeper analysis |
| Tool-using agents | Agents call external tools (PubMed API, MeSH lookup) |

## 9. Minimum Viable Scope

Full design covers 5 agents. Time-constrained minimum:

- **Must have (3):** Methodology Critic, Clinical Applicability, Summarization
- **Nice to have (2):** Retrieval, Statistical Reviewer

The minimum set tells a complete demo story: "evaluate research quality → assess clinical relevance → synthesize insights."
