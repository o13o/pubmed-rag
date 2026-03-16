# Plan INT: Integration (API Endpoint + DeepEval + Build Verify)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the POST /analyze API endpoint, add agent-based DeepEval metrics, and verify the full build.

**Architecture:** New FastAPI route that accepts search results and runs selected agents. DeepEval metrics reuse agent `run()` methods. Final build verification across backend and frontend.

**Tech Stack:** Python 3.11+, FastAPI, DeepEval, React 18, TypeScript

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Prerequisites:** Plan S, Plan A, Plan B, Plan C completed. (Plan B is required because Task 2 imports `StatisticalReviewerAgent` for the `StatisticalValidityMetric`.)

---

## Task 1: Create POST /analyze endpoint

**Files:**
- Create: `backend/src/api/routes/analyze.py`
- Modify: `backend/src/api/main.py`
- Create: `backend/tests/unit/test_api_analyze.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_api_analyze.py`:

```python
"""Tests for POST /analyze endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import AgentResult, Finding


@pytest.fixture
def client():
    with patch("src.api.main.connections"), \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient"), \
         patch("src.api.main.MeSHDatabase"), \
         patch("src.api.main.get_reranker"):

        mock_col.return_value = MagicMock(num_entities=100)

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


SAMPLE_RESULTS = [
    {
        "pmid": "111",
        "title": "RCT of Drug X",
        "abstract_text": "A randomized trial showed efficacy.",
        "score": 0.95,
        "year": 2023,
        "journal": "NEJM",
        "mesh_terms": ["Neoplasms"],
    },
]


@patch("src.api.routes.analyze.get_agents")
def test_analyze_returns_agent_results(mock_get_agents, client):
    mock_agent = MagicMock()
    mock_agent.name = "methodology_critic"
    mock_agent.run.return_value = AgentResult(
        agent_name="methodology_critic",
        summary="Good methodology.",
        findings=[Finding(label="RCT", detail="Well designed", severity="info")],
        confidence=0.85,
        score=8,
    )
    mock_get_agents.return_value = [mock_agent]

    response = client.post("/analyze", json={
        "query": "cancer treatment",
        "results": SAMPLE_RESULTS,
        "agents": ["methodology_critic"],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "cancer treatment"
    assert len(data["agent_results"]) == 1
    assert data["agent_results"][0]["agent_name"] == "methodology_critic"
    assert data["agent_results"][0]["score"] == 8


@patch("src.api.routes.analyze.get_agents")
def test_analyze_all_agents_when_none_specified(mock_get_agents, client):
    mock_agent = MagicMock()
    mock_agent.name = "summarization"
    mock_agent.run.return_value = AgentResult(
        agent_name="summarization",
        summary="Summary.",
        findings=[],
        confidence=0.9,
    )
    mock_get_agents.return_value = [mock_agent]

    response = client.post("/analyze", json={
        "query": "cancer treatment",
        "results": SAMPLE_RESULTS,
    })

    assert response.status_code == 200
    assert mock_get_agents.call_args.kwargs["names"] is None


def test_analyze_requires_query(client):
    response = client.post("/analyze", json={"results": SAMPLE_RESULTS})
    assert response.status_code == 422


def test_analyze_requires_results(client):
    response = client.post("/analyze", json={"query": "test"})
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_api_analyze.py::test_analyze_returns_agent_results -v`
Expected: FAIL — 404 (route not registered)

- [ ] **Step 3: Implement /analyze endpoint**

Create `backend/src/api/routes/analyze.py`:

```python
"""POST /analyze — multi-agent research analysis endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agents.registry import get_agents
from src.api.dependencies import get_llm
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, SearchResult

router = APIRouter()


class AnalyzeRequest(BaseModel):
    query: str
    results: list[SearchResult]
    agents: list[str] | None = None


class AnalyzeResponse(BaseModel):
    query: str
    agent_results: list[AgentResult]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(
    req: AnalyzeRequest,
    llm: LLMClient = Depends(get_llm),
):
    agents = get_agents(llm=llm, names=req.agents)
    agent_results = [agent.run(req.query, req.results) for agent in agents]
    return AnalyzeResponse(query=req.query, agent_results=agent_results)
```

- [ ] **Step 4: Register analyze router in main.py**

In `backend/src/api/main.py`, change the import line:

```python
from src.api.routes import analyze, ask, health, search
```

And in `create_app()`, add after the search router line:

```python
    app.include_router(analyze.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_api_analyze.py -v`
Expected: ALL 4 PASS

- [ ] **Step 6: Commit**

```bash
cd capstone && git add backend/src/api/routes/analyze.py backend/src/api/main.py backend/tests/unit/test_api_analyze.py
git commit -m "feat(api): add POST /analyze multi-agent analysis endpoint"
```

---

## Task 2: Add agent-based DeepEval custom metrics

**Files:**
- Modify: `backend/tests/eval/metrics/custom.py`
- Create: `backend/tests/unit/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_eval_metrics.py`:

```python
"""Tests for agent-based DeepEval custom metrics."""

import json
from unittest.mock import MagicMock, patch

from deepeval.test_case import LLMTestCase


def test_methodology_quality_metric():
    from tests.eval.metrics.custom import MethodologyQualityMetric

    with patch("tests.eval.metrics.custom.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "Adequate methodology.",
            "findings": [],
            "confidence": 0.8,
            "score": 7,
        })
        mock_llm_cls.return_value = mock_llm

        metric = MethodologyQualityMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Drug X is effective.",
            retrieval_context=[
                "PMID: 111\nRCT of Drug X\nA randomized trial of 500 patients.",
            ],
        )
        score = metric.measure(test_case)
        assert score == 0.7  # 7/10
        assert metric.is_successful()


def test_clinical_relevance_metric():
    from tests.eval.metrics.custom import ClinicalRelevanceMetric

    with patch("tests.eval.metrics.custom.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "Highly applicable.",
            "findings": [],
            "confidence": 0.9,
            "score": 9,
        })
        mock_llm_cls.return_value = mock_llm

        metric = ClinicalRelevanceMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Drug X is effective.",
            retrieval_context=[
                "PMID: 111\nRCT of Drug X\nA randomized trial.",
            ],
        )
        score = metric.measure(test_case)
        assert score == 0.9  # 9/10
        assert metric.is_successful()


def test_metric_with_no_context():
    from tests.eval.metrics.custom import MethodologyQualityMetric

    with patch("tests.eval.metrics.custom.LLMClient"):
        metric = MethodologyQualityMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Some answer.",
            retrieval_context=[],
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert not metric.is_successful()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_eval_metrics.py::test_methodology_quality_metric -v`
Expected: FAIL — `ImportError: cannot import name 'MethodologyQualityMetric'`

- [ ] **Step 3: Add agent-based metrics to custom.py**

Append to `backend/tests/eval/metrics/custom.py`:

```python
from src.agents.methodology_critic import MethodologyCriticAgent
from src.agents.statistical_reviewer import StatisticalReviewerAgent
from src.agents.clinical_applicability import ClinicalApplicabilityAgent
from src.shared.llm import LLMClient
from src.shared.models import SearchResult


def _parse_retrieval_context(test_case: LLMTestCase) -> list[SearchResult]:
    """Convert DeepEval retrieval_context strings to SearchResult objects."""
    results = []
    for i, ctx in enumerate(test_case.retrieval_context or []):
        lines = ctx.strip().split("\n")
        pmid = lines[0].replace("PMID: ", "").strip() if lines else str(i)
        title = lines[1].strip() if len(lines) > 1 else ""
        abstract = "\n".join(lines[2:]).strip() if len(lines) > 2 else ctx
        results.append(SearchResult(
            pmid=pmid, title=title, abstract_text=abstract,
            score=1.0, year=2023, journal="", mesh_terms=[],  # year/journal are fallback defaults
        ))
    return results


class MethodologyQualityMetric(BaseMetric):
    """Evaluate study methodology quality using MethodologyCriticAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = MethodologyCriticAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Methodology Quality"


class StatisticalValidityMetric(BaseMetric):
    """Evaluate statistical validity using StatisticalReviewerAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = StatisticalReviewerAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Statistical Validity"


class ClinicalRelevanceMetric(BaseMetric):
    """Evaluate clinical relevance using ClinicalApplicabilityAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = ClinicalApplicabilityAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Clinical Relevance"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_eval_metrics.py -v`
Expected: ALL 3 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/tests/eval/metrics/custom.py backend/tests/unit/test_eval_metrics.py
git commit -m "feat(eval): add agent-based DeepEval metrics with tests"
```

---

## Task 3: Verify full build

- [ ] **Step 1: Run all backend tests**

Run: `cd capstone/backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run frontend build**

Run: `cd capstone/frontend && npm run build`
Expected: `dist/` directory created, no errors

- [ ] **Step 3: Commit if any cleanup needed**

If there are uncommitted fixes from the verification steps:

```bash
cd capstone && git add -A
git commit -m "chore: verify full build for multi-agent feature"
```
