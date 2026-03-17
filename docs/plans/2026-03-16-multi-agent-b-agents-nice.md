# Plan B: Nice-to-Have Agents (Retrieval, StatisticalReviewer) + Registry Tests

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 2 nice-to-have agents (Retrieval, StatisticalReviewer) and add registry integration tests.

**Architecture:** Same pattern as Plan A agents. Registry tests verify all 5 agents load correctly.

**Tech Stack:** Python 3.11+, Pydantic v2, LiteLLM

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Prerequisites:** Plan S completed. Plan A must also be completed before registry tests (Task 3) can pass.

**Note:** This plan is skippable if time is tight. The system works with 3 agents from Plan A.

---

## Task 1: Implement RetrievalAgent

**Files:**
- Create: `backend/src/agents/retrieval.py`
- Create: `backend/tests/unit/test_agent_retrieval.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_agent_retrieval.py`:

```python
"""Tests for RetrievalAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized trial showed efficacy.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def test_retrieval_returns_agent_result():
    from src.agents.retrieval import RetrievalAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Good relevance coverage with one gap.",
        "findings": [
            {"label": "High relevance", "detail": "Paper directly addresses the query", "severity": "info"},
            {"label": "Missing RCTs", "detail": "No large-scale RCTs in results", "severity": "warning"},
        ],
        "confidence": 0.8,
    })

    agent = RetrievalAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "retrieval"
    assert result.score is None
    assert result.confidence == 0.8


def test_retrieval_handles_llm_failure():
    from src.agents.retrieval import RetrievalAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = RetrievalAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "retrieval"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_agent_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RetrievalAgent**

Create `backend/src/agents/retrieval.py`:

```python
"""Retrieval Agent — evaluates relevance, coverage, and gaps in search results."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical literature search expert. Analyze the provided search results and evaluate their relevance and coverage for the given query.

Evaluate:
- Relevance of each result to the query
- Coverage of different aspects of the query topic
- Gaps in the retrieved literature (missing study types, populations, time periods)
- Diversity of sources (journals, study types, geographic regions)
- Whether the top results adequately address the query

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment of retrieval quality",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0
}

Note: Do NOT include a "score" field. Retrieval evaluation does not score.
Return ONLY the JSON object, no explanation."""


class RetrievalAgent:
    name = "retrieval"
    description = "Evaluates relevance, coverage, and gaps in search results"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        results_text = "\n\n".join(
            f"PMID: {r.pmid} | Score: {r.score:.3f} | Year: {r.year}\n"
            f"Title: {r.title}\nJournal: {r.journal}\nMeSH: {', '.join(r.mesh_terms)}\n"
            f"Abstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nSearch results to evaluate ({len(results)} results):\n{results_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
            )
        except Exception as e:
            logger.warning("RetrievalAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_agent_retrieval.py -v`
Expected: ALL 2 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/agents/retrieval.py backend/tests/unit/test_agent_retrieval.py
git commit -m "feat(agents): implement RetrievalAgent"
```

---

## Task 2: Implement StatisticalReviewerAgent

**Files:**
- Create: `backend/src/agents/statistical_reviewer.py`
- Create: `backend/tests/unit/test_agent_statistical.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_agent_statistical.py`:

```python
"""Tests for StatisticalReviewerAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial of 500 patients showed Drug X reduced mortality by 30% (p<0.001).",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def test_statistical_reviewer_returns_agent_result():
    from src.agents.statistical_reviewer import StatisticalReviewerAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Statistical methods are generally sound with one concern.",
        "findings": [
            {"label": "Significant result", "detail": "p<0.001 in RCT", "severity": "info"},
            {"label": "Large sample", "detail": "n=500 is adequately powered", "severity": "info"},
        ],
        "confidence": 0.7,
        "score": 8,
    })

    agent = StatisticalReviewerAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "statistical_reviewer"
    assert result.score == 8
    assert result.confidence == 0.7


def test_statistical_reviewer_handles_llm_failure():
    from src.agents.statistical_reviewer import StatisticalReviewerAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = StatisticalReviewerAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "statistical_reviewer"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_agent_statistical.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement StatisticalReviewerAgent**

Create `backend/src/agents/statistical_reviewer.py`:

```python
"""Statistical Reviewer Agent — analyzes statistical methods and validity."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a biostatistics expert. Analyze the provided research abstracts and evaluate their statistical methods and validity.

For each abstract, evaluate:
- Statistical methods used (t-test, chi-square, regression, survival analysis, etc.)
- Sample size adequacy and power
- P-values and confidence intervals reported
- Effect size magnitude and clinical significance
- Potential statistical biases or methodological flaws

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = poor statistical rigor, 4-6 = moderate, 7-9 = strong, 10 = exemplary.
Return ONLY the JSON object, no explanation."""


class StatisticalReviewerAgent:
    name = "statistical_reviewer"
    description = "Analyzes statistical methods, significance, and sample sizes"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to evaluate:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=data.get("score"),
            )
        except Exception as e:
            logger.warning("StatisticalReviewerAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_agent_statistical.py -v`
Expected: ALL 2 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/agents/statistical_reviewer.py backend/tests/unit/test_agent_statistical.py
git commit -m "feat(agents): implement StatisticalReviewerAgent"
```

---

## Task 3: Registry integration tests (requires all 5 agents)

**Files:**
- Create: `backend/tests/unit/test_agent_registry.py`

**Note:** This task requires Plan A to be completed (all 5 agents must exist). These are integration tests over already-existing code — tests pass immediately because all agents and registry already exist. No "verify failure" step is needed.

- [ ] **Step 1: Write registry tests**

Create `backend/tests/unit/test_agent_registry.py`:

```python
"""Tests for agent registry."""

from unittest.mock import MagicMock

from src.agents.registry import get_agents


def test_get_agents_returns_all():
    llm = MagicMock()
    agents = get_agents(llm=llm)
    assert len(agents) == 5
    names = {a.name for a in agents}
    assert names == {
        "retrieval", "methodology_critic", "statistical_reviewer",
        "clinical_applicability", "summarization",
    }


def test_get_agents_filters_by_name():
    llm = MagicMock()
    agents = get_agents(llm=llm, names=["methodology_critic", "summarization"])
    assert len(agents) == 2
    names = {a.name for a in agents}
    assert names == {"methodology_critic", "summarization"}


def test_get_agents_empty_list():
    """names=[] (explicit empty) returns no agents, unlike names=None (return all)."""
    llm = MagicMock()
    agents = get_agents(llm=llm, names=[])
    assert len(agents) == 0
```

- [ ] **Step 2: Run tests**

Run: `cd backend && uv run pytest tests/unit/test_agent_registry.py -v`
Expected: ALL 3 PASS

- [ ] **Step 3: Run all agent tests together**

Run: `cd backend && uv run pytest tests/unit/test_agent_*.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd capstone && git add backend/tests/unit/test_agent_registry.py
git commit -m "test(agents): add registry integration tests"
```
