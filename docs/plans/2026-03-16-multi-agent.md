# Multi-Agent Research Analysis Layer — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 independent research analysis agents (Retrieval, Methodology Critic, Statistical Reviewer, Clinical Applicability, Summarization) exposed via `POST /analyze`, with agent logic reusable as DeepEval metrics.

**Architecture:** New `src/agents/` package with a `BaseAgent` protocol. Each agent is a specialized LLM prompt returning structured JSON (`AgentResult`). A registry maps agent names to classes (lazy evaluation — registry is built on each `get_agents()` call, not at import time). A new `/analyze` route accepts search results and runs selected agents. Frontend adds an "Analyze" button and `AgentResultsPanel` component.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, LiteLLM, DeepEval, React 18, TypeScript, Tailwind CSS

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Minimum Viable Scope:** Methodology Critic, Clinical Applicability, Summarization (3 agents). Retrieval and Statistical Reviewer are nice-to-have.

---

## File Structure

```
backend/src/agents/
    __init__.py                    # Package init, re-exports
    base.py                        # BaseAgent Protocol
    registry.py                    # get_agents() with lazy registry
    retrieval.py                   # RetrievalAgent
    methodology_critic.py          # MethodologyCriticAgent
    statistical_reviewer.py        # StatisticalReviewerAgent
    clinical_applicability.py      # ClinicalApplicabilityAgent
    summarization.py               # SummarizationAgent

backend/src/shared/models.py      # Add Finding, AgentResult (existing file)
backend/src/api/main.py           # Register analyze router (existing file)
backend/src/api/routes/analyze.py # POST /analyze endpoint
                                   # Note: AnalyzeRequest/AnalyzeResponse defined locally
                                   # in the route file, following the existing pattern
                                   # (AskRequest/AskResponse are in ask.py, not models.py).
                                   # Spec lists routes/__init__.py as modified, but it is
                                   # empty and main.py imports submodules directly — no
                                   # __init__.py change needed.

backend/tests/unit/test_agents.py         # Agent unit tests (one file per agent, registry separate)
backend/tests/unit/test_api_analyze.py    # /analyze endpoint tests
backend/tests/unit/test_eval_metrics.py   # DeepEval agent-based metric tests

backend/tests/eval/metrics/custom.py      # Add agent-based metrics (existing file)

frontend/src/types/index.ts               # Add agent types (existing file)
frontend/src/lib/api.ts                   # Add analyzeQuery() (existing file)
frontend/src/App.tsx                      # Add agentResults state + Analyze button (existing file)
frontend/src/components/AgentResultsPanel.tsx  # New component
frontend/vite.config.ts                   # Add /analyze proxy (existing file)
```

---

## Chunk 1: Models + BaseAgent + Registry

### Task 1: Add Finding and AgentResult models to shared/models.py

**Files:**
- Modify: `backend/src/shared/models.py`
- Test: `backend/tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/test_models.py`:

```python
from src.shared.models import AgentResult, Finding


def test_finding_model():
    f = Finding(label="Weak sample", detail="n=12 is underpowered", severity="warning")
    assert f.label == "Weak sample"
    assert f.severity == "warning"


def test_agent_result_model():
    result = AgentResult(
        agent_name="methodology_critic",
        summary="Study design is adequate.",
        findings=[Finding(label="RCT", detail="3/5 are RCTs", severity="info")],
        confidence=0.85,
        score=7,
    )
    assert result.agent_name == "methodology_critic"
    assert result.score == 7
    assert result.details is None
    assert len(result.findings) == 1


def test_agent_result_without_score():
    result = AgentResult(
        agent_name="summarization",
        summary="Overall consensus.",
        findings=[],
        confidence=0.9,
    )
    assert result.score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_models.py::test_finding_model -v`
Expected: FAIL — `ImportError: cannot import name 'AgentResult'`

- [ ] **Step 3: Write minimal implementation**

Edit `backend/src/shared/models.py`:

First, add `Any` to imports. The existing file has a docstring on lines 1-5, then `from pydantic import BaseModel, Field` on line 7. Add `from typing import Any` as a new line before the pydantic import (line 7), preserving the docstring:

```python
from typing import Any

from pydantic import BaseModel, Field
```

Then append after the `IngestReport` class at the end of the file:

```python
class Finding(BaseModel):
    label: str
    detail: str
    severity: str  # "info" | "warning" | "critical"


class AgentResult(BaseModel):
    agent_name: str
    summary: str
    findings: list[Finding]
    confidence: float
    score: int | None = None
    details: dict[str, Any] | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_models.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/shared/models.py backend/tests/unit/test_models.py
git commit -m "feat(agents): add Finding and AgentResult models"
```

---

### Task 2: Create BaseAgent protocol, package init, and registry

**Files:**
- Create: `backend/src/agents/__init__.py`
- Create: `backend/src/agents/base.py`
- Create: `backend/src/agents/registry.py`

No tests in this task — registry tests are deferred to Task 7 (after all agents exist) because the registry imports all agent modules.

- [ ] **Step 1: Create package init**

Create `backend/src/agents/__init__.py`:

```python
"""Multi-agent research analysis layer."""
```

- [ ] **Step 2: Create BaseAgent protocol**

Create `backend/src/agents/base.py`:

```python
"""BaseAgent protocol — interface for all analysis agents.

Current implementations use LLM + specialized prompt.
Future implementations may incorporate tools (PubMed API, MeSH lookup)
without changing this interface.
"""

from typing import Protocol

from src.shared.models import AgentResult, SearchResult


class BaseAgent(Protocol):
    name: str
    description: str

    def run(self, query: str, results: list[SearchResult]) -> AgentResult: ...
```

- [ ] **Step 3: Create registry with lazy evaluation**

Create `backend/src/agents/registry.py`:

```python
"""Agent registry — maps agent names to classes.

Registry is built lazily inside get_agents() to avoid import errors
when individual agent modules don't exist yet during development.
"""

from src.agents.base import BaseAgent
from src.shared.llm import LLMClient


def get_agents(llm: LLMClient, names: list[str] | None = None) -> list[BaseAgent]:
    """Return agent instances. If names is None, return all."""
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent
    from src.agents.methodology_critic import MethodologyCriticAgent
    from src.agents.retrieval import RetrievalAgent
    from src.agents.statistical_reviewer import StatisticalReviewerAgent
    from src.agents.summarization import SummarizationAgent

    registry: dict[str, type] = {
        "retrieval": RetrievalAgent,
        "methodology_critic": MethodologyCriticAgent,
        "statistical_reviewer": StatisticalReviewerAgent,
        "clinical_applicability": ClinicalApplicabilityAgent,
        "summarization": SummarizationAgent,
    }

    if names is not None:
        registry = {k: v for k, v in registry.items() if k in names}
    return [cls(llm=llm) for cls in registry.values()]
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/agents/
git commit -m "feat(agents): add BaseAgent protocol and registry scaffold"
```

---

## Chunk 2: Agent Implementations (Must Have)

### Task 3: Implement MethodologyCriticAgent

**Files:**
- Create: `backend/src/agents/methodology_critic.py`
- Create: `backend/tests/unit/test_agent_methodology.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_agent_methodology.py`:

```python
"""Tests for MethodologyCriticAgent."""

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
        SearchResult(
            pmid="222", title="Observational Study of Drug X",
            abstract_text="An observational cohort study of 50 patients suggested Drug X may improve outcomes, though selection bias is a limitation.",
            score=0.88, year=2022, journal="BMJ", mesh_terms=["Neoplasms"],
        ),
    ]


def test_methodology_critic_returns_agent_result():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Mixed study designs with moderate rigor.",
        "findings": [
            {"label": "RCT present", "detail": "1/2 studies is an RCT", "severity": "info"},
            {"label": "Selection bias", "detail": "Observational study lacks matching", "severity": "warning"},
        ],
        "confidence": 0.8,
        "score": 6,
    })

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "methodology_critic"
    assert result.score == 6
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_methodology_critic_handles_llm_failure():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "methodology_critic"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0


def test_methodology_critic_handles_invalid_json():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_methodology.py::test_methodology_critic_returns_agent_result -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.methodology_critic'`

- [ ] **Step 3: Implement MethodologyCriticAgent**

Create `backend/src/agents/methodology_critic.py`:

```python
"""Methodology Critic Agent — evaluates study design and methodological rigor."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research methodology expert. Analyze the provided research abstracts and evaluate their study design and methodological rigor.

For each abstract, assess:
- Study design type (RCT, cohort, case-control, case report, meta-analysis, etc.)
- Sample size adequacy
- Bias risk (selection, information, confounding)
- Control group presence and appropriateness
- Blinding and randomization quality

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = poor methodology, 4-6 = moderate, 7-9 = strong, 10 = exceptional.
Return ONLY the JSON object, no explanation."""


class MethodologyCriticAgent:
    name = "methodology_critic"
    description = "Evaluates study design, bias risk, and methodological rigor"

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
            logger.warning("MethodologyCriticAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_methodology.py -v`
Expected: ALL 3 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/agents/methodology_critic.py backend/tests/unit/test_agent_methodology.py
git commit -m "feat(agents): implement MethodologyCriticAgent"
```

---

### Task 4: Implement ClinicalApplicabilityAgent

**Files:**
- Create: `backend/src/agents/clinical_applicability.py`
- Create: `backend/tests/unit/test_agent_clinical.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_agent_clinical.py`:

```python
"""Tests for ClinicalApplicabilityAgent."""

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


def test_clinical_applicability_returns_agent_result():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Findings applicable to adult oncology patients.",
        "findings": [
            {"label": "Population match", "detail": "Studies cover adult patients", "severity": "info"},
            {"label": "Dosage unspecified", "detail": "No dosage guidance provided", "severity": "warning"},
        ],
        "confidence": 0.75,
        "score": 7,
    })

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "clinical_applicability"
    assert result.score == 7


def test_clinical_applicability_handles_failure():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("timeout")

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_clinical.py::test_clinical_applicability_returns_agent_result -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ClinicalApplicabilityAgent**

Create `backend/src/agents/clinical_applicability.py`:

```python
"""Clinical Applicability Agent — assesses real-world clinical relevance."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical medicine expert. Analyze the provided research abstracts and assess their real-world clinical applicability.

For each abstract, evaluate:
- Target patient population (age, condition severity, comorbidities)
- Clinical setting applicability (primary care, specialist, hospital)
- Treatment feasibility (availability, cost, implementation complexity)
- Generalizability of findings to broader patient populations
- Safety considerations and contraindications mentioned

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = low applicability, 4-6 = moderate, 7-9 = high, 10 = directly actionable.
Return ONLY the JSON object, no explanation."""


class ClinicalApplicabilityAgent:
    name = "clinical_applicability"
    description = "Assesses real-world clinical relevance and applicability"

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
            logger.warning("ClinicalApplicabilityAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_clinical.py -v`
Expected: ALL 2 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/agents/clinical_applicability.py backend/tests/unit/test_agent_clinical.py
git commit -m "feat(agents): implement ClinicalApplicabilityAgent"
```

---

### Task 5: Implement SummarizationAgent

**Files:**
- Create: `backend/src/agents/summarization.py`
- Create: `backend/tests/unit/test_agent_summarization.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_agent_summarization.py`:

```python
"""Tests for SummarizationAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial of 500 patients showed Drug X reduced mortality by 30%.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Observational Study of Drug X",
            abstract_text="An observational cohort study suggested Drug X may improve outcomes.",
            score=0.88, year=2022, journal="BMJ", mesh_terms=["Neoplasms"],
        ),
    ]


def test_summarization_returns_agent_result():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Drug X shows promise in cancer treatment with strong RCT evidence.",
        "findings": [
            {"label": "Consistent efficacy", "detail": "Both studies report positive outcomes", "severity": "info"},
            {"label": "Conflicting methods", "detail": "RCT vs observational yields different confidence", "severity": "warning"},
        ],
        "confidence": 0.85,
    })

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "summarization"
    assert result.score is None  # Summarization does not score
    assert len(result.findings) == 2


def test_summarization_handles_failure():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("timeout")

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_summarization.py::test_summarization_returns_agent_result -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement SummarizationAgent**

Create `backend/src/agents/summarization.py`:

```python
"""Summarization Agent — synthesizes insights across multiple research studies."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research synthesis expert. Analyze the provided research abstracts and produce a comprehensive synthesis of findings.

Your synthesis should:
- Identify consensus findings across studies
- Highlight conflicting or contradictory results
- Note gaps in the evidence base
- Identify emerging trends or promising directions
- Assess the overall strength of evidence

Return your analysis as a JSON object with these exact fields:
{
  "summary": "2-3 sentence synthesis of key insights",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0
}

Note: Do NOT include a "score" field. Summarization does not score.
Return ONLY the JSON object, no explanation."""


class SummarizationAgent:
    name = "summarization"
    description = "Synthesizes insights across multiple research studies"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to synthesize:\n{abstracts_text}"

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
            logger.warning("SummarizationAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_summarization.py -v`
Expected: ALL 2 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/agents/summarization.py backend/tests/unit/test_agent_summarization.py
git commit -m "feat(agents): implement SummarizationAgent"
```

---

## Chunk 3: Agent Implementations (Nice to Have)

### Task 6: Implement RetrievalAgent

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_retrieval.py -v`
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

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/agents/retrieval.py backend/tests/unit/test_agent_retrieval.py
git commit -m "feat(agents): implement RetrievalAgent"
```

---

### Task 7: Implement StatisticalReviewerAgent + registry tests

**Files:**
- Create: `backend/src/agents/statistical_reviewer.py`
- Create: `backend/tests/unit/test_agent_statistical.py`
- Create: `backend/tests/unit/test_agent_registry.py`

- [ ] **Step 1: Write the failing test for StatisticalReviewerAgent**

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_statistical.py -v`
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

- [ ] **Step 4: Run StatisticalReviewer test**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_statistical.py -v`
Expected: PASS

- [ ] **Step 5: Now write registry tests (all agents exist)**

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
    llm = MagicMock()
    agents = get_agents(llm=llm, names=[])
    assert len(agents) == 0
```

- [ ] **Step 6: Run all agent tests**

Run: `cd capstone/backend && uv run pytest tests/unit/test_agent_*.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add backend/src/agents/statistical_reviewer.py backend/tests/unit/test_agent_statistical.py backend/tests/unit/test_agent_registry.py
git commit -m "feat(agents): implement StatisticalReviewerAgent + registry tests"
```

---

## Chunk 4: API Endpoint

### Task 8: Create POST /analyze endpoint

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
    # agents=None → get_agents called with names=None
    call_kwargs = mock_get_agents.call_args
    assert call_kwargs.kwargs.get("names") is None or call_kwargs[1].get("names") is None


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

- [ ] **Step 6: Run all existing tests to verify no regressions**

Run: `cd capstone/backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add backend/src/api/routes/analyze.py backend/src/api/main.py backend/tests/unit/test_api_analyze.py
git commit -m "feat(api): add POST /analyze multi-agent analysis endpoint"
```

---

## Chunk 5: DeepEval Metrics

### Task 9: Add agent-based DeepEval custom metrics

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
git add backend/tests/eval/metrics/custom.py backend/tests/unit/test_eval_metrics.py
git commit -m "feat(eval): add agent-based DeepEval metrics with tests"
```

---

## Chunk 6: Frontend

### Task 10: Add TypeScript types for agent analysis

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

### Task 11: Add analyzeQuery API function and Vite proxy

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

### Task 12: Create AgentResultsPanel component

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

### Task 13: Wire up Analyze button in App.tsx

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

---

### Task 14: Verify full build

- [ ] **Step 1: Run all backend tests**

Run: `cd capstone/backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run frontend build**

Run: `cd capstone/frontend && npm run build`
Expected: `dist/` directory created, no errors

- [ ] **Step 3: Commit if any cleanup needed**

If there are uncommitted fixes from the verification steps:

```bash
git add -A
git commit -m "chore: verify full build for multi-agent feature"
```
