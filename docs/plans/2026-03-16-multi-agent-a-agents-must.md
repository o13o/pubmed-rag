# Plan A: Must-Have Agents (MethodologyCritic, ClinicalApplicability, Summarization)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 3 must-have analysis agents: MethodologyCritic, ClinicalApplicability, Summarization.

**Architecture:** Each agent is a class with `__init__(llm)` and `run(query, results) -> AgentResult`. Uses LLM + specialized system prompt, returns structured JSON parsed into AgentResult.

**Tech Stack:** Python 3.11+, Pydantic v2, LiteLLM

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Prerequisites:** Plan S completed (models, base, registry exist).

---

## Task 1: Implement MethodologyCriticAgent

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
    assert result.confidence == 0.8
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
    assert result.score is None


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

Run: `cd backend && uv run pytest tests/unit/test_agent_methodology.py::test_methodology_critic_returns_agent_result -v`
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

Run: `cd backend && uv run pytest tests/unit/test_agent_methodology.py -v`
Expected: ALL 3 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/agents/methodology_critic.py backend/tests/unit/test_agent_methodology.py
git commit -m "feat(agents): implement MethodologyCriticAgent"
```

---

## Task 2: Implement ClinicalApplicabilityAgent

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
    assert result.confidence == 0.75
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_clinical_applicability_handles_invalid_json():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None


def test_clinical_applicability_handles_failure():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("timeout")

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_agent_clinical.py::test_clinical_applicability_returns_agent_result -v`
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

Run: `cd backend && uv run pytest tests/unit/test_agent_clinical.py -v`
Expected: ALL 3 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/agents/clinical_applicability.py backend/tests/unit/test_agent_clinical.py
git commit -m "feat(agents): implement ClinicalApplicabilityAgent"
```

---

## Task 3: Implement SummarizationAgent

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
    assert result.confidence == 0.85
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_summarization_handles_invalid_json():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None


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

Run: `cd backend && uv run pytest tests/unit/test_agent_summarization.py::test_summarization_returns_agent_result -v`
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
  "summary": "1-2 sentence synthesis of key insights",
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

Run: `cd backend && uv run pytest tests/unit/test_agent_summarization.py -v`
Expected: ALL 3 PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/agents/summarization.py backend/tests/unit/test_agent_summarization.py
git commit -m "feat(agents): implement SummarizationAgent"
```
