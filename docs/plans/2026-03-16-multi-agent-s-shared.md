# Plan S: Shared Models + BaseAgent + Registry

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Finding/AgentResult models, BaseAgent protocol, and agent registry scaffold.

**Architecture:** Models in `shared/models.py` (existing pattern), protocol in `agents/base.py`, lazy registry in `agents/registry.py`.

**Tech Stack:** Python 3.11+, Pydantic v2

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

**Prerequisites:** None — this runs first.

---

## Task 1: Add Finding and AgentResult models to shared/models.py

**Files:**
- Modify: `backend/src/shared/models.py`
- Test: `backend/tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/test_models.py`. Add `AgentResult, Finding` to the existing import block (lines 3-6), then append the test functions at the end of the file:

```python
# Add to the existing import block at the top:
# from src.shared.models import (
#     ..., AgentResult, Finding,
# )

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

Run: `cd backend && uv run pytest tests/unit/test_models.py::test_finding_model -v`
Expected: FAIL — `ImportError: cannot import name 'AgentResult'`

- [ ] **Step 3: Write minimal implementation**

Edit `backend/src/shared/models.py`:

Add `from typing import Any` immediately before the existing `from pydantic import BaseModel, Field` line, preserving the docstring:

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

Run: `cd backend && uv run pytest tests/unit/test_models.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd capstone && git add backend/src/shared/models.py backend/tests/unit/test_models.py
git commit -m "feat(agents): add Finding and AgentResult models"
```

> **Note:** `AnalyzeRequest` and `AnalyzeResponse` Pydantic models are deferred to Plan INT (the `/analyze` endpoint plan), as they are route-specific rather than shared models.

---

## Task 2: Create BaseAgent protocol, package init, and registry

**Files:**
- Create: `backend/src/agents/__init__.py`
- Create: `backend/src/agents/base.py`
- Create: `backend/src/agents/registry.py`

No tests in this task — registry tests are deferred to Plan B Task 3 (after all agents exist) because the registry imports all agent modules. The `BaseAgent` protocol and `__init__.py` are pure declarations with no runtime behavior, so testing is deferred to integration tests.

> **Spec deviation (intentional):** The spec's `get_agents(names)` signature has no `llm` parameter, but agents require an `LLMClient` instance. The registry adds `llm: LLMClient` as a required first parameter for dependency injection. This is a deliberate improvement over the spec.

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

    registry: dict[str, type[BaseAgent]] = {
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
cd capstone && git add backend/src/agents/
git commit -m "feat(agents): add BaseAgent protocol and registry scaffold"
```
