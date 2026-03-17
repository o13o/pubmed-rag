# Phase B-1: Output Guardrails

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build output validation (citation grounding, hallucination detection, medical terminology validation, treatment recommendation guard, disclaimer) and lightweight input classification.

**Architecture:** `GuardrailValidator` class with dependency injection. Three internal components: LLM validator (single call for grounding + hallucination + treatment), MeSH validator (DuckDB term check), disclaimer (fixed text). Input guardrail is a separate lightweight LLM classifier.

**Tech Stack:** pydantic, LiteLLM, DuckDB (MeSH)

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md) — Section 4

**Dependency:** B-S (shared models + config) must be merged first. Uses `GuardrailWarning`, `ValidatedResponse` from `shared/models.py`.

---

## Chunk 1: Output Guardrails

### Task 1: Output Guardrails — LLM Validator + MeSH Validator + Disclaimer

**Files:**
- Create: `backend/src/guardrails/output.py`
- Create: `backend/tests/unit/test_guardrails_output.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_guardrails_output.py
"""Tests for output guardrails."""

import json
from unittest.mock import MagicMock

import pytest

from src.shared.models import (
    Citation,
    GuardrailWarning,
    RAGResponse,
    SearchResult,
    ValidatedResponse,
)
from src.guardrails.output import GuardrailValidator, MEDICAL_DISCLAIMER


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def search_results():
    return [
        SearchResult(
            pmid="111", title="Cancer Treatment Study",
            abstract_text="Drug X showed 40% improvement in survival rates.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


@pytest.fixture
def rag_response():
    return RAGResponse(
        answer="Drug X showed 40% improvement [PMID: 111]. Drug Y is also effective.",
        citations=[Citation(pmid="111", title="Cancer Treatment Study", journal="Nature", year=2023, relevance_score=0.95)],
        query="cancer treatment",
    )


def test_validate_grounded_response(mock_llm, mesh_db, rag_response, search_results):
    """Fully grounded response should pass with is_grounded=True."""
    mock_llm.complete.return_value = json.dumps([])  # No issues found

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(rag_response, search_results)

    assert isinstance(result, ValidatedResponse)
    assert result.is_grounded is True
    assert result.disclaimer == MEDICAL_DISCLAIMER
    assert result.answer == rag_response.answer


def test_validate_ungrounded_response(mock_llm, mesh_db, rag_response, search_results):
    """Response with ungrounded claims should have warnings and is_grounded=False."""
    mock_llm.complete.return_value = json.dumps([
        {"check": "citation_grounding", "severity": "error",
         "message": "Claim about Drug Y not supported by any abstract",
         "span": "Drug Y is also effective"},
    ])

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(rag_response, search_results)

    assert result.is_grounded is False
    assert len(result.warnings) >= 1
    grounding_warnings = [w for w in result.warnings if w.check == "citation_grounding"]
    assert len(grounding_warnings) == 1


def test_validate_hallucination_detection(mock_llm, mesh_db, rag_response, search_results):
    """Hallucinated facts should be flagged as warnings."""
    mock_llm.complete.return_value = json.dumps([
        {"check": "hallucination", "severity": "warning",
         "message": "Statistic not found in source material",
         "span": "some hallucinated fact"},
    ])

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(rag_response, search_results)

    hallucination_warnings = [w for w in result.warnings if w.check == "hallucination"]
    assert len(hallucination_warnings) == 1


def test_validate_treatment_recommendation(mock_llm, mesh_db, rag_response, search_results):
    """Unqualified treatment recommendations should be flagged."""
    mock_llm.complete.return_value = json.dumps([
        {"check": "treatment_recommendation", "severity": "warning",
         "message": "Definitive recommendation without hedging",
         "span": "Patients should take Drug X"},
    ])

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(rag_response, search_results)

    treatment_warnings = [w for w in result.warnings if w.check == "treatment_recommendation"]
    assert len(treatment_warnings) == 1


def test_validate_mesh_terminology(mock_llm, mesh_db, rag_response, search_results):
    """Medical terms not in MeSH should get terminology warnings."""
    # LLM finds no issues
    mock_llm.complete.return_value = json.dumps([])

    # Response that mentions a real MeSH term and a fake one
    response = RAGResponse(
        answer="Neoplasms treatment involves FakeDrugXYZ therapy.",
        citations=[], query="test",
    )

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(response, search_results)

    # The validator extracts capitalized medical-looking terms and checks MeSH
    # "Neoplasms" is valid MeSH, "FakeDrugXYZ" is not
    assert result.disclaimer == MEDICAL_DISCLAIMER


def test_validate_disclaimer_always_present(mock_llm, mesh_db, search_results):
    """Disclaimer should always be present, even with empty response."""
    mock_llm.complete.return_value = json.dumps([])
    response = RAGResponse(answer="No results found.", citations=[], query="test")

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(response, search_results)

    assert MEDICAL_DISCLAIMER in result.disclaimer


def test_validate_malformed_llm_response(mock_llm, mesh_db, rag_response, search_results):
    """If LLM returns non-JSON, validator should not crash."""
    mock_llm.complete.return_value = "This is not JSON"

    validator = GuardrailValidator(llm=mock_llm, mesh_db=mesh_db)
    result = validator.validate(rag_response, search_results)

    # Should still return a valid response, just without LLM-based warnings
    assert isinstance(result, ValidatedResponse)
    assert result.disclaimer == MEDICAL_DISCLAIMER
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_guardrails_output.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.guardrails.output'`

- [ ] **Step 3: Implement output guardrails**

```python
# src/guardrails/output.py
"""Output validation: citation grounding, hallucination detection, terminology check, disclaimer."""

import json
import logging
import re

from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    GuardrailWarning,
    RAGResponse,
    SearchResult,
    ValidatedResponse,
)

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = (
    "Disclaimer: This information is generated from research abstracts and is intended "
    "for educational purposes only. It does not constitute medical advice. Always consult "
    "a qualified healthcare professional for medical decisions."
)

VALIDATION_SYSTEM_PROMPT = """You are a medical content validator. Given an answer and its source abstracts, check for:
1. Citation grounding: Is each claim in the answer supported by a cited abstract?
2. Hallucination: Are there facts (drug names, statistics, outcomes) not found in source material?
3. Treatment recommendations: Are there definitive treatment recommendations without hedging language?

Return a JSON array of issues. Each issue has: check, severity, message, span.
- check: "citation_grounding" | "hallucination" | "treatment_recommendation"
- severity: "error" for ungrounded claims, "warning" for others
- message: brief description
- span: the problematic text from the answer

If no issues found, return an empty array: []
Return ONLY the JSON array, no explanation."""


class GuardrailValidator:
    """Output guardrails with dependency injection."""

    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase):
        self.llm = llm
        self.mesh_db = mesh_db

    def validate(
        self, response: RAGResponse, search_results: list[SearchResult]
    ) -> ValidatedResponse:
        """Run all output validation checks."""
        warnings: list[GuardrailWarning] = []

        # 1. LLM-based validation (grounding + hallucination + treatment)
        llm_warnings = self._llm_validate(response, search_results)
        warnings.extend(llm_warnings)

        # 2. MeSH terminology validation
        mesh_warnings = self._mesh_validate(response.answer)
        warnings.extend(mesh_warnings)

        # Determine grounding status
        has_grounding_errors = any(
            w.check == "citation_grounding" and w.severity == "error"
            for w in warnings
        )

        return ValidatedResponse(
            answer=response.answer,
            citations=response.citations,
            query=response.query,
            warnings=warnings,
            disclaimer=MEDICAL_DISCLAIMER,
            is_grounded=not has_grounding_errors,
        )

    def _llm_validate(
        self, response: RAGResponse, search_results: list[SearchResult]
    ) -> list[GuardrailWarning]:
        """Use LLM to check grounding, hallucination, and treatment recommendations."""
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in search_results
        )

        user_prompt = f"""Answer to validate:
{response.answer}

Source abstracts:
{abstracts_text}

Check the answer against the source abstracts and return a JSON array of issues."""

        try:
            result = self.llm.complete(
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            issues = json.loads(result.strip())
            if not isinstance(issues, list):
                return []
            return [
                GuardrailWarning(
                    check=issue.get("check", "unknown"),
                    severity=issue.get("severity", "warning"),
                    message=issue.get("message", ""),
                    span=issue.get("span", ""),
                )
                for issue in issues
            ]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM validation failed: %s", e)
            return []

    def _mesh_validate(self, answer: str) -> list[GuardrailWarning]:
        """Check medical terms in the answer against MeSH vocabulary."""
        warnings = []
        # Extract capitalized multi-word terms that look medical
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)
        # Deduplicate
        seen = set()
        for term in terms:
            if term in seen or len(term) < 4:
                continue
            seen.add(term)
            if not self.mesh_db.validate_term(term):
                warnings.append(
                    GuardrailWarning(
                        check="terminology",
                        severity="warning",
                        message=f"Term '{term}' not found in MeSH vocabulary",
                        span=term,
                    )
                )
        return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_guardrails_output.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/guardrails/output.py backend/tests/unit/test_guardrails_output.py
git commit -m "feat(guardrails): add output validation (LLM + MeSH + disclaimer)"
```

---

### Task 2: Input Guardrail (Lightweight)

**Files:**
- Create: `backend/src/guardrails/input.py`
- Create: `backend/tests/unit/test_guardrails_input.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_guardrails_input.py
"""Tests for input guardrails."""

from unittest.mock import MagicMock

from src.guardrails.input import classify_medical_relevance


def test_medical_query_classified_relevant():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "yes"
    result = classify_medical_relevance("breast cancer treatment", mock_llm)
    assert result.is_relevant is True


def test_non_medical_query_classified_irrelevant():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "no"
    result = classify_medical_relevance("best pizza in town", mock_llm)
    assert result.is_relevant is False
    assert len(result.warning) > 0


def test_malformed_llm_response_defaults_relevant():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "maybe something else"
    result = classify_medical_relevance("some query", mock_llm)
    # Default to relevant (don't block queries)
    assert result.is_relevant is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_guardrails_input.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement input guardrail**

```python
# src/guardrails/input.py
"""Lightweight input classification: is the query medical/biomedical?"""

import logging

from pydantic import BaseModel

from src.shared.llm import LLMClient

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """Is the following query related to medical or biomedical research?
Answer with only "yes" or "no".

Query: "{query}"
Answer:"""


class RelevanceResult(BaseModel):
    is_relevant: bool
    warning: str = ""


def classify_medical_relevance(query: str, llm: LLMClient) -> RelevanceResult:
    """Classify whether a query is medical/biomedical (soft warning, does not block)."""
    try:
        response = llm.complete(
            system_prompt="You classify queries as medical or non-medical. Answer only yes or no.",
            user_prompt=CLASSIFICATION_PROMPT.format(query=query),
        )
        answer = response.strip().lower()
        if answer.startswith("no"):
            return RelevanceResult(
                is_relevant=False,
                warning="This query may not be related to medical research. Results may be less relevant.",
            )
        return RelevanceResult(is_relevant=True)
    except Exception as e:
        logger.warning("Input classification failed: %s", e)
        return RelevanceResult(is_relevant=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_guardrails_input.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/guardrails/input.py backend/tests/unit/test_guardrails_input.py
git commit -m "feat(guardrails): add lightweight input medical topic classifier"
```

---

### Task 3: Guardrails Public Interface

**Files:**
- Modify: `backend/src/guardrails/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
# src/guardrails/__init__.py
"""Guardrails module - public interface."""

from src.guardrails.output import GuardrailValidator, MEDICAL_DISCLAIMER
from src.guardrails.input import classify_medical_relevance

__all__ = ["GuardrailValidator", "MEDICAL_DISCLAIMER", "classify_medical_relevance"]
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/guardrails/__init__.py
git commit -m "feat(guardrails): add public interface"
```
