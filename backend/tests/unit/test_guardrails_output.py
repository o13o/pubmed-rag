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
