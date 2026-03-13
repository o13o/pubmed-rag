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
