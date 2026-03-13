"""Tests for RAG chain with reranker and guardrails."""

from unittest.mock import MagicMock, patch

from src.shared.models import (
    Citation, GuardrailWarning, RAGResponse, SearchFilters,
    SearchResult, ValidatedResponse,
)
from src.rag.chain import ask


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Title 1",
            abstract_text="Abstract about cancer treatment.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_returns_validated_response(mock_expander_cls, mock_search):
    """With guardrails enabled, ask() should return ValidatedResponse."""
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "Based on PMID: 111, cancer treatment shows...",  # RAG answer
        "[]",  # Guardrail validation (no issues)
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    response = ask(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=True,
    )

    assert isinstance(response, ValidatedResponse)
    assert response.query == "cancer treatment"
    assert len(response.answer) > 0
    assert len(response.citations) == 1
    assert response.disclaimer != ""
    mock_reranker.rerank.assert_called_once()


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_without_guardrails(mock_expander_cls, mock_search):
    """With guardrails disabled, ask() should return RAGResponse."""
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Based on PMID: 111, cancer treatment shows..."

    response = ask(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        guardrails_enabled=False,
    )

    assert isinstance(response, RAGResponse)
    assert not isinstance(response, ValidatedResponse)


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_with_no_results(mock_expander_cls, mock_search):
    """Empty search results should still produce a response."""
    mock_search.return_value = []
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="unknown query")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "No relevant research was found.",  # RAG answer
        "[]",  # Guardrail validation
    ]

    response = ask(
        query="unknown query",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        guardrails_enabled=True,
    )

    assert isinstance(response, ValidatedResponse)
    assert response.citations == []
