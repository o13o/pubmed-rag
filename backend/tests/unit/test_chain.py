"""Tests for RAG chain."""

import re
from unittest.mock import MagicMock, patch

from src.shared.models import Citation, RAGResponse, SearchFilters, SearchResult
from src.rag.chain import ask


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Title 1", abstract_text="Abstract about cancer treatment.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_returns_rag_response(mock_expander_cls, mock_search):
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
    )

    assert isinstance(response, RAGResponse)
    assert response.query == "cancer treatment"
    assert len(response.answer) > 0
    assert len(response.citations) == 1


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_with_no_results(mock_expander_cls, mock_search):
    mock_search.return_value = []
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="unknown query")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "No relevant research was found."

    response = ask(
        query="unknown query",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
    )

    assert isinstance(response, RAGResponse)
    assert response.citations == []
