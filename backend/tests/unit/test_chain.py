"""Tests for RAG chain with reranker and guardrails."""

from unittest.mock import MagicMock, patch

from src.shared.models import (
    Citation, GuardrailWarning, RAGResponse, SearchFilters,
    SearchResult, ValidatedResponse,
)
from src.rag.chain import ask, ask_stream


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Title 1",
            abstract_text="Abstract about cancer treatment.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


def _mock_search_client(results=None):
    client = MagicMock()
    client.search.return_value = results if results is not None else _mock_search_results()
    return client


@patch("src.rag.chain.QueryExpander")
def test_ask_returns_validated_response(mock_expander_cls):
    """With guardrails enabled, ask() should return ValidatedResponse."""
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
        search_client=_mock_search_client(),
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


@patch("src.rag.chain.QueryExpander")
def test_ask_without_guardrails(mock_expander_cls):
    """With guardrails disabled, ask() should return RAGResponse."""
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Based on PMID: 111, cancer treatment shows..."

    response = ask(
        query="cancer treatment",
        search_client=_mock_search_client(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        guardrails_enabled=False,
    )

    assert isinstance(response, RAGResponse)
    assert not isinstance(response, ValidatedResponse)


@patch("src.rag.chain.QueryExpander")
def test_ask_with_no_results(mock_expander_cls):
    """Empty search results should still produce a response."""
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
        search_client=_mock_search_client(results=[]),
        llm=mock_llm,
        mesh_db=MagicMock(),
        guardrails_enabled=True,
    )

    assert isinstance(response, ValidatedResponse)
    assert response.citations == []


@patch("src.rag.chain.QueryExpander")
def test_ask_stream_yields_token_and_done_events(mock_expander_cls):
    """ask_stream() should yield token events then a done event."""
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete_stream.return_value = iter(["Based on ", "research..."])
    # Guardrail validation LLM call
    mock_llm.complete.return_value = "[]"

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    events = list(ask_stream(
        query="cancer treatment",
        search_client=_mock_search_client(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=True,
    ))

    # Should have 2 token events + 1 done event
    token_events = [e for e in events if e["event"] == "token"]
    done_events = [e for e in events if e["event"] == "done"]

    assert len(token_events) == 2
    assert token_events[0]["data"]["text"] == "Based on "
    assert token_events[1]["data"]["text"] == "research..."

    assert len(done_events) == 1
    done_data = done_events[0]["data"]
    assert "citations" in done_data
    assert "warnings" in done_data
    assert "disclaimer" in done_data
    assert "is_grounded" in done_data
    assert len(done_data["citations"]) == 1


@patch("src.rag.chain.QueryExpander")
def test_ask_stream_without_guardrails(mock_expander_cls):
    """ask_stream() without guardrails should still yield done with empty warnings."""
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete_stream.return_value = iter(["answer"])

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    events = list(ask_stream(
        query="cancer treatment",
        search_client=_mock_search_client(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=False,
    ))

    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["data"]["warnings"] == []
    assert done_events[0]["data"]["disclaimer"] == ""


@patch("src.rag.chain.QueryExpander")
def test_ask_stream_yields_error_on_exception(mock_expander_cls):
    """ask_stream() should yield an error event if an exception occurs."""
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="test")
    mock_expander_cls.return_value = mock_expander

    mock_search_client = MagicMock()
    mock_search_client.search.side_effect = RuntimeError("Milvus connection lost")

    events = list(ask_stream(
        query="test",
        search_client=mock_search_client,
        llm=MagicMock(),
        mesh_db=MagicMock(),
    ))

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "Milvus connection lost" in events[0]["data"]["message"]
