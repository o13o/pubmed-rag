"""Tests for POST /ask endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import Citation, RAGResponse, ValidatedResponse


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


@patch("src.api.routes.ask.rag_ask")
def test_ask_returns_answer(mock_ask, client):
    mock_ask.return_value = ValidatedResponse(
        answer="Test answer with [PMID: 123].",
        citations=[Citation(pmid="123", title="Test", journal="Nature", year=2023, relevance_score=0.95)],
        query="test query",
        warnings=[],
        disclaimer="Disclaimer text.",
        is_grounded=True,
    )
    response = client.post("/ask", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["query"] == "test query"
    assert len(data["citations"]) == 1


def test_ask_requires_query(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


@patch("src.api.routes.ask.ask_stream")
def test_ask_stream_returns_sse(mock_ask_stream, client):
    mock_ask_stream.return_value = iter([
        {"event": "token", "data": {"text": "Hello"}},
        {"event": "token", "data": {"text": " world"}},
        {"event": "done", "data": {
            "citations": [{"pmid": "123", "title": "Test", "journal": "Nature", "year": 2023, "relevance_score": 0.95}],
            "warnings": [],
            "disclaimer": "Disclaimer.",
            "is_grounded": True,
        }},
    ])

    response = client.post("/ask", json={"query": "test", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "event: token" in body
    assert '"text": "Hello"' in body or '"text":"Hello"' in body
    assert "event: done" in body
    assert '"citations"' in body


@patch("src.api.routes.ask.ask_stream")
def test_ask_stream_sse_headers(mock_ask_stream, client):
    mock_ask_stream.return_value = iter([
        {"event": "done", "data": {"citations": [], "warnings": [], "disclaimer": "", "is_grounded": True}},
    ])

    response = client.post("/ask", json={"query": "test", "stream": True})
    assert response.headers.get("cache-control") == "no-cache"


@patch("src.api.routes.ask.rag_ask")
def test_ask_stream_false_returns_json(mock_ask, client):
    """stream=false (default) should still return JSON."""
    mock_ask.return_value = ValidatedResponse(
        answer="Test answer.", citations=[], query="test",
        warnings=[], disclaimer="Disclaimer.", is_grounded=True,
    )
    response = client.post("/ask", json={"query": "test", "stream": False})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer."
