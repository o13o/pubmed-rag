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
