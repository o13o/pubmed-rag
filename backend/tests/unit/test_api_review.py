"""Tests for POST /review endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import (
    AgentResult,
    Citation,
    LiteratureReview,
    SearchResult,
)


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


def _mock_review():
    return LiteratureReview(
        query="test query",
        overview="Overview text",
        main_findings="Findings text",
        gaps_and_conflicts="Gaps text",
        recommendations="Recs text",
        citations=[Citation(pmid="123", title="Test", journal="J", year=2023, relevance_score=0.9)],
        search_results=[SearchResult(pmid="123", title="Test", abstract_text="Abstract", score=0.9, year=2023, journal="J", mesh_terms=[])],
        agent_results=[AgentResult(agent_name="test", summary="ok", findings=[], confidence=0.8)],
        agents_succeeded=1,
        agents_failed=0,
    )


@patch("src.api.routes.review.ReviewPipeline")
def test_review_returns_literature_review(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _mock_review()
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["overview"] == "Overview text"
    assert len(data["citations"]) == 1
    assert data["agents_succeeded"] == 1
    mock_pipeline.run.assert_called_once()


@patch("src.api.routes.review.ReviewPipeline")
def test_review_passes_filters(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _mock_review()
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={
        "query": "cancer",
        "year_min": 2020,
        "year_max": 2025,
        "top_k": 5,
    })
    assert response.status_code == 200
    call_args = mock_pipeline.run.call_args
    filters = call_args[0][1]  # second positional arg
    assert filters.year_min == 2020
    assert filters.year_max == 2025
    assert filters.top_k == 5


def test_review_requires_query(client):
    response = client.post("/review", json={})
    assert response.status_code == 422


@patch("src.api.routes.review.ReviewPipeline")
def test_review_empty_results_returns_404(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = ValueError("No results found for query: test")
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test"})
    assert response.status_code == 404
    assert "No results" in response.json()["detail"]


@patch("src.api.routes.review.ReviewPipeline")
def test_review_synthesizer_error_returns_502(mock_pipeline_cls, client):
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = RuntimeError("LLM API error")
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post("/review", json={"query": "test"})
    assert response.status_code == 502
    assert response.json()["detail"]  # non-empty error detail
