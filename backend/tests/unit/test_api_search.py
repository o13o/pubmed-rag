"""Tests for POST /search endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import SearchResult


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


@patch("src.retrieval.client.LocalSearchClient.search")
def test_search_returns_results(mock_search, client):
    mock_search.return_value = [
        SearchResult(
            pmid="123", title="Test", abstract_text="Abstract",
            score=0.95, year=2023, journal="Nature", mesh_terms=[],
        ),
    ]
    response = client.post("/search", json={"query": "cancer treatment"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["pmid"] == "123"


@patch("src.retrieval.client.LocalSearchClient.search")
def test_search_passes_publication_types_filter(mock_search, client):
    mock_search.return_value = []
    response = client.post("/search", json={
        "query": "cancer",
        "publication_types": ["Randomized Controlled Trial", "Meta-Analysis"],
        "mesh_categories": ["Neoplasms"],
    })
    assert response.status_code == 200
    call_args = mock_search.call_args
    filters = call_args[0][1]  # second positional arg to search(query, filters)
    assert filters.publication_types == ["Randomized Controlled Trial", "Meta-Analysis"]
    assert filters.mesh_categories == ["Neoplasms"]


def test_search_requires_query(client):
    response = client.post("/search", json={})
    assert response.status_code == 422
