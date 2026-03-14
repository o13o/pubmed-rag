"""Tests for API health endpoint and app factory."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked services."""
    with patch("src.api.main.connections") as mock_conn, \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient") as mock_llm_cls, \
         patch("src.api.main.MeSHDatabase") as mock_mesh_cls, \
         patch("src.api.main.get_reranker") as mock_reranker:

        mock_col_instance = MagicMock()
        mock_col_instance.num_entities = 1000
        mock_col.return_value = mock_col_instance

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_cors_headers(client):
    response = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-origin") == "*"
