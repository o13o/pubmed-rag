"""Tests for POST /analyze endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import AgentResult, Finding


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


SAMPLE_RESULTS = [
    {
        "pmid": "111",
        "title": "RCT of Drug X",
        "abstract_text": "A randomized trial showed efficacy.",
        "score": 0.95,
        "year": 2023,
        "journal": "NEJM",
        "mesh_terms": ["Neoplasms"],
    },
]


@patch("src.api.routes.analyze.get_agents")
def test_analyze_returns_agent_results(mock_get_agents, client):
    mock_agent = MagicMock()
    mock_agent.name = "methodology_critic"
    mock_agent.run.return_value = AgentResult(
        agent_name="methodology_critic",
        summary="Good methodology.",
        findings=[Finding(label="RCT", detail="Well designed", severity="info")],
        confidence=0.85,
        score=8,
    )
    mock_get_agents.return_value = [mock_agent]

    response = client.post("/analyze", json={
        "query": "cancer treatment",
        "results": SAMPLE_RESULTS,
        "agents": ["methodology_critic"],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "cancer treatment"
    assert len(data["agent_results"]) == 1
    assert data["agent_results"][0]["agent_name"] == "methodology_critic"
    assert data["agent_results"][0]["score"] == 8


@patch("src.api.routes.analyze.get_agents")
def test_analyze_all_agents_when_none_specified(mock_get_agents, client):
    mock_agent = MagicMock()
    mock_agent.name = "summarization"
    mock_agent.run.return_value = AgentResult(
        agent_name="summarization",
        summary="Summary.",
        findings=[],
        confidence=0.9,
    )
    mock_get_agents.return_value = [mock_agent]

    response = client.post("/analyze", json={
        "query": "cancer treatment",
        "results": SAMPLE_RESULTS,
    })

    assert response.status_code == 200
    assert mock_get_agents.call_args.kwargs["names"] is None


def test_analyze_requires_query(client):
    response = client.post("/analyze", json={"results": SAMPLE_RESULTS})
    assert response.status_code == 422


def test_analyze_requires_results(client):
    response = client.post("/analyze", json={"query": "test"})
    assert response.status_code == 422
