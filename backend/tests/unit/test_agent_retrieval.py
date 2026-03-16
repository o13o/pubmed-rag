"""Tests for RetrievalAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized trial showed efficacy.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def test_retrieval_returns_agent_result():
    from src.agents.retrieval import RetrievalAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Good relevance coverage with one gap.",
        "findings": [
            {"label": "High relevance", "detail": "Paper directly addresses the query", "severity": "info"},
            {"label": "Missing RCTs", "detail": "No large-scale RCTs in results", "severity": "warning"},
        ],
        "confidence": 0.8,
    })

    agent = RetrievalAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "retrieval"
    assert result.score is None
    assert result.confidence == 0.8


def test_retrieval_handles_llm_failure():
    from src.agents.retrieval import RetrievalAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = RetrievalAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "retrieval"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None
