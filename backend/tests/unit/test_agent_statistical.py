"""Tests for StatisticalReviewerAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial of 500 patients showed Drug X reduced mortality by 30% (p<0.001).",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def test_statistical_reviewer_returns_agent_result():
    from src.agents.statistical_reviewer import StatisticalReviewerAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Statistical methods are generally sound with one concern.",
        "findings": [
            {"label": "Significant result", "detail": "p<0.001 in RCT", "severity": "info"},
            {"label": "Large sample", "detail": "n=500 is adequately powered", "severity": "info"},
        ],
        "confidence": 0.7,
        "score": 8,
    })

    agent = StatisticalReviewerAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "statistical_reviewer"
    assert result.score == 8
    assert result.confidence == 0.7


def test_statistical_reviewer_handles_llm_failure():
    from src.agents.statistical_reviewer import StatisticalReviewerAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = StatisticalReviewerAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "statistical_reviewer"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None
