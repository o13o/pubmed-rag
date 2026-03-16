"""Tests for TrendAnalysisAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="Immunotherapy in Lung Cancer 2020",
            abstract_text="Checkpoint inhibitors showed 40% response rate in NSCLC.",
            score=0.95, year=2020, journal="JCO", mesh_terms=["Immunotherapy", "Lung Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Immunotherapy Combinations 2023",
            abstract_text="Combination checkpoint therapy achieved 60% response rate in advanced NSCLC.",
            score=0.90, year=2023, journal="NEJM", mesh_terms=["Immunotherapy", "Lung Neoplasms"],
        ),
    ]


def test_trend_analysis_returns_agent_result():
    from src.agents.trend_analysis import TrendAnalysisAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Immunotherapy research shows strong upward trend.",
        "findings": [
            {"label": "Immunotherapy growth", "detail": "Increasing focus on combination therapies", "severity": "info"},
        ],
        "confidence": 0.9,
        "trends": [
            {"topic": "Combination immunotherapy", "direction": "increasing", "period": "2020-2023", "evidence_count": 2},
        ],
    })

    agent = TrendAnalysisAgent(llm=mock_llm)
    result = agent.run("lung cancer immunotherapy", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "trend_analysis"
    assert result.score is None
    assert result.confidence == 0.9
    assert len(result.findings) == 1
    assert result.details is not None
    assert len(result.details["trends"]) == 1
    assert result.details["trends"][0]["direction"] == "increasing"
    mock_llm.complete.assert_called_once()


def test_trend_analysis_handles_llm_failure():
    from src.agents.trend_analysis import TrendAnalysisAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = TrendAnalysisAgent(llm=mock_llm)
    result = agent.run("lung cancer", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "trend_analysis"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None


def test_trend_analysis_handles_invalid_json():
    from src.agents.trend_analysis import TrendAnalysisAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Not valid JSON"

    agent = TrendAnalysisAgent(llm=mock_llm)
    result = agent.run("lung cancer", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
