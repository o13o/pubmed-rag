"""Tests for SummarizationAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial of 500 patients showed Drug X reduced mortality by 30%.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Observational Study of Drug X",
            abstract_text="An observational cohort study suggested Drug X may improve outcomes.",
            score=0.88, year=2022, journal="BMJ", mesh_terms=["Neoplasms"],
        ),
    ]


def test_summarization_returns_agent_result():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Drug X shows promise in cancer treatment with strong RCT evidence.",
        "findings": [
            {"label": "Consistent efficacy", "detail": "Both studies report positive outcomes", "severity": "info"},
            {"label": "Conflicting methods", "detail": "RCT vs observational yields different confidence", "severity": "warning"},
        ],
        "confidence": 0.85,
    })

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "summarization"
    assert result.score is None
    assert result.confidence == 0.85
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_summarization_handles_invalid_json():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None


def test_summarization_handles_failure():
    from src.agents.summarization import SummarizationAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("timeout")

    agent = SummarizationAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None
