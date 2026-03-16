"""Tests for MethodologyCriticAgent."""

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
        SearchResult(
            pmid="222", title="Observational Study of Drug X",
            abstract_text="An observational cohort study of 50 patients suggested Drug X may improve outcomes, though selection bias is a limitation.",
            score=0.88, year=2022, journal="BMJ", mesh_terms=["Neoplasms"],
        ),
    ]


def test_methodology_critic_returns_agent_result():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Mixed study designs with moderate rigor.",
        "findings": [
            {"label": "RCT present", "detail": "1/2 studies is an RCT", "severity": "info"},
            {"label": "Selection bias", "detail": "Observational study lacks matching", "severity": "warning"},
        ],
        "confidence": 0.8,
        "score": 6,
    })

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "methodology_critic"
    assert result.score == 6
    assert result.confidence == 0.8
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_methodology_critic_handles_llm_failure():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "methodology_critic"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None


def test_methodology_critic_handles_invalid_json():
    from src.agents.methodology_critic import MethodologyCriticAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = MethodologyCriticAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
