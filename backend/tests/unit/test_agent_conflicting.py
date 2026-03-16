"""Tests for ConflictingFindingsAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="Drug X Improves Survival",
            abstract_text="A randomized trial showed Drug X improved overall survival by 25% (p<0.01).",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Drug X Shows No Benefit",
            abstract_text="A multicenter study found no significant survival benefit from Drug X (p=0.45).",
            score=0.88, year=2023, journal="Lancet", mesh_terms=["Neoplasms"],
        ),
    ]


def test_conflicting_findings_returns_agent_result():
    from src.agents.conflicting_findings import ConflictingFindingsAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "1 conflicting pair found regarding Drug X efficacy.",
        "findings": [
            {"label": "Drug X efficacy", "detail": "PMID 111 shows benefit, PMID 222 shows none", "severity": "critical"},
        ],
        "confidence": 0.85,
        "conflicts": [
            {"pmid_a": "111", "pmid_b": "222", "topic": "Drug X survival benefit", "description": "Contradictory survival outcomes"},
        ],
    })

    agent = ConflictingFindingsAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "conflicting_findings"
    assert result.score is None
    assert result.confidence == 0.85
    assert len(result.findings) == 1
    assert result.details is not None
    assert len(result.details["conflicts"]) == 1
    mock_llm.complete.assert_called_once()


def test_conflicting_findings_handles_llm_failure():
    from src.agents.conflicting_findings import ConflictingFindingsAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = ConflictingFindingsAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "conflicting_findings"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None


def test_conflicting_findings_handles_invalid_json():
    from src.agents.conflicting_findings import ConflictingFindingsAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Not valid JSON"

    agent = ConflictingFindingsAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
