"""Tests for ClinicalApplicabilityAgent."""

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


def test_clinical_applicability_returns_agent_result():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Findings applicable to adult oncology patients.",
        "findings": [
            {"label": "Population match", "detail": "Studies cover adult patients", "severity": "info"},
            {"label": "Dosage unspecified", "detail": "No dosage guidance provided", "severity": "warning"},
        ],
        "confidence": 0.75,
        "score": 7,
    })

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "clinical_applicability"
    assert result.score == 7
    assert result.confidence == 0.75
    assert len(result.findings) == 2
    mock_llm.complete.assert_called_once()


def test_clinical_applicability_handles_invalid_json():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "This is not JSON at all"

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None


def test_clinical_applicability_handles_failure():
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("timeout")

    agent = ClinicalApplicabilityAgent(llm=mock_llm)
    result = agent.run("cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
    assert result.score is None
