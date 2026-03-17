"""Tests for ReviewSynthesizer."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, Citation, Finding, LiteratureReview, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial showed Drug X reduced mortality.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def _mock_agent_results():
    return [
        AgentResult(
            agent_name="methodology_critic", summary="Strong RCT design",
            findings=[Finding(label="RCT", detail="Well-designed", severity="info")],
            confidence=0.9, score=8,
        ),
        AgentResult(
            agent_name="statistical_reviewer", summary="Significant results",
            findings=[], confidence=0.85, score=7,
        ),
    ]


def test_synthesizer_returns_literature_review():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "overview": "This review covers Drug X.",
        "main_findings": "Drug X reduced mortality.",
        "gaps_and_conflicts": "Limited sample diversity.",
        "recommendations": "Larger multi-center trials needed.",
    })

    synth = ReviewSynthesizer(llm=mock_llm)
    result = synth.run("cancer treatment", _mock_results(), _mock_agent_results())

    assert isinstance(result, LiteratureReview)
    assert result.query == "cancer treatment"
    assert result.overview == "This review covers Drug X."
    assert result.main_findings == "Drug X reduced mortality."
    assert result.agents_succeeded == 2
    assert result.agents_failed == 0
    assert len(result.citations) == 1
    assert result.citations[0].pmid == "111"
    mock_llm.complete.assert_called_once()


def test_synthesizer_counts_failed_agents():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "overview": "Overview", "main_findings": "Findings",
        "gaps_and_conflicts": "Gaps", "recommendations": "Recs",
    })

    agent_results = [
        AgentResult(agent_name="ok_agent", summary="Good", findings=[], confidence=0.9),
        AgentResult(agent_name="bad_agent", summary="Analysis failed: timeout", findings=[], confidence=0.0),
    ]

    synth = ReviewSynthesizer(llm=mock_llm)
    result = synth.run("query", _mock_results(), agent_results)

    assert result.agents_succeeded == 1
    assert result.agents_failed == 1


def test_synthesizer_handles_llm_failure():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("API error")

    synth = ReviewSynthesizer(llm=mock_llm)
    try:
        synth.run("query", _mock_results(), _mock_agent_results())
        assert False, "Should have raised"
    except RuntimeError:
        pass  # Expected — pipeline catches this at Stage 3
