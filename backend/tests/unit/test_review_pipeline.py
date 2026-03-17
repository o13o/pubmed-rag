"""Tests for ReviewPipeline — 3-stage A2A handoff."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, Finding, LiteratureReview, SearchFilters, SearchResult


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Study A", abstract_text="Abstract A.",
            score=0.9, year=2023, journal="NEJM", mesh_terms=[],
        ),
    ]


def _mock_agent_result(name):
    return AgentResult(
        agent_name=name, summary=f"{name} analysis",
        findings=[Finding(label="test", detail="detail", severity="info")],
        confidence=0.8, score=7,
    )


def test_pipeline_runs_three_stages():
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = _mock_search_results()

    mock_llm = MagicMock()
    # Agent LLM calls return valid JSON
    agent_json = json.dumps({
        "summary": "Agent analysis", "findings": [{"label": "ok", "detail": "fine", "severity": "info"}],
        "confidence": 0.8, "score": 7,
    })
    # ReviewSynthesizer LLM call returns review JSON
    review_json = json.dumps({
        "overview": "Overview", "main_findings": "Findings",
        "gaps_and_conflicts": "Gaps", "recommendations": "Recs",
    })
    mock_llm.complete.side_effect = [agent_json] * 6 + [review_json]

    pipeline = ReviewPipeline(search_client=mock_search, llm=mock_llm)
    result = pipeline.run("test query", SearchFilters())

    assert isinstance(result, LiteratureReview)
    assert result.query == "test query"
    assert result.overview == "Overview"
    assert len(result.search_results) == 1
    assert len(result.agent_results) == 6
    # 6 agent calls + 1 synthesizer call
    assert mock_llm.complete.call_count == 7
    mock_search.search.assert_called_once()


def test_pipeline_continues_with_partial_agent_failure():
    """Agents catch their own LLM errors internally and return degraded
    AgentResult with confidence=0.0.  The pipeline (and ReviewSynthesizer)
    counts confidence==0.0 as "failed".  This test verifies end-to-end
    graceful degradation through the agents' internal error handling."""
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = _mock_search_results()

    mock_llm = MagicMock()
    agent_json = json.dumps({
        "summary": "ok", "findings": [], "confidence": 0.8, "score": 7,
    })
    review_json = json.dumps({
        "overview": "O", "main_findings": "F",
        "gaps_and_conflicts": "G", "recommendations": "R",
    })
    # First agent succeeds, rest get LLM errors (agents catch internally,
    # returning degraded AgentResult with confidence=0.0), then synthesizer succeeds
    mock_llm.complete.side_effect = [agent_json] + [RuntimeError("timeout")] * 5 + [review_json]

    pipeline = ReviewPipeline(search_client=mock_search, llm=mock_llm)
    result = pipeline.run("test", SearchFilters())

    assert isinstance(result, LiteratureReview)
    assert result.agents_succeeded >= 1
    assert result.agents_failed >= 1


def test_pipeline_raises_on_empty_search():
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = []

    pipeline = ReviewPipeline(search_client=mock_search, llm=MagicMock())

    try:
        pipeline.run("test", SearchFilters())
        assert False, "Should have raised"
    except ValueError as e:
        assert "No results" in str(e)
