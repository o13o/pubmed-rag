"""Tests for LiteratureReview model."""

from src.shared.models import AgentResult, Citation, LiteratureReview, SearchResult


def test_literature_review_round_trip():
    review = LiteratureReview(
        query="test query",
        overview="Overview text",
        main_findings="Findings text",
        gaps_and_conflicts="Gaps text",
        recommendations="Recs text",
        citations=[Citation(pmid="123", title="Test", journal="J", year=2023, relevance_score=0.9)],
        search_results=[SearchResult(pmid="123", title="Test", abstract_text="Abstract", score=0.9, year=2023, journal="J", mesh_terms=[])],
        agent_results=[AgentResult(agent_name="test", summary="ok", findings=[], confidence=0.8)],
        agents_succeeded=1,
        agents_failed=0,
    )
    data = review.model_dump()
    restored = LiteratureReview(**data)
    assert restored.query == "test query"
    assert restored.agents_succeeded == 1
    assert len(restored.citations) == 1
    assert len(restored.search_results) == 1
    assert len(restored.agent_results) == 1
