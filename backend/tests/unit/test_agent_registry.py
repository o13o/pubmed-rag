"""Tests for agent registry."""

from unittest.mock import MagicMock

from src.agents.registry import get_agents


def test_get_agents_returns_all():
    llm = MagicMock()
    agents = get_agents(llm=llm)
    assert len(agents) == 8
    names = {a.name for a in agents}
    assert names == {
        "retrieval", "methodology_critic", "statistical_reviewer",
        "clinical_applicability", "summarization",
        "conflicting_findings", "trend_analysis", "knowledge_graph",
    }


def test_get_agents_filters_by_name():
    llm = MagicMock()
    agents = get_agents(llm=llm, names=["methodology_critic", "summarization"])
    assert len(agents) == 2
    names = {a.name for a in agents}
    assert names == {"methodology_critic", "summarization"}


def test_get_agents_empty_list():
    """names=[] (explicit empty) returns no agents, unlike names=None (return all)."""
    llm = MagicMock()
    agents = get_agents(llm=llm, names=[])
    assert len(agents) == 0
