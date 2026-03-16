"""Tests for KnowledgeGraphAgent."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="Trastuzumab in HER2+ Breast Cancer",
            abstract_text="Trastuzumab targets HER2 receptor and improves progression-free survival in HER2-positive breast cancer.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Breast Neoplasms", "Trastuzumab"],
        ),
        SearchResult(
            pmid="222", title="HER2 Biomarker in Prognosis",
            abstract_text="HER2 overexpression is associated with aggressive tumor behavior and poor prognosis without targeted therapy.",
            score=0.88, year=2022, journal="JCO", mesh_terms=["Breast Neoplasms", "Biomarkers"],
        ),
    ]


def test_knowledge_graph_returns_agent_result():
    from src.agents.knowledge_graph import KnowledgeGraphAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "summary": "Extracted 4 entities and 3 relationships around HER2+ breast cancer.",
        "findings": [
            {"label": "Key pathway", "detail": "Trastuzumab targets HER2 in breast cancer", "severity": "info"},
        ],
        "confidence": 0.85,
        "nodes": [
            {"id": "breast_cancer", "label": "Breast Cancer", "type": "disease"},
            {"id": "trastuzumab", "label": "Trastuzumab", "type": "treatment"},
            {"id": "her2", "label": "HER2", "type": "biomarker"},
            {"id": "pfs", "label": "Progression-Free Survival", "type": "outcome"},
        ],
        "edges": [
            {"source": "trastuzumab", "target": "breast_cancer", "relation": "treats"},
            {"source": "trastuzumab", "target": "her2", "relation": "inhibits"},
            {"source": "her2", "target": "breast_cancer", "relation": "associated_with"},
        ],
    })

    agent = KnowledgeGraphAgent(llm=mock_llm)
    result = agent.run("HER2 breast cancer treatment", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "knowledge_graph"
    assert result.score is None
    assert result.confidence == 0.85
    assert len(result.findings) == 1
    assert result.details is not None
    assert len(result.details["nodes"]) == 4
    assert len(result.details["edges"]) == 3
    mock_llm.complete.assert_called_once()


def test_knowledge_graph_handles_llm_failure():
    from src.agents.knowledge_graph import KnowledgeGraphAgent

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM timeout")

    agent = KnowledgeGraphAgent(llm=mock_llm)
    result = agent.run("breast cancer", _mock_results())

    assert isinstance(result, AgentResult)
    assert result.agent_name == "knowledge_graph"
    assert "failed" in result.summary.lower()
    assert result.confidence == 0.0
    assert result.score is None


def test_knowledge_graph_handles_invalid_json():
    from src.agents.knowledge_graph import KnowledgeGraphAgent

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Not valid JSON"

    agent = KnowledgeGraphAgent(llm=mock_llm)
    result = agent.run("breast cancer", _mock_results())

    assert isinstance(result, AgentResult)
    assert "failed" in result.summary.lower()
