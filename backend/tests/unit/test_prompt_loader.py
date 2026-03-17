"""Tests for the prompt loader."""

import pytest

from src.shared.prompt_loader import _cache, load_prompt


@pytest.fixture(autouse=True)
def _clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def test_load_prompt_returns_dict():
    data = load_prompt("rag/system")
    assert isinstance(data, dict)
    assert "version" in data
    assert "system" in data


def test_load_prompt_has_version():
    data = load_prompt("rag/system")
    assert data["version"] == "1.0"


def test_load_prompt_caches():
    data1 = load_prompt("rag/system")
    data2 = load_prompt("rag/system")
    assert data1 is data2


def test_load_prompt_missing_file():
    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        load_prompt("nonexistent/prompt")


def test_all_prompt_files_loadable():
    names = [
        "rag/system",
        "guardrails/input",
        "guardrails/output",
        "retrieval/query_expander",
        "retrieval/reranker",
        "transcribe/image",
        "transcribe/document",
        "agents/methodology_critic",
        "agents/statistical_reviewer",
        "agents/clinical_applicability",
        "agents/retrieval",
        "agents/summarization",
        "agents/conflicting_findings",
        "agents/trend_analysis",
        "agents/knowledge_graph",
        "agents/review_synthesizer",
    ]
    for name in names:
        data = load_prompt(name)
        assert "version" in data, f"{name} missing version"
        assert "description" in data, f"{name} missing description"
        assert "system" in data, f"{name} missing system prompt"
