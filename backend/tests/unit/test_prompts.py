"""Tests for prompt templates."""

from src.rag.prompts import build_system_prompt, build_user_prompt
from src.shared.models import SearchResult


def test_system_prompt_contains_instructions():
    prompt = build_system_prompt()
    assert "cite" in prompt.lower() or "citation" in prompt.lower()
    assert "PMID" in prompt


def test_user_prompt_includes_query_and_abstracts():
    results = [
        SearchResult(
            pmid="111", title="Title 1", abstract_text="Abstract 1",
            score=0.95, year=2023, journal="J1", mesh_terms=["Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Title 2", abstract_text="Abstract 2",
            score=0.90, year=2024, journal="J2", mesh_terms=[],
        ),
    ]
    prompt = build_user_prompt("test query", results)
    assert "test query" in prompt
    assert "PMID: 111" in prompt
    assert "Title 1" in prompt
    assert "Abstract 1" in prompt
    assert "PMID: 222" in prompt


def test_user_prompt_empty_results():
    prompt = build_user_prompt("test query", [])
    assert "test query" in prompt
    assert "no relevant" in prompt.lower() or "no abstracts" in prompt.lower()
