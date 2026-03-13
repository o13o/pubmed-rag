"""Tests for Milvus vector search."""

from unittest.mock import MagicMock, patch

from src.shared.models import SearchFilters, SearchResult
from src.retrieval.search import build_filter_expression, parse_search_results


def test_build_filter_no_filters():
    filters = SearchFilters()
    expr = build_filter_expression(filters)
    assert expr == ""


def test_build_filter_year_range():
    filters = SearchFilters(year_min=2022, year_max=2024)
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert "year <= 2024" in expr


def test_build_filter_year_min_only():
    filters = SearchFilters(year_min=2023)
    expr = build_filter_expression(filters)
    assert "year >= 2023" in expr
    assert "year <=" not in expr


def test_build_filter_journals():
    filters = SearchFilters(journals=["Nature", "Science"])
    expr = build_filter_expression(filters)
    assert 'journal in ["Nature", "Science"]' in expr


def test_build_filter_combined():
    filters = SearchFilters(year_min=2022, journals=["Nature"])
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert "journal" in expr
    assert " and " in expr


def test_parse_search_results():
    """Parse raw Milvus search results into SearchResult models.

    Note: Milvus COSINE metric returns similarity (0-1, higher is better) as distance.
    """
    entity_data = {
        "pmid": "123",
        "title": "Test Title",
        "abstract_text": "Test abstract",
        "year": 2023,
        "journal": "Nature",
        "mesh_terms": '["Neoplasms"]',
    }
    mock_hit = MagicMock()
    mock_hit.entity.get = lambda k: entity_data[k]
    mock_hit.distance = 0.95  # Milvus COSINE: higher = more similar

    results = parse_search_results([mock_hit])
    assert len(results) == 1
    assert results[0].pmid == "123"
    assert results[0].score == 0.95
    assert results[0].mesh_terms == ["Neoplasms"]
