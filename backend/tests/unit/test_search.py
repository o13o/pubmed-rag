"""Tests for Milvus vector search."""

from unittest.mock import MagicMock, patch

from src.shared.models import SearchFilters, SearchResult
from src.retrieval.search import build_filter_expression, parse_search_results, _resolve_search_mode


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
        "publication_types": '["Journal Article"]',
    }
    mock_hit = MagicMock()
    mock_hit.entity.get = lambda k: entity_data.get(k)
    mock_hit.distance = 0.95  # Milvus COSINE: higher = more similar

    results = parse_search_results([mock_hit])
    assert len(results) == 1
    assert results[0].pmid == "123"
    assert results[0].score == 0.95
    assert results[0].mesh_terms == ["Neoplasms"]
    assert results[0].publication_types == ["Journal Article"]


def test_resolve_search_mode_from_filters():
    filters = SearchFilters(search_mode="hybrid")
    assert _resolve_search_mode(filters) == "hybrid"


def test_resolve_search_mode_default_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    filters = SearchFilters()  # search_mode is None
    # Should fall back to config default ("dense")
    mode = _resolve_search_mode(filters)
    assert mode == "dense"


def test_build_filter_publication_types_single():
    filters = SearchFilters(publication_types=["Review"])
    expr = build_filter_expression(filters)
    assert 'publication_types like "%Review%"' in expr


def test_build_filter_publication_types_multiple_or():
    filters = SearchFilters(publication_types=["Review", "Meta-Analysis"])
    expr = build_filter_expression(filters)
    assert 'publication_types like "%Review%"' in expr
    assert 'publication_types like "%Meta-Analysis%"' in expr
    assert " or " in expr


def test_build_filter_mesh_categories_single():
    filters = SearchFilters(mesh_categories=["Neoplasms"])
    expr = build_filter_expression(filters)
    assert 'mesh_terms like "%Neoplasms%"' in expr


def test_build_filter_mesh_categories_multiple_or():
    filters = SearchFilters(mesh_categories=["Neoplasms", "Cardiovascular Diseases"])
    expr = build_filter_expression(filters)
    assert 'mesh_terms like "%Neoplasms%"' in expr
    assert 'mesh_terms like "%Cardiovascular Diseases%"' in expr
    assert " or " in expr


def test_build_filter_combined_year_and_publication_types():
    filters = SearchFilters(year_min=2023, publication_types=["Randomized Controlled Trial"])
    expr = build_filter_expression(filters)
    assert "year >= 2023" in expr
    assert 'publication_types like "%Randomized Controlled Trial%"' in expr
    assert " and " in expr


def test_build_filter_combined_all():
    filters = SearchFilters(
        year_min=2022,
        publication_types=["Review"],
        mesh_categories=["Neoplasms"],
    )
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert 'publication_types like "%Review%"' in expr
    assert 'mesh_terms like "%Neoplasms%"' in expr
    # Should have AND between groups
    assert expr.count(" and ") >= 2


def test_build_filter_sanitize_percent():
    filters = SearchFilters(publication_types=["Review%injection"])
    expr = build_filter_expression(filters)
    assert "%" not in expr.replace('like "%', "").replace('%"', "")


def test_build_filter_sanitize_quote():
    filters = SearchFilters(publication_types=['Review"injection'])
    expr = build_filter_expression(filters)
    # The sanitized value should not contain double quotes
    assert 'Reviewinjection' in expr


def test_build_filter_sanitize_backslash():
    filters = SearchFilters(mesh_categories=["Neoplasms\\test"])
    expr = build_filter_expression(filters)
    assert "\\" not in expr.replace('like "%', "").replace('%"', "")


def test_parse_search_results_includes_publication_types():
    entity_data = {
        "pmid": "456",
        "title": "Test Title",
        "abstract_text": "Test abstract",
        "year": 2023,
        "journal": "Nature",
        "mesh_terms": '["Neoplasms"]',
        "publication_types": '["Journal Article", "Review"]',
    }
    mock_hit = MagicMock()
    mock_hit.entity.get = lambda k: entity_data.get(k)
    mock_hit.distance = 0.90

    results = parse_search_results([mock_hit])
    assert len(results) == 1
    assert results[0].publication_types == ["Journal Article", "Review"]
