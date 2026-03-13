"""Tests for MeSH-based query expansion."""

from unittest.mock import MagicMock, patch

import duckdb
import pytest

from src.shared.mesh_db import MeSHDatabase
from src.retrieval.query_expander import QueryExpander


@pytest.fixture
def mesh_db():
    db = MeSHDatabase(":memory:")
    db._init_schema()
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D001943', 'Breast Neoplasms', ['C04.588.180']),
        ('D020370', 'Osteoarthritis, Knee', ['C05.550.114.606'])
    """)
    db.conn.execute("""
        INSERT INTO mesh_synonyms VALUES
        ('Cancer', 'D009369'),
        ('Breast Cancer', 'D001943'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Knee Pain', 'D020370')
    """)
    return db


def test_expand_with_mesh_terms(mesh_db):
    """When LLM extracts keywords that match MeSH, expand with descriptors and children."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["cancer", "treatment"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("What are the latest cancer treatments?")

    assert "cancer" in result.original_query.lower()
    assert "Neoplasms" in result.mesh_terms
    # Breast Neoplasms is a child of Neoplasms (C04 → C04.588.180)
    assert "Breast Neoplasms" in result.child_terms
    assert result.expanded_query != result.original_query


def test_expand_no_mesh_match(mesh_db):
    """When keywords don't match MeSH, return original query unchanged."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["completely_unknown_term"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("completely unknown medical query")

    assert result.mesh_terms == []
    assert result.expanded_query == result.original_query


def test_expand_query_format(mesh_db):
    """Expanded query should include original + MeSH terms."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["knee pain"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("knee pain treatment options")

    assert "knee pain treatment options" in result.expanded_query
    assert "Osteoarthritis, Knee" in result.expanded_query
