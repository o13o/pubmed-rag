"""Tests for chunker (1 abstract = 1 chunk per ADR-0001)."""

from src.shared.models import Article
from src.ingestion.chunker import chunk_article


def _make_article(**overrides) -> Article:
    defaults = {
        "pmid": "123", "title": "Test Title", "abstract": "Test abstract.",
        "authors": ["John Doe"], "year": 2023, "journal": "Test Journal",
        "mesh_terms": ["Neoplasms", "Cardiovascular Diseases"],
        "keywords": ["cancer"], "publication_types": ["Journal Article"],
    }
    defaults.update(overrides)
    return Article(**defaults)


def test_chunk_produces_correct_text_format():
    article = _make_article()
    chunk = chunk_article(article)
    assert chunk.chunk_text == "Title: Test Title\nAbstract: Test abstract.\nMeSH: Neoplasms; Cardiovascular Diseases"


def test_chunk_preserves_pmid():
    article = _make_article(pmid="999")
    chunk = chunk_article(article)
    assert chunk.pmid == "999"


def test_chunk_with_no_mesh_terms():
    article = _make_article(mesh_terms=[])
    chunk = chunk_article(article)
    assert chunk.chunk_text == "Title: Test Title\nAbstract: Test abstract."


def test_chunk_carries_article_reference():
    article = _make_article()
    chunk = chunk_article(article)
    assert chunk.title == article.title
    assert chunk.abstract_text == article.abstract
    assert chunk.year == article.year
    assert chunk.journal == article.journal
