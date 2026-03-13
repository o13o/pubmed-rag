"""Tests for shared Pydantic models."""

from src.shared.models import (
    Article, Chunk, Citation, IngestReport, RAGResponse, SearchFilters, SearchResult,
)


def test_article_creation():
    a = Article(
        pmid="12345", title="Test Title", abstract="Test abstract.",
        authors=["John Doe"], year=2023, journal="Test Journal",
        mesh_terms=["Neoplasms"], keywords=["cancer"],
        publication_types=["Journal Article"],
    )
    assert a.pmid == "12345"
    assert a.year == 2023


def test_article_defaults():
    a = Article(pmid="1", title="T", abstract="A", year=2023)
    assert a.authors == []
    assert a.mesh_terms == []
    assert a.keywords == []
    assert a.publication_types == []
    assert a.journal == ""


def test_chunk_creation():
    c = Chunk(
        pmid="1", chunk_text="Title: T\nAbstract: A", title="T",
        abstract_text="A", year=2023, journal="J",
        authors="[]", mesh_terms="[]", keywords="[]", publication_types="[]",
    )
    assert c.pmid == "1"
    assert "Title: T" in c.chunk_text


def test_search_result_with_score():
    sr = SearchResult(
        pmid="1", title="T", abstract_text="A", score=0.95,
        year=2023, journal="J", mesh_terms=["Neoplasms"],
    )
    assert sr.score == 0.95


def test_search_filters_defaults():
    f = SearchFilters()
    assert f.year_min is None
    assert f.year_max is None
    assert f.journals == []
    assert f.top_k == 10


def test_search_filters_custom():
    f = SearchFilters(year_min=2022, year_max=2024, journals=["Nature"], top_k=5)
    assert f.year_min == 2022
    assert f.top_k == 5


def test_rag_response():
    r = RAGResponse(
        answer="Based on the evidence...",
        citations=[Citation(pmid="1", title="T", relevance_score=0.9)],
        query="test query",
    )
    assert len(r.citations) == 1
    assert r.query == "test query"
    assert r.citations[0].pmid == "1"


def test_ingest_report():
    r = IngestReport(total_articles=100, total_chunks=100, upserted=100, source_path="/tmp/test.jsonl")
    assert r.total_articles == 100
