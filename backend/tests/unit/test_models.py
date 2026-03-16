"""Tests for shared Pydantic models."""

from src.shared.models import (
    AgentResult, Article, Chunk, Citation, Finding, GuardrailWarning, IngestReport,
    RAGResponse, SearchFilters, SearchResult, ValidatedResponse,
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


def test_guardrail_warning():
    w = GuardrailWarning(
        check="citation_grounding", severity="error",
        message="Claim not supported", span="Drug X cures cancer",
    )
    assert w.check == "citation_grounding"
    assert w.severity == "error"
    assert w.span == "Drug X cures cancer"


def test_guardrail_warning_defaults():
    w = GuardrailWarning(check="hallucination", severity="warning", message="test")
    assert w.span == ""


def test_validated_response():
    vr = ValidatedResponse(
        answer="Test answer",
        citations=[Citation(pmid="123", title="T")],
        query="test query",
        warnings=[GuardrailWarning(check="hallucination", severity="warning", message="Possible hallucination")],
        disclaimer="This is not medical advice.",
        is_grounded=True,
    )
    assert vr.is_grounded is True
    assert len(vr.warnings) == 1
    assert vr.disclaimer == "This is not medical advice."


def test_search_filters_search_mode():
    f = SearchFilters(search_mode="hybrid")
    assert f.search_mode == "hybrid"


def test_search_filters_search_mode_default():
    f = SearchFilters()
    assert f.search_mode is None


def test_ingest_report():
    r = IngestReport(total_articles=100, total_chunks=100, upserted=100, source_path="/tmp/test.jsonl")
    assert r.total_articles == 100


def test_finding_model():
    f = Finding(label="Weak sample", detail="n=12 is underpowered", severity="warning")
    assert f.label == "Weak sample"
    assert f.severity == "warning"


def test_agent_result_model():
    result = AgentResult(
        agent_name="methodology_critic",
        summary="Study design is adequate.",
        findings=[Finding(label="RCT", detail="3/5 are RCTs", severity="info")],
        confidence=0.85,
        score=7,
    )
    assert result.agent_name == "methodology_critic"
    assert result.score == 7
    assert result.details is None
    assert len(result.findings) == 1


def test_agent_result_without_score():
    result = AgentResult(
        agent_name="summarization",
        summary="Overall consensus.",
        findings=[],
        confidence=0.9,
    )
    assert result.score is None
