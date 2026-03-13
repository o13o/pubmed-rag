"""Pydantic models shared across all modules.

These models serve as the inter-module communication contracts.
When splitting into microservices, these become API schemas.
"""

from pydantic import BaseModel, Field


class Article(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: int
    journal: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    pmid: str
    chunk_text: str
    title: str
    abstract_text: str
    year: int
    journal: str
    authors: str
    mesh_terms: str
    keywords: str
    publication_types: str


class SearchFilters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    top_k: int = 10


class SearchResult(BaseModel):
    pmid: str
    title: str
    abstract_text: str
    score: float
    year: int
    journal: str
    mesh_terms: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    pmid: str
    title: str
    journal: str = ""
    year: int = 0
    relevance_score: float = 0.0


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    query: str


class IngestReport(BaseModel):
    total_articles: int
    total_chunks: int
    upserted: int
    source_path: str
