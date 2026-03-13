"""Pydantic models - stub for A-2 parallel development. Will be replaced by A-3."""

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


class IngestReport(BaseModel):
    total_articles: int
    total_chunks: int
    upserted: int
    source_path: str
