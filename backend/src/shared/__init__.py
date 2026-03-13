"""Shared module - public interface."""

from src.shared.config import Settings, get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    Article, Chunk, Citation, IngestReport, RAGResponse, SearchFilters, SearchResult,
)

__all__ = [
    "Article", "Chunk", "Citation", "IngestReport", "LLMClient", "MeSHDatabase",
    "RAGResponse", "SearchFilters", "SearchResult", "Settings", "get_settings",
]
