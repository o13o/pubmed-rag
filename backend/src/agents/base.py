"""BaseAgent protocol — interface for all analysis agents.

Current implementations use LLM + specialized prompt.
Future implementations may incorporate tools (PubMed API, MeSH lookup)
without changing this interface.
"""

from typing import Protocol

from src.shared.models import AgentResult, SearchResult


class BaseAgent(Protocol):
    name: str
    description: str

    def run(self, query: str, results: list[SearchResult]) -> AgentResult: ...
