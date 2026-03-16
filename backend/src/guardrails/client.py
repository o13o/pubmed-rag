"""Guardrail client abstraction for monolith/microservice dual deployment.

LocalGuardrailClient: runs validation in-process (monolith mode).
A RemoteGuardrailClient can be added when guardrails are extracted
into a separate service.
"""

from __future__ import annotations

from typing import Protocol

from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import RAGResponse, SearchResult, ValidatedResponse


class GuardrailClient(Protocol):
    def validate(
        self, response: RAGResponse, results: list[SearchResult]
    ) -> ValidatedResponse: ...


class LocalGuardrailClient:
    """Monolith mode — runs guardrails in-process."""

    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase) -> None:
        from src.guardrails.output import GuardrailValidator

        self._validator = GuardrailValidator(llm=llm, mesh_db=mesh_db)

    def validate(
        self, response: RAGResponse, results: list[SearchResult]
    ) -> ValidatedResponse:
        return self._validator.validate(response, results)
