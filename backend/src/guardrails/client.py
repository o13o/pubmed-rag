"""Guardrail client abstraction for monolith/microservice dual deployment.

LocalGuardrailClient: runs validation in-process (monolith mode).
RemoteGuardrailClient: placeholder for HTTP call (microservice mode).
"""

from __future__ import annotations

from typing import Protocol

from src.shared.models import RAGResponse, SearchResult, ValidatedResponse


class GuardrailClient(Protocol):
    def validate(
        self, response: RAGResponse, results: list[SearchResult]
    ) -> ValidatedResponse: ...


class LocalGuardrailClient:
    """Monolith mode — runs guardrails in-process."""

    def __init__(self, llm, mesh_db) -> None:
        self._llm = llm
        self._mesh_db = mesh_db

    def validate(
        self, response: RAGResponse, results: list[SearchResult]
    ) -> ValidatedResponse:
        from src.guardrails.output import GuardrailValidator

        validator = GuardrailValidator(llm=self._llm, mesh_db=self._mesh_db)
        return validator.validate(response, results)
