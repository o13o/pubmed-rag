"""Retrieval Agent — evaluates relevance, coverage, and gaps in search results."""

import logging

from src.agents import parse_llm_json
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult
from src.shared.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("agents/retrieval")


class RetrievalAgent:
    name = "retrieval"
    description = "Evaluates relevance, coverage, and gaps in search results"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        results_text = "\n\n".join(
            f"PMID: {r.pmid} | Score: {r.score:.3f} | Year: {r.year}\n"
            f"Title: {r.title}\nJournal: {r.journal}\nMeSH: {', '.join(r.mesh_terms)}\n"
            f"Abstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nSearch results to evaluate ({len(results)} results):\n{results_text}"

        try:
            raw = self.llm.complete(system_prompt=_PROMPT["system"], user_prompt=user_prompt)
            data = parse_llm_json(raw)
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
            )
        except Exception as e:
            logger.warning("RetrievalAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
