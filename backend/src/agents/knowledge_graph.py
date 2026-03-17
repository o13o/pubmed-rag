"""Knowledge Graph Agent — extracts entity-relationship graphs from research abstracts."""

import logging

from src.agents import parse_llm_json
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult
from src.shared.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("agents/knowledge_graph")


class KnowledgeGraphAgent:
    name = "knowledge_graph"
    description = "Extracts entity-relationship graphs connecting diseases, treatments, and outcomes"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts for knowledge extraction:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=_PROMPT["system"], user_prompt=user_prompt)
            data = parse_llm_json(raw)
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
                details={
                    "nodes": data.get("nodes", []),
                    "edges": data.get("edges", []),
                },
            )
        except Exception as e:
            logger.warning("KnowledgeGraphAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
