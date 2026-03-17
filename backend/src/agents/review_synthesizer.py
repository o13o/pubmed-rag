"""ReviewSynthesizer — generates structured literature review from agent analyses.

Stage 3 of the review pipeline. Does NOT implement BaseAgent protocol
(different signature — takes agent_results as additional input).
"""

import logging

from src.agents import parse_llm_json
from src.shared.llm import LLMClient
from src.shared.models import (
    AgentResult, Citation, LiteratureReview, SearchResult,
)
from src.shared.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("agents/review_synthesizer")


class ReviewSynthesizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        query: str,
        results: list[SearchResult],
        agent_results: list[AgentResult],
    ) -> LiteratureReview:
        user_prompt = self._build_user_prompt(query, results, agent_results)
        raw = self.llm.complete(system_prompt=_PROMPT["system"], user_prompt=user_prompt)
        data = parse_llm_json(raw)

        citations = [
            Citation(
                pmid=r.pmid, title=r.title, journal=r.journal,
                year=r.year, relevance_score=r.score,
            )
            for r in results
        ]

        failed = sum(1 for a in agent_results if a.confidence == 0.0)

        return LiteratureReview(
            query=query,
            overview=data.get("overview", ""),
            main_findings=data.get("main_findings", ""),
            gaps_and_conflicts=data.get("gaps_and_conflicts", ""),
            recommendations=data.get("recommendations", ""),
            citations=citations,
            search_results=results,
            agent_results=agent_results,
            agents_succeeded=len(agent_results) - failed,
            agents_failed=failed,
        )

    def _build_user_prompt(
        self,
        query: str,
        results: list[SearchResult],
        agent_results: list[AgentResult],
    ) -> str:
        abstracts = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        analyses = "\n\n".join(
            f"Agent: {a.agent_name}\nSummary: {a.summary}\n"
            f"Findings: {'; '.join(f.label + ': ' + f.detail for f in a.findings)}\n"
            f"Confidence: {a.confidence}"
            for a in agent_results
        )
        return (
            f"Query: {query}\n\n"
            f"=== Retrieved Abstracts ===\n{abstracts}\n\n"
            f"=== Agent Analyses ===\n{analyses}"
        )
