"""ReviewPipeline — 3-stage A2A agent pipeline for literature review generation.

Stage 1: Search via SearchClient
Stage 2: 6 analysis agents in parallel (ThreadPoolExecutor)
Stage 3: ReviewSynthesizer merges all outputs into LiteratureReview
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.clinical_applicability import ClinicalApplicabilityAgent
from src.agents.conflicting_findings import ConflictingFindingsAgent
from src.agents.knowledge_graph import KnowledgeGraphAgent
from src.agents.methodology_critic import MethodologyCriticAgent
from src.agents.review_synthesizer import ReviewSynthesizer
from src.agents.statistical_reviewer import StatisticalReviewerAgent
from src.agents.trend_analysis import TrendAnalysisAgent
from src.retrieval.client import SearchClient
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, LiteratureReview, SearchFilters, SearchResult

logger = logging.getLogger(__name__)

PIPELINE_AGENTS = [
    MethodologyCriticAgent,
    StatisticalReviewerAgent,
    ClinicalApplicabilityAgent,
    ConflictingFindingsAgent,
    TrendAnalysisAgent,
    KnowledgeGraphAgent,
]


class ReviewPipeline:
    def __init__(self, search_client: SearchClient, llm: LLMClient):
        self.search_client = search_client
        self.llm = llm

    def run(self, query: str, filters: SearchFilters) -> LiteratureReview:
        # Stage 1: Search
        results = self.search_client.search(query, filters)
        if not results:
            raise ValueError(f"No results found for query: {query}")
        logger.info("Stage 1 complete: %d results", len(results))

        # Stage 2: Parallel agent analysis
        agent_results = self._run_agents(query, results)
        logger.info("Stage 2 complete: %d agent results", len(agent_results))

        # Stage 3: Synthesize review
        review = ReviewSynthesizer(self.llm).run(query, results, agent_results)
        logger.info("Stage 3 complete: literature review generated")
        return review

    def _run_agents(self, query: str, results: list[SearchResult]) -> list[AgentResult]:
        """Run all pipeline agents in parallel.

        Note: existing agents catch their own LLM exceptions internally and
        return degraded AgentResult(confidence=0.0).  The except block below
        is a safety net for unexpected errors (e.g., import failures, thread
        issues) that bypass the agent's internal handling.
        """
        agents = [cls(llm=self.llm) for cls in PIPELINE_AGENTS]
        agent_results: list[AgentResult] = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(agent.run, query, results): agent
                for agent in agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    result = future.result()
                    agent_results.append(result)
                except Exception as e:
                    logger.warning("Agent %s failed: %s", agent.name, e)
                    agent_results.append(AgentResult(
                        agent_name=agent.name,
                        summary=f"Analysis failed: {e}",
                        findings=[],
                        confidence=0.0,
                    ))

        return agent_results
