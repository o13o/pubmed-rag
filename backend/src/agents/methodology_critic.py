"""Methodology Critic Agent — evaluates study design and methodological rigor."""

import logging

from src.agents import parse_llm_json
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research methodology expert. Analyze the provided research abstracts and evaluate their study design and methodological rigor.

For each abstract, assess:
- Study design type (RCT, cohort, case-control, case report, meta-analysis, etc.)
- Sample size adequacy
- Bias risk (selection, information, confounding)
- Control group presence and appropriateness
- Blinding and randomization quality

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = poor methodology, 4-6 = moderate, 7-9 = strong, 10 = exceptional.
Return ONLY the JSON object, no explanation."""


class MethodologyCriticAgent:
    name = "methodology_critic"
    description = "Evaluates study design, bias risk, and methodological rigor"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to evaluate:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = parse_llm_json(raw)
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=data.get("score"),
            )
        except Exception as e:
            logger.warning("MethodologyCriticAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
