"""Knowledge Graph Agent — extracts entity-relationship graphs from research abstracts."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a biomedical knowledge extraction expert. Analyze the provided research abstracts and extract a knowledge graph of entities and their relationships.

Entity types to extract:
- disease: diseases, conditions, syndromes
- treatment: drugs, therapies, interventions, procedures
- outcome: clinical outcomes, endpoints, side effects
- gene: genes, proteins, biomarkers
- biomarker: diagnostic or prognostic markers

Relationship types:
- treats: treatment -> disease
- causes: entity -> disease/outcome
- associated_with: any entity -> any entity
- inhibits: treatment -> gene/biomarker
- indicates: biomarker -> disease

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overview of the extracted graph",
  "findings": [
    {"label": "short label", "detail": "key relationship description", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "nodes": [
    {"id": "unique_id", "label": "human readable name", "type": "disease|treatment|outcome|gene|biomarker"}
  ],
  "edges": [
    {"source": "node_id", "target": "node_id", "relation": "treats|causes|associated_with|inhibits|indicates"}
  ]
}

Return ONLY the JSON object, no explanation."""


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
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
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
