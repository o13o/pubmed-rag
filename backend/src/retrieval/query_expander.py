"""MeSH-based query expansion: LLM keyword extraction + DuckDB MeSH hierarchy lookup.

Flow (per spec Section 4.2):
1. LLM extracts medical keywords from natural language
2. DuckDB MeSH lookup: keywords → descriptors + synonyms + child terms
3. Build expanded query (original + MeSH terms)
"""

import json
import logging

from pydantic import BaseModel, Field

from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase

logger = logging.getLogger(__name__)

KEYWORD_EXTRACTION_PROMPT = """Extract medical/biomedical keywords from the user's query.
Return ONLY a JSON array of keyword strings, no explanation.
Focus on diseases, conditions, treatments, drugs, anatomical terms.

Example:
Query: "What are the latest treatments for knee osteoarthritis in elderly patients?"
Output: ["knee osteoarthritis", "treatment", "elderly"]

Query: "{query}"
Output:"""


class ExpandedQuery(BaseModel):
    """Result of query expansion."""

    original_query: str
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    child_terms: list[str] = Field(default_factory=list)
    expanded_query: str = ""


class QueryExpander:
    """Expand queries using LLM keyword extraction + MeSH hierarchy lookup."""

    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase):
        self.llm = llm
        self.mesh_db = mesh_db

    def _extract_keywords(self, query: str) -> list[str]:
        """Use LLM to extract medical keywords from the query."""
        prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
        response = self.llm.complete(
            system_prompt="You extract medical keywords from queries. Return only a JSON array.",
            user_prompt=prompt,
        )
        try:
            keywords = json.loads(response.strip())
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords if k]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM keyword response: %s", response)
        return []

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query with MeSH terms.

        1. Extract keywords via LLM
        2. Look up each keyword in MeSH (name or synonym)
        3. Get child terms for matched descriptors
        4. Build expanded query string
        """
        keywords = self._extract_keywords(query)
        logger.debug("Extracted keywords: %s", keywords)

        mesh_terms = []
        child_terms = []
        seen = set()

        for keyword in keywords:
            result = self.mesh_db.lookup(keyword)
            if result is None:
                continue

            name = result["name"]
            if name not in seen:
                mesh_terms.append(name)
                seen.add(name)

            # Get child terms via tree number prefix match
            for tree_num in result.get("tree_numbers", []):
                children = self.mesh_db.get_children(tree_num)
                for child in children:
                    if child["name"] not in seen:
                        child_terms.append(child["name"])
                        seen.add(child["name"])

        # Build expanded query
        all_terms = mesh_terms + child_terms
        if all_terms:
            expanded = f"{query} ({'; '.join(all_terms)})"
        else:
            expanded = query

        result = ExpandedQuery(
            original_query=query,
            keywords=keywords,
            mesh_terms=mesh_terms,
            child_terms=child_terms,
            expanded_query=expanded,
        )
        logger.info("Query expanded: %d MeSH terms, %d children", len(mesh_terms), len(child_terms))
        return result
