# Review Pipeline — Plan A: Backend Core

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement LiteratureReview model, ReviewSynthesizer, ReviewPipeline, and prompt YAML.

**Architecture:** 3-stage pipeline — Search → 6 agents parallel → ReviewSynthesizer. Data flows via Pydantic models. This plan covers the core logic; the API route and frontend are separate plans (B and C).

**Tech Stack:** Python, Pydantic, concurrent.futures, LiteLLM, YAML prompts

**Parallel with:** Plan B (API route), Plan C (frontend) — Plan B depends on this plan completing first. Plan C depends on Plan B.

---

### Task 1: Add LiteratureReview model

**Files:**
- Modify: `backend/src/shared/models.py`
- Test: `backend/tests/unit/test_models_review.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for LiteratureReview model."""

from src.shared.models import AgentResult, Citation, LiteratureReview, SearchResult


def test_literature_review_round_trip():
    review = LiteratureReview(
        query="test query",
        overview="Overview text",
        main_findings="Findings text",
        gaps_and_conflicts="Gaps text",
        recommendations="Recs text",
        citations=[Citation(pmid="123", title="Test", journal="J", year=2023, relevance_score=0.9)],
        search_results=[SearchResult(pmid="123", title="Test", abstract_text="Abstract", score=0.9, year=2023, journal="J", mesh_terms=[])],
        agent_results=[AgentResult(agent_name="test", summary="ok", findings=[], confidence=0.8)],
        agents_succeeded=1,
        agents_failed=0,
    )
    data = review.model_dump()
    restored = LiteratureReview(**data)
    assert restored.query == "test query"
    assert restored.agents_succeeded == 1
    assert len(restored.citations) == 1
    assert len(restored.search_results) == 1
    assert len(restored.agent_results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_models_review.py -v`
Expected: FAIL — `LiteratureReview` not defined

- [ ] **Step 3: Add LiteratureReview to models.py**

Append to `backend/src/shared/models.py`:

```python
class LiteratureReview(BaseModel):
    query: str
    overview: str
    main_findings: str
    gaps_and_conflicts: str
    recommendations: str
    citations: list[Citation]
    search_results: list[SearchResult]
    agent_results: list[AgentResult]
    agents_succeeded: int
    agents_failed: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_models_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/shared/models.py backend/tests/unit/test_models_review.py
git commit -m "feat: add LiteratureReview model"
```

---

### Task 2: Create ReviewSynthesizer prompt YAML

**Files:**
- Create: `backend/prompts/agents/review_synthesizer.yaml`
- Test: `backend/tests/unit/test_prompts.py` (existing test covers prompt loading)

- [ ] **Step 1: Create the prompt file**

```yaml
version: "1.0"
description: "Review Synthesizer — generates structured literature review from agent analyses"

system: |
  You are a medical research review synthesizer. You receive:
  1. A research query
  2. Retrieved research abstracts
  3. Analysis results from specialized agents (methodology critic, statistical reviewer, clinical applicability assessor, conflicting findings detector, trend analyzer, knowledge graph mapper)

  Synthesize all inputs into a structured literature review. Return a JSON object with these exact fields:
  {
    "overview": "Brief context and scope of the review (2-3 sentences)",
    "main_findings": "Key results and consensus across studies (1-2 paragraphs)",
    "gaps_and_conflicts": "Contradictions, evidence gaps, and methodological concerns (1-2 paragraphs)",
    "recommendations": "Research directions, clinical implications, and next steps (1-2 paragraphs)"
  }

  Guidelines:
  - Ground all claims in the provided abstracts and agent analyses
  - Reference specific PMIDs when citing findings
  - Highlight where agents agree or disagree
  - Be concise but comprehensive
  - Return ONLY the JSON object, no explanation
```

- [ ] **Step 2: Add to prompt loader test**

In `backend/tests/unit/test_prompt_loader.py`, add `"agents/review_synthesizer"` to the `names` list in `test_all_prompt_files_loadable` (after `"agents/knowledge_graph"`):

```python
        "agents/knowledge_graph",
        "agents/review_synthesizer",
```

- [ ] **Step 3: Run prompt loader test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_prompt_loader.py::test_all_prompt_files_loadable -v`
Expected: FAIL — prompt file not found

- [ ] **Step 4: Create the prompt file**

(Create the YAML file shown in Step 1 above.)

- [ ] **Step 5: Run prompt loader test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_prompt_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/prompts/agents/review_synthesizer.yaml backend/tests/unit/test_prompt_loader.py
git commit -m "feat: add review synthesizer prompt YAML"
```

---

### Task 3: Implement ReviewSynthesizer

**Files:**
- Create: `backend/src/agents/review_synthesizer.py`
- Test: `backend/tests/unit/test_review_synthesizer.py`

- [ ] **Step 1: Write the tests**

```python
"""Tests for ReviewSynthesizer."""

import json
from unittest.mock import MagicMock

from src.shared.models import AgentResult, Citation, Finding, LiteratureReview, SearchResult


def _mock_results():
    return [
        SearchResult(
            pmid="111", title="RCT of Drug X",
            abstract_text="A randomized controlled trial showed Drug X reduced mortality.",
            score=0.95, year=2023, journal="NEJM", mesh_terms=["Neoplasms"],
        ),
    ]


def _mock_agent_results():
    return [
        AgentResult(
            agent_name="methodology_critic", summary="Strong RCT design",
            findings=[Finding(label="RCT", detail="Well-designed", severity="info")],
            confidence=0.9, score=8,
        ),
        AgentResult(
            agent_name="statistical_reviewer", summary="Significant results",
            findings=[], confidence=0.85, score=7,
        ),
    ]


def test_synthesizer_returns_literature_review():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "overview": "This review covers Drug X.",
        "main_findings": "Drug X reduced mortality.",
        "gaps_and_conflicts": "Limited sample diversity.",
        "recommendations": "Larger multi-center trials needed.",
    })

    synth = ReviewSynthesizer(llm=mock_llm)
    result = synth.run("cancer treatment", _mock_results(), _mock_agent_results())

    assert isinstance(result, LiteratureReview)
    assert result.query == "cancer treatment"
    assert result.overview == "This review covers Drug X."
    assert result.main_findings == "Drug X reduced mortality."
    assert result.agents_succeeded == 2
    assert result.agents_failed == 0
    assert len(result.citations) == 1
    assert result.citations[0].pmid == "111"
    mock_llm.complete.assert_called_once()


def test_synthesizer_counts_failed_agents():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.return_value = json.dumps({
        "overview": "Overview", "main_findings": "Findings",
        "gaps_and_conflicts": "Gaps", "recommendations": "Recs",
    })

    agent_results = [
        AgentResult(agent_name="ok_agent", summary="Good", findings=[], confidence=0.9),
        AgentResult(agent_name="bad_agent", summary="Analysis failed: timeout", findings=[], confidence=0.0),
    ]

    synth = ReviewSynthesizer(llm=mock_llm)
    result = synth.run("query", _mock_results(), agent_results)

    assert result.agents_succeeded == 1
    assert result.agents_failed == 1


def test_synthesizer_handles_llm_failure():
    from src.agents.review_synthesizer import ReviewSynthesizer

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("API error")

    synth = ReviewSynthesizer(llm=mock_llm)
    try:
        synth.run("query", _mock_results(), _mock_agent_results())
        assert False, "Should have raised"
    except RuntimeError:
        pass  # Expected — pipeline catches this at Stage 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_review_synthesizer.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement ReviewSynthesizer**

Create `backend/src/agents/review_synthesizer.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_review_synthesizer.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/agents/review_synthesizer.py backend/tests/unit/test_review_synthesizer.py
git commit -m "feat: implement ReviewSynthesizer with tests"
```

---

### Task 4: Implement ReviewPipeline

**Files:**
- Create: `backend/src/agents/pipeline.py`
- Test: `backend/tests/unit/test_review_pipeline.py`

- [ ] **Step 1: Write the tests**

```python
"""Tests for ReviewPipeline — 3-stage A2A handoff."""

import json
from unittest.mock import MagicMock, patch

from src.shared.models import AgentResult, Finding, LiteratureReview, SearchFilters, SearchResult


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Study A", abstract_text="Abstract A.",
            score=0.9, year=2023, journal="NEJM", mesh_terms=[],
        ),
    ]


def _mock_agent_result(name):
    return AgentResult(
        agent_name=name, summary=f"{name} analysis",
        findings=[Finding(label="test", detail="detail", severity="info")],
        confidence=0.8, score=7,
    )


def test_pipeline_runs_three_stages():
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = _mock_search_results()

    mock_llm = MagicMock()
    # Agent LLM calls return valid JSON
    agent_json = json.dumps({
        "summary": "Agent analysis", "findings": [{"label": "ok", "detail": "fine", "severity": "info"}],
        "confidence": 0.8, "score": 7,
    })
    # ReviewSynthesizer LLM call returns review JSON
    review_json = json.dumps({
        "overview": "Overview", "main_findings": "Findings",
        "gaps_and_conflicts": "Gaps", "recommendations": "Recs",
    })
    mock_llm.complete.side_effect = [agent_json] * 6 + [review_json]

    pipeline = ReviewPipeline(search_client=mock_search, llm=mock_llm)
    result = pipeline.run("test query", SearchFilters())

    assert isinstance(result, LiteratureReview)
    assert result.query == "test query"
    assert result.overview == "Overview"
    assert len(result.search_results) == 1
    assert len(result.agent_results) == 6
    # 6 agent calls + 1 synthesizer call
    assert mock_llm.complete.call_count == 7
    mock_search.search.assert_called_once()


def test_pipeline_continues_with_partial_agent_failure():
    """Agents catch their own LLM errors internally and return degraded
    AgentResult with confidence=0.0.  The pipeline (and ReviewSynthesizer)
    counts confidence==0.0 as "failed".  This test verifies end-to-end
    graceful degradation through the agents' internal error handling."""
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = _mock_search_results()

    mock_llm = MagicMock()
    agent_json = json.dumps({
        "summary": "ok", "findings": [], "confidence": 0.8, "score": 7,
    })
    review_json = json.dumps({
        "overview": "O", "main_findings": "F",
        "gaps_and_conflicts": "G", "recommendations": "R",
    })
    # First agent succeeds, rest get LLM errors (agents catch internally,
    # returning degraded AgentResult with confidence=0.0), then synthesizer succeeds
    mock_llm.complete.side_effect = [agent_json] + [RuntimeError("timeout")] * 5 + [review_json]

    pipeline = ReviewPipeline(search_client=mock_search, llm=mock_llm)
    result = pipeline.run("test", SearchFilters())

    assert isinstance(result, LiteratureReview)
    assert result.agents_succeeded >= 1
    assert result.agents_failed >= 1


def test_pipeline_raises_on_empty_search():
    from src.agents.pipeline import ReviewPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = []

    pipeline = ReviewPipeline(search_client=mock_search, llm=MagicMock())

    try:
        pipeline.run("test", SearchFilters())
        assert False, "Should have raised"
    except ValueError as e:
        assert "No results" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_review_pipeline.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement ReviewPipeline**

Create `backend/src/agents/pipeline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_review_pipeline.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run all unit tests to check for regressions**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add backend/src/agents/pipeline.py backend/tests/unit/test_review_pipeline.py
git commit -m "feat: implement ReviewPipeline with parallel agent execution"
```
