ADR: Evaluation Strategy — DeepEval with Agent-Based Custom Metrics

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The requirements specify:

- "DeepEval with domain-specific medical metrics" (Requirement 2)
- "Custom evaluation: clinical relevance, evidence quality, study design assessment" (Requirement 2)
- "LLM-as-judge with medical knowledge validation" (Requirement 2)

RAG evaluation requires measuring both retrieval quality (are the right documents found?) and generation quality (is the answer faithful, relevant, and safe?). Standard metrics exist for general RAG, but medical domain evaluation needs additional checks: Is the methodology of cited studies sound? Are statistical claims valid? Is the answer clinically applicable?

## Decision

Use **DeepEval** as the evaluation framework with two categories of metrics:

1. **Standard RAG metrics** — Faithfulness, Answer Relevancy, Contextual Precision
2. **Custom domain metrics** — Citation Presence, Medical Disclaimer, and three agent-backed metrics (Methodology Quality, Statistical Validity, Clinical Relevance)

## Metrics

### Standard Metrics (DeepEval built-in)

| Metric | Threshold | What It Measures |
|--------|-----------|-----------------|
| **Faithfulness** | 0.7 | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | 0.7 | Does the answer address the user's query? |
| **Contextual Precision** | 0.7 | Are the most relevant contexts ranked higher in the retrieval? |

All three use `gpt-4o-mini` as the LLM judge. Threshold 0.7 was chosen as a reasonable baseline — strict enough to catch poor responses, lenient enough to account for medical complexity where answers may reference multiple aspects of the query.

### Custom Metrics: Rule-Based

| Metric | Threshold | What It Measures |
|--------|-----------|-----------------|
| **Citation Presence** | 0.5 | Does the answer contain at least one PMID citation? |
| **Medical Disclaimer** | 1.0 | Does the response include a medical disclaimer? |

These are simple pattern-matching checks (regex for PMIDs, keyword search for disclaimer phrases). They cost nothing (no LLM calls) and verify structural requirements of every response.

### Custom Metrics: Agent-Backed

| Metric | Agent Used | Threshold | What It Measures |
|--------|-----------|-----------|-----------------|
| **Methodology Quality** | `MethodologyCriticAgent` | 0.5 | Study design rigor, bias risk, randomization quality |
| **Statistical Validity** | `StatisticalReviewerAgent` | 0.5 | Statistical methods, significance, sample sizes |
| **Clinical Relevance** | `ClinicalApplicabilityAgent` | 0.5 | Real-world clinical applicability |

Each agent-backed metric:

1. Parses DeepEval's `retrieval_context` strings back into `SearchResult` objects
2. Instantiates the corresponding agent with an LLM client
3. Calls `agent.run(query, results)` to get an `AgentResult` with a 1-10 score
4. Normalizes the score to 0.0-1.0 for DeepEval compatibility

## Key Design Decision: Agent Reuse as Metrics

The analysis agents (`methodology_critic`, `statistical_reviewer`, `clinical_applicability`) serve dual purposes:

1. **Runtime analysis** — called via `/analyze` endpoint for user-facing research evaluation
2. **Evaluation metrics** — wrapped as DeepEval `BaseMetric` subclasses for automated quality testing

This reuse means:

- No duplicated evaluation logic — the same prompt and parsing code runs in both contexts
- Improvements to an agent automatically improve the corresponding evaluation metric
- Agent quality can be benchmarked through the evaluation suite itself

The agents were designed with this dual use in mind: they accept `(query, list[SearchResult])` and return structured `AgentResult` with a numeric score, which maps naturally to DeepEval's `measure() → float` interface.

## Evaluation Dataset

**File:** `tests/eval/dataset.json`

10 curated medical queries covering diverse topics:

- Oncology (pancreatic cancer, CAR-T therapy, colorectal surgery)
- Infectious disease (mRNA vaccines, long COVID)
- Chronic conditions (knee osteoarthritis, diabetes, Alzheimer's)
- Emerging therapies (CRISPR, GLP-1 agonists, gut microbiome/immunotherapy)

Each entry has a `query` and `notes` describing expected coverage areas. The dataset does not include expected answers (ground truth) — metrics are computed against retrieval context, not pre-defined answers. This is intentional: the system retrieves from a live Milvus index, and the "correct" answer depends on what's been ingested.

## Running Evaluations

```bash
# Requires: OPENAI_API_KEY, running Milvus, ingested data
uv run pytest tests/eval/test_rag_evaluation.py -v
```

The test parametrizes over the dataset, running all metrics on each query. Results are displayed per-query and per-metric via DeepEval's reporting.

## Alternatives Considered

### RAGAS

- Popular RAG evaluation framework
- Similar metrics (faithfulness, relevancy, context precision)
- Less flexible for custom metrics compared to DeepEval's `BaseMetric` subclassing

**Not chosen** because DeepEval's extensible `BaseMetric` made it straightforward to wrap agents as evaluation metrics. RAGAS would have required more adaptation for the agent-backed metrics.

### Manual evaluation only

- Human expert reviews each response
- Gold standard for medical accuracy
- Does not scale; not reproducible

**Not chosen** as the sole approach. Automated metrics provide repeatable, consistent evaluation. Manual review is valuable but complementary, not a replacement.

### BioASQ / PubMedQA benchmarks

- Established biomedical QA benchmarks with ground truth
- Useful for comparing against published baselines
- Requires mapping our system's output format to benchmark expectations

**Deferred** for future work. The current evaluation uses our own query set against our own corpus. Benchmark comparison would be valuable but is out of scope for the PoC.

## Consequences

### Positive

- 8 metrics covering retrieval quality, generation quality, structural requirements, and domain-specific assessment
- Agent reuse eliminates duplicated evaluation logic
- Fully automated — runs in CI with `pytest`
- Extensible — adding a new agent automatically enables a new evaluation metric

### Trade-offs

- Agent-backed metrics require LLM calls — expensive to run frequently (~3 LLM calls per metric per query = ~90 calls for the full suite)
- No ground-truth answers — metrics measure consistency and quality heuristics, not absolute correctness
- Dataset is small (10 queries) — sufficient for PoC validation but not statistically robust
- Evaluation requires a running Milvus instance with ingested data — cannot run in a pure unit test environment
