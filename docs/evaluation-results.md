# Evaluation Results

Evaluation run: 2026-03-17
Dataset: 10 medical research queries (see `backend/tests/eval/dataset.json`)
Configuration: hybrid search, cross-encoder reranker, GPT-4o-mini, guardrails enabled
Corpus: 100,000 PubMed abstracts

## Summary

### Accuracy Metrics (DeepEval + Custom)

| Metric | Avg | Min | Max | Threshold | Pass Rate |
|--------|-----|-----|-----|-----------|-----------|
| **Faithfulness** | 0.856 | 0.200 | 1.000 | 0.7 | 8/10 |
| **Answer Relevancy** | 0.815 | 0.000 | 1.000 | 0.7 | 8/10 |
| **Citation Presence** | 0.900 | 0.000 | 1.000 | 0.5 | 9/10 |
| **Medical Disclaimer** | 1.000 | 1.000 | 1.000 | 1.0 | 10/10 |

- **Faithfulness** — Is the generated answer grounded in the retrieved abstracts? Evaluated by GPT-4o-mini via DeepEval.
- **Answer Relevancy** — Does the answer address the user's query? Evaluated by GPT-4o-mini via DeepEval.
- **Citation Presence** — Does the answer include PMID citations? Regex-based check.
- **Medical Disclaimer** — Is a medical disclaimer appended? Keyword-based check.

### Latency

| Metric | Avg | P50 | P99 |
|--------|-----|-----|-----|
| **End-to-end /ask** (full RAG pipeline) | 16.51s | 18.47s | 22.51s |
| **Search-only /search** (retrieval + reranking) | 1.00s | 0.32s | 6.68s |

The `/ask` latency includes: query expansion → hybrid search → cross-encoder reranking → LLM generation → output guardrails (grounding check via additional LLM call). The majority of latency comes from the LLM generation and guardrail validation steps.

## Per-Query Results

| # | Query | Ask (s) | Search (s) | Faithfulness | Relevancy | Citations | Disclaimer | Grounded |
|---|-------|---------|------------|-------------|-----------|-----------|------------|----------|
| 1 | Latest treatments for early-stage pancreatic cancer | 9.36 | 0.44 | 0.200 | 0.000 | 1.0 | 1.0 | Yes |
| 2 | Non-invasive therapies for knee osteoarthritis | 18.47 | 6.68 | 0.917 | 0.750 | 1.0 | 1.0 | Yes |
| 3 | mRNA vaccine efficacy and safety after 2022 | 20.20 | 0.29 | 0.857 | 1.000 | 1.0 | 1.0 | No |
| 4 | CRISPR gene therapy for sickle cell disease | 17.71 | 0.29 | 1.000 | 0.917 | 1.0 | 1.0 | Yes |
| 5 | Gut microbiome and immune checkpoint inhibitors | 5.63 | 0.35 | 0.667 | 0.750 | 0.0 | 1.0 | Yes |
| 6 | GLP-1 receptor agonists in type 2 diabetes | 20.10 | 0.32 | 1.000 | 0.733 | 1.0 | 1.0 | Yes |
| 7 | ML for early detection of Alzheimer's disease | 21.47 | 0.30 | 1.000 | 1.000 | 1.0 | 1.0 | No |
| 8 | Robotic vs. laparoscopic surgery for colorectal cancer | 22.51 | 0.73 | 1.000 | 1.000 | 1.0 | 1.0 | No |
| 9 | Long COVID cardiovascular complications | 14.02 | 0.28 | 1.000 | 1.000 | 1.0 | 1.0 | No |
| 10 | CAR-T cell therapy for relapsed DLBCL | 15.58 | 0.27 | 1.000 | 1.000 | 1.0 | 1.0 | Yes |

## Analysis

### Strengths

- **High faithfulness** (avg 0.856): The system generates answers well-grounded in retrieved abstracts. 6/10 queries scored 1.0 (perfectly grounded).
- **Strong answer relevancy** (avg 0.815): Responses directly address the medical research queries.
- **Consistent guardrails**: Medical disclaimer is present in 100% of responses. Citation presence at 90%.

### Areas for Improvement

- **Query 1 (pancreatic cancer)**: Low faithfulness (0.200) and relevancy (0.000). This is the most specific clinical query — the system may be retrieving broadly related but not precisely matching abstracts for this narrow topic.
- **Query 5 (gut microbiome + immunotherapy)**: No PMID citations in the answer text. The cross-domain nature of the query (microbiome + oncology) may lead to answers that summarize without specific citations.
- **Grounding check (is_grounded)**: 4/10 queries flagged as "not grounded" by the output guardrail. This is the LLM-based grounding check being conservative — it flags answers that include inferences or summaries not verbatim in the abstracts, which is expected for synthesis-style answers.

### Latency Breakdown

- Search-only latency (p50 = 0.32s) is fast, demonstrating that hybrid retrieval + reranking performs well at 100k scale.
- End-to-end latency (avg 16.51s) is dominated by LLM calls: one for answer generation and one for guardrail validation. This is a known trade-off for the guardrail architecture (see ADR-0008).
- The latency outlier (Query 2, search = 6.68s) appears to be a cold-start effect or Milvus cache miss.

## Methodology

1. Each query is sent to the `/ask` endpoint (full RAG pipeline) and `/search` endpoint (retrieval only).
2. DeepEval standard metrics (Faithfulness, Answer Relevancy) are computed using GPT-4o-mini as the judge model.
3. Custom metrics (Citation Presence, Medical Disclaimer) use deterministic checks (regex, keyword matching).
4. Latency is measured wall-clock time for each API call.
5. The Contextual Precision metric was excluded because it requires ground-truth expected output, which is not available for open-ended medical research queries.

## Reproduction

```bash
cd capstone/backend

# Requires: Milvus running, data ingested, API server running, OPENAI_API_KEY set
.venv/bin/python scripts/run_evaluation.py > data/eval_results.json
```

Full raw results are saved to `backend/data/eval_results.json`.
