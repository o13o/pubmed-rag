# Evaluation Results (Full Metrics)

Evaluation run: 2026-03-17
Dataset: 10 medical research queries (see `backend/tests/eval/dataset.json`)
Configuration: hybrid search, cross-encoder reranker, GPT-4o-mini, guardrails enabled
Corpus: 100,000 PubMed abstracts

## Summary

### Standard Metrics (DeepEval)

| Metric | Threshold | Pass Rate | Description |
|--------|-----------|-----------|-------------|
| **Faithfulness** | 0.7 | 9/10 | Is the answer grounded in retrieved abstracts? |
| **Answer Relevancy** | 0.7 | 8/10 | Does the answer address the user's query? |
| **Contextual Relevancy** | 0.5 | 7/10 | Are the retrieved contexts relevant to the query? |

### Custom Metrics

| Metric | Threshold | Pass Rate | Description |
|--------|-----------|-----------|-------------|
| **Citation Presence** | 0.5 | 9/10 | Does the answer include PMID citations? (regex) |
| **Medical Disclaimer** | 1.0 | 10/10 | Is a medical disclaimer appended? (keyword) |
| **Methodology Quality** | 0.5 | 10/10 | Study design and methodological rigor (via MethodologyCriticAgent) |
| **Statistical Validity** | 0.5 | 9/10 | Statistical methods and significance (via StatisticalReviewerAgent) |
| **Clinical Relevance** | 0.5 | 10/10 | Real-world clinical applicability (via ClinicalApplicabilityAgent) |

### Overall

- **6/10 queries passed all 8 metrics**
- **4/10 queries failed on 1-2 metrics** (retrieval quality issues, not system errors)
- Total evaluation time: ~14 minutes (10 queries x 8 metrics, sequential execution)

## Per-Query Results

| # | Query | Faith. | Ans.Rel. | Ctx.Rel. | Citation | Disclaimer | Method. | Stat. | Clinical | Result |
|---|-------|--------|----------|----------|----------|------------|---------|-------|----------|--------|
| 1 | Latest treatments for early-stage pancreatic cancer | PASS | 0.17 FAIL | 0.30 FAIL | PASS | PASS | PASS | PASS | PASS | FAIL |
| 2 | Non-invasive therapies for knee osteoarthritis | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3 | mRNA vaccine efficacy after 2022 | PASS | PASS | 0.43 FAIL | PASS | PASS | PASS | PASS | PASS | FAIL |
| 4 | CRISPR gene therapy for sickle cell disease | PASS | PASS | 0.18 FAIL | PASS | PASS | PASS | 0.4 FAIL | PASS | FAIL |
| 5 | Gut microbiome + immune checkpoint inhibitors | PASS | 0.33 FAIL | PASS | 0.0 FAIL | PASS | PASS | PASS | PASS | FAIL |
| 6 | GLP-1 receptor agonists in type 2 diabetes | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 7 | ML for early detection of Alzheimer's disease | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 8 | Robotic vs. laparoscopic surgery for colorectal cancer | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 9 | Long COVID cardiovascular complications | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 10 | CAR-T cell therapy for relapsed DLBCL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Failure Analysis

### Query 1: Pancreatic cancer (Answer Relevancy 0.17, Contextual Relevancy 0.30)
The query specifies "early-stage" pancreatic cancer, but the retrieved abstracts primarily discuss advanced/metastatic PDAC. The 100k corpus may lack sufficient early-stage-specific literature, and the semantic search does not distinguish disease stage effectively.

### Query 3: mRNA vaccine after 2022 (Contextual Relevancy 0.43)
The temporal filter "after 2022" is expressed in natural language rather than as a metadata filter. Many retrieved abstracts predate 2022. Using `year_min: 2022` as an explicit filter would improve results.

### Query 4: CRISPR for sickle cell (Contextual Relevancy 0.18, Statistical Validity 0.4)
Retrieved contexts discuss CRISPR gene therapy broadly rather than sickle cell disease specifically. The corpus may have limited sickle-cell-specific CRISPR studies within the 100k sample.

### Query 5: Gut microbiome + checkpoint inhibitors (Answer Relevancy 0.33, Citation Presence 0.0)
Cross-domain query spanning microbiome and oncology. The LLM answer included extensive disclaimers that diluted relevancy scoring. No PMID citations were embedded in the answer text.

## Agent-Based Custom Metrics

The three new custom metrics leverage the multi-agent system as DeepEval evaluators:

- **Methodology Quality** (MethodologyCriticAgent): Evaluates study design, bias risk, and methodological rigor across retrieved abstracts. Scored 1-10, normalized to 0-1. Passed 10/10 at threshold 0.5.
- **Statistical Validity** (StatisticalReviewerAgent): Analyzes statistical methods, significance reporting, and sample sizes. Passed 9/10 (failed on CRISPR query where abstracts lacked rigorous statistical data).
- **Clinical Relevance** (ClinicalApplicabilityAgent): Assesses real-world clinical applicability of retrieved research. Passed 10/10.

These metrics demonstrate that the multi-agent analysis system can serve dual purposes: runtime research analysis and offline RAG quality evaluation.

## Comparison with Basic Evaluation

| Metric | Basic Run (5 metrics) | Full Run (8 metrics) |
|--------|----------------------|---------------------|
| Faithfulness | 8/10 pass | 9/10 pass |
| Answer Relevancy | 8/10 pass | 8/10 pass |
| Citation Presence | 9/10 pass | 9/10 pass |
| Medical Disclaimer | 10/10 pass | 10/10 pass |
| Contextual Relevancy | — | 7/10 pass |
| Methodology Quality | — | 10/10 pass |
| Statistical Validity | — | 9/10 pass |
| Clinical Relevance | — | 10/10 pass |

The Contextual Relevancy metric (replacing Contextual Precision which required expected_output) is the strictest metric, correctly identifying retrieval gaps for niche queries in the 100k sample.

## Reproduction

```bash
cd backend

# Requires: Milvus running, data ingested, OPENAI_API_KEY set
# Note: stop the backend server first to avoid DuckDB lock conflicts
uv run pytest tests/eval/test_rag_evaluation.py -v
```
