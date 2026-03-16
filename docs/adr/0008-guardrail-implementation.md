ADR: Guardrail Implementation for Medical Accuracy

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The system generates answers from medical research abstracts. Incorrect or ungrounded medical information can be harmful. The requirements specify:

- "Medical terminology validation guardrails" (Requirement 1)
- "Medical accuracy guardrails and fact-checking" (Requirement 2)
- "LLM-as-judge with medical knowledge validation" (Requirement 2)

The guardrail system must balance safety (catching hallucinations and ungrounded claims) with usability (not blocking legitimate queries or over-warning).

## Decision

Implement a **two-layer guardrail system** with soft warnings (non-blocking):

1. **Input guardrails** — classify whether the query is medical/biomedical
2. **Output guardrails** — validate the generated answer against source abstracts

Both layers produce warnings that are returned to the user but do not block the response.

## Layer 1: Input Guardrails

**File:** `guardrails/input.py`

A lightweight LLM classification call determines whether the query is related to medical/biomedical research.

```
Query → LLM ("Is this medical? yes/no") → RelevanceResult(is_relevant, warning)
```

- If `no`: returns a soft warning ("This query may not be related to medical research. Results may be less relevant.")
- If `yes` or if classification fails: no warning, processing continues
- **Never blocks** — the query proceeds regardless of classification result

**Why soft warning, not hard block:** Users may have legitimate cross-domain queries (e.g., "AI applications in radiology"). Blocking non-medical queries would harm UX and require an appeal mechanism. A warning is informative without being obstructive.

## Layer 2: Output Guardrails

**File:** `guardrails/output.py`

Three validation checks run after LLM answer generation:

### Check 1: LLM-Based Validation (Grounding + Hallucination + Treatment Recommendations)

An LLM call reviews the generated answer against the source abstracts:

```
Answer + Source Abstracts → LLM Validator → list[GuardrailWarning]
```

The validator checks for:

| Check | Severity | What It Catches |
|-------|----------|-----------------|
| `citation_grounding` | error | Claims not supported by any cited abstract |
| `hallucination` | warning | Facts (drug names, statistics, outcomes) not found in source material |
| `treatment_recommendation` | warning | Definitive treatment recommendations without hedging language |

The LLM returns a structured JSON array of issues, each with `check`, `severity`, `message`, and `span` (the problematic text).

### Check 2: MeSH Terminology Validation

A non-LLM check that extracts capitalized multi-word terms from the answer and verifies them against the MeSH vocabulary in DuckDB.

```
Answer → regex extraction → MeSH lookup → terminology warnings
```

Terms not found in MeSH get a `terminology` warning with severity `warning`. This catches fabricated medical terms without requiring an LLM call.

### Check 3: Medical Disclaimer

A static disclaimer is always appended:

> "Disclaimer: This information is generated from research abstracts and is intended for educational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions."

## Grounding Status

The `is_grounded` boolean on the response is determined by whether any `citation_grounding` errors were found. If the answer contains claims that cannot be traced to a cited abstract, `is_grounded = false`. The frontend can use this to display a warning banner.

## Architecture

```
GuardrailClient (Protocol)
├── LocalGuardrailClient   — runs validation in-process (monolith mode)
└── (RemoteGuardrailClient — future: HTTP call to guardrail service)

LocalGuardrailClient
└── GuardrailValidator
    ├── _llm_validate()   — LLM-as-judge (grounding, hallucination, treatment)
    └── _mesh_validate()  — MeSH term lookup (DuckDB)
```

The `GuardrailClient` Protocol enables the same abstraction used for `SearchClient` — guardrails can be extracted into a separate microservice without changing the RAG chain.

## Integration in RAG Pipeline

`rag/chain.py` step 7:

```python
if guardrails_enabled:
    guardrail_client = LocalGuardrailClient(llm=llm, mesh_db=mesh_db)
    return guardrail_client.validate(rag_response, results)  # → ValidatedResponse
```

Guardrails can be disabled per-request via `guardrails_enabled: false` in the API request, useful for benchmarking or debugging.

## Alternatives Considered

### Hard blocking (reject non-medical queries)

- Requires high classification accuracy to avoid false positives
- Needs an appeal/override mechanism
- Over-engineering for a PoC

**Rejected:** Soft warnings achieve the safety goal without blocking legitimate edge cases.

### Rule-based output validation only (no LLM-as-judge)

- Cheaper (no additional LLM call)
- Limited to pattern matching (e.g., "contains PMID?")
- Cannot assess semantic grounding or detect hallucinated facts

**Rejected:** Medical accuracy requires semantic understanding that rules alone cannot provide.

### Guardrails AI framework

- Structured validation framework with pre-built validators
- Additional dependency; learning curve
- Less control over medical-specific validation logic

**Rejected:** The custom implementation is simpler and tailored to the specific checks needed. Two checks (LLM + MeSH) are manageable without a framework.

## Consequences

### Positive

- Catches hallucinated claims, ungrounded statements, and fabricated medical terms
- Non-blocking design preserves UX while adding safety
- MeSH validation is zero-cost (local DuckDB lookup)
- Structured warnings (`GuardrailWarning` model) enable frontend display

### Trade-offs

- One additional LLM call per request (~500-1000ms, ~200 tokens) for output validation
- LLM-as-judge accuracy depends on the judge model (currently gpt-4o-mini)
- MeSH regex extraction is heuristic — may miss terms or flag false positives
- Guardrails add latency to the critical path (not parallelizable since they need the generated answer)
