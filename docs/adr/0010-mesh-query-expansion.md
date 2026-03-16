ADR: MeSH-Based Query Expansion

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

Users search with natural language queries like "latest treatments for knee arthritis." The PubMed corpus uses standardized Medical Subject Headings (MeSH) terminology — the same concept may be indexed under "Osteoarthritis, Knee" with child terms like "Patellofemoral Pain Syndrome." Without expansion, the system may miss relevant abstracts that use different but related terminology.

The requirements specify "Query expansion using medical terminology (MeSH terms)" (Requirement 1).

## Decision

Implement a three-stage query expansion pipeline:

1. **LLM keyword extraction** — extract medical terms from the natural language query
2. **MeSH hierarchy lookup** — match keywords to MeSH descriptors and retrieve child terms via DuckDB
3. **Query augmentation** — append MeSH terms to the original query

Use a **local DuckDB database** built from NLM's MeSH descriptor XML (`desc2025.xml`) for all lookups.

## Pipeline

```
"latest treatments for knee arthritis"
  → LLM: ["knee arthritis", "treatment"]
  → MeSH lookup: "knee arthritis" → "Osteoarthritis, Knee" (descriptor)
  → Children: "Patellofemoral Pain Syndrome", ...
  → Expanded: "latest treatments for knee arthritis (Osteoarthritis, Knee; Patellofemoral Pain Syndrome)"
```

### Stage 1: LLM Keyword Extraction

The LLM receives the user query and returns a JSON array of medical/biomedical keywords. The prompt focuses extraction on diseases, conditions, treatments, drugs, and anatomical terms.

**Why LLM over NER/regex:** Medical queries use colloquial language ("knee arthritis" not "Osteoarthritis, Knee"). An LLM understands the intent and extracts clinically meaningful terms that a regex or simple NER would miss.

### Stage 2: MeSH Hierarchy Lookup

Each extracted keyword is looked up in DuckDB:

1. **Name match** — exact match against descriptor `name` field
2. **Synonym match** — match against `synonyms` (MeSH "entry terms")
3. **Child term retrieval** — for matched descriptors, find child descriptors via tree number prefix match

The MeSH hierarchy is a polyhierarchy (a term can appear in multiple trees). The lookup follows all tree paths to capture the full set of related terms.

### Stage 3: Query Augmentation

MeSH terms and child terms are appended to the original query in parentheses:

```
"{original_query} ({MeSH term 1}; {MeSH term 2}; {child term 1}; ...)"
```

The augmented query is then embedded and used for vector search. The parenthetical format ensures the original query semantics dominate while related terms broaden recall.

## Why DuckDB (Local) Over External APIs

| Option | Pros | Cons |
|--------|------|------|
| **DuckDB (chosen)** | Zero latency, offline, no rate limits | Requires build step, static snapshot |
| NLM MeSH API | Always current, no local storage | Network dependency, rate limits, ~100-500ms per call |
| UMLS API | Richer ontology (RxNorm, SNOMED CT) | Requires UMLS license, complex API, heavier |
| In-memory dict | Fastest | ~50MB RAM; no SQL flexibility for tree queries |

DuckDB was chosen because:

- **30k+ MeSH descriptors fit in ~50MB** — trivial storage cost
- **Tree number prefix queries** (`WHERE tree_number LIKE 'C05.550%'`) are natural in SQL
- **No network dependency** — works offline, no API keys, no rate limits
- **Build once, use everywhere** — `scripts/build_mesh_db.py` converts `desc2025.xml` → `mesh.duckdb`

## DuckDB Schema

Built by `scripts/build_mesh_db.py` from NLM's `desc2025.xml`:

```sql
descriptors(ui, name, tree_numbers JSON, synonyms JSON)
```

Lookup methods in `shared/mesh_db.py`:

- `lookup(term)` — find descriptor by name or synonym (case-insensitive)
- `get_children(tree_number)` — find child descriptors by tree number prefix
- `validate_term(term)` — check if a term exists in MeSH vocabulary (used by guardrails)

## Implementation

**Files:**

- `retrieval/query_expander.py` — `QueryExpander` class with `expand()` method
- `shared/mesh_db.py` — `MeSHDatabase` class wrapping DuckDB
- `scripts/build_mesh_db.py` — one-time build script

**Data model:**

```python
class ExpandedQuery(BaseModel):
    original_query: str
    keywords: list[str]       # LLM-extracted keywords
    mesh_terms: list[str]     # Matched MeSH descriptors
    child_terms: list[str]    # Child terms from hierarchy
    expanded_query: str       # Final augmented query string
```

## Consequences

### Positive

- Bridges the gap between colloquial user language and standardized PubMed terminology
- Child term expansion captures related concepts the user may not have thought of
- DuckDB is fast (~1ms per lookup), offline, and reusable by both query expansion and guardrail validation
- `ExpandedQuery` model makes the expansion transparent and debuggable

### Trade-offs

- One additional LLM call per query for keyword extraction (~200-500ms, ~100 tokens)
- MeSH is a static snapshot (desc2025.xml) — new terms added after the build are not captured
- Over-expansion risk: too many child terms can dilute the query embedding. Mitigated by the parenthetical format, which keeps the original query dominant
- LLM keyword extraction may occasionally produce non-medical terms, which simply fail the MeSH lookup (harmless)
