# Dynamic Filtering — Plan 1: Backend Search Core

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `publication_types` and `mesh_categories` like-based filters to `build_filter_expression()`, add `publication_types` to search output fields and `SearchResult` model.

**Architecture:** Extend `build_filter_expression()` with OR-joined `like` clauses per filter field, AND-joined across fields. Add `_sanitize_like_value()` to prevent pattern injection. Add `publication_types` to `OUTPUT_FIELDS` and `SearchResult` for result display.

**Tech Stack:** Python, Pydantic, Milvus `like` operator, pytest

**Spec:** `docs/specs/2026-03-18-dynamic-filtering-design.md`

**Parallelism:** This plan has NO dependencies on Plans 2 or 3. Can run in parallel.

---

### Task 1: Add `_sanitize_like_value()` + filter expression logic

**Files:**
- Modify: `backend/src/retrieval/search.py:17,25-37`
- Test: `backend/tests/unit/test_search.py`

- [ ] **Step 1: Write failing tests for new filter expressions**

Add to `backend/tests/unit/test_search.py`:

```python
def test_build_filter_publication_types_single():
    filters = SearchFilters(publication_types=["Review"])
    expr = build_filter_expression(filters)
    assert 'publication_types like "%Review%"' in expr


def test_build_filter_publication_types_multiple_or():
    filters = SearchFilters(publication_types=["Review", "Meta-Analysis"])
    expr = build_filter_expression(filters)
    assert 'publication_types like "%Review%"' in expr
    assert 'publication_types like "%Meta-Analysis%"' in expr
    assert " or " in expr


def test_build_filter_mesh_categories_single():
    filters = SearchFilters(mesh_categories=["Neoplasms"])
    expr = build_filter_expression(filters)
    assert 'mesh_terms like "%Neoplasms%"' in expr


def test_build_filter_mesh_categories_multiple_or():
    filters = SearchFilters(mesh_categories=["Neoplasms", "Cardiovascular Diseases"])
    expr = build_filter_expression(filters)
    assert 'mesh_terms like "%Neoplasms%"' in expr
    assert 'mesh_terms like "%Cardiovascular Diseases%"' in expr
    assert " or " in expr


def test_build_filter_combined_year_and_publication_types():
    filters = SearchFilters(year_min=2023, publication_types=["Randomized Controlled Trial"])
    expr = build_filter_expression(filters)
    assert "year >= 2023" in expr
    assert 'publication_types like "%Randomized Controlled Trial%"' in expr
    assert " and " in expr


def test_build_filter_combined_all():
    filters = SearchFilters(
        year_min=2022,
        publication_types=["Review"],
        mesh_categories=["Neoplasms"],
    )
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert 'publication_types like "%Review%"' in expr
    assert 'mesh_terms like "%Neoplasms%"' in expr
    # Should have AND between groups
    assert expr.count(" and ") >= 2


def test_build_filter_sanitize_percent():
    filters = SearchFilters(publication_types=["Review%injection"])
    expr = build_filter_expression(filters)
    assert "%" not in expr.replace('like "%', "").replace('%"', "")


def test_build_filter_sanitize_quote():
    filters = SearchFilters(publication_types=['Review"injection'])
    expr = build_filter_expression(filters)
    # The sanitized value should not contain double quotes
    assert 'Reviewinjection' in expr


def test_build_filter_sanitize_backslash():
    filters = SearchFilters(mesh_categories=["Neoplasms\\test"])
    expr = build_filter_expression(filters)
    assert "\\" not in expr.replace('like "%', "").replace('%"', "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/test_search.py -v -k "publication_types or mesh_categories or sanitize"`
Expected: FAIL — current `build_filter_expression()` does not handle these fields.

- [ ] **Step 3: Implement `_sanitize_like_value()` and extend `build_filter_expression()`**

Edit `backend/src/retrieval/search.py`. Add `_sanitize_like_value` before `build_filter_expression`, then add two new blocks inside `build_filter_expression`:

```python
def _sanitize_like_value(value: str) -> str:
    """Strip characters that could alter like pattern or break expression syntax."""
    return value.replace("%", "").replace('"', "").replace("\\", "")


def build_filter_expression(filters: SearchFilters) -> str:
    """Build a Milvus boolean filter expression from SearchFilters."""
    conditions = []

    if filters.year_min is not None:
        conditions.append(f"year >= {filters.year_min}")
    if filters.year_max is not None:
        conditions.append(f"year <= {filters.year_max}")
    if filters.journals:
        journals_str = json.dumps(filters.journals)
        conditions.append(f"journal in {journals_str}")
    if filters.publication_types:
        pt_clauses = [
            f'publication_types like "%{_sanitize_like_value(pt)}%"'
            for pt in filters.publication_types
        ]
        conditions.append(f"({' or '.join(pt_clauses)})")
    if filters.mesh_categories:
        # mesh_categories filter maps to the mesh_terms Milvus field
        mc_clauses = [
            f'mesh_terms like "%{_sanitize_like_value(mc)}%"'
            for mc in filters.mesh_categories
        ]
        conditions.append(f"({' or '.join(mc_clauses)})")

    return " and ".join(conditions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/test_search.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/retrieval/search.py backend/tests/unit/test_search.py
git commit -m "feat: add publication_types and mesh_categories to build_filter_expression"
```

---

### Task 2: Add `publication_types` to `OUTPUT_FIELDS`, `SearchResult`, and `parse_search_results()`

**Files:**
- Modify: `backend/src/shared/models.py:47-54`
- Modify: `backend/src/retrieval/search.py:17,51-73`
- Test: `backend/tests/unit/test_search.py`

- [ ] **Step 1: Write failing test for parse_search_results with publication_types**

Add to `backend/tests/unit/test_search.py`:

```python
def test_parse_search_results_includes_publication_types():
    entity_data = {
        "pmid": "456",
        "title": "Test Title",
        "abstract_text": "Test abstract",
        "year": 2023,
        "journal": "Nature",
        "mesh_terms": '["Neoplasms"]',
        "publication_types": '["Journal Article", "Review"]',
    }
    mock_hit = MagicMock()
    mock_hit.entity.get = lambda k: entity_data.get(k)
    mock_hit.distance = 0.90

    results = parse_search_results([mock_hit])
    assert len(results) == 1
    assert results[0].publication_types == ["Journal Article", "Review"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_search.py::test_parse_search_results_includes_publication_types -v`
Expected: FAIL — `SearchResult` has no `publication_types` field.

- [ ] **Step 3: Add `publication_types` to `SearchResult` model**

Edit `backend/src/shared/models.py`, add field to `SearchResult` class after `mesh_terms`:

```python
class SearchResult(BaseModel):
    pmid: str
    title: str
    abstract_text: str
    score: float
    year: int
    journal: str
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Add `publication_types` to `OUTPUT_FIELDS` and `parse_search_results()`**

Edit `backend/src/retrieval/search.py`:

Update `OUTPUT_FIELDS` (line 17):
```python
OUTPUT_FIELDS = ["pmid", "title", "abstract_text", "year", "journal", "mesh_terms", "publication_types"]
```

Update `parse_search_results()` to parse `publication_types` the same way as `mesh_terms`:

```python
def parse_search_results(hits: list) -> list[SearchResult]:
    results = []
    for hit in hits:
        mesh_raw = hit.entity.get("mesh_terms")
        mesh_terms = json.loads(mesh_raw) if isinstance(mesh_raw, str) else mesh_raw

        pt_raw = hit.entity.get("publication_types")
        publication_types = json.loads(pt_raw) if isinstance(pt_raw, str) else pt_raw

        results.append(
            SearchResult(
                pmid=hit.entity.get("pmid"),
                title=hit.entity.get("title"),
                abstract_text=hit.entity.get("abstract_text"),
                score=hit.distance,
                year=hit.entity.get("year"),
                journal=hit.entity.get("journal"),
                mesh_terms=mesh_terms if mesh_terms else [],
                publication_types=publication_types if publication_types else [],
            )
        )
    return results
```

- [ ] **Step 5: Run all search tests**

Run: `cd backend && uv run pytest tests/unit/test_search.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full unit test suite to check for regressions**

Run: `cd backend && uv run pytest tests/unit/ -v`
Expected: ALL PASS. The new `publication_types` field has a default of `[]`, so existing test fixtures that create `SearchResult` without it will still work.

- [ ] **Step 7: Commit**

```bash
git add backend/src/shared/models.py backend/src/retrieval/search.py backend/tests/unit/test_search.py
git commit -m "feat: add publication_types to OUTPUT_FIELDS and SearchResult"
```
