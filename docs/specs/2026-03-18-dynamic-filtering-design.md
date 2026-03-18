# Dynamic Filtering: Publication Types & Disease Area

Date: 2026-03-18
Status: Approved

## Problem

`statements.md` requires "Dynamic filtering by research attributes (disease area, publication year, clinical trial stage)". Year and journal filters are implemented, but `publication_types` and `mesh_categories` (disease area) filters are not — despite the data and Milvus schema already supporting them.

## Constraints

- Milvus stores `publication_types` and `mesh_terms` as JSON-string VARCHAR (e.g., `["Journal Article", "Review"]`)
- Milvus `like` operator works on these fields: verified at ~20-40ms overhead on 100k records
- No schema change or data re-ingestion required
- Frontend should not become cluttered — advanced filters hidden by default

## Design

### Filter Logic

- Same field, multiple values: **OR** (broadens results)
- Different fields: **AND** (narrows results)

Example: `publication_types=[RCT, Meta-Analysis]`, `mesh_categories=[Neoplasms]`, `year_min=2023`

```
year >= 2023
AND (publication_types like "%Randomized Controlled Trial%" OR publication_types like "%Meta-Analysis%")
AND (mesh_terms like "%Neoplasms%")
```

### Backend Changes

#### Input Sanitization

Add a `_sanitize_like_value()` helper in `build_filter_expression()` that strips `%`, `"`, and `\` characters from user-supplied values before interpolation into `like` expressions. This prevents pattern injection and expression syntax breakage.

```python
def _sanitize_like_value(value: str) -> str:
    return value.replace("%", "").replace('"', "").replace("\\", "")
```

#### `build_filter_expression()` in `backend/src/retrieval/search.py`

Add two new filter blocks after existing `year`/`journal` logic:

```python
if filters.publication_types:
    pt_clauses = [
        f'publication_types like "%{_sanitize_like_value(pt)}%"'
        for pt in filters.publication_types
    ]
    conditions.append(f"({' or '.join(pt_clauses)})")

if filters.mesh_categories:
    # mesh_categories maps to the mesh_terms Milvus field via like substring match
    mc_clauses = [
        f'mesh_terms like "%{_sanitize_like_value(mc)}%"'
        for mc in filters.mesh_categories
    ]
    conditions.append(f"({' or '.join(mc_clauses)})")
```

`SearchFilters.publication_types` and `SearchFilters.mesh_categories` already exist in the model with `list[str]` defaults — no model change needed.

#### `OUTPUT_FIELDS` and `SearchResult`

Add `publication_types` to `OUTPUT_FIELDS` in `search.py` so filtered results include the field. Add `publication_types: list[str]` to the `SearchResult` model and parse the JSON string in `parse_search_results()`, same pattern as `mesh_terms`.

#### API Request Models

Add `publication_types` and `mesh_categories` fields to:

- `AskRequest` in `backend/src/api/routes/ask.py`
- `SearchRequest` in `backend/src/api/routes/search.py`
- `ReviewRequest` in `backend/src/api/routes/review.py`

All as `list[str] = Field(default_factory=list)`. Pass through to `SearchFilters` constructor. Follow the existing pattern for filter field pass-through (each route constructs `SearchFilters` inline).

### Frontend Changes

#### Presets

Publication Types (~7 items):
- Review
- Systematic Review
- Meta-Analysis
- Randomized Controlled Trial
- Case Reports
- Clinical Trial
- Observational Study

Disease Areas (~10 items, matching `data_pipeline/config.yaml` categories):
- Neoplasms
- Cardiovascular Diseases
- Infectious Diseases
- Nervous System Diseases
- Respiratory Tract Diseases
- Digestive System Diseases
- Urogenital Diseases
- Musculoskeletal Diseases
- Nutritional and Metabolic Diseases
- Immune System Diseases

These are defined as constants in the frontend only. The backend accepts any string.

#### `FilterPanel.tsx`

- Add collapsible "Advanced Filters" section, **hidden by default**
- Two checkbox groups: Publication Types, Disease Area
- Show badge with count of active advanced filters when collapsed
- Selected values passed to API as `publication_types` and `mesh_categories` arrays
- Existing "Clear" button resets advanced filters too

#### `api.ts` and `types/index.ts`

- Add `publication_types: string[]` and `mesh_categories: string[]` to request types and `SearchResult` type

### Tests

- `test_search.py`: Test `build_filter_expression()` with new filter types (single, multiple, combined with year), sanitization of special characters
- `test_api_search.py`: Test new parameters propagate to `SearchFilters`
- `test_api_ask.py`: Test both synchronous and SSE streaming paths pass new filters through
- `test_api_review.py`: Test new parameters propagate to `SearchFilters`
- No breaking changes: default values are empty lists, existing behavior unchanged

## Approach Rationale

Chose `like` on VARCHAR over ARRAY migration because:
- Zero schema change, zero data re-ingestion
- Verified working and performant on current data
- Isolated to `build_filter_expression()` — can swap to `array_contains` later if schema migrates to ARRAY type
