# Dynamic Filtering — Plan 3: Frontend

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add collapsible Advanced Filters (Publication Types + Disease Area checkboxes) to the FilterPanel, wire them through types and API calls.

**Architecture:** Add fields to TypeScript interfaces, add preset constants and collapsible checkbox UI to FilterPanel, pass selected values through all API call sites in App.tsx.

**Tech Stack:** React 19, TypeScript, Tailwind CSS 4

**Spec:** `docs/specs/2026-03-18-dynamic-filtering-design.md`

**Parallelism:** This plan has NO dependencies on Plans 1 or 2. Can run in parallel. Frontend changes are entirely in `frontend/src/`.

---

### Task 1: Update TypeScript types

**Files:**
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Add fields to `Filters`, request types, and `SearchResult`**

Edit `frontend/src/types/index.ts`:

Add `publication_types` and `mesh_categories` to `Filters`:

```typescript
export interface Filters {
  year_min?: number;
  year_max?: number;
  top_k: number;
  search_mode: string;
  publication_types: string[];
  mesh_categories: string[];
}
```

Add to `AskRequest`:

```typescript
export interface AskRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
  publication_types?: string[];
  mesh_categories?: string[];
  stream?: boolean;
}
```

Add to `SearchRequest`:

```typescript
export interface SearchRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
  publication_types?: string[];
  mesh_categories?: string[];
}
```

Add to `SearchResult`:

```typescript
export interface SearchResult {
  pmid: string;
  title: string;
  abstract_text: string;
  score: number;
  year: number;
  journal: string;
  mesh_terms: string[];
  publication_types: string[];
}
```

Add to `ReviewRequest`:

```typescript
export interface ReviewRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  journals?: string[];
  top_k?: number;
  search_mode?: string;
  publication_types?: string[];
  mesh_categories?: string[];
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: Errors about missing `publication_types` and `mesh_categories` in `Filters` initialization in `App.tsx`. This is expected — we'll fix it in Task 3.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "feat: add publication_types and mesh_categories to frontend types"
```

---

### Task 2: Add collapsible Advanced Filters to FilterPanel

**Files:**
- Modify: `frontend/src/components/FilterPanel.tsx`

- [ ] **Step 1: Add preset constants and collapsible checkbox UI**

Replace the entire `frontend/src/components/FilterPanel.tsx` with:

```tsx
import { useState } from "react";
import type { Filters, Mode } from "../types";

const PUBLICATION_TYPE_PRESETS = [
  "Review",
  "Systematic Review",
  "Meta-Analysis",
  "Randomized Controlled Trial",
  "Case Reports",
  "Clinical Trial",
  "Observational Study",
];

const DISEASE_AREA_PRESETS = [
  "Neoplasms",
  "Cardiovascular Diseases",
  "Infectious Diseases",
  "Nervous System Diseases",
  "Respiratory Tract Diseases",
  "Digestive System Diseases",
  "Urogenital Diseases",
  "Musculoskeletal Diseases",
  "Nutritional and Metabolic Diseases",
  "Immune System Diseases",
];

interface Props {
  mode: Mode;
  filters: Filters;
  onModeChange: (mode: Mode) => void;
  onFiltersChange: (filters: Filters) => void;
  onClear?: () => void;
  showClear?: boolean;
}

export function FilterPanel({
  mode,
  filters,
  onModeChange,
  onFiltersChange,
  onClear,
  showClear,
}: Props) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const advancedCount =
    (filters.publication_types?.length ?? 0) +
    (filters.mesh_categories?.length ?? 0);

  const toggleValue = (
    field: "publication_types" | "mesh_categories",
    value: string,
  ) => {
    const current = filters[field] ?? [];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    onFiltersChange({ ...filters, [field]: next });
  };

  return (
    <div className="border-t border-gray-800 bg-gray-950">
      <div className="flex items-center gap-3 flex-wrap px-4 py-2">
        {/* Mode toggle */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => onModeChange("ask")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              mode === "ask"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Ask
          </button>
          <button
            onClick={() => onModeChange("search")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              mode === "search"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Search
          </button>
        </div>

        <span className="text-gray-700">|</span>

        {/* Search mode */}
        <div className="flex items-center gap-1">
          <button
            onClick={() =>
              onFiltersChange({ ...filters, search_mode: "dense" })
            }
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              filters.search_mode === "dense"
                ? "bg-emerald-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Dense
          </button>
          <button
            onClick={() =>
              onFiltersChange({ ...filters, search_mode: "hybrid" })
            }
            className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
              filters.search_mode === "hybrid"
                ? "bg-emerald-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Hybrid
          </button>
        </div>

        <span className="text-gray-700">|</span>

        {/* Year range */}
        <div className="flex items-center gap-1">
          <label className="text-xs text-gray-500">Year</label>
          <input
            type="number"
            value={filters.year_min ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_min: e.target.value
                  ? Number(e.target.value)
                  : undefined,
              })
            }
            placeholder="min"
            className="w-16 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-gray-200 placeholder-gray-600"
          />
          <span className="text-xs text-gray-600">-</span>
          <input
            type="number"
            value={filters.year_max ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_max: e.target.value
                  ? Number(e.target.value)
                  : undefined,
              })
            }
            placeholder="max"
            className="w-16 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-gray-200 placeholder-gray-600"
          />
        </div>

        <span className="text-gray-700">|</span>

        {/* Top K */}
        <div className="flex items-center gap-1">
          <label className="text-xs text-gray-500">Top K</label>
          <input
            type="number"
            value={filters.top_k}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                top_k: Number(e.target.value) || 10,
              })
            }
            className="w-12 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-xs text-gray-200"
          />
        </div>

        <span className="text-gray-700">|</span>

        {/* Advanced Filters toggle */}
        <button
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          <span>{advancedOpen ? "▾" : "▸"}</span>
          <span>Filters</span>
          {advancedCount > 0 && (
            <span className="bg-blue-600 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full leading-none">
              {advancedCount}
            </span>
          )}
        </button>

        {/* Clear */}
        {showClear && onClear && (
          <>
            <span className="text-gray-700">|</span>
            <button
              onClick={onClear}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              Clear
            </button>
          </>
        )}
      </div>

      {/* Advanced Filters panel */}
      {advancedOpen && (
        <div className="px-4 pb-3 pt-1 border-t border-gray-800/50 grid grid-cols-2 gap-4">
          {/* Publication Types */}
          <div>
            <h4 className="text-[11px] font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Publication Type
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {PUBLICATION_TYPE_PRESETS.map((pt) => {
                const active = filters.publication_types?.includes(pt);
                return (
                  <button
                    key={pt}
                    onClick={() => toggleValue("publication_types", pt)}
                    className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                      active
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-400 hover:text-gray-200"
                    }`}
                  >
                    {pt}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Disease Area */}
          <div>
            <h4 className="text-[11px] font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Disease Area
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {DISEASE_AREA_PRESETS.map((da) => {
                const active = filters.mesh_categories?.includes(da);
                return (
                  <button
                    key={da}
                    onClick={() => toggleValue("mesh_categories", da)}
                    className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                      active
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-400 hover:text-gray-200"
                    }`}
                  >
                    {da}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/FilterPanel.tsx
git commit -m "feat: add collapsible Advanced Filters panel with publication type and disease area"
```

---

### Task 3: Wire filters through App.tsx API calls

**Files:**
- Modify: `frontend/src/App.tsx:31-34,112-119,170-176,267-273`

- [ ] **Step 1: Update initial `filters` state**

Edit `frontend/src/App.tsx`, update the `useState<Filters>` initializer (line 31-34):

```typescript
  const [filters, setFilters] = useState<Filters>({
    top_k: 10,
    search_mode: "dense",
    publication_types: [],
    mesh_categories: [],
  });
```

- [ ] **Step 2: Pass new filter fields in `askQueryStream` call**

In `handleSend`, update the `askQueryStream` `req` object (around line 112-119):

```typescript
          req: {
            query,
            year_min: filters.year_min,
            year_max: filters.year_max,
            top_k: filters.top_k,
            search_mode: filters.search_mode,
            publication_types: filters.publication_types,
            mesh_categories: filters.mesh_categories,
            stream: true,
          },
```

- [ ] **Step 3: Pass new filter fields in `searchQuery` call**

In `handleSend` search mode path (around line 170-176):

```typescript
        const res = await searchQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
          publication_types: filters.publication_types,
          mesh_categories: filters.mesh_categories,
        });
```

- [ ] **Step 4: Pass new filter fields in `reviewQuery` call**

In `handleReview` (around line 267-273):

```typescript
      const result = await reviewQuery({
        query,
        year_min: filters.year_min,
        year_max: filters.year_max,
        top_k: filters.top_k,
        search_mode: filters.search_mode,
        publication_types: filters.publication_types,
        mesh_categories: filters.mesh_categories,
      });
```

- [ ] **Step 5: Update `handleClear` to reset advanced filters**

In `handleClear` (around line 74), add `setFilters((prev) => ({ ...prev, publication_types: [], mesh_categories: [] }))` to reset advanced filters when clearing conversation. This matches the spec requirement and keeps UX consistent — Clear means full reset.

- [ ] **Step 6: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 7: Verify dev server starts**

Run: `cd frontend && npm run dev`
Expected: Compiles successfully. Open browser and verify:
1. FilterPanel shows existing controls (mode, search mode, year, top_k)
2. "Filters" toggle button appears
3. Clicking it reveals Publication Type and Disease Area checkbox groups
4. Selecting checkboxes shows a count badge
5. Checkbox selections persist across toggle open/close

- [ ] **Step 8: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire publication_types and mesh_categories through App.tsx API calls"
```
