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
          <span>{advancedOpen ? "\u25BE" : "\u25B8"}</span>
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
