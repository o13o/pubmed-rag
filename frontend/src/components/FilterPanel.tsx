import type { Filters, Mode } from "../types";

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
  return (
    <div className="flex items-center gap-3 flex-wrap px-4 py-2 border-t border-gray-800 bg-gray-950">
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
              year_min: e.target.value ? Number(e.target.value) : undefined,
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
              year_max: e.target.value ? Number(e.target.value) : undefined,
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
  );
}
