import type { Filters, Mode } from "../types";

interface Props {
  mode: Mode;
  filters: Filters;
  onModeChange: (mode: Mode) => void;
  onFiltersChange: (filters: Filters) => void;
}

export function FilterPanel({
  mode,
  filters,
  onModeChange,
  onFiltersChange,
}: Props) {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Mode
      </h3>
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => onModeChange("ask")}
          className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === "ask"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Ask
        </button>
        <button
          onClick={() => onModeChange("search")}
          className={`flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            mode === "search"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
        >
          Search
        </button>
      </div>

      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Filters
      </h3>
      <div className="space-y-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Year Min</label>
          <input
            type="number"
            value={filters.year_min ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_min: e.target.value ? Number(e.target.value) : undefined,
              })
            }
            placeholder="e.g. 2020"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 placeholder-gray-600"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Year Max</label>
          <input
            type="number"
            value={filters.year_max ?? ""}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                year_max: e.target.value ? Number(e.target.value) : undefined,
              })
            }
            placeholder="e.g. 2024"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 placeholder-gray-600"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Top K</label>
          <input
            type="number"
            value={filters.top_k}
            onChange={(e) =>
              onFiltersChange({
                ...filters,
                top_k: Number(e.target.value) || 10,
              })
            }
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Search Mode
          </label>
          <div className="flex gap-2">
            <button
              onClick={() =>
                onFiltersChange({ ...filters, search_mode: "dense" })
              }
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
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
              className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                filters.search_mode === "hybrid"
                  ? "bg-emerald-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              Hybrid
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
