import type { Citation, SearchResult } from "../types";

interface Props {
  citations: Citation[];
  searchResults: SearchResult[];
  mode: "ask" | "search";
}

export function ResultsPanel({ citations, searchResults, mode }: Props) {
  if (mode === "ask") {
    if (citations.length === 0) {
      return (
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Citations
          </h3>
          <p className="text-xs text-gray-600">
            Ask a question to see citations.
          </p>
        </div>
      );
    }

    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Citations ({citations.length})
        </h3>
        <div className="space-y-2">
          {citations.map((c) => (
            <div
              key={c.pmid}
              className="bg-gray-800 rounded p-2 border border-gray-700"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono text-blue-400">
                  PMID: {c.pmid}
                </span>
                <span className="text-xs font-mono text-emerald-400">
                  {c.relevance_score.toFixed(3)}
                </span>
              </div>
              <div className="text-xs text-gray-300 leading-snug">
                {c.title}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {c.journal} ({c.year})
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Search mode
  if (searchResults.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Search Results
        </h3>
        <p className="text-xs text-gray-600">Search to see results.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Search Results ({searchResults.length})
      </h3>
      <div className="space-y-2">
        {searchResults.map((r) => (
          <div
            key={r.pmid}
            className="bg-gray-800 rounded p-2 border border-gray-700"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-blue-400">
                PMID: {r.pmid}
              </span>
              <span className="text-xs font-mono text-emerald-400">
                {r.score.toFixed(4)}
              </span>
            </div>
            <div className="text-xs text-gray-300 leading-snug">{r.title}</div>
            <div className="text-xs text-gray-500 mt-1">
              {r.journal} ({r.year})
            </div>
            <div className="text-xs text-gray-400 mt-1 line-clamp-3">
              {r.abstract_text}
            </div>
            {r.mesh_terms.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {r.mesh_terms.slice(0, 5).map((term) => (
                  <span
                    key={term}
                    className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded"
                  >
                    {term}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
