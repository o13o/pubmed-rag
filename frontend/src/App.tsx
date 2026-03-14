import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { askQuery, searchQuery } from "./lib/api";
import type {
  Citation,
  Filters,
  Message,
  Mode,
  SearchResult,
} from "./types";

function App() {
  const [mode, setMode] = useState<Mode>("ask");
  const [filters, setFilters] = useState<Filters>({
    top_k: 10,
    search_mode: "dense",
  });
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (query: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      if (mode === "ask") {
        const res = await askQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
        });
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: res.answer,
          citations: res.citations,
          warnings: res.warnings,
          disclaimer: res.disclaimer,
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setCitations(res.citations);
      } else {
        const res = await searchQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
        });
        const infoMsg: Message = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Found ${res.total} results.`,
        };
        setMessages((prev) => [...prev, infoMsg]);
        setSearchResults(res.results);
      }
    } catch (err) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "error",
        content:
          err instanceof Error ? err.message : "Backend unavailable",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <h1 className="text-lg font-bold tracking-tight">
          <span className="text-blue-400">PubMed</span> RAG
        </h1>
        <span className="text-xs text-gray-600">
          {mode === "ask" ? "Ask Mode" : "Search Mode"} ·{" "}
          {filters.search_mode}
        </span>
      </header>
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 flex flex-col min-w-0">
          <ChatPanel
            messages={messages}
            loading={loading}
            onSend={handleSend}
          />
        </main>
        <aside className="w-80 border-l border-gray-800 p-4 overflow-y-auto space-y-4">
          <FilterPanel
            mode={mode}
            filters={filters}
            onModeChange={setMode}
            onFiltersChange={setFilters}
          />
          <ResultsPanel
            citations={citations}
            searchResults={searchResults}
            mode={mode}
          />
        </aside>
      </div>
    </div>
  );
}

export default App;
