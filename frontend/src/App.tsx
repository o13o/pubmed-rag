import { useState, useRef } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { AgentResultsPanel } from "./components/AgentResultsPanel";
import { analyzeQuery, askQueryStream, searchQuery } from "./lib/api";
import type {
  AgentResult,
  Citation,
  Filters,
  Message,
  Mode,
  SearchResult,
  SSEDoneEvent,
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
  const [agentResults, setAgentResults] = useState<AgentResult[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const handleSend = async (query: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setAgentResults([]);

    try {
      if (mode === "ask") {
        const assistantId = crypto.randomUUID();
        const assistantMsg: Message = {
          id: assistantId,
          role: "assistant",
          content: "",
        };
        setMessages((prev) => [...prev, assistantMsg]);

        abortRef.current = new AbortController();

        await askQueryStream(
          {
            query,
            year_min: filters.year_min,
            year_max: filters.year_max,
            top_k: filters.top_k,
            search_mode: filters.search_mode,
            stream: true,
          },
          // onToken
          (text: string) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + text }
                  : m,
              ),
            );
          },
          // onDone
          (data: SSEDoneEvent) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      citations: data.citations,
                      warnings: data.warnings,
                      disclaimer: data.disclaimer,
                    }
                  : m,
              ),
            );
            setCitations(data.citations);
            setLoading(false);
          },
          // onError
          (error: Error) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, role: "error", content: error.message }
                  : m,
              ),
            );
            setLoading(false);
          },
          abortRef.current.signal,
        );
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
      abortRef.current = null;
    }
  };

  const handleAnalyze = async () => {
    if (searchResults.length === 0 && citations.length === 0) return;
    setAnalyzing(true);
    setAgentResults([]);
    try {
      // In search mode, use searchResults directly.
      // In ask mode, searchResults may be empty — use citations as a fallback.
      // Note: citations lack abstract_text, so agent analysis will be limited
      // to title/metadata. For full analysis, use search mode first.
      const results: SearchResult[] =
        searchResults.length > 0
          ? searchResults
          : citations.map((c) => ({
              pmid: c.pmid,
              title: c.title,
              abstract_text: "",
              score: c.relevance_score,
              year: c.year,
              journal: c.journal,
              mesh_terms: [],
            }));
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      const query = lastUserMsg?.content ?? "";
      const res = await analyzeQuery({ query, results });
      setAgentResults(res.agent_results);
    } catch (err) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "error",
        content: err instanceof Error ? err.message : "Analysis failed",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setAnalyzing(false);
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
          {(searchResults.length > 0 || citations.length > 0) && (
            <button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {analyzing ? "Analyzing..." : "Analyze with Agents"}
            </button>
          )}
          <AgentResultsPanel agentResults={agentResults} loading={analyzing} />
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
