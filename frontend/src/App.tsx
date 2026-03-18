import { useState, useRef, useCallback, useEffect } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { FilterPanel } from "./components/FilterPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { AgentResultsPanel } from "./components/AgentResultsPanel";
import { ReviewPanel } from "./components/ReviewPanel";
import { analyzeQuery, askQueryStream, reviewQuery, searchQuery } from "./lib/api";
import type {
  AgentResult,
  Citation,
  Filters,
  LiteratureReview,
  Message,
  Mode,
  SearchResult,
} from "./types";

const AGENT_NAMES = [
  "retrieval",
  "methodology_critic",
  "statistical_reviewer",
  "clinical_applicability",
  "summarization",
  "conflicting_findings",
  "trend_analysis",
  "knowledge_graph",
];

type SidebarTab = "citations" | "analyze" | "review";

function App() {
  const [mode, setMode] = useState<Mode>("ask");
  const [activeTab, setActiveTab] = useState<SidebarTab>("citations");
  const [filters, setFilters] = useState<Filters>({
    top_k: 10,
    search_mode: "dense",
    publication_types: [],
    mesh_categories: [],
  });
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [streamStage, setStreamStage] = useState<string | null>(null);
  const [agentResults, setAgentResults] = useState<AgentResult[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [reviewResult, setReviewResult] = useState<LiteratureReview | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(600);
  const dragging = useRef(false);

  const handleMouseDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const newWidth = window.innerWidth - e.clientX;
      setSidebarWidth(Math.max(200, Math.min(600, newWidth)));
    };
    const handleMouseUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  const handleClear = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setCitations([]);
    setSearchResults([]);
    setAgentResults([]);
    setLoading(false);
    setStreamStage(null);
    setAnalyzing(false);
    setReviewResult(null);
    setReviewing(false);
    setFilters((prev) => ({ ...prev, publication_types: [], mesh_categories: [] }));
  };

  const handleSend = async (query: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setStreamStage(null);
    setAgentResults([]);
    setReviewResult(null);

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

        await askQueryStream({
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
          onToken: (text) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + text }
                  : m,
              ),
            );
          },
          onDone: (data) => {
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
            setStreamStage(null);
          },
          onError: (error) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, role: "error", content: error.message }
                  : m,
              ),
            );
            setLoading(false);
            setStreamStage(null);
          },
          signal: abortRef.current.signal,
          onCitations: ({ citations: earlyCitations, search_results }) => {
            setCitations(earlyCitations);
            if (search_results) {
              setSearchResults(search_results);
            }
          },
          onStatus: (stage) => {
            setStreamStage(stage);
          },
        });
      } else {
        const res = await searchQuery({
          query,
          year_min: filters.year_min,
          year_max: filters.year_max,
          top_k: filters.top_k,
          search_mode: filters.search_mode,
          publication_types: filters.publication_types,
          mesh_categories: filters.mesh_categories,
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
      setStreamStage(null);
      abortRef.current = null;
    }
  };

  const handleAnalyze = async () => {
    if (searchResults.length === 0 && citations.length === 0) return;
    setAnalyzing(true);
    setAgentResults([]);

    try {
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
              publication_types: [],
            }));
      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      const query = lastUserMsg?.content ?? "";

      // Run agents in batches of 3 to avoid API rate limits
      const BATCH_SIZE = 3;
      for (let i = 0; i < AGENT_NAMES.length; i += BATCH_SIZE) {
        const batch = AGENT_NAMES.slice(i, i + BATCH_SIZE);
        const promises = batch.map(async (agentName) => {
          try {
            const res = await analyzeQuery({ query, results, agents: [agentName] });
            setAgentResults((prev) => [...prev, ...res.agent_results]);
          } catch (err) {
            console.error(`Agent ${agentName} failed:`, err);
            setAgentResults((prev) => [
              ...prev,
              {
                agent_name: agentName,
                summary: `Failed: ${err instanceof Error ? err.message : "Unknown error"}`,
                findings: [],
                confidence: 0,
                score: null,
                details: null,
              },
            ]);
          }
        });
        await Promise.all(promises);
      }
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

  const handleReview = async () => {
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    const query = lastUserMsg?.content ?? "";
    if (!query) return;

    setReviewing(true);
    setReviewResult(null);

    try {
      const result = await reviewQuery({
        query,
        year_min: filters.year_min,
        year_max: filters.year_max,
        top_k: filters.top_k,
        search_mode: filters.search_mode,
        publication_types: filters.publication_types,
        mesh_categories: filters.mesh_categories,
      });
      setReviewResult(result);
    } catch (err) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "error",
        content: err instanceof Error ? err.message : "Review generation failed",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setReviewing(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-2 flex-shrink-0">
        <h1
          onClick={handleClear}
          className="text-lg font-bold tracking-tight cursor-pointer"
        >
          <span className="text-blue-400">PubMed</span> RAG
        </h1>
      </header>
      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 flex flex-col min-w-0">
          <ChatPanel
            messages={messages}
            loading={loading}
            streamStage={streamStage}
            onSend={handleSend}
          />
          <FilterPanel
            mode={mode}
            filters={filters}
            onModeChange={setMode}
            onFiltersChange={setFilters}
            onClear={handleClear}
            showClear={messages.length > 0}
          />
        </main>
        <div
          onMouseDown={handleMouseDown}
          className="w-1 cursor-col-resize bg-gray-800 hover:bg-blue-500 transition-colors flex-shrink-0"
        />
        <aside
          style={{ width: sidebarWidth }}
          className="flex-shrink-0 flex flex-col overflow-hidden"
        >
          <div className="flex border-b border-gray-800 flex-shrink-0">
            {([
              ["citations", "Citations"],
              ["analyze", "Analyze"],
              ["review", "Review"],
            ] as const).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`flex-1 px-2 py-2 text-xs font-medium transition-colors ${
                  activeTab === key
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4 scroll-pt-4">
            {activeTab === "citations" && (
              <ResultsPanel
                citations={citations}
                searchResults={searchResults}
                mode={mode}
              />
            )}
            {activeTab === "analyze" && (
              <>
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing || (searchResults.length === 0 && citations.length === 0)}
                  className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                >
                  {analyzing ? "Analyzing..." : "Analyze with Agents"}
                </button>
                {searchResults.length === 0 && citations.length === 0 && !analyzing && agentResults.length === 0 && (
                  <p className="text-xs text-gray-600 text-center">
                    Run a query first to enable agent analysis.
                  </p>
                )}
                <AgentResultsPanel agentResults={agentResults} loading={analyzing} totalAgents={AGENT_NAMES.length} />
              </>
            )}
            {activeTab === "review" && (
              <>
                <button
                  onClick={handleReview}
                  disabled={reviewing || (searchResults.length === 0 && citations.length === 0)}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                >
                  {reviewing ? "Generating Review..." : "Generate Literature Review"}
                </button>
                {searchResults.length === 0 && citations.length === 0 && !reviewing && !reviewResult && (
                  <p className="text-xs text-gray-600 text-center">
                    Run a query first to generate a literature review.
                  </p>
                )}
                <ReviewPanel review={reviewResult} loading={reviewing} />
              </>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
