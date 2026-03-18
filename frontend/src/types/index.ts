export interface Citation {
  pmid: string;
  title: string;
  journal: string;
  year: number;
  relevance_score: number;
}

export interface Warning {
  check: string;
  severity: string;
  message: string;
  span: string;
}

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

export interface AskResponse {
  answer: string;
  citations: Citation[];
  query: string;
  warnings: Warning[];
  disclaimer: string;
  is_grounded: boolean;
}

export interface SearchRequest {
  query: string;
  year_min?: number;
  year_max?: number;
  top_k?: number;
  search_mode?: string;
  publication_types?: string[];
  mesh_categories?: string[];
}

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

export interface SearchResponse {
  results: SearchResult[];
  total: number;
}

export type Mode = "ask" | "search";

export interface Filters {
  year_min?: number;
  year_max?: number;
  top_k: number;
  search_mode: string;
  publication_types: string[];
  mesh_categories: string[];
}

export interface SSETokenEvent {
  text: string;
}

export interface SSEDoneEvent {
  citations: Citation[];
  warnings: Warning[];
  disclaimer: string;
  is_grounded: boolean;
}

export interface SSEErrorEvent {
  message: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  citations?: Citation[];
  warnings?: Warning[];
  disclaimer?: string;
}

export interface Finding {
  label: string;
  detail: string;
  severity: "info" | "warning" | "critical";
}

export interface AgentResult {
  agent_name: string;
  summary: string;
  findings: Finding[];
  confidence: number;
  score: number | null;
  details: Record<string, unknown> | null;
}

export interface AnalyzeRequest {
  query: string;
  results: SearchResult[];
  agents?: string[];
}

export interface AnalyzeResponse {
  query: string;
  agent_results: AgentResult[];
}

export interface TranscribeResponse {
  text: string;
  media_type: "audio" | "image" | "document";
}

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

export interface LiteratureReview {
  query: string;
  overview: string;
  main_findings: string;
  gaps_and_conflicts: string;
  recommendations: string;
  citations: Citation[];
  search_results: SearchResult[];
  agent_results: AgentResult[];
  agents_succeeded: number;
  agents_failed: number;
}
