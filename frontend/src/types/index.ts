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
}

export interface SearchResult {
  pmid: string;
  title: string;
  abstract_text: string;
  score: number;
  year: number;
  journal: string;
  mesh_terms: string[];
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
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  citations?: Citation[];
  warnings?: Warning[];
  disclaimer?: string;
}
