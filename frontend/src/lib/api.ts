import type {
  AskRequest,
  AskResponse,
  SearchRequest,
  SearchResponse,
  SSEDoneEvent,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function askQuery(req: AskRequest): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function searchQuery(
  req: SearchRequest
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function askQueryStream(
  req: AskRequest & { stream: true },
  onToken: (text: string) => void,
  onDone: (data: SSEDoneEvent) => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!res.ok) {
    onError(new Error(`API error: ${res.status} ${res.statusText}`));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError(new Error("No response body"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          let data: unknown;
          try {
            data = JSON.parse(line.slice(6));
          } catch {
            onError(new Error("Failed to parse SSE data: " + line));
            return;
          }
          if (currentEvent === "token") {
            onToken((data as { text: string }).text);
          } else if (currentEvent === "done") {
            onDone(data as SSEDoneEvent);
          } else if (currentEvent === "error") {
            onError(new Error((data as { message: string }).message));
          }
          currentEvent = "";
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
