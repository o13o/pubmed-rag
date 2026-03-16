import { useState, useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import { transcribeFile } from "../lib/api";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  loading: boolean;
  onSend: (query: string) => void;
}

export function ChatPanel({ messages, loading, onSend }: Props) {
  const [input, setInput] = useState("");
  const [transcribing, setTranscribing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;
    onSend(query);
    setInput("");
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Reset so the same file can be re-selected
    e.target.value = "";

    setTranscribing(true);
    try {
      const result = await transcribeFile(file);
      setInput(result.text);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Transcription failed";
      alert(msg);
    } finally {
      setTranscribing(false);
    }
  };

  const isDisabled = loading || transcribing;

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 pt-4 pb-2 space-y-2">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-600">
            <p>Ask a question about medical research.</p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {loading && messages.length > 0 && messages[messages.length - 1].role !== "assistant" && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-800 text-gray-400 rounded-lg px-4 py-2">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form
        onSubmit={handleSubmit}
        className="border-t border-gray-800 p-4 flex gap-2"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isDisabled}
          title="Upload audio or image"
          className="bg-gray-800 hover:bg-gray-700 disabled:bg-gray-800 disabled:text-gray-600 text-gray-400 hover:text-gray-200 px-3 py-2 rounded-lg transition-colors"
        >
          {transcribing ? (
            <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
          )}
        </button>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={transcribing ? "Transcribing..." : "Ask a question..."}
          disabled={isDisabled}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={isDisabled || !input.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-6 py-2 rounded-lg font-medium transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}
