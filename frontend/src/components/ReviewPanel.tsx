import { useState } from "react";
import type { LiteratureReview } from "../types";

interface Props {
  review: LiteratureReview | null;
  loading: boolean;
}

function Section({ title, content }: { title: string; content: string }) {
  return (
    <div className="mb-4">
      <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
        {title}
      </h4>
      <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
        {content}
      </p>
    </div>
  );
}

export function ReviewPanel({ review, loading }: Props) {
  const [showAgents, setShowAgents] = useState(false);

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Literature Review
        </h3>
        <p className="text-xs text-gray-500 animate-pulse">
          Generating literature review... This may take a minute.
        </p>
      </div>
    );
  }

  if (!review) return null;

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Literature Review
      </h3>

      <Section title="Overview" content={review.overview} />
      <Section title="Main Findings" content={review.main_findings} />
      <Section title="Gaps & Conflicts" content={review.gaps_and_conflicts} />
      <Section title="Recommendations" content={review.recommendations} />

      <div className="border-t border-gray-700 pt-3 mt-3">
        <p className="text-xs text-gray-500">
          {review.citations.length} citations &middot;{" "}
          {review.agents_succeeded} agents succeeded
          {review.agents_failed > 0 && (
            <span className="text-yellow-500">
              {" "}&middot; {review.agents_failed} failed
            </span>
          )}
        </p>
        <button
          onClick={() => setShowAgents(!showAgents)}
          className="text-xs text-blue-400 hover:text-blue-300 mt-1"
        >
          {showAgents ? "Hide" : "Show"} agent details
        </button>
        {showAgents && (
          <div className="mt-2 space-y-2">
            {review.agent_results.map((a) => (
              <div
                key={a.agent_name}
                className="bg-gray-800 rounded p-2 border border-gray-700"
              >
                <span className="text-xs font-semibold text-gray-300">
                  {a.agent_name}
                </span>
                <span className="text-xs text-gray-500 ml-2">
                  confidence: {a.confidence.toFixed(2)}
                </span>
                <p className="text-xs text-gray-400 mt-1">{a.summary}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
