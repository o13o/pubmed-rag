import type { AgentResult } from "../types";

interface Props {
  agentResults: AgentResult[];
  loading: boolean;
}

const SEVERITY_COLORS = {
  info: "text-blue-400",
  warning: "text-yellow-400",
  critical: "text-red-400",
};

const AGENT_LABELS: Record<string, string> = {
  retrieval: "Retrieval",
  methodology_critic: "Methodology Critic",
  statistical_reviewer: "Statistical Reviewer",
  clinical_applicability: "Clinical Applicability",
  summarization: "Summarization",
};

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 7
      ? "bg-emerald-600"
      : score >= 4
        ? "bg-yellow-600"
        : "bg-red-600";
  return (
    <span
      className={`${color} text-white text-xs font-bold px-2 py-0.5 rounded-full`}
    >
      {score}/10
    </span>
  );
}

export function AgentResultsPanel({ agentResults, loading }: Props) {
  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Agent Analysis
        </h3>
        <p className="text-xs text-gray-500 animate-pulse">Analyzing...</p>
      </div>
    );
  }

  if (agentResults.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Agent Analysis ({agentResults.length})
      </h3>
      <div className="space-y-3">
        {agentResults.map((r) => (
          <div
            key={r.agent_name}
            className="bg-gray-800 rounded p-3 border border-gray-700"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-gray-300">
                {AGENT_LABELS[r.agent_name] ?? r.agent_name}
              </span>
              {r.score !== null && <ScoreBadge score={r.score} />}
            </div>
            <p className="text-xs text-gray-400 mb-2">{r.summary}</p>
            {r.findings.length > 0 && (
              <div className="space-y-1">
                {r.findings.map((f, i) => (
                  <div key={i} className="flex gap-2 text-xs">
                    <span
                      className={
                        SEVERITY_COLORS[
                          f.severity as keyof typeof SEVERITY_COLORS
                        ] ?? "text-gray-400"
                      }
                    >
                      [{f.label}]
                    </span>
                    <span className="text-gray-500">{f.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
