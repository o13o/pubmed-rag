const STAGE_CONFIG: Record<string, { label: string; icon: string }> = {
  expanding: { label: "Expanding query with MeSH terms", icon: "search" },
  searching: { label: "Searching PubMed abstracts", icon: "search" },
  reranking: { label: "Reranking results", icon: "filter" },
  generating: { label: "Generating answer", icon: "sparkle" },
  validating: { label: "Validating response", icon: "shield" },
};

function PulsingDots() {
  return (
    <span className="inline-flex items-center gap-1 ml-1">
      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse-dot" />
      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse-dot-d1" />
      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse-dot-d2" />
    </span>
  );
}

function StageIcon({ type }: { type: string }) {
  if (type === "search") {
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    );
  }
  if (type === "filter") {
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
      </svg>
    );
  }
  if (type === "sparkle") {
    return (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
      </svg>
    );
  }
  // shield
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  );
}

interface Props {
  stage: string | null;
}

export function StreamingIndicator({ stage }: Props) {
  const config = stage ? STAGE_CONFIG[stage] : null;
  const label = config?.label ?? "Processing";
  const icon = config?.icon ?? "search";

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-gray-800 rounded-lg px-4 py-3 max-w-[80%] min-w-[260px]">
        {/* Skeleton lines */}
        <div className="space-y-2.5 mb-3">
          <div className="h-3 bg-gray-700 rounded-sm animate-shimmer w-[85%]" />
          <div className="h-3 bg-gray-700 rounded-sm animate-shimmer-d1 w-[70%]" />
          <div className="h-3 bg-gray-700 rounded-sm animate-shimmer-d2 w-[55%]" />
        </div>
        {/* Stage label */}
        <div className="flex items-center gap-2 text-sm text-blue-400 pt-1 border-t border-gray-700">
          <span className="animate-pulse">
            <StageIcon type={icon} />
          </span>
          <span>{label}</span>
          <PulsingDots />
        </div>
      </div>
    </div>
  );
}
