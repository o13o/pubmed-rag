"""Run RAG evaluation via API and collect metrics.

Usage:
    cd capstone/backend
    .venv/bin/python scripts/run_evaluation.py

Requires: API server running at localhost:8000, OPENAI_API_KEY set.
Outputs: JSON results to stdout, summary table to stderr.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
DATASET_PATH = Path(__file__).parent.parent / "tests" / "eval" / "dataset.json"
TIMEOUT = 120.0


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def run_query(client: httpx.Client, query: str, top_k: int = 10) -> dict:
    """Call /ask and /search, return combined result with timing."""
    # /ask for RAG answer
    start = time.time()
    ask_resp = client.post(
        f"{API_BASE}/ask",
        json={
            "query": query,
            "top_k": top_k,
            "search_mode": "hybrid",
            "guardrails_enabled": True,
            "stream": False,
        },
        timeout=TIMEOUT,
    )
    ask_latency = time.time() - start
    ask_resp.raise_for_status()
    ask_data = ask_resp.json()

    # /search for retrieval context (used by DeepEval metrics)
    start = time.time()
    search_resp = client.post(
        f"{API_BASE}/search",
        json={"query": query, "top_k": top_k, "search_mode": "hybrid"},
        timeout=TIMEOUT,
    )
    search_latency = time.time() - start
    search_resp.raise_for_status()
    search_data = search_resp.json()

    contexts = [
        f"PMID: {r['pmid']}\n{r['title']}\n{r.get('abstract_text', '')}"
        for r in search_data.get("results", [])
    ]

    return {
        "query": query,
        "answer": ask_data.get("answer", ""),
        "citations": ask_data.get("citations", []),
        "warnings": ask_data.get("warnings", []),
        "disclaimer": ask_data.get("disclaimer", ""),
        "is_grounded": ask_data.get("is_grounded", None),
        "contexts": contexts,
        "ask_latency_s": round(ask_latency, 2),
        "search_latency_s": round(search_latency, 2),
    }


def evaluate_metrics(result: dict) -> dict:
    """Run DeepEval metrics on a single result."""
    test_case = LLMTestCase(
        input=result["query"],
        actual_output=result["answer"],
        retrieval_context=result["contexts"],
    )

    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        "contextual_precision": ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini"),
    }

    scores = {}
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores[name] = round(metric.score, 3)
        except Exception as e:
            scores[name] = f"error: {e}"

    # Custom metrics (simple checks)
    import re
    pmid_count = len(re.findall(r"PMID[:\s]+\d+", result["answer"]))
    scores["citation_presence"] = 1.0 if pmid_count > 0 else 0.0

    disclaimer_keywords = ["not medical advice", "consult", "healthcare professional", "disclaimer"]
    has_disclaimer = any(kw.lower() in result.get("disclaimer", "").lower() for kw in disclaimer_keywords)
    scores["medical_disclaimer"] = 1.0 if has_disclaimer else 0.0

    return scores


def main():
    dataset = load_dataset()
    client = httpx.Client()

    # Health check
    try:
        health = client.get(f"{API_BASE}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        print(f"API not reachable at {API_BASE}: {e}", file=sys.stderr)
        sys.exit(1)

    all_results = []

    for i, case in enumerate(dataset):
        query = case["query"]
        print(f"[{i+1}/{len(dataset)}] {query[:60]}...", file=sys.stderr)

        try:
            result = run_query(client, query)
            scores = evaluate_metrics(result)
            result["scores"] = scores
            all_results.append(result)

            print(
                f"  ask={result['ask_latency_s']}s  "
                f"faith={scores.get('faithfulness', '?')}  "
                f"relevancy={scores.get('answer_relevancy', '?')}  "
                f"precision={scores.get('contextual_precision', '?')}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            all_results.append({"query": query, "error": str(e)})

    # Output full results as JSON
    json.dump(all_results, sys.stdout, indent=2, ensure_ascii=False)
    print(file=sys.stdout)

    # Summary to stderr
    print("\n" + "=" * 70, file=sys.stderr)
    print("EVALUATION SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    successful = [r for r in all_results if "scores" in r]
    if not successful:
        print("No successful evaluations.", file=sys.stderr)
        return

    metric_names = ["faithfulness", "answer_relevancy", "contextual_precision",
                    "citation_presence", "medical_disclaimer"]

    for metric in metric_names:
        values = [r["scores"][metric] for r in successful if isinstance(r["scores"].get(metric), (int, float))]
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric:25s}  avg={avg:.3f}  min={min(values):.3f}  max={max(values):.3f}  n={len(values)}", file=sys.stderr)

    latencies = [r["ask_latency_s"] for r in successful]
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    print(f"\n  ask_latency (s)           avg={sum(latencies)/len(latencies):.2f}  p50={p50:.2f}  p99={p99:.2f}", file=sys.stderr)

    search_latencies = [r["search_latency_s"] for r in successful]
    sorted_sl = sorted(search_latencies)
    sp50 = sorted_sl[len(sorted_sl) // 2]
    print(f"  search_latency (s)        avg={sum(search_latencies)/len(search_latencies):.2f}  p50={sp50:.2f}", file=sys.stderr)

    print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()
