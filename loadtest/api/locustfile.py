"""Locust load test for PubMed RAG API.

Usage:
    cd loadtest/api
    pip install locust
    locust                          # Web UI at http://localhost:8089
    locust --headless -u 10 -r 2 -t 60s  # Headless: 10 users, ramp 2/s, 60s
"""

import random

from locust import HttpUser, between, task

SAMPLE_QUERIES = [
    "What are the latest treatments for early-stage pancreatic cancer?",
    "Non-invasive therapy for knee arthritis",
    "mRNA vaccine efficacy studies published after 2022",
    "CRISPR gene therapy for sickle cell disease",
    "Machine learning applications in radiology diagnosis",
    "Immunotherapy checkpoint inhibitors for melanoma",
    "Gut microbiome influence on mental health",
    "Antibiotic resistance mechanisms in hospital-acquired infections",
    "Stem cell therapy for spinal cord injury",
    "Long COVID neurological symptoms and treatment",
]

SEARCH_MODES = ["dense", "hybrid"]


class PubMedRAGUser(HttpUser):
    """Simulates a researcher interacting with the PubMed RAG system."""

    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def _random_query(self) -> str:
        return random.choice(SAMPLE_QUERIES)

    @task(5)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(10)
    def search(self):
        query = self._random_query()
        self.client.post(
            "/search",
            json={
                "query": query,
                "top_k": 5,
                "search_mode": random.choice(SEARCH_MODES),
            },
            name="/search",
        )

    @task(3)
    def search_with_year_filter(self):
        query = self._random_query()
        self.client.post(
            "/search",
            json={
                "query": query,
                "top_k": 5,
                "year_min": random.randint(2018, 2024),
                "search_mode": "dense",
            },
            name="/search [year filter]",
        )

    @task(2)
    def ask(self):
        query = self._random_query()
        self.client.post(
            "/ask",
            json={
                "query": query,
                "top_k": 5,
                "search_mode": "dense",
                "guardrails_enabled": True,
            },
            name="/ask",
            timeout=60,
        )

    @task(1)
    def analyze(self):
        """Search first, then send results to /analyze."""
        query = self._random_query()
        search_resp = self.client.post(
            "/search",
            json={
                "query": query,
                "top_k": 5,
                "search_mode": "dense",
            },
            name="/analyze [prefetch]",
        )
        if search_resp.status_code != 200:
            return

        results = search_resp.json().get("results", [])
        if not results:
            return

        self.client.post(
            "/analyze",
            json={
                "query": query,
                "results": results,
                "agents": ["methodology_critic", "summarization"],
            },
            name="/analyze",
            timeout=120,
        )
