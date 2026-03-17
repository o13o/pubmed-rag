"""Locust load test for Milvus search performance (bypasses API/LLM layer).

Usage:
    cd loadtest/milvus
    pip install locust numpy pymilvus
    locust                                # Web UI at http://localhost:8089
    locust --headless -u 10 -r 2 -t 60s  # Headless: 10 users, ramp 2/s, 60s

Environment variables:
    MILVUS_HOST       (default: localhost)
    MILVUS_PORT       (default: 19530)
    MILVUS_COLLECTION (default: pubmed_abstracts)
    TOP_K             (default: 10)
"""

import os
import random
import time

import numpy as np
from locust import User, between, events, task
from pymilvus import AnnSearchRequest, Collection, RRFRanker, connections

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "pubmed_abstracts")
TOP_K = int(os.getenv("TOP_K", "10"))
EMBEDDING_DIM = 1536

OUTPUT_FIELDS = ["pmid", "title", "year", "journal"]

SAMPLE_QUERIES = [
    "treatments for early-stage pancreatic cancer",
    "non-invasive therapy for knee arthritis",
    "mRNA vaccine efficacy studies",
    "CRISPR gene therapy sickle cell disease",
    "machine learning radiology diagnosis",
    "immunotherapy checkpoint inhibitors melanoma",
    "gut microbiome mental health",
    "antibiotic resistance hospital infections",
    "stem cell therapy spinal cord injury",
    "long COVID neurological symptoms treatment",
]


def _random_vector() -> list[float]:
    """Generate a random normalized 1536-dim vector."""
    vec = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _random_year_filter() -> str:
    year_min = random.randint(2015, 2022)
    year_max = year_min + random.randint(1, 3)
    return f"year >= {year_min} and year <= {year_max}"


def _fire_request(environment, request_type, name, start_time, exc=None):
    """Report a request to Locust's event system."""
    elapsed_ms = (time.time() - start_time) * 1000
    if exc:
        environment.events.request.fire(
            request_type=request_type,
            name=name,
            response_time=elapsed_ms,
            response_length=0,
            exception=exc,
        )
    else:
        environment.events.request.fire(
            request_type=request_type,
            name=name,
            response_time=elapsed_ms,
            response_length=0,
            exception=None,
        )


# Global connection — shared across all Locust users in the same worker process
_collection = None


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Connect to Milvus once when the test starts."""
    global _collection
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    _collection = Collection(MILVUS_COLLECTION)
    _collection.load()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Disconnect from Milvus when the test ends."""
    connections.disconnect("default")


class MilvusSearchUser(User):
    """Simulates concurrent search load against Milvus directly."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def dense_search(self):
        start = time.time()
        try:
            _collection.search(
                data=[_random_vector()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=TOP_K,
                output_fields=OUTPUT_FIELDS,
            )
            _fire_request(self.environment, "milvus", "dense_search", start)
        except Exception as e:
            _fire_request(self.environment, "milvus", "dense_search", start, exc=e)

    @task(5)
    def dense_search_filtered(self):
        start = time.time()
        try:
            _collection.search(
                data=[_random_vector()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=TOP_K,
                expr=_random_year_filter(),
                output_fields=OUTPUT_FIELDS,
            )
            _fire_request(self.environment, "milvus", "dense_search [filtered]", start)
        except Exception as e:
            _fire_request(self.environment, "milvus", "dense_search [filtered]", start, exc=e)

    @task(5)
    def hybrid_search(self):
        start = time.time()
        try:
            dense_req = AnnSearchRequest(
                data=[_random_vector()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=TOP_K,
            )
            sparse_req = AnnSearchRequest(
                data=[random.choice(SAMPLE_QUERIES)],
                anns_field="chunk_text_sparse",
                param={"metric_type": "BM25"},
                limit=TOP_K,
            )
            _collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=RRFRanker(k=60),
                limit=TOP_K,
                output_fields=OUTPUT_FIELDS,
            )
            _fire_request(self.environment, "milvus", "hybrid_search", start)
        except Exception as e:
            _fire_request(self.environment, "milvus", "hybrid_search", start, exc=e)

    @task(3)
    def hybrid_search_filtered(self):
        start = time.time()
        try:
            expr = _random_year_filter()
            dense_req = AnnSearchRequest(
                data=[_random_vector()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=TOP_K,
                expr=expr,
            )
            sparse_req = AnnSearchRequest(
                data=[random.choice(SAMPLE_QUERIES)],
                anns_field="chunk_text_sparse",
                param={"metric_type": "BM25"},
                limit=TOP_K,
                expr=expr,
            )
            _collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=RRFRanker(k=60),
                limit=TOP_K,
                output_fields=OUTPUT_FIELDS,
            )
            _fire_request(self.environment, "milvus", "hybrid_search [filtered]", start)
        except Exception as e:
            _fire_request(self.environment, "milvus", "hybrid_search [filtered]", start, exc=e)
