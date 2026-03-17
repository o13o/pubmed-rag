# Phase A-2: Ingestion Pipeline

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingestion pipeline that reads PubMed JSONL, chunks documents, generates embeddings, and upserts into Milvus.

**Architecture:** Three components in `ingestion/`: loader (JSONL parsing), chunker (text formatting), embedder (OpenAI API + Milvus upsert). Each takes and returns Pydantic models from `shared/models.py`.

**Tech Stack:** Python 3.11+, openai (embedding API), pymilvus, pydantic

**Spec:** [2026-03-14-pubmed-rag-system-design.md](../specs/2026-03-14-pubmed-rag-system-design.md) - Section 4.1, Section 5

**Dependency:** This plan uses `Article` and `Chunk` models from A-3 (`shared/models.py`). If running in parallel with A-3, define minimal stub types in tests and integrate after merge.

---

## Chunk 1: Loader and Chunker

### Task 1: JSONL Loader

**Files:**
- Create: `backend/src/ingestion/loader.py`
- Create: `backend/tests/unit/test_loader.py`

**JSONL input format** (from playground pipeline `sampled.jsonl`):
```json
{"pmid": "12345678", "title": "Article title", "abstract": "Abstract text", "authors": ["First Last"], "publication_date": "2023", "mesh_terms": ["Neoplasms"], "keywords": ["keyword1"], "publication_types": ["Journal Article"], "language": "eng", "journal": "Journal of Medicine"}
```

- [ ] **Step 1: Write failing tests for loader**

```python
# tests/unit/test_loader.py
"""Tests for JSONL loader."""

import json
import tempfile
from pathlib import Path

from src.ingestion.loader import load_articles


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_single_record():
    record = {
        "pmid": "111",
        "title": "Test Title",
        "abstract": "Test abstract text.",
        "authors": ["John Doe"],
        "publication_date": "2023",
        "mesh_terms": ["Neoplasms"],
        "keywords": ["cancer"],
        "publication_types": ["Journal Article"],
        "language": "eng",
        "journal": "Test Journal",
    }
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write(json.dumps(record) + "\n")
        path = Path(f.name)

    articles = load_articles(path)
    assert len(articles) == 1
    a = articles[0]
    assert a.pmid == "111"
    assert a.title == "Test Title"
    assert a.abstract == "Test abstract text."
    assert a.year == 2023
    assert a.mesh_terms == ["Neoplasms"]
    assert a.keywords == ["cancer"]
    assert a.publication_types == ["Journal Article"]


def test_load_skips_records_without_abstract():
    records = [
        {"pmid": "1", "title": "T", "abstract": "Has abstract", "authors": [],
         "publication_date": "2023", "mesh_terms": [], "keywords": [],
         "publication_types": [], "language": "eng", "journal": "J"},
        {"pmid": "2", "title": "T", "abstract": "", "authors": [],
         "publication_date": "2023", "mesh_terms": [], "keywords": [],
         "publication_types": [], "language": "eng", "journal": "J"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = Path(f.name)

    articles = load_articles(path)
    assert len(articles) == 1
    assert articles[0].pmid == "1"


def test_load_extracts_year_from_publication_date():
    record = {
        "pmid": "1", "title": "T", "abstract": "A", "authors": [],
        "publication_date": "2024-03-15", "mesh_terms": [], "keywords": [],
        "publication_types": [], "language": "eng", "journal": "J",
    }
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write(json.dumps(record) + "\n")
        path = Path(f.name)

    articles = load_articles(path)
    assert articles[0].year == 2024
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_loader.py -v
```

Expected: FAIL - `ModuleNotFoundError: No module named 'src.ingestion.loader'`

- [ ] **Step 3: Implement loader**

```python
# src/ingestion/loader.py
"""Load PubMed JSONL into Article models."""

import json
import logging
from pathlib import Path

from src.shared.models import Article

logger = logging.getLogger(__name__)


def _extract_year(publication_date: str) -> int:
    """Extract year from publication_date string (e.g., '2023' or '2023-03-15')."""
    if len(publication_date) >= 4:
        return int(publication_date[:4])
    raise ValueError(f"Cannot extract year from: {publication_date}")


def load_articles(path: Path) -> list[Article]:
    """Load articles from a JSONL file. Skips records without abstracts."""
    articles = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            if not raw.get("abstract", "").strip():
                logger.debug("Skipping PMID %s: no abstract", raw.get("pmid"))
                continue

            try:
                article = Article(
                    pmid=raw["pmid"],
                    title=raw["title"],
                    abstract=raw["abstract"],
                    authors=raw.get("authors", []),
                    year=_extract_year(raw.get("publication_date", "")),
                    journal=raw.get("journal", ""),
                    mesh_terms=raw.get("mesh_terms", []),
                    keywords=raw.get("keywords", []),
                    publication_types=raw.get("publication_types", []),
                )
                articles.append(article)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping line %d: %s", line_num, e)

    logger.info("Loaded %d articles from %s", len(articles), path)
    return articles
```

Note: This depends on `src.shared.models.Article` from A-3. If A-3 is not yet merged, create a minimal stub:

```python
# src/shared/models.py (minimal stub for A-2 development)
from pydantic import BaseModel


class Article(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str] = []
    year: int
    journal: str = ""
    mesh_terms: list[str] = []
    keywords: list[str] = []
    publication_types: list[str] = []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_loader.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/ingestion/loader.py backend/tests/unit/test_loader.py
git commit -m "feat(ingestion): add JSONL loader with Article model parsing"
```

---

### Task 2: Chunker

**Files:**
- Create: `backend/src/ingestion/chunker.py`
- Create: `backend/tests/unit/test_chunker.py`

Per ADR-0001: 1 abstract = 1 chunk. Format: `"Title: {title}\nAbstract: {abstract}\nMeSH: {term1}; {term2}; ..."`

- [ ] **Step 1: Write failing tests for chunker**

```python
# tests/unit/test_chunker.py
"""Tests for chunker (1 abstract = 1 chunk per ADR-0001)."""

from src.shared.models import Article
from src.ingestion.chunker import chunk_article


def _make_article(**overrides) -> Article:
    defaults = {
        "pmid": "123",
        "title": "Test Title",
        "abstract": "Test abstract.",
        "authors": ["John Doe"],
        "year": 2023,
        "journal": "Test Journal",
        "mesh_terms": ["Neoplasms", "Cardiovascular Diseases"],
        "keywords": ["cancer"],
        "publication_types": ["Journal Article"],
    }
    defaults.update(overrides)
    return Article(**defaults)


def test_chunk_produces_correct_text_format():
    article = _make_article()
    chunk = chunk_article(article)
    assert chunk.chunk_text == "Title: Test Title\nAbstract: Test abstract.\nMeSH: Neoplasms; Cardiovascular Diseases"


def test_chunk_preserves_pmid():
    article = _make_article(pmid="999")
    chunk = chunk_article(article)
    assert chunk.pmid == "999"


def test_chunk_with_no_mesh_terms():
    article = _make_article(mesh_terms=[])
    chunk = chunk_article(article)
    assert chunk.chunk_text == "Title: Test Title\nAbstract: Test abstract."


def test_chunk_carries_article_reference():
    article = _make_article()
    chunk = chunk_article(article)
    assert chunk.title == article.title
    assert chunk.abstract_text == article.abstract
    assert chunk.year == article.year
    assert chunk.journal == article.journal
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_chunker.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement chunker**

```python
# src/ingestion/chunker.py
"""Chunk articles into indexable text per ADR-0001."""

import json

from src.shared.models import Article, Chunk


def chunk_article(article: Article) -> Chunk:
    """Convert an Article into a single Chunk (1 abstract = 1 chunk).

    Text format per ADR-0001:
        Title: {title}
        Abstract: {abstract}
        MeSH: {term1}; {term2}; ...
    """
    parts = [f"Title: {article.title}", f"Abstract: {article.abstract}"]
    if article.mesh_terms:
        parts.append(f"MeSH: {'; '.join(article.mesh_terms)}")
    chunk_text = "\n".join(parts)

    return Chunk(
        pmid=article.pmid,
        chunk_text=chunk_text,
        title=article.title,
        abstract_text=article.abstract,
        year=article.year,
        journal=article.journal,
        authors=json.dumps(article.authors),
        mesh_terms=json.dumps(article.mesh_terms),
        keywords=json.dumps(article.keywords),
        publication_types=json.dumps(article.publication_types),
    )
```

Note: This depends on `Chunk` model from A-3. Add to stub if needed:

```python
class Chunk(BaseModel):
    pmid: str
    chunk_text: str
    title: str
    abstract_text: str
    year: int
    journal: str
    authors: str  # JSON string for Milvus VARCHAR
    mesh_terms: str  # JSON string
    keywords: str  # JSON string
    publication_types: str  # JSON string
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_chunker.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/ingestion/chunker.py backend/tests/unit/test_chunker.py
git commit -m "feat(ingestion): add chunker (1 abstract = 1 chunk per ADR-0001)"
```

---

## Chunk 2: Embedder and Ingestion Orchestrator

### Task 3: Embedder (OpenAI + Milvus Upsert)

**Files:**
- Create: `backend/src/ingestion/embedder.py`
- Create: `backend/tests/unit/test_embedder.py`
- Create: `backend/tests/integration/test_embedder_milvus.py`

- [ ] **Step 1: Write unit tests for embedder (with mocked OpenAI)**

```python
# tests/unit/test_embedder.py
"""Tests for embedder (OpenAI API calls mocked)."""

from unittest.mock import MagicMock, patch

from src.ingestion.embedder import generate_embeddings


def test_generate_embeddings_calls_openai():
    texts = ["Hello world", "Test text"]
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]

    with patch("src.ingestion.embedder._get_openai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        embeddings = generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536


def test_generate_embeddings_batches_large_input():
    texts = [f"text {i}" for i in range(250)]

    def make_response(n):
        mock = MagicMock()
        mock.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
        return mock

    with patch("src.ingestion.embedder._get_openai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            make_response(100), make_response(100), make_response(50),
        ]
        mock_get_client.return_value = mock_client
        embeddings = generate_embeddings(texts, batch_size=100)

    assert mock_client.embeddings.create.call_count == 3  # 100 + 100 + 50
    assert len(embeddings) == 250
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_embedder.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement embedder**

```python
# src/ingestion/embedder.py
"""Generate embeddings and upsert into Milvus."""

import json
import logging
import time

from openai import OpenAI
from pymilvus import Collection

from src.shared.config import get_settings
from src.shared.models import Chunk

logger = logging.getLogger(__name__)


def _get_openai_client() -> OpenAI:
    """Lazy OpenAI client initialization (avoids import-time OPENAI_API_KEY check)."""
    return OpenAI()


def generate_embeddings(
    texts: list[str],
    batch_size: int = 100,
    max_retries: int = 3,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI API.

    Batches requests per spec Section 11 (configurable batch size, retry with backoff).
    Uses embedding model from shared config to stay consistent with search.
    """
    client = _get_openai_client()
    settings = get_settings()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in response.data])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Embedding API error (attempt %d): %s. Retrying in %ds...", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    raise

        logger.debug("Embedded batch %d-%d (%d texts)", i, i + len(batch), len(batch))

    return all_embeddings


def upsert_chunks(
    collection: Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Upsert chunks with their embeddings into Milvus. Idempotent by PMID."""
    data = [
        {
            "pmid": chunk.pmid,
            "embedding": emb,
            "title": chunk.title,
            "abstract_text": chunk.abstract_text,
            "chunk_text": chunk.chunk_text,
            "year": chunk.year,
            "journal": chunk.journal,
            "authors": chunk.authors,
            "mesh_terms": chunk.mesh_terms,
            "publication_types": chunk.publication_types,
            "keywords": chunk.keywords,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    result = collection.upsert(data)
    logger.info("Upserted %d chunks into Milvus", len(data))
    return result.upsert_count
```

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_embedder.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/ingestion/embedder.py backend/tests/unit/test_embedder.py
git commit -m "feat(ingestion): add embedder with batching and retry"
```

---

### Task 4: Ingestion Orchestrator

**Files:**
- Create: `backend/src/ingestion/pipeline.py`
- Modify: `backend/src/ingestion/__init__.py`
- Create: `backend/tests/unit/test_pipeline.py`

- [ ] **Step 1: Write test for pipeline orchestration**

```python
# tests/unit/test_pipeline.py
"""Tests for ingestion pipeline orchestrator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion.pipeline import ingest


def _write_sample_jsonl(path: Path, n: int = 3) -> None:
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "pmid": str(i),
                "title": f"Title {i}",
                "abstract": f"Abstract text {i}",
                "authors": ["Author"],
                "publication_date": "2023",
                "mesh_terms": ["Neoplasms"],
                "keywords": [],
                "publication_types": ["Journal Article"],
                "language": "eng",
                "journal": "Journal",
            }
            f.write(json.dumps(record) + "\n")


@patch("src.ingestion.pipeline.upsert_chunks")
@patch("src.ingestion.pipeline.generate_embeddings")
def test_ingest_loads_chunks_and_upserts(mock_embed, mock_upsert):
    mock_embed.return_value = [[0.1] * 1536] * 3
    mock_upsert.return_value = 3

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    _write_sample_jsonl(path)

    mock_collection = MagicMock()
    report = ingest(path, mock_collection)

    assert report.total_articles == 3
    assert report.total_chunks == 3
    assert report.upserted == 3
    mock_embed.assert_called_once()
    mock_upsert.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
uv run pytest tests/unit/test_pipeline.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement pipeline orchestrator**

```python
# src/ingestion/pipeline.py
"""Ingestion pipeline: JSONL → Article → Chunk → Embed → Milvus."""

import logging
from pathlib import Path

from pymilvus import Collection

from src.ingestion.chunker import chunk_article
from src.ingestion.embedder import generate_embeddings, upsert_chunks
from src.ingestion.loader import load_articles
from src.shared.models import IngestReport

logger = logging.getLogger(__name__)


def ingest(source_path: Path, collection: Collection, batch_size: int = 100) -> IngestReport:
    """Run the full ingestion pipeline.

    1. Load articles from JSONL
    2. Chunk each article (1 abstract = 1 chunk)
    3. Generate embeddings
    4. Upsert into Milvus
    """
    logger.info("Starting ingestion from %s", source_path)

    articles = load_articles(source_path)
    logger.info("Loaded %d articles", len(articles))

    chunks = [chunk_article(a) for a in articles]
    logger.info("Created %d chunks", len(chunks))

    texts = [c.chunk_text for c in chunks]
    embeddings = generate_embeddings(texts, batch_size=batch_size)
    logger.info("Generated %d embeddings", len(embeddings))

    upserted = upsert_chunks(collection, chunks, embeddings)

    report = IngestReport(
        total_articles=len(articles),
        total_chunks=len(chunks),
        upserted=upserted,
        source_path=str(source_path),
    )
    logger.info("Ingestion complete: %s", report)
    return report
```

Note: Depends on `IngestReport` from A-3. Add to stub if needed:

```python
class IngestReport(BaseModel):
    total_articles: int
    total_chunks: int
    upserted: int
    source_path: str
```

- [ ] **Step 4: Update `__init__.py` with public interface**

```python
# src/ingestion/__init__.py
"""Ingestion module - public interface."""

from src.ingestion.pipeline import ingest

__all__ = ["ingest"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_pipeline.py tests/unit/test_loader.py tests/unit/test_chunker.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/ingestion/ backend/tests/unit/test_pipeline.py
git commit -m "feat(ingestion): add pipeline orchestrator (load → chunk → embed → upsert)"
```
