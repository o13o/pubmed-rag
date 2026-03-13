# Phase A-3: Shared Module (Models, Config, LiteLLM, MeSH DuckDB)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cross-cutting shared module: Pydantic models, configuration management, LiteLLM wrapper, and DuckDB-backed MeSH lookup.

**Architecture:** `shared/` is the dependency-free foundation that all other modules import from. No module-to-module dependencies within shared; each file is independently testable.

**Tech Stack:** Pydantic v2, pydantic-settings, LiteLLM, DuckDB, pytest

**Spec:** [2026-03-14-pubmed-rag-system-design.md](../specs/2026-03-14-pubmed-rag-system-design.md) - Sections 3.3, 6

---

## Chunk 1: Pydantic Models and Config

### Task 1: Pydantic Models

**Files:**
- Create: `capstone/backend/src/shared/models.py`
- Create: `capstone/backend/tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# tests/unit/test_models.py
"""Tests for shared Pydantic models."""

import json

from src.shared.models import (
    Article,
    Chunk,
    Citation,
    IngestReport,
    RAGResponse,
    SearchFilters,
    SearchResult,
)


def test_article_creation():
    a = Article(
        pmid="12345",
        title="Test Title",
        abstract="Test abstract.",
        authors=["John Doe"],
        year=2023,
        journal="Test Journal",
        mesh_terms=["Neoplasms"],
        keywords=["cancer"],
        publication_types=["Journal Article"],
    )
    assert a.pmid == "12345"
    assert a.year == 2023


def test_article_defaults():
    a = Article(pmid="1", title="T", abstract="A", year=2023)
    assert a.authors == []
    assert a.mesh_terms == []
    assert a.keywords == []
    assert a.publication_types == []
    assert a.journal == ""


def test_chunk_creation():
    c = Chunk(
        pmid="1",
        chunk_text="Title: T\nAbstract: A",
        title="T",
        abstract_text="A",
        year=2023,
        journal="J",
        authors="[]",
        mesh_terms="[]",
        keywords="[]",
        publication_types="[]",
    )
    assert c.pmid == "1"
    assert "Title: T" in c.chunk_text


def test_search_result_with_score():
    sr = SearchResult(
        pmid="1",
        title="T",
        abstract_text="A",
        score=0.95,
        year=2023,
        journal="J",
        mesh_terms=["Neoplasms"],
    )
    assert sr.score == 0.95


def test_search_filters_defaults():
    f = SearchFilters()
    assert f.year_min is None
    assert f.year_max is None
    assert f.journals == []
    assert f.top_k == 10


def test_search_filters_custom():
    f = SearchFilters(year_min=2022, year_max=2024, journals=["Nature"], top_k=5)
    assert f.year_min == 2022
    assert f.top_k == 5


def test_rag_response():
    r = RAGResponse(
        answer="Based on the evidence...",
        citations=[
            Citation(pmid="1", title="T", relevance_score=0.9),
        ],
        query="test query",
    )
    assert len(r.citations) == 1
    assert r.query == "test query"
    assert r.citations[0].pmid == "1"


def test_ingest_report():
    r = IngestReport(total_articles=100, total_chunks=100, upserted=100, source_path="/tmp/test.jsonl")
    assert r.total_articles == 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_models.py -v
```

Expected: FAIL - no module

- [ ] **Step 3: Implement models**

```python
# src/shared/models.py
"""Pydantic models shared across all modules.

These models serve as the inter-module communication contracts.
When splitting into microservices, these become API schemas.
"""

from pydantic import BaseModel, Field


class Article(BaseModel):
    """Raw article parsed from PubMed JSONL."""

    pmid: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: int
    journal: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """Indexed chunk for Milvus storage.

    Scalar fields (authors, mesh_terms, etc.) are JSON strings
    because Milvus VARCHAR stores them as-is.
    """

    pmid: str
    chunk_text: str
    title: str
    abstract_text: str
    year: int
    journal: str
    authors: str  # JSON string
    mesh_terms: str  # JSON string
    keywords: str  # JSON string
    publication_types: str  # JSON string


class SearchFilters(BaseModel):
    """Filters for metadata-based search narrowing."""

    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    top_k: int = 10


class SearchResult(BaseModel):
    """A single search result returned from Milvus."""

    pmid: str
    title: str
    abstract_text: str
    score: float
    year: int
    journal: str
    mesh_terms: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A single citation referencing a source abstract."""

    pmid: str
    title: str
    journal: str = ""
    year: int = 0
    relevance_score: float = 0.0


class RAGResponse(BaseModel):
    """Response from the RAG chain."""

    answer: str
    citations: list[Citation]
    query: str


class IngestReport(BaseModel):
    """Report from the ingestion pipeline."""

    total_articles: int
    total_chunks: int
    upserted: int
    source_path: str
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_models.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/shared/models.py capstone/backend/tests/unit/test_models.py
git commit -m "feat(shared): add Pydantic models for inter-module contracts"
```

---

### Task 2: Configuration Management

**Files:**
- Create: `capstone/backend/src/shared/config.py`
- Create: `capstone/backend/tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/unit/test_config.py
"""Tests for configuration management."""

import os
from unittest.mock import patch

from src.shared.config import Settings


def test_default_settings():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        s = Settings()
    assert s.milvus_host == "localhost"
    assert s.milvus_port == 19530
    assert s.llm_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
    assert s.embedding_batch_size == 100
    assert s.top_k == 10


def test_settings_from_env():
    env = {
        "OPENAI_API_KEY": "sk-test",
        "MILVUS_HOST": "milvus-server",
        "MILVUS_PORT": "29530",
        "LLM_MODEL": "gpt-4o",
        "TOP_K": "20",
    }
    with patch.dict(os.environ, env, clear=False):
        s = Settings()
    assert s.milvus_host == "milvus-server"
    assert s.milvus_port == 29530
    assert s.llm_model == "gpt-4o"
    assert s.top_k == 20
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_config.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement config**

```python
# src/shared/config.py
"""Application settings via pydantic-settings (env vars + .env)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration. All values can be overridden via environment variables."""

    # OpenAI
    openai_api_key: str = ""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "pubmed_abstracts"

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 30

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100

    # Retrieval
    top_k: int = 10

    # MeSH DuckDB
    mesh_db_path: str = "data/mesh.duckdb"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_config.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/shared/config.py capstone/backend/tests/unit/test_config.py
git commit -m "feat(shared): add pydantic-settings configuration management"
```

---

## Chunk 2: LiteLLM Wrapper

### Task 3: LiteLLM Wrapper

**Files:**
- Create: `capstone/backend/src/shared/llm.py`
- Create: `capstone/backend/tests/unit/test_llm.py`

- [ ] **Step 1: Write failing tests for LLM wrapper**

```python
# tests/unit/test_llm.py
"""Tests for LiteLLM wrapper."""

from unittest.mock import MagicMock, patch

from src.shared.llm import LLMClient


def test_llm_client_complete():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(total_tokens=50)
        mock_completion.return_value = mock_response

        client = LLMClient(model="gpt-4o-mini")
        result = client.complete(
            system_prompt="You are helpful.",
            user_prompt="Hello",
        )

    assert result == "Test response"
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert len(call_kwargs["messages"]) == 2


def test_llm_client_uses_configured_model():
    with patch("src.shared.llm.litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = MagicMock(total_tokens=10)
        mock_completion.return_value = mock_response

        client = LLMClient(model="claude-sonnet-4-20250514")
        client.complete(system_prompt="sys", user_prompt="usr")

    call_kwargs = mock_completion.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


def test_llm_client_default_model():
    client = LLMClient()
    assert client.model == "gpt-4o-mini"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_llm.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement LLM wrapper**

```python
# src/shared/llm.py
"""LiteLLM wrapper for model-agnostic LLM calls."""

import logging

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """Thin wrapper around LiteLLM for consistent LLM access.

    Supports any model LiteLLM supports: OpenAI, Anthropic, etc.
    Default: gpt-4o-mini (per spec).
    """

    def __init__(self, model: str = "gpt-4o-mini", timeout: int = 30):
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Send a completion request and return the response text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
        )

        result = response.choices[0].message.content
        tokens = response.usage.total_tokens
        logger.debug("LLM call: model=%s, tokens=%d", self.model, tokens)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_llm.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/shared/llm.py capstone/backend/tests/unit/test_llm.py
git commit -m "feat(shared): add LiteLLM wrapper for model-agnostic LLM calls"
```

---

## Chunk 3: MeSH DuckDB Lookup

### Task 4: MeSH XML Parser and DuckDB Loader

**Files:**
- Create: `capstone/backend/src/shared/mesh_db.py`
- Create: `capstone/backend/tests/unit/test_mesh_db.py`
- Create: `capstone/backend/scripts/build_mesh_db.py`

The MeSH XML (`desc2025.xml`) has this structure per descriptor:
```xml
<DescriptorRecord>
  <DescriptorUI>D009369</DescriptorUI>
  <DescriptorName><String>Neoplasms</String></DescriptorName>
  <TreeNumberList>
    <TreeNumber>C04</TreeNumber>
  </TreeNumberList>
  <ConceptList>
    <Concept>
      <TermList>
        <Term><String>Neoplasms</String></Term>
        <Term><String>Cancer</String></Term>
        <Term><String>Tumors</String></Term>
      </TermList>
    </Concept>
  </ConceptList>
</DescriptorRecord>
```

- [ ] **Step 1: Write failing tests for MeSH DB**

```python
# tests/unit/test_mesh_db.py
"""Tests for DuckDB-backed MeSH lookup."""

import duckdb
import pytest

from src.shared.mesh_db import MeSHDatabase


@pytest.fixture
def mesh_db():
    """Create an in-memory MeSH database with test data."""
    db = MeSHDatabase(":memory:")
    db._init_schema()

    # Insert test descriptors
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D002318', 'Cardiovascular Diseases', ['C14']),
        ('D003324', 'Coronary Artery Disease', ['C14.280.647.250']),
        ('D006333', 'Heart Failure', ['C14.280.434']),
        ('D020370', 'Osteoarthritis, Knee', ['C05.550.114.606'])
    """)

    # Insert test synonyms
    db.conn.execute("""
        INSERT INTO mesh_synonyms VALUES
        ('Cancer', 'D009369'),
        ('Tumors', 'D009369'),
        ('Malignancy', 'D009369'),
        ('Heart Disease', 'D002318'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Degenerative Arthritis of Knee', 'D020370')
    """)

    return db


def test_lookup_by_name(mesh_db):
    result = mesh_db.lookup("Neoplasms")
    assert result is not None
    assert result["descriptor_ui"] == "D009369"
    assert result["name"] == "Neoplasms"


def test_lookup_by_synonym(mesh_db):
    result = mesh_db.lookup("Cancer")
    assert result is not None
    assert result["name"] == "Neoplasms"


def test_lookup_case_insensitive(mesh_db):
    result = mesh_db.lookup("cancer")
    assert result is not None
    assert result["name"] == "Neoplasms"


def test_lookup_not_found(mesh_db):
    result = mesh_db.lookup("nonexistent_term")
    assert result is None


def test_get_children(mesh_db):
    """C14 → should find C14.280.647.250 and C14.280.434."""
    children = mesh_db.get_children("C14")
    names = [c["name"] for c in children]
    assert "Coronary Artery Disease" in names
    assert "Heart Failure" in names
    assert "Cardiovascular Diseases" not in names  # parent itself excluded


def test_get_synonyms(mesh_db):
    synonyms = mesh_db.get_synonyms("D009369")
    assert "Cancer" in synonyms
    assert "Tumors" in synonyms
    assert "Malignancy" in synonyms


def test_validate_term_exists(mesh_db):
    assert mesh_db.validate_term("Neoplasms") is True
    assert mesh_db.validate_term("Cancer") is True  # synonym


def test_validate_term_not_exists(mesh_db):
    assert mesh_db.validate_term("FakeDrug123") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_mesh_db.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement MeSH database**

```python
# src/shared/mesh_db.py
"""DuckDB-backed MeSH lookup for query expansion and terminology validation.

Provides:
- Term lookup (name or synonym → descriptor)
- Hierarchy traversal (parent tree number → child descriptors)
- Synonym resolution (descriptor_ui → all synonyms)
- Term validation (does this term exist in MeSH?)

See spec Section 6 for schema and query patterns.
"""

import logging
from functools import lru_cache

import duckdb

logger = logging.getLogger(__name__)


class MeSHDatabase:
    """DuckDB-backed MeSH lookup."""

    def __init__(self, db_path: str = "data/mesh.duckdb"):
        self.conn = duckdb.connect(db_path)

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_descriptors (
                descriptor_ui VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                tree_numbers VARCHAR[] NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_synonyms (
                synonym VARCHAR NOT NULL,
                descriptor_ui VARCHAR NOT NULL REFERENCES mesh_descriptors(descriptor_ui)
            )
        """)

    def lookup(self, term: str) -> dict | None:
        """Look up a term by exact name or synonym (case-insensitive).

        Returns: {"descriptor_ui": ..., "name": ..., "tree_numbers": [...]} or None
        """
        # Try descriptor name first
        row = self.conn.execute(
            "SELECT descriptor_ui, name, tree_numbers FROM mesh_descriptors WHERE name ILIKE ?",
            [term],
        ).fetchone()

        if row:
            return {"descriptor_ui": row[0], "name": row[1], "tree_numbers": row[2]}

        # Try synonym
        row = self.conn.execute(
            """SELECT d.descriptor_ui, d.name, d.tree_numbers
               FROM mesh_synonyms s
               JOIN mesh_descriptors d ON s.descriptor_ui = d.descriptor_ui
               WHERE s.synonym ILIKE ?""",
            [term],
        ).fetchone()

        if row:
            return {"descriptor_ui": row[0], "name": row[1], "tree_numbers": row[2]}

        return None

    def get_children(self, tree_number: str) -> list[dict]:
        """Get all child descriptors under a tree number (prefix match).

        Excludes the exact tree number itself (only returns children).
        """
        rows = self.conn.execute(
            """SELECT DISTINCT d.descriptor_ui, d.name
               FROM mesh_descriptors d, unnest(d.tree_numbers) AS t(tn)
               WHERE tn LIKE ? AND tn != ?""",
            [f"{tree_number}.%", tree_number],
        ).fetchall()

        return [{"descriptor_ui": r[0], "name": r[1]} for r in rows]

    def get_synonyms(self, descriptor_ui: str) -> list[str]:
        """Get all synonyms (entry terms) for a descriptor."""
        rows = self.conn.execute(
            "SELECT synonym FROM mesh_synonyms WHERE descriptor_ui = ?",
            [descriptor_ui],
        ).fetchall()
        return [r[0] for r in rows]

    def validate_term(self, term: str) -> bool:
        """Check if a term exists as a descriptor name or synonym."""
        row = self.conn.execute(
            """SELECT EXISTS(SELECT 1 FROM mesh_descriptors WHERE name ILIKE ?)
               OR EXISTS(SELECT 1 FROM mesh_synonyms WHERE synonym ILIKE ?)""",
            [term, term],
        ).fetchone()
        return bool(row and row[0])

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_mesh_db.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/shared/mesh_db.py capstone/backend/tests/unit/test_mesh_db.py
git commit -m "feat(shared): add DuckDB-backed MeSH lookup (hierarchy, synonyms, validation)"
```

---

### Task 5: MeSH XML → DuckDB Build Script

**Files:**
- Create: `capstone/backend/scripts/build_mesh_db.py`

- [ ] **Step 1: Write the build script**

```python
# scripts/build_mesh_db.py
"""Parse NLM MeSH XML (desc2025.xml) and build DuckDB database.

Usage:
    uv run python scripts/build_mesh_db.py --input data/desc2025.xml --output data/mesh.duckdb

Download desc2025.xml from: https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/
"""

import argparse
import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_mesh_xml(xml_path: Path) -> tuple[list[tuple], list[tuple]]:
    """Parse MeSH XML and extract descriptors and synonyms.

    Returns: (descriptors, synonyms) where
        descriptors = [(descriptor_ui, name, tree_numbers), ...]
        synonyms = [(synonym_text, descriptor_ui), ...]
    """
    logger.info("Parsing %s ...", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    descriptors = []
    synonyms = []

    for record in root.findall("DescriptorRecord"):
        ui = record.findtext("DescriptorUI", default="")
        name = record.findtext("DescriptorName/String", default="")

        tree_numbers = [tn.text for tn in record.findall("TreeNumberList/TreeNumber") if tn.text]

        if ui and name:
            descriptors.append((ui, name, tree_numbers))

        # Extract entry terms (synonyms) from all concepts
        for concept in record.findall("ConceptList/Concept"):
            for term in concept.findall("TermList/Term"):
                term_text = term.findtext("String", default="")
                if term_text and term_text != name:
                    synonyms.append((term_text, ui))

    logger.info("Parsed %d descriptors, %d synonyms", len(descriptors), len(synonyms))
    return descriptors, synonyms


def build_duckdb(descriptors: list[tuple], synonyms: list[tuple], output_path: Path) -> None:
    """Create DuckDB database from parsed MeSH data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to rebuild
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    conn.execute("""
        CREATE TABLE mesh_descriptors (
            descriptor_ui VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            tree_numbers VARCHAR[] NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE mesh_synonyms (
            synonym VARCHAR NOT NULL,
            descriptor_ui VARCHAR NOT NULL REFERENCES mesh_descriptors(descriptor_ui)
        )
    """)

    # Batch insert descriptors
    conn.executemany(
        "INSERT INTO mesh_descriptors VALUES (?, ?, ?)",
        descriptors,
    )

    # Batch insert synonyms
    conn.executemany(
        "INSERT INTO mesh_synonyms VALUES (?, ?)",
        synonyms,
    )

    # Create indexes for fast lookup
    conn.execute("CREATE INDEX idx_desc_name ON mesh_descriptors(name)")
    conn.execute("CREATE INDEX idx_syn_text ON mesh_synonyms(synonym)")
    conn.execute("CREATE INDEX idx_syn_ui ON mesh_synonyms(descriptor_ui)")

    count_d = conn.execute("SELECT COUNT(*) FROM mesh_descriptors").fetchone()[0]
    count_s = conn.execute("SELECT COUNT(*) FROM mesh_synonyms").fetchone()[0]
    logger.info("Built DuckDB: %d descriptors, %d synonyms → %s", count_d, count_s, output_path)

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build MeSH DuckDB from XML")
    parser.add_argument("--input", required=True, help="Path to MeSH XML (e.g., desc2025.xml)")
    parser.add_argument("--output", default="data/mesh.duckdb", help="Output DuckDB path")
    args = parser.parse_args()

    descriptors, synonyms = parse_mesh_xml(Path(args.input))
    build_duckdb(descriptors, synonyms, Path(args.output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add capstone/backend/scripts/build_mesh_db.py
git commit -m "feat(mesh): add MeSH XML → DuckDB build script"
```

---

### Task 6: Shared conftest.py

**Files:**
- Create: `capstone/backend/tests/conftest.py`

- [ ] **Step 1: Create conftest with shared fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""

import os

import duckdb
import pytest

from src.shared.mesh_db import MeSHDatabase


@pytest.fixture
def mesh_db():
    """In-memory MeSH database with test data."""
    db = MeSHDatabase(":memory:")
    db._init_schema()
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D001943', 'Breast Neoplasms', ['C04.588.180']),
        ('D002318', 'Cardiovascular Diseases', ['C14']),
        ('D003324', 'Coronary Artery Disease', ['C14.280.647.250']),
        ('D006333', 'Heart Failure', ['C14.280.434']),
        ('D020370', 'Osteoarthritis, Knee', ['C05.550.114.606'])
    """)
    db.conn.execute("""
        INSERT INTO mesh_synonyms VALUES
        ('Cancer', 'D009369'),
        ('Tumors', 'D009369'),
        ('Malignancy', 'D009369'),
        ('Breast Cancer', 'D001943'),
        ('Heart Disease', 'D002318'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Knee Pain', 'D020370'),
        ('Degenerative Arthritis of Knee', 'D020370')
    """)
    yield db
    db.close()


@pytest.fixture
def mock_env(monkeypatch):
    """Set minimal env vars for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
```

- [ ] **Step 2: Commit**

```bash
git add capstone/backend/tests/conftest.py
git commit -m "test: add shared conftest.py with MeSH and env fixtures"
```

---

### Task 7: Update shared `__init__.py`

**Files:**
- Modify: `capstone/backend/src/shared/__init__.py`

- [ ] **Step 1: Define public interface**

```python
# src/shared/__init__.py
"""Shared module - public interface."""

from src.shared.config import Settings, get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    Article,
    Chunk,
    Citation,
    IngestReport,
    RAGResponse,
    SearchFilters,
    SearchResult,
)

__all__ = [
    "Article",
    "Chunk",
    "Citation",
    "IngestReport",
    "LLMClient",
    "MeSHDatabase",
    "RAGResponse",
    "SearchFilters",
    "SearchResult",
    "Settings",
    "get_settings",
]
```

- [ ] **Step 2: Run all shared module tests**

```bash
cd capstone/backend
uv run pytest tests/unit/test_models.py tests/unit/test_config.py tests/unit/test_llm.py tests/unit/test_mesh_db.py -v
```

Expected: All tests PASS (21 total).

- [ ] **Step 3: Commit**

```bash
git add capstone/backend/src/shared/__init__.py
git commit -m "feat(shared): define public interface in __init__.py"
```
