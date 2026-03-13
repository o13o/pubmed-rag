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
                "pmid": str(i), "title": f"Title {i}", "abstract": f"Abstract text {i}",
                "authors": ["Author"], "publication_date": "2023",
                "mesh_terms": ["Neoplasms"], "keywords": [],
                "publication_types": ["Journal Article"], "language": "eng", "journal": "Journal",
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
