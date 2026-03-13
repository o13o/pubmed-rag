"""Tests for JSONL loader."""

import json
import tempfile
from pathlib import Path

from src.ingestion.loader import load_articles


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
