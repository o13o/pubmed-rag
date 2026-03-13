import json
import pytest
from pathlib import Path

from sample import extract_year, filter_records, match_mesh_category

# Test data: realistic records for filtering/sampling tests
RECORDS = [
    {"pmid": "1", "title": "T1", "abstract": "Abstract text", "authors": [], "publication_date": "2023-01-01", "mesh_terms": ["Lung Neoplasms"], "keywords": [], "publication_types": [], "language": "eng", "journal": "J1"},
    {"pmid": "2", "title": "T2", "abstract": "Abstract text", "authors": [], "publication_date": "2023-06-15", "mesh_terms": ["Cardiovascular Diseases"], "keywords": [], "publication_types": [], "language": "eng", "journal": "J2"},
    {"pmid": "3", "title": "T3", "abstract": "", "authors": [], "publication_date": "2023-03-01", "mesh_terms": [], "keywords": [], "publication_types": [], "language": "eng", "journal": "J3"},
    {"pmid": "4", "title": "T4", "abstract": "Abstract text", "authors": [], "publication_date": "2023-04-01", "mesh_terms": [], "keywords": [], "publication_types": [], "language": "fra", "journal": "J4"},
    {"pmid": "5", "title": "T5", "abstract": "Abstract text", "authors": [], "publication_date": "2019-01-01", "mesh_terms": [], "keywords": [], "publication_types": [], "language": "eng", "journal": "J5"},
    {"pmid": "6", "title": "T6", "abstract": "Abstract text", "authors": [], "publication_date": "2021-01-01", "mesh_terms": ["Infectious Diseases"], "keywords": [], "publication_types": [], "language": "eng", "journal": "J6"},
]


def test_filter_records():
    config = {
        "years": [2021, 2022, 2023, 2024, 2025],
        "language": "eng",
        "require_abstract": True,
    }
    filtered = filter_records(RECORDS, config)
    pmids = [r["pmid"] for r in filtered]
    # pmid 3: no abstract -> excluded
    # pmid 4: language=fra -> excluded
    # pmid 5: year=2019 -> excluded
    assert pmids == ["1", "2", "6"]


def test_match_mesh_category_substring():
    """Substring match: 'Lung Neoplasms' matches category 'Neoplasms'."""
    assert match_mesh_category(["Lung Neoplasms", "Drug Therapy"], "Neoplasms") is True


def test_match_mesh_category_exact():
    """Exact match also works."""
    assert match_mesh_category(["Cardiovascular Diseases"], "Cardiovascular Diseases") is True


def test_match_mesh_category_no_match():
    assert match_mesh_category(["Drug Therapy"], "Neoplasms") is False


def test_match_mesh_category_empty():
    assert match_mesh_category([], "Neoplasms") is False
