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


from sample import stratified_sample


def _make_records(year, n, mesh_prefix="General"):
    """Helper to generate n records for a given year."""
    return [
        {
            "pmid": f"{year}-{mesh_prefix}-{i}",
            "title": f"T-{i}",
            "abstract": "Abstract text",
            "authors": [],
            "publication_date": f"{year}-01-01",
            "mesh_terms": [f"{mesh_prefix} Disease"],
            "keywords": [],
            "publication_types": [],
            "language": "eng",
            "journal": "J",
        }
        for i in range(n)
    ]


def test_stratified_sample_basic():
    """With 2 years, per_year=5, no MeSH coverage, should get 10 total."""
    records_2023 = _make_records(2023, 20)
    records_2024 = _make_records(2024, 20)
    all_records = records_2023 + records_2024

    sampling_config = {
        "n_max": 10,
        "seed": 42,
        "allocation": "equal_per_year",
        "min_coverage": {"enabled": False},
    }
    config = {
        "years": [2023, 2024],
        "language": "eng",
        "require_abstract": True,
        "sampling": sampling_config,
    }

    result, audit = stratified_sample(all_records, config)
    assert len(result) == 10
    # 5 per year
    years = [extract_year(r["publication_date"]) for r in result]
    assert years.count(2023) == 5
    assert years.count(2024) == 5


def test_stratified_sample_with_mesh_coverage():
    """MeSH min coverage should guarantee category representation."""
    # 100 records in 2023: 90 Neoplasms, 10 Infectious
    neo_records = _make_records(2023, 90, "Neoplasms")
    inf_records = _make_records(2023, 10, "Infectious Diseases")
    all_records = neo_records + inf_records

    sampling_config = {
        "n_max": 20,
        "seed": 42,
        "allocation": "equal_per_year",
        "min_coverage": {
            "enabled": True,
            "per_category_per_year": 5,
            "mesh_categories": ["Neoplasms", "Infectious Diseases"],
        },
    }
    config = {
        "years": [2023],
        "language": "eng",
        "require_abstract": True,
        "sampling": sampling_config,
    }

    result, audit = stratified_sample(all_records, config)
    assert len(result) == 20

    # At least 5 Infectious Diseases records guaranteed
    inf_count = sum(1 for r in result if match_mesh_category(r["mesh_terms"], "Infectious Diseases"))
    assert inf_count >= 5


def test_stratified_sample_reproducible():
    """Same seed should produce same results."""
    records = _make_records(2023, 100)
    sampling_config = {
        "n_max": 10,
        "seed": 42,
        "allocation": "equal_per_year",
        "min_coverage": {"enabled": False},
    }
    config = {
        "years": [2023],
        "language": "eng",
        "require_abstract": True,
        "sampling": sampling_config,
    }

    result1, _ = stratified_sample(records, config)
    result2, _ = stratified_sample(records, config)
    assert [r["pmid"] for r in result1] == [r["pmid"] for r in result2]


def test_stratified_sample_shortfall_logged():
    """When a category has fewer records than min, log shortfall."""
    # Only 2 Infectious records, but min is 5
    neo_records = _make_records(2023, 50, "Neoplasms")
    inf_records = _make_records(2023, 2, "Infectious Diseases")
    all_records = neo_records + inf_records

    sampling_config = {
        "n_max": 20,
        "seed": 42,
        "allocation": "equal_per_year",
        "min_coverage": {
            "enabled": True,
            "per_category_per_year": 5,
            "mesh_categories": ["Neoplasms", "Infectious Diseases"],
        },
    }
    config = {
        "years": [2023],
        "language": "eng",
        "require_abstract": True,
        "sampling": sampling_config,
    }

    result, audit = stratified_sample(all_records, config)
    # Should still succeed, not error
    assert len(result) == 20
    # Shortfall should be logged (check by category name, not by list order)
    assert len(audit["shortfalls"]) > 0
    inf_shortfalls = [s for s in audit["shortfalls"] if s["category"] == "Infectious Diseases"]
    assert len(inf_shortfalls) == 1
    assert inf_shortfalls[0]["available"] == 2
    assert inf_shortfalls[0]["requested"] == 5
