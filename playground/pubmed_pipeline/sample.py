"""Filter and sample PubMed records per ADR 0002."""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def extract_year(publication_date: str) -> int | None:
    """Extract year as int from publication_date string."""
    if len(publication_date) >= 4:
        try:
            return int(publication_date[:4])
        except ValueError:
            return None
    return None


def match_mesh_category(mesh_terms: list[str], category: str) -> bool:
    """Check if any MeSH term contains the category name as a substring."""
    return any(category in term for term in mesh_terms)


def filter_records(records: list[dict], config: dict) -> list[dict]:
    """Apply year, language, and abstract filters per ADR 0002."""
    target_years = set(config["years"])
    target_lang = config["language"]
    require_abstract = config["require_abstract"]

    filtered = []
    for r in records:
        year = extract_year(r.get("publication_date", ""))
        if year is None or year not in target_years:
            continue
        if r.get("language", "") != target_lang:
            continue
        if require_abstract and not r.get("abstract", "").strip():
            continue
        filtered.append(r)
    return filtered
