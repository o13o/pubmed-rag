# PubMed Download & Sampling Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-script pipeline that downloads PubMed data from HuggingFace and produces a filtered, year-stratified, MeSH-coverage-guaranteed JSONL sample with audit log.

**Architecture:** Two independent Python scripts connected by an intermediate JSONL file contract. `download_hf.py` fetches and converts HuggingFace PubMed data to raw JSONL. `sample.py` reads raw JSONL, applies filters and stratified sampling with MeSH minimum coverage, and outputs final JSONL + audit log.

**Tech Stack:** Python 3.10+, `datasets` (HuggingFace), `pyyaml`, `xml.etree.ElementTree`, standard library (`json`, `random`, `collections`, `datetime`, `pathlib`, `argparse`)

**Spec:** `docs/specs/2026-03-13-pubmed-download-pipeline-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `data_pipeline/config.yaml` | Pipeline configuration (years, language, sampling params, MeSH categories, paths) |
| `data_pipeline/download_hf.py` | Download PubMed from HuggingFace, parse XML, write raw JSONL |
| `data_pipeline/sample.py` | Read raw JSONL, filter, stratified sample with MeSH coverage, write final JSONL + audit log |
| `data_pipeline/README.md` | Usage instructions |
| `data_pipeline/tests/test_download_hf.py` | Tests for XML parsing and record conversion |
| `data_pipeline/tests/test_sample.py` | Tests for filtering, sampling, MeSH matching |
| `data_pipeline/conftest.py` | pytest path configuration for bare module imports |
| `data_pipeline/data/.gitignore` | Ignore generated data files |
| `data_pipeline/data/raw/.gitkeep` | Placeholder for raw output directory |
| `data_pipeline/data/processed/.gitkeep` | Placeholder for processed output directory |

---

## Chunk 1: Project Scaffolding & Config

### Task 1: Create directory structure and config.yaml

**Files:**
- Create: `data_pipeline/config.yaml`
- Create: `data_pipeline/conftest.py`
- Create: `data_pipeline/data/.gitignore`
- Create: `data_pipeline/data/raw/.gitkeep`
- Create: `data_pipeline/data/processed/.gitkeep`

- [ ] **Step 1: Create directory placeholders and support files**

```bash
mkdir -p data_pipeline/data/raw
mkdir -p data_pipeline/data/processed
mkdir -p data_pipeline/tests
touch data_pipeline/data/raw/.gitkeep
touch data_pipeline/data/processed/.gitkeep
touch data_pipeline/tests/__init__.py
```

Create `data_pipeline/conftest.py` to ensure pytest can import modules from the project root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

Create `data_pipeline/data/.gitignore` to prevent committing generated data:

```gitignore
*.jsonl
*.json
!.gitkeep
!.gitignore
```

- [ ] **Step 2: Create config.yaml**

```yaml
years: [2021, 2022, 2023, 2024, 2025]
language: "eng"
require_abstract: true
sampling:
  n_max: 100000
  seed: 42
  allocation: "equal_per_year"  # Only equal_per_year is implemented
  min_coverage:
    enabled: true
    per_category_per_year: 500
    mesh_categories:
      - "Neoplasms"
      - "Cardiovascular Diseases"
      - "Infectious Diseases"
      - "Nervous System Diseases"
      - "Respiratory Tract Diseases"
      - "Digestive System Diseases"
      - "Urogenital Diseases"
      - "Musculoskeletal Diseases"
      - "Nutritional and Metabolic Diseases"
      - "Immune System Diseases"

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
```

- [ ] **Step 3: Commit**

```bash
cd data_pipeline/pubmed_pipeline
git add .
git commit -m "scaffold: pubmed_pipeline directory structure and config.yaml"
```

---

## Chunk 2: download_hf.py — XML Parsing & Record Conversion

### Task 2: Write tests for XML-to-JSONL record conversion

**Files:**
- Create: `data_pipeline/tests/test_download_hf.py`

- [ ] **Step 1: Write test for parsing a complete MedlineCitation XML**

The HuggingFace `ncbi/pubmed` dataset stores each record as a dict with a `MedlineCitation` key containing XML. Write a test that verifies the `parse_medline_xml` function correctly extracts all fields from a realistic XML sample.

```python
import pytest
from download_hf import parse_medline_xml

SAMPLE_XML = """\
<MedlineCitation>
  <PMID>12345678</PMID>
  <Article>
    <Journal>
      <Title>Journal of Testing</Title>
      <JournalIssue>
        <PubDate>
          <Year>2023</Year>
          <Month>05</Month>
          <Day>15</Day>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Test Article Title</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Background text.</AbstractText>
      <AbstractText Label="METHODS">Methods text.</AbstractText>
    </Abstract>
    <AuthorList>
      <Author>
        <LastName>Smith</LastName>
        <ForeName>John</ForeName>
      </Author>
      <Author>
        <LastName>Doe</LastName>
        <ForeName>Jane</ForeName>
      </Author>
    </AuthorList>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
      <PublicationType>Randomized Controlled Trial</PublicationType>
    </PublicationTypeList>
  </Article>
  <MeshHeadingList>
    <MeshHeading>
      <DescriptorName>Lung Neoplasms</DescriptorName>
    </MeshHeading>
    <MeshHeading>
      <DescriptorName>Drug Therapy</DescriptorName>
    </MeshHeading>
  </MeshHeadingList>
  <KeywordList>
    <Keyword>cancer</Keyword>
    <Keyword>treatment</Keyword>
  </KeywordList>
</MedlineCitation>"""


def test_parse_medline_xml_complete():
    record = parse_medline_xml(SAMPLE_XML)
    assert record["pmid"] == "12345678"
    assert record["title"] == "Test Article Title"
    assert record["abstract"] == "BACKGROUND: Background text. METHODS: Methods text."
    assert record["authors"] == ["John Smith", "Jane Doe"]
    assert record["publication_date"] == "2023-05-15"
    assert record["mesh_terms"] == ["Lung Neoplasms", "Drug Therapy"]
    assert record["keywords"] == ["cancer", "treatment"]
    assert record["publication_types"] == ["Journal Article", "Randomized Controlled Trial"]
    assert record["language"] == "eng"
    assert record["journal"] == "Journal of Testing"
```

- [ ] **Step 2: Write test for XML with missing optional fields**

```python
MINIMAL_XML = """\
<MedlineCitation>
  <PMID>99999999</PMID>
  <Article>
    <Journal>
      <Title>Minimal Journal</Title>
      <JournalIssue>
        <PubDate>
          <Year>2024</Year>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Minimal Title</ArticleTitle>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
    </PublicationTypeList>
  </Article>
</MedlineCitation>"""


def test_parse_medline_xml_minimal():
    record = parse_medline_xml(MINIMAL_XML)
    assert record["pmid"] == "99999999"
    assert record["title"] == "Minimal Title"
    assert record["abstract"] == ""
    assert record["authors"] == []
    assert record["publication_date"] == "2024"
    assert record["mesh_terms"] == []
    assert record["keywords"] == []
    assert record["publication_types"] == ["Journal Article"]
    assert record["language"] == "eng"
    assert record["journal"] == "Minimal Journal"
```

- [ ] **Step 3: Write test for year extraction from publication_date**

```python
def test_parse_medline_xml_year_only_date():
    """Year-only PubDate should produce just the year string."""
    record = parse_medline_xml(MINIMAL_XML)
    assert record["publication_date"] == "2024"


FULL_DATE_XML = """\
<MedlineCitation>
  <PMID>11111111</PMID>
  <Article>
    <Journal>
      <Title>Date Journal</Title>
      <JournalIssue>
        <PubDate>
          <Year>2022</Year>
          <Month>12</Month>
          <Day>01</Day>
        </PubDate>
      </JournalIssue>
    </Journal>
    <ArticleTitle>Date Test</ArticleTitle>
    <Language>eng</Language>
    <PublicationTypeList>
      <PublicationType>Journal Article</PublicationType>
    </PublicationTypeList>
  </Article>
</MedlineCitation>"""


def test_parse_medline_xml_full_date():
    record = parse_medline_xml(FULL_DATE_XML)
    assert record["publication_date"] == "2022-12-01"
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_download_hf.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'download_hf'`

- [ ] **Step 5: Commit test file**

```bash
git add tests/test_download_hf.py
git commit -m "test: add XML parsing tests for download_hf"
```

### Task 3: Implement parse_medline_xml and download_hf.py

**Files:**
- Create: `data_pipeline/download_hf.py`

- [ ] **Step 1: Implement parse_medline_xml function**

```python
"""Download PubMed data from HuggingFace and convert to intermediate JSONL."""

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


def parse_medline_xml(xml_string: str) -> dict:
    """Parse a MedlineCitation XML string into the intermediate JSONL schema."""
    root = ET.fromstring(xml_string)

    # PMID
    pmid = root.findtext("PMID", default="")

    # Article fields
    article = root.find("Article")
    title = article.findtext("ArticleTitle", default="") if article is not None else ""
    language = article.findtext("Language", default="") if article is not None else ""

    # Journal
    journal = ""
    if article is not None:
        journal = article.findtext("Journal/Title", default="")

    # Publication date
    pub_date = ""
    if article is not None:
        pub_date_el = article.find("Journal/JournalIssue/PubDate")
        if pub_date_el is not None:
            year = pub_date_el.findtext("Year", default="")
            month = pub_date_el.findtext("Month", default="")
            day = pub_date_el.findtext("Day", default="")
            if year and month and day:
                pub_date = f"{year}-{month}-{day}"
            elif year:
                pub_date = year

    # Abstract (multiple sections joined)
    abstract = ""
    if article is not None:
        abstract_el = article.find("Abstract")
        if abstract_el is not None:
            parts = []
            for text_el in abstract_el.findall("AbstractText"):
                label = text_el.get("Label", "")
                text = text_el.text or ""
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            abstract = " ".join(parts)

    # Authors
    authors = []
    if article is not None:
        for author in article.findall("AuthorList/Author"):
            last = author.findtext("LastName", default="")
            fore = author.findtext("ForeName", default="")
            if fore and last:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)

    # Publication types
    publication_types = []
    if article is not None:
        for pt in article.findall("PublicationTypeList/PublicationType"):
            if pt.text:
                publication_types.append(pt.text)

    # MeSH terms
    mesh_terms = []
    for heading in root.findall("MeshHeadingList/MeshHeading/DescriptorName"):
        if heading.text:
            mesh_terms.append(heading.text)

    # Keywords
    keywords = []
    for kw in root.findall("KeywordList/Keyword"):
        if kw.text:
            keywords.append(kw.text)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_date": pub_date,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "publication_types": publication_types,
        "language": language,
        "journal": journal,
    }
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_download_hf.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 3: Add the main download logic**

Append to `download_hf.py`:

```python
def extract_year(publication_date: str) -> str:
    """Extract 4-digit year from publication_date string."""
    return publication_date[:4] if len(publication_date) >= 4 else ""


def main():
    parser = argparse.ArgumentParser(description="Download PubMed from HuggingFace")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process (for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_years = set(str(y) for y in config["years"])
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "pubmed_raw.jsonl"

    from datasets import load_dataset

    ds = load_dataset("ncbi/pubmed", streaming=True, split="train")

    count = 0
    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for record in ds:
            count += 1
            if args.limit is not None and count > args.limit:
                break

            xml_str = record.get("MedlineCitation", {}).get("value", "")
            if not xml_str:
                continue

            try:
                parsed = parse_medline_xml(xml_str)
            except ET.ParseError:
                continue

            year = extract_year(parsed["publication_date"])
            if year not in target_years:
                continue

            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            written += 1

            if written % 10000 == 0:
                print(f"  Written {written} records (scanned {count})...", file=sys.stderr)

    print(f"Done. Scanned {count}, written {written} to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add download_hf.py
git commit -m "feat: implement download_hf.py with XML parsing and HuggingFace streaming"
```

---

## Chunk 3: sample.py — Filtering, Sampling & Audit Log

### Task 4: Write tests for filtering logic

**Files:**
- Create: `data_pipeline/tests/test_sample.py`

- [ ] **Step 1: Write test fixtures — sample records**

```python
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
```

- [ ] **Step 2: Write test for filter_records**

Note: `filter_records`, `match_mesh_category`, and `extract_year` are already imported at the top of the file.

```python
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
```

- [ ] **Step 3: Write test for MeSH category matching**

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_sample.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'sample'`

- [ ] **Step 5: Commit test file**

```bash
git add tests/test_sample.py
git commit -m "test: add filter and MeSH matching tests for sample.py"
```

### Task 5: Implement filtering and MeSH matching

**Files:**
- Create: `data_pipeline/sample.py`

- [ ] **Step 1: Implement filter_records and match_mesh_category**

```python
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
```

- [ ] **Step 2: Run tests to verify filter and MeSH tests pass**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_sample.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add sample.py
git commit -m "feat: implement filter_records and match_mesh_category"
```

### Task 6: Write tests for stratified sampling logic

**Files:**
- Modify: `data_pipeline/tests/test_sample.py`

- [ ] **Step 1: Write test for stratified_sample**

Add the following imports and test code to `tests/test_sample.py`. Note: `extract_year`, `filter_records`, `match_mesh_category` are already imported at the top of the file.

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_sample.py::test_stratified_sample_basic -v
```

Expected: FAIL — `ImportError: cannot import name 'stratified_sample'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_sample.py
git commit -m "test: add stratified sampling tests"
```

### Task 7: Implement stratified_sample

**Files:**
- Modify: `data_pipeline/sample.py`

- [ ] **Step 1: Implement stratified_sample function**

Append to `sample.py`:

```python
def stratified_sample(records: list[dict], config: dict) -> tuple[list[dict], dict]:
    """
    Year-stratified sampling with optional MeSH minimum coverage.
    Returns (sampled_records, audit_info).
    """
    sampling = config["sampling"]
    n_max = sampling["n_max"]
    seed = sampling["seed"]
    years = config["years"]
    per_year_quota = n_max // len(years)

    rng = random.Random(seed)

    # Group by year
    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        year = extract_year(r.get("publication_date", ""))
        if year in years:
            by_year[year].append(r)

    sampled = []
    audit_per_year = {}
    shortfalls = []

    min_cov = sampling.get("min_coverage", {})
    cov_enabled = min_cov.get("enabled", False)
    categories = min_cov.get("mesh_categories", [])
    min_per_cat = min_cov.get("per_category_per_year", 0)

    for year in sorted(years):
        pool = by_year.get(year, [])
        population = len(pool)
        selected_set: set[str] = set()
        year_selected: list[dict] = []
        per_category_audit = {}

        if cov_enabled and categories:
            for cat in categories:
                cat_records = [r for r in pool if match_mesh_category(r["mesh_terms"], cat) and r["pmid"] not in selected_set]
                available = len(cat_records)
                to_select = min(min_per_cat, available)

                if available < min_per_cat:
                    shortfalls.append({
                        "year": year,
                        "category": cat,
                        "requested": min_per_cat,
                        "available": available,
                    })

                chosen = rng.sample(cat_records, to_select) if to_select > 0 else []
                for r in chosen:
                    if r["pmid"] not in selected_set:
                        selected_set.add(r["pmid"])
                        year_selected.append(r)

                per_category_audit[cat] = {"available": available, "selected": len(chosen)}

        # Fill remaining quota randomly
        remaining_quota = per_year_quota - len(year_selected)
        if remaining_quota > 0:
            remaining_pool = [r for r in pool if r["pmid"] not in selected_set]
            fill_count = min(remaining_quota, len(remaining_pool))
            fill = rng.sample(remaining_pool, fill_count) if fill_count > 0 else []
            year_selected.extend(fill)

        sampled.extend(year_selected)
        audit_per_year[str(year)] = {
            "population": population,
            "selected": len(year_selected),
            "per_category": per_category_audit,
        }

    audit = {
        "per_year": audit_per_year,
        "total_selected": len(sampled),
        "shortfalls": shortfalls,
    }
    return sampled, audit
```

- [ ] **Step 2: Run all sample tests**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/test_sample.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add sample.py
git commit -m "feat: implement stratified_sample with MeSH coverage"
```

### Task 8: Add main CLI to sample.py

**Files:**
- Modify: `data_pipeline/sample.py`

- [ ] **Step 1: Implement main function**

Append to `sample.py`:

```python
def load_records(raw_dir: Path) -> list[dict]:
    """Load all JSONL files from raw_dir."""
    records = []
    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Filter and sample PubMed records")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records...", file=sys.stderr)
    all_records = load_records(raw_dir)
    print(f"Loaded {len(all_records)} records", file=sys.stderr)

    print("Filtering...", file=sys.stderr)
    filtered = filter_records(all_records, config)
    print(f"After filtering: {len(filtered)} records", file=sys.stderr)

    print("Sampling...", file=sys.stderr)
    sampled, audit = stratified_sample(filtered, config)
    print(f"Sampled: {len(sampled)} records", file=sys.stderr)

    # Write sampled JSONL
    output_path = processed_dir / "sampled.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written to {output_path}", file=sys.stderr)

    # Write audit log
    audit["config"] = config
    audit["timestamp"] = datetime.now(timezone.utc).isoformat()
    audit["sampled_pmids"] = [r["pmid"] for r in sampled]

    audit_path = processed_dir / "audit_log.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Audit log written to {audit_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to make sure nothing broke**

```bash
cd data_pipeline/pubmed_pipeline
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add sample.py
git commit -m "feat: add CLI main to sample.py with JSONL and audit log output"
```

---

## Chunk 4: README & Final Verification

### Task 9: Write README.md

**Files:**
- Create: `data_pipeline/README.md`

- [ ] **Step 1: Create README.md**

```markdown
# PubMed Pipeline (Playground)

Toy implementation of the PubMed data download & sampling pipeline.
See `docs/specs/2026-03-13-pubmed-download-pipeline-design.md` for full design.

## Setup

```bash
pip install datasets pyyaml
```

## Usage

### 1. Download from HuggingFace

```bash
# Small test (first 10,000 records)
python download_hf.py --limit 10000

# Full download (warning: slow, large output)
python download_hf.py
```

Output: `data/raw/pubmed_raw.jsonl`

### 2. Filter & Sample

```bash
python sample.py
```

Output:
- `data/processed/sampled.jsonl` — Sampled records
- `data/processed/audit_log.json` — Audit log with per-year/per-category stats

### Configuration

Edit `config.yaml` to adjust:
- `sampling.n_max` — Total sample size (default: 100,000)
- `sampling.seed` — Random seed for reproducibility
- `sampling.min_coverage.per_category_per_year` — MeSH category minimum

### Running Tests

```bash
python -m pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README for pubmed_pipeline playground"
```

### Task 10: End-to-end smoke test with --limit

- [ ] **Step 1: Verify HuggingFace record structure**

Before running the full pipeline, verify the actual record structure from HuggingFace matches our assumption. Run:

```bash
cd data_pipeline/pubmed_pipeline
python -c "
from datasets import load_dataset
ds = load_dataset('ncbi/pubmed', streaming=True, split='train')
record = next(iter(ds))
print(type(record))
print(list(record.keys()))
for k, v in record.items():
    print(f'{k}: {type(v).__name__} = {str(v)[:200]}')
"
```

Expected: record should contain a `MedlineCitation` key. Inspect the output and adjust the XML extraction path in `download_hf.py` if the structure differs from `record.get("MedlineCitation", {}).get("value", "")`.

- [ ] **Step 2: Run download with small limit**

```bash
python download_hf.py --limit 1000
```

Expected: `data/raw/pubmed_raw.jsonl` created with records. Check output count in stderr.

- [ ] **Step 3: Run sample.py**

```bash
python sample.py
```

Expected: `data/processed/sampled.jsonl` and `data/processed/audit_log.json` created. Check stderr for counts.

- [ ] **Step 4: Inspect audit log**

```bash
python -c "import json; print(json.dumps(json.load(open('data/processed/audit_log.json')), indent=2))" | head -50
```

Verify: per_year population/selected counts are present, shortfalls logged for any under-represented categories.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Final commit**

```bash
git add data_pipeline/
git commit -m "feat: pubmed_pipeline playground — complete toy implementation"
```
