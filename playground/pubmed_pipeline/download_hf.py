"""Download PubMed baseline data from NLM FTP and convert to intermediate JSONL."""

import argparse
import gzip
import json
import sys
import urllib.request
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


def extract_year(publication_date: str) -> str:
    """Extract 4-digit year from publication_date string."""
    return publication_date[:4] if len(publication_date) >= 4 else ""


FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


def list_baseline_files(base_url: str = FTP_BASE_URL) -> list[str]:
    """List .xml.gz files from NLM baseline directory."""
    import re

    resp = urllib.request.urlopen(base_url, timeout=30)
    content = resp.read().decode()
    files = sorted(set(re.findall(r"pubmed\d+n\d+\.xml\.gz", content)))
    return files


def download_and_parse_baseline_file(url: str, target_years: set[str]) -> list[dict]:
    """Download a single .xml.gz baseline file, parse MedlineCitations, return records."""
    print(f"  Downloading {url}...", file=sys.stderr)
    resp = urllib.request.urlopen(url, timeout=300)
    data = gzip.decompress(resp.read())
    xml_str = data.decode("utf-8")

    records = []
    # PubMed baseline XML has <PubmedArticleSet> root with <PubmedArticle> children
    # Each <PubmedArticle> contains a <MedlineCitation>
    root = ET.fromstring(xml_str)
    for article_el in root.findall("PubmedArticle"):
        citation_el = article_el.find("MedlineCitation")
        if citation_el is None:
            continue

        citation_xml = ET.tostring(citation_el, encoding="unicode")
        try:
            parsed = parse_medline_xml(citation_xml)
        except ET.ParseError:
            continue

        year = extract_year(parsed["publication_date"])
        if year in target_years:
            records.append(parsed)

    return records


def main():
    parser = argparse.ArgumentParser(description="Download PubMed baseline from NLM FTP")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, default=None, help="Max baseline files to process (for testing)")
    parser.add_argument("--start", type=int, default=0, help="Start from this file index (0-based)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_years = set(str(y) for y in config["years"])
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "pubmed_raw.jsonl"

    print("Listing baseline files...", file=sys.stderr)
    files = list_baseline_files()
    print(f"Found {len(files)} baseline files", file=sys.stderr)

    files = files[args.start:]
    if args.limit is not None:
        files = files[:args.limit]
    print(f"Processing {len(files)} files (start={args.start}, limit={args.limit})", file=sys.stderr)

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, filename in enumerate(files, 1):
            url = FTP_BASE_URL + filename
            try:
                records = download_and_parse_baseline_file(url, target_years)
            except Exception as e:
                print(f"  Error processing {filename}: {e}", file=sys.stderr)
                continue

            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_written += len(records)
            print(f"  [{i}/{len(files)}] {filename}: {len(records)} records (total: {total_written})", file=sys.stderr)

    print(f"Done. Written {total_written} records to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
