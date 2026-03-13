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
