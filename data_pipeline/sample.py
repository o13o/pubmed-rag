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