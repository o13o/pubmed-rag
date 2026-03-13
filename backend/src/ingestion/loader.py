"""Load PubMed JSONL into Article models."""

import json
import logging
from pathlib import Path

from src.shared.models import Article

logger = logging.getLogger(__name__)


def _extract_year(publication_date: str) -> int:
    if len(publication_date) >= 4:
        return int(publication_date[:4])
    raise ValueError(f"Cannot extract year from: {publication_date}")


def load_articles(path: Path) -> list[Article]:
    articles = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            if not raw.get("abstract", "").strip():
                logger.debug("Skipping PMID %s: no abstract", raw.get("pmid"))
                continue

            try:
                article = Article(
                    pmid=raw["pmid"],
                    title=raw["title"],
                    abstract=raw["abstract"],
                    authors=raw.get("authors", []),
                    year=_extract_year(raw.get("publication_date", "")),
                    journal=raw.get("journal", ""),
                    mesh_terms=raw.get("mesh_terms", []),
                    keywords=raw.get("keywords", []),
                    publication_types=raw.get("publication_types", []),
                )
                articles.append(article)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping line %d: %s", line_num, e)

    logger.info("Loaded %d articles from %s", len(articles), path)
    return articles
