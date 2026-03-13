"""Chunk articles into indexable text per ADR-0001."""

import json

from src.shared.models import Article, Chunk


def chunk_article(article: Article) -> Chunk:
    parts = [f"Title: {article.title}", f"Abstract: {article.abstract}"]
    if article.mesh_terms:
        parts.append(f"MeSH: {'; '.join(article.mesh_terms)}")
    chunk_text = "\n".join(parts)

    return Chunk(
        pmid=article.pmid,
        chunk_text=chunk_text,
        title=article.title,
        abstract_text=article.abstract,
        year=article.year,
        journal=article.journal,
        authors=json.dumps(article.authors),
        mesh_terms=json.dumps(article.mesh_terms),
        keywords=json.dumps(article.keywords),
        publication_types=json.dumps(article.publication_types),
    )
