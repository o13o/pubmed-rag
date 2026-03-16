"""Export pubmed_abstracts collection from Milvus to Parquet.

Usage:
    python -m src.ingestion.export_collection --output data/export/pubmed_abstracts.parquet
"""

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pymilvus import Collection, connections

from src.ingestion.milvus_setup import COLLECTION_NAME

logger = logging.getLogger(__name__)

EXPORT_FIELDS = [
    "pmid",
    "embedding",
    "title",
    "abstract_text",
    "chunk_text",
    "year",
    "journal",
    "authors",
    "mesh_terms",
    "publication_types",
    "keywords",
]

BATCH_SIZE = 1000


def export_to_parquet(
    output_path: Path,
    host: str = "localhost",
    port: str = "19530",
) -> int:
    connections.connect("default", host=host, port=port)
    col = Collection(COLLECTION_NAME)
    col.load()

    iterator = col.query_iterator(
        output_fields=EXPORT_FIELDS,
        batch_size=BATCH_SIZE,
    )

    writer = None
    total = 0

    try:
        while True:
            batch = iterator.next()
            if not batch:
                break

            table = _batch_to_table(batch)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            total += len(batch)
            logger.info("Exported %d records so far...", total)
    finally:
        iterator.close()
        if writer is not None:
            writer.close()

    logger.info("Export complete: %d records -> %s", total, output_path)
    return total


def _batch_to_table(batch: list[dict]) -> pa.Table:
    columns: dict[str, list] = {field: [] for field in EXPORT_FIELDS}
    for row in batch:
        for field in EXPORT_FIELDS:
            columns[field].append(row[field])

    schema = pa.schema([
        pa.field("pmid", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("title", pa.string()),
        pa.field("abstract_text", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("journal", pa.string()),
        pa.field("authors", pa.string()),
        pa.field("mesh_terms", pa.string()),
        pa.field("publication_types", pa.string()),
        pa.field("keywords", pa.string()),
    ])
    return pa.table(columns, schema=schema)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Export Milvus collection to Parquet")
    parser.add_argument("--output", type=Path, default=Path("data/export/pubmed_abstracts.parquet"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    args = parser.parse_args()

    total = export_to_parquet(args.output, host=args.host, port=args.port)
    print(f"Exported {total} records to {args.output}")


if __name__ == "__main__":
    main()
