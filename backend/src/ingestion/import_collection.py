"""Import pubmed_abstracts collection from Parquet into Milvus.

Usage:
    python -m src.ingestion.import_collection --input data/export/pubmed_abstracts.parquet
"""

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq
from pymilvus import Collection

from src.ingestion.milvus_setup import COLLECTION_NAME, create_collection

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


def import_from_parquet(
    input_path: Path,
    host: str = "localhost",
    port: str = "19530",
    recreate: bool = False,
) -> int:
    collection = create_collection(host=host, port=port, recreate=recreate)

    parquet_file = pq.ParquetFile(str(input_path))
    total = 0

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        rows = _batch_to_rows(batch)
        collection.upsert(rows)
        total += len(rows)
        logger.info("Imported %d records so far...", total)

    collection.flush()
    logger.info("Import complete: %d records from %s", total, input_path)
    return total


def _batch_to_rows(batch) -> list[dict]:
    table = batch.to_pydict()
    n = len(table["pmid"])
    rows = []
    for i in range(n):
        row = {
            "pmid": table["pmid"][i],
            "embedding": table["embedding"][i],
            "title": table["title"][i],
            "abstract_text": table["abstract_text"][i],
            "chunk_text": table["chunk_text"][i],
            "year": table["year"][i],
            "journal": table["journal"][i],
            "authors": table["authors"][i],
            "mesh_terms": table["mesh_terms"][i],
            "publication_types": table["publication_types"][i],
            "keywords": table["keywords"][i],
        }
        rows.append(row)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Import Parquet into Milvus collection")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection before import")
    args = parser.parse_args()

    total = import_from_parquet(args.input, host=args.host, port=args.port, recreate=args.recreate)
    print(f"Imported {total} records from {args.input}")


if __name__ == "__main__":
    main()
