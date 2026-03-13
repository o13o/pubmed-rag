"""Create the pubmed_abstracts collection in Milvus."""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

COLLECTION_NAME = "pubmed_abstracts"
EMBEDDING_DIM = 1536


def get_schema() -> CollectionSchema:
    """Define the pubmed_abstracts collection schema per spec Section 5.

    Includes BM25 Function for hybrid search (Phase B).
    """
    fields = [
        FieldSchema(name="pmid", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(
            name="chunk_text", dtype=DataType.VARCHAR, max_length=12000,
            enable_analyzer=True, enable_match=True,
        ),
        FieldSchema(name="chunk_text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="year", dtype=DataType.INT16),
        FieldSchema(name="journal", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="mesh_terms", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="publication_types", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=5000),
    ]

    schema = CollectionSchema(fields, description="PubMed abstracts for RAG")

    # BM25 Function: chunk_text (VARCHAR) → chunk_text_sparse (SPARSE_FLOAT_VECTOR)
    bm25_function = Function(
        name="text_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["chunk_text"],
        output_field_names=["chunk_text_sparse"],
    )
    schema.add_function(bm25_function)

    return schema


def create_collection(
    host: str = "localhost", port: str = "19530", recreate: bool = False,
) -> Collection:
    """Create the collection and indexes. Idempotent unless recreate=True.

    Args:
        recreate: If True, drops existing collection and recreates with new schema.
                  Required when upgrading from Phase A schema (no BM25) to Phase B.
    """
    connections.connect("default", host=host, port=port)

    if utility.has_collection(COLLECTION_NAME):
        if recreate:
            Collection(COLLECTION_NAME).drop()
        else:
            col = Collection(COLLECTION_NAME)
            field_names = [f.name for f in col.schema.fields]
            if "chunk_text_sparse" not in field_names:
                import logging
                logging.getLogger(__name__).warning(
                    "Collection '%s' exists but lacks BM25 fields. "
                    "Run with recreate=True and re-ingest data for hybrid search.",
                    COLLECTION_NAME,
                )
            return col

    schema = get_schema()
    collection = Collection(COLLECTION_NAME, schema)

    # Dense vector index (HNSW)
    collection.create_index(
        "embedding",
        {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
    )

    # Sparse vector index (BM25)
    collection.create_index(
        "chunk_text_sparse",
        {"metric_type": "BM25", "index_type": "AUTOINDEX"},
    )

    return collection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()
    col = create_collection(recreate=args.recreate)
    print(f"Collection '{col.name}' ready. Fields: {[f.name for f in col.schema.fields]}")
