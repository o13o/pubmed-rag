"""Create the pubmed_abstracts collection in Milvus."""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION_NAME = "pubmed_abstracts"
EMBEDDING_DIM = 1536


def get_schema() -> CollectionSchema:
    """Define the pubmed_abstracts collection schema per spec Section 5."""
    fields = [
        FieldSchema(name="pmid", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=12000),
        FieldSchema(name="year", dtype=DataType.INT16),
        FieldSchema(name="journal", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="mesh_terms", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="publication_types", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=5000),
    ]
    return CollectionSchema(fields, description="PubMed abstracts for RAG")


def create_collection(host: str = "localhost", port: str = "19530") -> Collection:
    """Create the collection and HNSW index. Idempotent."""
    connections.connect("default", host=host, port=port)

    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    schema = get_schema()
    collection = Collection(COLLECTION_NAME, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index("embedding", index_params)

    return collection


if __name__ == "__main__":
    col = create_collection()
    print(f"Collection '{col.name}' ready. Fields: {[f.name for f in col.schema.fields]}")
