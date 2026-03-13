"""Test Milvus connectivity and collection setup.

These tests require a running Milvus instance.
Run with: uv run pytest tests/integration/ -v
"""

import pytest
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


@pytest.fixture(autouse=True)
def milvus_connection():
    connections.connect("default", host="localhost", port="19530")
    yield
    connections.disconnect("default")


def test_milvus_is_reachable():
    """Milvus should be reachable on localhost:19530."""
    assert connections.has_connection("default")


def test_create_and_drop_collection():
    """Should be able to create and drop a test collection."""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    schema = CollectionSchema(fields, description="test collection")

    col = Collection("test_connectivity", schema)
    assert utility.has_collection("test_connectivity")
    col.drop()
    assert not utility.has_collection("test_connectivity")


def test_pubmed_collection_setup():
    """Should create the pubmed_abstracts collection with correct schema."""
    from src.ingestion.milvus_setup import COLLECTION_NAME, create_collection

    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()

    col = create_collection()
    assert utility.has_collection(COLLECTION_NAME)

    field_names = [f.name for f in col.schema.fields]
    assert "pmid" in field_names
    assert "embedding" in field_names
    assert "title" in field_names
    assert "year" in field_names
    assert "mesh_terms" in field_names

    col.drop()
