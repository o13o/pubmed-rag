"""Tests for embedder (OpenAI API calls mocked)."""

from unittest.mock import MagicMock, patch

from src.ingestion.embedder import generate_embeddings


def test_generate_embeddings_calls_openai():
    texts = ["Hello world", "Test text"]
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]

    with patch("src.ingestion.embedder._get_openai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        embeddings = generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536


def test_generate_embeddings_batches_large_input():
    texts = [f"text {i}" for i in range(250)]

    def make_response(n):
        mock = MagicMock()
        mock.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
        return mock

    with patch("src.ingestion.embedder._get_openai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            make_response(100), make_response(100), make_response(50),
        ]
        mock_get_client.return_value = mock_client
        embeddings = generate_embeddings(texts, batch_size=100)

    assert mock_client.embeddings.create.call_count == 3
    assert len(embeddings) == 250
