"""Tests for POST /transcribe endpoint."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.main.connections"), \
         patch("src.api.main.Collection") as mock_col, \
         patch("src.api.main.LLMClient"), \
         patch("src.api.main.MeSHDatabase"), \
         patch("src.api.main.get_reranker"):

        mock_col.return_value = MagicMock(num_entities=100)

        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


@patch("src.api.routes.transcribe._transcribe_audio")
def test_transcribe_audio(mock_transcribe, client):
    mock_transcribe.return_value = "cancer treatment options"
    audio_bytes = b"\x00\x01\x02\x03" * 100
    response = client.post(
        "/transcribe",
        files={"file": ("recording.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "cancer treatment options"
    assert data["media_type"] == "audio"
    mock_transcribe.assert_called_once()


@patch("src.api.routes.transcribe._transcribe_image")
def test_transcribe_image(mock_transcribe, client):
    mock_transcribe.return_value = "chart showing survival rates"
    # Minimal valid PNG header
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    response = client.post(
        "/transcribe",
        files={"file": ("figure.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "chart showing survival rates"
    assert data["media_type"] == "image"
    mock_transcribe.assert_called_once()


def test_transcribe_rejects_unsupported_mime(client):
    response = client.post(
        "/transcribe",
        files={"file": ("archive.zip", io.BytesIO(b"PK\x03\x04"), "application/zip")},
    )
    assert response.status_code == 415


def test_transcribe_rejects_oversized_file(client):
    big_bytes = b"\x00" * (26 * 1024 * 1024)  # 26MB > 25MB limit
    response = client.post(
        "/transcribe",
        files={"file": ("big.mp3", io.BytesIO(big_bytes), "audio/mpeg")},
    )
    assert response.status_code == 413
