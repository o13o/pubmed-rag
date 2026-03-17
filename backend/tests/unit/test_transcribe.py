"""Unit tests for POST /transcribe — document upload (PDF/TXT/DOCX)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.shared.models import Citation


# --- Use the same pattern as test_api_ask.py to avoid pymilvus import ---


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


@pytest.fixture
def transcribe_module():
    """Import transcribe module after pymilvus is patched."""
    with patch("src.api.main.connections"), \
         patch("src.api.main.Collection"), \
         patch("src.api.main.LLMClient"), \
         patch("src.api.main.MeSHDatabase"), \
         patch("src.api.main.get_reranker"):
        from src.api.routes.transcribe import (
            MAX_DOCUMENT_SIZE,
            MAX_EXTRACTED_CHARS,
            MAX_FILE_SIZE,
            _extract_docx,
            _extract_pdf,
            _extract_txt,
            _summarize_document,
        )
        return {
            "MAX_DOCUMENT_SIZE": MAX_DOCUMENT_SIZE,
            "MAX_EXTRACTED_CHARS": MAX_EXTRACTED_CHARS,
            "MAX_FILE_SIZE": MAX_FILE_SIZE,
            "_extract_docx": _extract_docx,
            "_extract_pdf": _extract_pdf,
            "_extract_txt": _extract_txt,
            "_summarize_document": _summarize_document,
        }


# --- extraction helpers ---


def test_extract_txt_utf8(transcribe_module):
    assert transcribe_module["_extract_txt"](b"hello world") == "hello world"


def test_extract_txt_unicode(transcribe_module):
    text = "日本語テスト"
    assert transcribe_module["_extract_txt"](text.encode("utf-8")) == text


def test_extract_pdf(transcribe_module):
    page1 = MagicMock()
    page1.get_text.return_value = "Page one text"
    page2 = MagicMock()
    page2.get_text.return_value = "Page two text"

    mock_doc = MagicMock()
    mock_doc.__iter__ = lambda self: iter([page1, page2])

    mock_pymupdf = MagicMock()
    mock_pymupdf.open.return_value = mock_doc

    with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
        result = transcribe_module["_extract_pdf"](b"fake-pdf-bytes")

    assert result == "Page one text\nPage two text"
    mock_doc.close.assert_called_once()


def test_extract_docx(transcribe_module):
    para1 = MagicMock()
    para1.text = "First paragraph"
    para2 = MagicMock()
    para2.text = "Second paragraph"

    mock_doc = MagicMock()
    mock_doc.paragraphs = [para1, para2]

    mock_docx = MagicMock()
    mock_docx.Document.return_value = mock_doc

    with patch.dict("sys.modules", {"docx": mock_docx}):
        result = transcribe_module["_extract_docx"](b"fake-docx-bytes")

    assert result == "First paragraph\nSecond paragraph"


# --- summarize ---


def _mock_openai_module(response_text):
    """Create a mock openai module that returns response_text from chat completions."""
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_result = MagicMock()
    mock_result.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_result

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai, mock_client


def test_summarize_document_truncates_to_limit(transcribe_module):
    limit = transcribe_module["MAX_EXTRACTED_CHARS"]
    long_text = "a" * (limit + 5000)

    mock_openai, mock_client = _mock_openai_module("summarized query")

    with patch.dict("sys.modules", {"openai": mock_openai}):
        result = transcribe_module["_summarize_document"](long_text)

    assert result == "summarized query"
    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args[1]["messages"][1]["content"]
    assert len(user_content) == limit


def test_summarize_document_short_text_not_truncated(transcribe_module):
    short_text = "short medical text"

    mock_openai, mock_client = _mock_openai_module("query from short text")

    with patch.dict("sys.modules", {"openai": mock_openai}):
        result = transcribe_module["_summarize_document"](short_text)

    assert result == "query from short text"
    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args[1]["messages"][1]["content"]
    assert user_content == short_text


# --- endpoint tests via TestClient ---


def test_endpoint_rejects_unsupported_type(client):
    resp = client.post(
        "/transcribe",
        files={"file": ("test.zip", b"fake", "application/zip")},
    )
    assert resp.status_code == 415


def test_endpoint_rejects_oversized_document(client):
    from src.api.routes.transcribe import MAX_DOCUMENT_SIZE

    resp = client.post(
        "/transcribe",
        files={"file": ("big.pdf", b"x" * (MAX_DOCUMENT_SIZE + 1), "application/pdf")},
    )
    assert resp.status_code == 413


def test_endpoint_rejects_oversized_audio(client):
    from src.api.routes.transcribe import MAX_FILE_SIZE

    resp = client.post(
        "/transcribe",
        files={"file": ("big.wav", b"x" * (MAX_FILE_SIZE + 1), "audio/wav")},
    )
    assert resp.status_code == 413


def test_endpoint_rejects_empty_document(client):
    resp = client.post(
        "/transcribe",
        files={"file": ("empty.txt", b"   ", "text/plain")},
    )
    assert resp.status_code == 422


@patch("src.api.routes.transcribe._summarize_document", return_value="search query")
@patch("src.api.routes.transcribe._extract_pdf", return_value="extracted text")
def test_endpoint_pdf_success(mock_extract, mock_summarize, client):
    resp = client.post(
        "/transcribe",
        files={"file": ("paper.pdf", b"fake-pdf", "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "search query"
    assert data["media_type"] == "document"


@patch("src.api.routes.transcribe._summarize_document", return_value="search query")
def test_endpoint_txt_success(mock_summarize, client):
    resp = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"medical research text", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "search query"
    assert data["media_type"] == "document"


@patch("src.api.routes.transcribe._summarize_document", return_value="search query")
@patch("src.api.routes.transcribe._extract_docx", return_value="docx text")
def test_endpoint_docx_success(mock_extract, mock_summarize, client):
    resp = client.post(
        "/transcribe",
        files={
            "file": (
                "report.docx",
                b"fake-docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "search query"
    assert data["media_type"] == "document"
